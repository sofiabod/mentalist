"""Capture real scheduling boundaries without making extra model/tool calls."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from eval import live_ab


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _run(model_fn, exec_fn, *, max_steps=8):
    return live_ab.run_agent(
        exec_fn, "test task", base_url="http://unused", model="stub",
        api_key="unused", max_steps=max_steps, model_fn=model_fn)


def test_timings_capture_all_model_calls_and_only_executed_tools(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    responses = ["```bash\necho first\n```", "```bash\necho second\n```", "DONE"]
    durations = [0.25, 0.5, 0.125]
    model_inputs, tool_calls = [], []

    def model_fn(base_url, model, api_key, messages):
        index = len(model_inputs)
        model_inputs.append([dict(message) for message in messages])
        clock.advance(durations[index])
        return responses[index]

    def exec_fn(command):
        tool_calls.append(command)
        clock.advance(0.125)
        return f"result:{command}"

    result = _run(model_fn, exec_fn)
    assert tool_calls == result["commands"] == ["echo first", "echo second"]
    assert result["steps"] == 2
    assert len(model_inputs) == 3
    assert result["stop_reason"] == "done"
    assert result["timing_clock"] == "monotonic_elapsed_seconds"
    assert result["wall_s"] == 1.125
    assert result["timings"] == [
        {"model_step": 0, "command_index": 0,
         "model_start_s": 0.0, "model_end_s": 0.25,
         "tool_start_s": 0.25, "tool_end_s": 0.375},
        {"model_step": 1, "command_index": 1,
         "model_start_s": 0.375, "model_end_s": 0.875,
         "tool_start_s": 0.875, "tool_end_s": 1.0},
        {"model_step": 2, "command_index": None,
         "model_start_s": 1.0, "model_end_s": 1.125,
         "tool_start_s": None, "tool_end_s": None},
    ]
    assert model_inputs[1][-2:] == [
        {"role": "assistant", "content": responses[0]},
        {"role": "user", "content": "Output:\nresult:echo first"},
    ]


@pytest.mark.parametrize(("response", "reason"), [
    ("DONE", "done"), ("  DONE with explanation", "done"),
    ("No bash command here", "unparseable"),
])
def test_terminal_response_records_model_time_without_a_tool(monkeypatch, response, reason):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    calls = []

    def model_fn(*args):
        calls.append("model")
        clock.advance(0.25)
        return response

    result = _run(model_fn, lambda command: pytest.fail("unexpected tool call"))
    assert calls == ["model"]
    assert result["stop_reason"] == reason
    assert result["steps"] == 0 and result["commands"] == []
    assert result["wall_s"] == 0.25
    assert result["timings"] == [
        {"model_step": 0, "command_index": None,
         "model_start_s": 0.0, "model_end_s": 0.25,
         "tool_start_s": None, "tool_end_s": None},
    ]


@pytest.mark.parametrize("max_steps", [0, 2])
def test_step_limit_records_reason_without_an_extra_model_call(monkeypatch, max_steps):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    model_calls, tool_calls = [], []

    def model_fn(*args):
        model_calls.append(True)
        clock.advance(0.25)
        return "```bash\necho repeat\n```"

    def exec_fn(command):
        tool_calls.append(command)
        clock.advance(0.125)
        return "repeat"

    result = _run(model_fn, exec_fn, max_steps=max_steps)
    assert result["stop_reason"] == "max_steps"
    assert result["steps"] == len(model_calls) == len(tool_calls) == max_steps
    assert len(result["timings"]) == max_steps
    assert result["wall_s"] == 0.375 * max_steps
    for index, timing in enumerate(result["timings"]):
        assert timing["model_step"] == timing["command_index"] == index
        assert (0 <= timing["model_start_s"] <= timing["model_end_s"]
                <= timing["tool_start_s"] <= timing["tool_end_s"] <= result["wall_s"])


def test_model_exception_still_propagates_without_calling_a_tool():
    def model_fn(*args):
        raise TimeoutError("model stalled")

    with pytest.raises(TimeoutError, match="model stalled"):
        _run(model_fn, lambda command: pytest.fail("unexpected tool call"))


def test_tool_exception_still_propagates_without_retry():
    calls = []

    def exec_fn(command):
        calls.append(command)
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        _run(lambda *args: "```bash\necho fail\n```", exec_fn)
    assert calls == ["echo fail"]


def test_live_wrapper_preserves_timings_in_context_and_arm_artifact(monkeypatch, tmp_path):
    from eval import sfx_live_agent

    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    responses = iter(["```bash\necho stable\n```", "DONE"])
    model_calls, tool_calls = [], []

    def model_fn(*args):
        model_calls.append(True)
        clock.advance(0.25)
        return next(responses)

    monkeypatch.setattr(sfx_live_agent, "_resolve_model_fn", lambda spec: model_fn)

    async def execute(command, user, cwd=None):
        assert cwd == "/testbed"
        tool_calls.append((command, user))
        clock.advance(0.125)
        return SimpleNamespace(return_code=0, stdout="stable\n", stderr="")

    environment = SimpleNamespace(exec=execute, default_user="agent")
    context = SimpleNamespace(metadata={"existing": "retained"})
    wrapped = sfx_live_agent.build_live_agent(
        tmp_path, environment, config={"base_url": "http://unused",
                                      "model": "stub", "api_key": "unused",
                                      "repo": "/testbed"})
    asyncio.run(wrapped.run("test task", environment, context))
    trajectory = context.metadata["sfx_live_trajectory"]
    assert trajectory["stop_reason"] == "done"
    assert trajectory["steps"] == 1
    assert len(trajectory["timings"]) == len(model_calls) == 2
    assert tool_calls == [("echo stable", "agent")]
    assert trajectory["timings"][0]["tool_end_s"] == 0.375
    assert trajectory["wall_s"] == 0.625

    agent = sfx_live_agent.SFXLiveAgent.__new__(sfx_live_agent.SFXLiveAgent)
    agent.arm, agent.depth = "ON", 1
    agent.logs_dir = tmp_path
    agent._counts, agent._receipts, agent._raw = {}, [], []
    agent._session = "timing-test"
    agent._persist(context)

    persisted = json.loads((tmp_path / "sfx-live-ON.json").read_text())
    assert persisted["sfx_live_trajectory"] == trajectory
    assert context.metadata["sfx_live"]["sfx_live_trajectory"] == trajectory
    assert context.metadata["existing"] == "retained"


def test_persist_remains_compatible_with_wrappers_without_trajectory(tmp_path):
    from eval.sfx_live_agent import SFXLiveAgent

    agent = SFXLiveAgent.__new__(SFXLiveAgent)
    agent.arm, agent.depth = "OFF", 1
    agent.logs_dir = tmp_path
    agent._counts, agent._receipts, agent._raw = {}, [], []
    agent._session = "stub-test"
    context = SimpleNamespace(metadata=None)
    agent._persist(context)
    persisted = json.loads((tmp_path / "sfx-live-OFF.json").read_text())
    assert "sfx_live_trajectory" not in persisted
    assert context.metadata["sfx_live"] == persisted


def _stream_run(stream_fn, exec_fn, **kwargs):
    return live_ab.run_agent(
        exec_fn, "test task", base_url="http://unused", model="stub-stream",
        api_key="unused", max_steps=kwargs.pop("max_steps", 8),
        stream_fn=stream_fn, **kwargs)


def test_delta_capture_has_linear_size_without_truncating_long_response():
    def capture(chunks):
        callback_lengths = []

        def source(*args, **kwargs):
            yield {"type": "delta", "content": "DONE "}
            for _ in range(chunks):
                yield {"type": "delta", "content": "x" * 64}
            yield {"type": "done", "finish_reason": "stop"}

        def callback(event):
            if event["event"] == "model_delta":
                callback_lengths.append(len(event["text"]))

        record = _stream_run(source, lambda command: pytest.fail("unexpected tool"),
                             on_stream_event=callback)
        rows = record["stream_events"]
        deltas = [row for row in rows if row["event"] == "model_delta"]
        full = "DONE " + "x" * (64 * chunks)
        assert len(deltas) == chunks + 1
        assert all("text" not in row for row in deltas)
        assert "".join(row["delta"] for row in deltas) == full
        assert rows[-1]["text"] == full
        assert callback_lengths == [5 + 64 * index for index in range(chunks + 1)]
        assert deltas[-1]["text_sha256"] == hashlib.sha256(full.encode()).hexdigest()
        return len(json.dumps(record))

    small, large = capture(512), capture(1024)
    assert large < small * 2.2


def test_compact_unicode_capture_preserves_cumulative_callback_and_exact_fingerprint():
    parts, callbacks = ["DO", "NE caf", "é", "🙂"], []

    def source(*args, **kwargs):
        for part in parts:
            yield {"type": "delta", "content": part}
        yield {"type": "done", "finish_reason": "stop"}

    record = _stream_run(source, lambda command: pytest.fail("unexpected tool"),
                         on_stream_event=callbacks.append)
    deltas = [row for row in record["stream_events"] if row["event"] == "model_delta"]
    callback_deltas = [row for row in callbacks if row["event"] == "model_delta"]
    for index, (row, callback) in enumerate(zip(deltas, callback_deltas)):
        prefix = "".join(parts[:index + 1])
        assert callback["text"] == prefix
        assert row["text_chars"] == len(prefix)
        assert row["text_sha256"] == hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        assert row["raw_stream_event"] == {"type": "delta", "content": parts[index]}
    assert record["stream_events"][-1]["text"] == "".join(parts)


def test_streamed_content_arrives_before_execution_with_usage_and_raw_timestamps(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    requests, callbacks, executed = [], [], []
    content = "```bash\nprintf 'café'\n```"

    def stream_fn(base_url, model, api_key, messages, **config):
        requests.append(([dict(row) for row in messages], config))
        if len(requests) == 1:
            clock.advance(0.1)
            yield {"type": "delta", "content": "", "reasoning": "considering the task"}
            clock.advance(0.2)
            yield {"type": "delta", "content": content[:10]}
            clock.advance(0.3)
            yield {"type": "delta", "content": content[10:]}
            clock.advance(0.4)
            yield {"type": "usage", "usage": {"completion_tokens": 17}}
        else:
            clock.advance(0.2)
            yield {"type": "delta", "content": "DONE"}
            yield {"type": "usage", "usage": {"completion_tokens": 3}}
        yield {"type": "done", "finish_reason": "stop"}

    def exec_fn(command):
        assert callbacks[-2]["event"] == "model_end"
        assert callbacks[-2]["command"] == command
        assert callbacks[-1]["event"] == "tool_start"
        executed.append(command)
        clock.advance(0.25)
        return "café"

    result = _stream_run(stream_fn, exec_fn, on_stream_event=callbacks.append,
                         temperature=0.7, seed=42, max_tokens=77)
    assert executed == result["commands"] == ["printf 'café'"]
    assert requests[0][1] == {"temperature": 0.7, "seed": 42, "max_tokens": 77}
    assert requests[1][0][-2:] == [{"role": "assistant", "content": content},
                                  {"role": "user", "content": "Output:\ncafé"}]
    assert result["timing_source"] == "injected_stream"
    assert result["token_usage_source"] == "injected_usage"
    assert result["model"] == "stub-stream" and result["temperature"] == 0.7
    assert result["seed"] == 42 and result["max_tokens"] == 77
    assert result["completion_tokens"] == 20  # Four text/reasoning chunks are NOT four tokens.
    assert result["mean_generation_latency_s"] == pytest.approx(0.6)
    assert result["mean_request_s_per_completion_token"] == pytest.approx(0.06)
    assert result["wall_s"] == pytest.approx(1.45)
    assert result["stream_events"] == [
        {key: value for key, value in event.items() if key != "_monotonic_origin"
         and not (event["event"] == "model_delta" and key == "text")}
        for event in callbacks]
    assert result["stream_event_format"] == "delta_v1"
    assert all(event["_monotonic_origin"] == callbacks[0]["_monotonic_origin"]
               for event in callbacks)
    assert [row["event"] for row in callbacks] == [
        "model_start", "model_delta", "model_delta", "model_delta", "model_usage",
        "model_end", "tool_start", "tool_end", "model_start", "model_delta",
        "model_usage", "model_end",
    ]
    deltas = [row for row in callbacks if row["event"] == "model_delta"]
    assert [row["text"] for row in deltas] == ["", content[:10], content, "DONE"]
    assert deltas[0]["raw_stream_event"]["reasoning"] == "considering the task"
    assert [row["elapsed_s"] for row in deltas] == pytest.approx([0.1, 0.3, 0.6, 1.45])
    first = result["timings"][0]
    assert first["first_token_chunk_s"] == pytest.approx(0.1)
    assert first["first_reasoning_s"] == pytest.approx(0.1)
    assert first["first_content_s"] == pytest.approx(0.3)
    assert first["model_end_s"] == pytest.approx(1.0)
    assert first["finish_reason"] == "stop"
    assert first["stream_callback_s"] == 0
    assert callbacks[-1]["command"] is None


def test_callback_overhead_is_measured_and_not_called_pure_decode_time(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)

    def stream_fn(*args, **kwargs):
        clock.advance(0.5)
        yield {"type": "delta", "content": "DONE"}
        yield {"type": "usage", "usage": {"completion_tokens": 1}}
        yield {"type": "done", "finish_reason": "stop"}

    def callback(event):
        if event["event"] in ("model_delta", "model_end"):
            clock.advance(0.25)

    result = _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"),
                         on_stream_event=callback)
    timing = result["timings"][0]
    assert timing["generation_latency_s"] == 0.75
    assert timing["stream_callback_s"] == 0.5
    assert result["wall_s"] == 1.0
    assert "not isolated GPU decoding" in result["model_latency_semantics"]


def test_stream_request_origin_excludes_model_start_control_callback(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)

    def stream_fn(*args, **kwargs):
        clock.advance(0.5)
        yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}

    def callback(event):
        if event["event"] == "model_start":
            clock.advance(0.25)

    result = _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"),
                         on_stream_event=callback)
    timing = result["timings"][0]
    assert timing["model_start_s"] == 0.0
    assert timing["stream_request_start_s"] == 0.25
    assert timing["model_end_s"] == 0.75
    assert timing["stream_callback_s"] == 0.25


@pytest.mark.parametrize("response", ["DONE", "no parseable command"])
def test_terminal_stream_delivers_model_end_without_executing(response):
    callbacks = []

    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": response}
        yield {"type": "done", "finish_reason": "stop"}

    result = _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"),
                         on_stream_event=callbacks.append)
    assert callbacks[-1]["event"] == "model_end"
    assert callbacks[-1]["text"] == response and callbacks[-1]["command"] is None
    assert result["completion_tokens"] is None
    assert result["token_usage_source"] == "unavailable"
    assert result["mean_request_s_per_completion_token"] is None


@pytest.mark.parametrize("end", [None, "length", "content_filter", "error", "tool_calls"])
def test_incomplete_or_truncated_stream_never_executes_even_a_complete_bash_block(end):
    callbacks, closed = [], []

    def stream_fn(*args, **kwargs):
        try:
            yield {"type": "delta", "content": "```bash\necho unsafe\n```"}
            if end is not None:
                yield {"type": "done", "finish_reason": end}
        finally:
            closed.append(True)

    with pytest.raises(live_ab.ModelStreamError, match="did not finish normally"):
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"),
                    on_stream_event=callbacks.append)
    assert closed == [True]
    assert callbacks[-1]["event"] == "model_abort"
    assert callbacks[-1]["text"] == "```bash\necho unsafe\n```"
    assert not any(row["event"] == "model_end" for row in callbacks)


def test_stream_exception_aborts_pending_speculation_and_propagates():
    callbacks, closed = [], []

    def stream_fn(*args, **kwargs):
        try:
            yield {"type": "delta", "content": "```bash\necho unsafe\n```"}
            raise TimeoutError("model stalled")
        finally:
            closed.append(True)

    with pytest.raises(TimeoutError, match="model stalled"):
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"),
                    on_stream_event=callbacks.append)
    assert callbacks[-1]["event"] == "model_abort"
    assert callbacks[-1]["error_type"] == "TimeoutError"
    assert closed == [True]


def test_stream_callback_failure_closes_iterator_and_requests_abort():
    events, closed = [], []

    def stream_fn(*args, **kwargs):
        try:
            yield {"type": "delta", "content": "partial"}
            pytest.fail("must not resume failed stream")
        finally:
            closed.append(True)

    def callback(event):
        events.append(event["event"])
        if event["event"] == "model_delta":
            raise RuntimeError("control RPC failed")

    with pytest.raises(RuntimeError, match="control RPC failed"):
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"), on_stream_event=callback)
    assert events == ["model_start", "model_delta", "model_abort"]
    assert closed == [True]


def test_abort_callback_failure_does_not_mask_generation_error():
    def stream_fn(*args, **kwargs):
        raise TimeoutError("model stalled")

    def callback(event):
        if event["event"] == "model_abort":
            raise RuntimeError("abort failed")

    with pytest.raises(TimeoutError, match="model stalled") as error:
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"), on_stream_event=callback)
    assert "abort failed" in error.value.__notes__[0]


@pytest.mark.parametrize("bad_event", [
    "bad", {"type": "delta", "content": 3}, {"type": "delta", "reasoning": []},
    {"type": "usage", "usage": {"completion_tokens": -1}},
    {"type": "done"}, {"type": "unknown"},
])
def test_injected_stream_schema_is_checked_too(bad_event):
    def stream_fn(*args, **kwargs):
        yield bad_event

    with pytest.raises(live_ab.ModelStreamError):
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"))


def test_data_after_injected_done_is_rejected():
    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}
        yield {"type": "delta", "content": "late"}

    with pytest.raises(live_ab.ModelStreamError, match="after stream completion"):
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"))


def test_incomplete_usage_does_not_turn_observed_chunks_into_token_counts():
    requests = []

    def stream_fn(*args, **kwargs):
        requests.append(True)
        yield {"type": "delta", "content": "```bash\necho ok\n```" if len(requests) == 1 else "DONE"}
        if len(requests) == 1:
            yield {"type": "usage", "usage": {"completion_tokens": 10}}
        yield {"type": "done", "finish_reason": "stop"}

    result = _stream_run(stream_fn, lambda cmd: "ok")
    assert result["timings"][0]["completion_tokens"] == 10
    assert result["timings"][1]["completion_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["mean_request_s_per_completion_token"] is None


def test_nonstreaming_live_route_applies_requested_sampling_parameters(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": "DONE"}}]})

    monkeypatch.setattr("requests.post", post)
    result = live_ab.run_agent(
        lambda cmd: pytest.fail("unexpected tool"), "task", base_url="http://stub/v1",
        model="stub", api_key="unused", max_steps=1, temperature=0.7, seed=9, max_tokens=67)
    assert calls[0]["json"]["temperature"] == 0.7
    assert calls[0]["json"]["seed"] == 9
    assert calls[0]["json"]["max_tokens"] == 67
    assert result["timing_source"] == "live_response"  # Transport tested with an HTTP fixture.
    assert result["completion_tokens"] is None


def test_live_stream_transport_path_records_endpoint_usage(monkeypatch):
    # A fake HTTP response exercises the real transport; no server/model runs.
    closed = []
    values = [
        {"choices": [{"index": 0, "delta": {"content": "DONE"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 2}},
    ]
    lines = [line for value in values for line in (f"data: {json.dumps(value)}".encode(), b"")]
    lines.extend([b"data: [DONE]", b""])
    response = SimpleNamespace(raise_for_status=lambda: None,
                               iter_lines=lambda **kwargs: iter(lines),
                               close=lambda: closed.append(True))
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: response)
    result = _stream_run(live_ab._model_stream, lambda cmd: pytest.fail("unexpected tool"))
    assert result["timing_source"] == "live_stream"
    assert result["token_usage_source"] == "endpoint_usage"
    assert result["completion_tokens"] == 2
    assert closed == [True]


def test_recorded_stream_keeps_capture_provenance_and_is_never_called_live():
    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": "DONE"}
        yield {"type": "usage", "usage": {"completion_tokens": 2}}
        yield {"type": "done", "finish_reason": "stop"}

    stream_fn.sfx_stream_provenance = {
        "kind": "recorded_stream_replay", "source_timing_source": "live_stream",
        "capture_sha256": "a" * 64, "source_config": {"model": "captured-model"},
    }
    result = _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"))
    assert result["timing_source"] == "recorded_stream_replay"
    assert result["token_usage_source"] == "recorded_usage"
    assert result["completion_tokens"] == 2
    assert result["stream_provenance"] == stream_fn.sfx_stream_provenance
    stream_fn.sfx_stream_provenance["source_config"]["model"] = "changed"
    assert result["stream_provenance"]["source_config"]["model"] == "captured-model"


def test_non_replay_stream_metadata_does_not_turn_injected_stream_into_live():
    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}

    stream_fn.sfx_stream_provenance = {"kind": "injected_test", "label": "CPU fixture"}
    result = _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"))
    assert result["timing_source"] == "injected_stream"
    assert result["stream_provenance"]["label"] == "CPU fixture"


@pytest.mark.parametrize("failure", ["length", "disconnect"])
def test_failed_stream_attaches_prior_tools_and_partial_raw_events_without_token_estimate(
        monkeypatch, failure):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    partial = "```bash\ncat > next.py <<'EOF'\nprint('not committed')\nEOF\n"
    executed = []

    def stream_fn(base_url, model, api_key, messages, **kwargs):
        step = sum(row["role"] == "assistant" for row in messages)
        if step == 0:
            clock.advance(0.25)
            yield {"type": "delta", "content": "```bash\necho first\n```"}
            yield {"type": "usage", "usage": {"completion_tokens": 8}}
            yield {"type": "done", "finish_reason": "stop"}
        else:
            clock.advance(0.5)
            yield {"type": "delta", "content": partial, "reasoning": "unfinished thought"}
            yield {"type": "usage", "usage": {"completion_tokens": 19}}
            if failure == "disconnect":
                raise TimeoutError("controlled disconnect")
            yield {"type": "done", "finish_reason": "length"}

    def execute(command):
        executed.append(command)
        clock.advance(0.125)
        return "first"

    expected = TimeoutError if failure == "disconnect" else live_ab.ModelStreamError
    with pytest.raises(expected) as error:
        _stream_run(stream_fn, execute, temperature=0.7, seed=5, max_tokens=99)
    trajectory = error.value.sfx_live_trajectory
    assert trajectory["stop_reason"] == "model_error"
    assert trajectory["model_error"]["error_type"] == expected.__name__
    assert trajectory["model_error"]["model_step"] == 1
    assert trajectory["commands"] == executed == ["echo first"]
    assert trajectory["steps"] == 1
    assert trajectory["wall_s"] == 0.875
    assert trajectory["temperature"] == 0.7 and trajectory["seed"] == 5
    assert trajectory["max_tokens"] == 99 and trajectory["model"] == "stub-stream"
    assert trajectory["timing_source"] == "injected_stream"
    assert trajectory["completion_tokens"] is None
    assert trajectory["token_usage_source"] == "unavailable"
    assert trajectory["mean_generation_latency_s"] is None
    assert trajectory["mean_request_s_per_completion_token"] is None
    first, failed = trajectory["timings"]
    assert first["tool_end_s"] == 0.375 and first["completion_tokens"] == 8
    assert failed["model_start_s"] == 0.375 and failed["model_abort_s"] == 0.875
    assert failed["model_end_s"] is None and failed["tool_start_s"] is None
    assert failed["usage"] == {"completion_tokens": 19}  # Raw reported usage is not discarded.
    assert failed["finish_reason"] == ("length" if failure == "length" else None)
    deltas = [event for event in trajectory["stream_events"] if event["event"] == "model_delta"]
    assert deltas[-1]["raw_stream_event"] == {
        "type": "delta", "content": partial, "reasoning": "unfinished thought"}
    assert trajectory["stream_events"][-1]["event"] == "model_abort"
    assert trajectory["stream_events"][-1]["text"] == partial
    assert all("_monotonic_origin" not in event for event in trajectory["stream_events"])


@pytest.mark.parametrize("recorded", [False, True])
def test_failed_replay_retains_provenance_without_mislabeling_it_live(recorded):
    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": "partial"}
        raise TimeoutError("injected failure")

    if recorded:
        stream_fn.sfx_stream_provenance = {
            "kind": "recorded_stream_replay", "source_timing_source": "live_stream",
            "capture_sha256": "a" * 64,
        }
    with pytest.raises(TimeoutError) as error:
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"))
    trajectory = error.value.sfx_live_trajectory
    assert trajectory["timing_source"] == ("recorded_stream_replay" if recorded else "injected_stream")
    assert trajectory["stream_provenance"] == getattr(stream_fn, "sfx_stream_provenance", None)
    assert trajectory["completion_tokens"] is None


def test_live_transport_failure_attaches_partial_endpoint_trace(monkeypatch):
    # Real transport exercised with an HTTP fixture, not a live model/server.
    closed = []
    chunk = {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
    lines = [f"data: {json.dumps(chunk)}".encode(), b""]
    response = SimpleNamespace(raise_for_status=lambda: None,
                               iter_lines=lambda **kwargs: iter(lines),
                               close=lambda: closed.append(True))
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: response)
    with pytest.raises(live_ab.ModelStreamError, match="without \\[DONE\\]") as error:
        _stream_run(live_ab._model_stream, lambda cmd: pytest.fail("unexpected tool"))
    trajectory = error.value.sfx_live_trajectory
    assert trajectory["timing_source"] == "live_stream"
    assert [event["event"] for event in trajectory["stream_events"]] == [
        "model_start", "model_delta", "model_abort"]
    assert trajectory["stream_events"][1]["delta"] == "partial"
    assert trajectory["stop_reason"] == "model_error"
    assert closed == [True]


def test_nonstream_exception_also_retains_failure_timing_and_original_error(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    original = TimeoutError("ordinary model stalled")

    def model_fn(*args):
        clock.advance(0.5)
        raise original

    with pytest.raises(TimeoutError) as error:
        _run(model_fn, lambda cmd: pytest.fail("unexpected tool"))
    assert error.value is original
    trajectory = error.value.sfx_live_trajectory
    assert trajectory["timing_source"] == "injected_response"
    assert trajectory["timings"][0]["model_end_s"] is None
    assert trajectory["timings"][0]["model_abort_s"] == 0.5
    assert trajectory["stream_events"][-1]["event"] == "model_abort"
    assert trajectory["commands"] == [] and trajectory["completion_tokens"] is None


def test_iterator_close_failure_does_not_replace_callback_failure():
    original = RuntimeError("control failure")

    def stream_fn(*args, **kwargs):
        try:
            yield {"type": "delta", "content": "partial"}
        finally:
            raise ValueError("close failed")

    def callback(event):
        if event["event"] == "model_delta":
            raise original

    with pytest.raises(RuntimeError) as error:
        _stream_run(stream_fn, lambda cmd: pytest.fail("unexpected tool"), on_stream_event=callback)
    assert error.value is original
    assert "close failed" in error.value.__notes__[0]
    assert error.value.sfx_live_trajectory["stream_events"][-1]["event"] == "model_abort"
