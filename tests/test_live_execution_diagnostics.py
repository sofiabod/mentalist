import asyncio
import json
from types import SimpleNamespace

import pytest

from eval import live_ab, sfx_live_agent


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def run(exec_fn, **kwargs):
    return live_ab.run_agent(
        exec_fn, "CPU diagnostics", base_url="unused", model="injected",
        api_key="unused", max_steps=5, **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
def test_execution_error_retains_exact_attempt_and_prior_trajectory(monkeypatch, streaming):
    clock = Clock()
    monkeypatch.setattr(live_ab, "time", clock)
    commands = ["printf first", "pkill -f python3"]
    original = RuntimeError("mutation_end failed")
    requests, executions, callbacks = [], [], []

    def response(*args):
        index = len(requests)
        requests.append(True)
        clock.advance(0.25)
        return f"```bash\n{commands[index]}\n```"

    def stream(*args, **kwargs):
        yield {"type": "delta", "content": response(*args)}
        yield {"type": "usage", "usage": {"completion_tokens": 8}}
        yield {"type": "done", "finish_reason": "stop"}

    def execute(command):
        assert callbacks[-1]["event"] == "tool_start"
        assert callbacks[-1]["command"] == command
        executions.append(command)
        clock.advance(0.125)
        if len(executions) == 2:
            raise original
        return "first"

    config = {"stream_fn": stream} if streaming else {"model_fn": response}
    with pytest.raises(RuntimeError) as error:
        run(execute, on_stream_event=callbacks.append, **config)
    assert error.value is original
    trace = error.value.sfx_live_trajectory
    assert trace["stop_reason"] == "tool_error"
    assert "model_error" not in trace
    assert trace["commands"] == executions == commands
    assert len(requests) == 2
    assert trace["steps"] == 2 and trace["wall_s"] == 0.75
    assert trace["tool_error"] == {
        "model_step": 1, "command_index": 1, "command": commands[1],
        "phase": "execute", "error_type": "RuntimeError", "error": str(original),
    }
    first, failed = trace["timings"]
    assert first["tool_end_s"] == 0.375
    assert failed["model_start_s"] == 0.375 and failed["model_end_s"] == 0.625
    assert failed["tool_start_s"] == 0.625 and failed["tool_abort_s"] == 0.75
    assert failed["tool_end_s"] is None
    assert failed["command_index"] == 1
    assert trace["completion_tokens"] is None
    assert trace["token_usage_source"] == "unavailable"
    assert trace["mean_generation_latency_s"] is None
    assert trace["mean_request_s_per_completion_token"] is None
    if streaming:
        assert failed["stream_complete"] is True
        assert failed["usage"] == {"completion_tokens": 8}
    events = trace["stream_events"]
    assert events[-2]["event"] == "tool_start"
    assert events[-1]["event"] == "tool_abort"
    assert events[-1]["command"] == commands[1]
    assert not any(row["event"] == "tool_end" and row["model_step"] == 1 for row in events)
    assert json.loads(json.dumps(trace)) == trace


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, asyncio.CancelledError,
                                       KeyboardInterrupt, SystemExit])
def test_nonstream_error_records_tool_start_without_callback(error_type):
    original = error_type("controlled execution failure")
    executions = []

    def execute(command):
        executions.append(command)
        raise original

    with pytest.raises(error_type) as error:
        run(execute, model_fn=lambda *args: "```bash\nprintf noop\n```")
    assert error.value is original
    assert executions == ["printf noop"]
    trace = error.value.sfx_live_trajectory
    assert [row["event"] for row in trace["stream_events"]] == ["tool_start", "tool_abort"]
    assert trace["tool_error"]["error_type"] == error_type.__name__
    assert trace["timings"][0]["tool_end_s"] is None


@pytest.mark.parametrize("phase", ["tool_start", "execute", "tool_end"])
def test_callback_failures_preserve_original_error_and_execution_status(phase):
    original = RuntimeError("original phase failure")
    executions = []

    def callback(event):
        if event["event"] == phase:
            raise original
        if event["event"] == "tool_abort":
            raise ValueError("abort callback failure")

    def execute(command):
        executions.append(command)
        if phase == "execute":
            raise original
        return "ok"

    with pytest.raises(RuntimeError) as error:
        run(execute, model_fn=lambda *args: "```bash\nprintf noop\n```",
            on_stream_event=callback)
    assert error.value is original
    assert executions == ([] if phase == "tool_start" else ["printf noop"])
    trace = error.value.sfx_live_trajectory
    assert trace["tool_error"]["phase"] == phase
    assert (trace["timings"][0]["tool_end_s"] is not None) == (phase == "tool_end")
    assert trace["stream_events"][-1]["event"] == "tool_abort"
    assert any("abort callback failure" in note for note in original.__notes__)


def test_wrapped_agent_keeps_execution_failure_across_worker_thread(tmp_path, monkeypatch):
    original = RuntimeError("control socket unavailable")
    commands = []

    async def execute(command, **kwargs):
        commands.append(command)
        raise original

    def stream(*args, **kwargs):
        yield {"type": "delta", "content": "```bash\nprintf noop\n```"}
        yield {"type": "done", "finish_reason": "stop"}

    monkeypatch.setattr(live_ab, "_model_stream", stream)
    environment = SimpleNamespace(exec=execute, default_user=None)
    agent = sfx_live_agent.build_live_agent(tmp_path, environment, config={
        "base_url": "unused", "model": "injected", "api_key": "unused",
        "repo": "/app", "max_steps": 5, "streaming": True, "scaffold": "general",
    })
    context = SimpleNamespace(metadata={})
    with pytest.raises(RuntimeError) as error:
        asyncio.run(agent.run("CPU diagnostics", environment, context))
    assert error.value is original
    assert commands == ["printf noop"]
    trace = context.metadata["sfx_live_trajectory"]
    assert trace == original.sfx_live_trajectory
    assert trace["stop_reason"] == "tool_error"
    assert trace["commands"] == commands
    assert trace["scaffold"] == "general"
    assert trace["stream_events"][-1]["event"] == "tool_abort"
