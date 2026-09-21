import copy

import pytest

from eval import live_ab
from eval.sfx_live_agent import SFXLiveAgent


BUDGET = {
    "provider": "sglang", "protocol": "sfx-chat-token-budget-v1",
    "request_sha256": "a" * 64, "server_instance_id": "b" * 32,
    "renderer_sha256": "c" * 64, "prompt_tokens": 27336,
    "reserved_tokens": 0, "context_length": 32768, "safety_margin_tokens": 1,
    "requested_max_tokens": 8192, "effective_max_tokens": 5431,
}


def drive(monkeypatch, events, **options):
    requests = []

    def transport(*args, **kwargs):
        requests.append(kwargs)
        yield from events

    monkeypatch.setattr(live_ab, "_model_stream", transport)
    result = live_ab.run_agent(
        lambda command: pytest.fail("unexpected tool execution"), "task",
        base_url="http://unit.invalid/v1", model="unit", api_key="unit", max_steps=1,
        max_tokens=8192, stream_fn=transport, context_budget="sglang", **options)
    return result, requests


def test_budget_is_persisted_separately_from_actual_token_usage(monkeypatch):
    callbacks = []
    event = {"type": "context_budget", "budget": copy.deepcopy(BUDGET)}
    result, requests = drive(monkeypatch, [
        event, {"type": "delta", "content": "DONE"},
        {"type": "usage", "usage": {"completion_tokens": 1, "prompt_tokens": 27336}},
        {"type": "done", "finish_reason": "stop"},
    ], on_stream_event=callbacks.append)
    assert requests[0]["context_budget"] == "sglang"
    assert requests[0]["max_tokens"] == result["max_tokens"] == 8192
    timing = result["timings"][0]
    assert timing["context_budget"] == BUDGET
    assert timing["context_budget_s"] >= 0
    assert timing["stream_complete"] is True
    assert timing["completion_tokens"] == result["completion_tokens"] == 1
    assert [event["event"] for event in callbacks] == [
        "model_start", "model_context_budget", "model_delta", "model_usage", "model_end"]
    event["budget"]["effective_max_tokens"] = 7
    assert timing["context_budget"]["effective_max_tokens"] == 5431
    assert callbacks[1]["raw_stream_event"]["budget"]["effective_max_tokens"] == 5431


@pytest.mark.parametrize("field,value", [
    ("effective_max_tokens", 8192), ("requested_max_tokens", 5),
    ("prompt_tokens", True), ("safety_margin_tokens", 0),
    ("reserved_tokens", -1), ("server_instance_id", "unknown"),
    ("provider", "heuristic"), ("extra_prompt", "do not persist"),
])
def test_invalid_budget_aborts_before_output_or_execution(monkeypatch, field, value):
    budget = {**BUDGET, field: value}
    with pytest.raises(live_ab.ModelStreamError) as error:
        drive(monkeypatch, [{"type": "context_budget", "budget": budget}])
    trace = error.value.sfx_live_trajectory
    assert trace["commands"] == []
    assert "context_budget" not in trace["timings"][0]
    assert "do not persist" not in str(trace)


@pytest.mark.parametrize("events", [
    [{"type": "delta", "content": "DONE"}],
    [{"type": "done", "finish_reason": "stop"}],
    [{"type": "context_budget", "budget": BUDGET}] * 2,
    [{"type": "usage", "usage": {"completion_tokens": 1}},
     {"type": "context_budget", "budget": BUDGET}],
])
def test_missing_duplicate_or_late_budget_fails_closed(monkeypatch, events):
    with pytest.raises(live_ab.ModelStreamError):
        drive(monkeypatch, events)


@pytest.mark.parametrize("stream_fn", [None, lambda *args, **kwargs: iter(())])
def test_budget_cannot_be_silently_ignored_by_another_transport(stream_fn):
    with pytest.raises(ValueError, match="live fenced"):
        live_ab.run_agent(lambda _: None, "task", base_url="unused", model="unit",
                          api_key="unit", max_steps=1, stream_fn=stream_fn,
                          context_budget="sglang")


@pytest.mark.parametrize("options", [
    {"scaffold": "native"}, {"streaming": False},
    {"model_fn": "unused:injected"}, {"stream_model_fn": "unused:injected"},
])
def test_wrapper_rejects_unsupported_budget_combination(tmp_path, options):
    with pytest.raises(ValueError, match="live fenced"):
        SFXLiveAgent(tmp_path, context_budget="sglang", **options)


def test_complete_but_length_limited_stream_aborts_and_retains_evidence(monkeypatch):
    with pytest.raises(live_ab.ModelStreamError, match="did not finish normally") as error:
        drive(monkeypatch, [
            {"type": "context_budget", "budget": BUDGET},
            {"type": "delta", "content": "```bash\necho never-executed\n```"},
            {"type": "done", "finish_reason": "length"},
        ])
    trace = error.value.sfx_live_trajectory
    assert trace["commands"] == []
    assert trace["stop_reason"] == "model_error"
    assert trace["timings"][0]["stream_complete"] is True
    assert trace["timings"][0]["finish_reason"] == "length"
    assert trace["timings"][0]["context_budget"] == BUDGET


def test_exhausted_budget_never_generates_and_preserves_typed_failure(monkeypatch):
    from eval.model_stream import ContextBudgetExceeded

    budget = {**BUDGET, "prompt_tokens": 32768, "effective_max_tokens": 0}

    def transport(*args, **kwargs):
        yield {"type": "context_budget", "budget": budget}
        raise ContextBudgetExceeded(budget)

    monkeypatch.setattr(live_ab, "_model_stream", transport)
    with pytest.raises(ContextBudgetExceeded) as error:
        live_ab.run_agent(lambda _: pytest.fail("unexpected tool"), "task",
                          base_url="http://unit.invalid/v1", model="unit", api_key="unit",
                          max_steps=1, max_tokens=8192, stream_fn=transport, context_budget="sglang")
    trace = error.value.sfx_live_trajectory
    assert trace["commands"] == []
    assert trace["model_error"]["context_budget"] == budget
    assert trace["model_error"]["error_type"] == "ContextBudgetExceeded"
    assert trace["timings"][0]["stream_complete"] is False
    assert trace["timings"][0]["first_chunk_s"] is None


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 27337, "completion_tokens": 1},
    {"prompt_tokens": 27336, "completion_tokens": 5432},
    {"prompt_tokens": 27336},
])
def test_usage_mismatch_cannot_publish_a_tool(monkeypatch, usage):
    with pytest.raises(live_ab.ModelStreamError, match="usage differs"):
        drive(monkeypatch, [
            {"type": "context_budget", "budget": BUDGET},
            {"type": "delta", "content": "```bash\necho never-executed\n```"},
            {"type": "usage", "usage": usage},
            {"type": "done", "finish_reason": "stop"},
        ])
