"""CPU-only protocol fixtures; these are not recordings of live inference."""
import json

import pytest

from eval.model_stream import ContextBudgetExceeded, ModelStreamError, stream_chat_completion


class Response:
    def __init__(self, lines, *, status_error=None):
        self.lines = lines
        self.status_error = status_error
        self.closed = False
        self.iter_options = None

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_lines(self, **kwargs):
        self.iter_options = kwargs
        yield from self.lines

    def close(self):
        self.closed = True


def _choice(delta=None, *, reason=None, index=0):
    return {"choices": [{"index": index, "delta": delta or {}, "finish_reason": reason}]}


def _frame(value):
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [f"data: {payload}".encode(), b""]


def _request(monkeypatch, values, **kwargs):
    lines = [line for value in values for line in _frame(value)]
    response = Response(lines)
    calls = []

    def post(url, **request):
        calls.append((url, request))
        return response

    monkeypatch.setattr("requests.post", post)
    stream = stream_chat_completion(
        "http://local-stub/v1/", "stub-model", "unused", [{"role": "user", "content": "task"}],
        **kwargs)
    return stream, response, calls


def test_stream_request_preserves_content_reasoning_usage_and_closes(monkeypatch):
    stream, response, calls = _request(monkeypatch, [
        _choice({"role": "assistant", "content": ""}),
        _choice({"reasoning": None, "reasoning_content": "first thought"}),
        _choice({"reasoning": " then another"}),
        _choice({"content": "```bash\nprintf 'café'"}),
        _choice({"content": "\n```"}, reason="stop"),
        {"choices": [], "usage": {"prompt_tokens": 21, "completion_tokens": 17,
                                  "total_tokens": 38}},
        "[DONE]",
    ], temperature=0.7, seed=42, max_tokens=87)
    assert list(stream) == [
        {"type": "delta", "content": ""},
        {"type": "delta", "content": "", "reasoning": "first thought"},
        {"type": "delta", "content": "", "reasoning": " then another"},
        {"type": "delta", "content": "```bash\nprintf 'café'"},
        {"type": "delta", "content": "\n```"},
        {"type": "usage", "usage": {"prompt_tokens": 21, "completion_tokens": 17,
                                     "total_tokens": 38}},
        {"type": "done", "finish_reason": "stop"},
    ]
    assert response.closed
    assert response.iter_options == {"chunk_size": 1, "decode_unicode": False}
    assert calls == [("http://local-stub/v1/chat/completions", {
        "headers": {"Authorization": "Bearer unused"},
        "json": {"model": "stub-model", "messages": [{"role": "user", "content": "task"}],
                 "temperature": 0.7, "seed": 42, "max_tokens": 87,
                 "stream": True, "stream_options": {"include_usage": True}},
        "stream": True, "timeout": (15, 180),
    })]


def test_comments_metadata_and_multiline_sse_are_supported(monkeypatch):
    stream, response, _ = _request(monkeypatch, [_choice(reason="stop"), "[DONE]"])
    response.lines[:0] = [
        b": heartbeat", b"", b"event: message", b"id: 1", b"retry: 1000",
        b'data: {"choices":',
        b'data: [{"index":0,"delta":{"content":"DONE"},"finish_reason":null}]}', b"",
    ]
    assert list(stream) == [{"type": "delta", "content": "DONE"},
                            {"type": "done", "finish_reason": "stop"}]
    assert response.closed


@pytest.mark.parametrize(("values", "message"), [
    (["{"], "Malformed SSE JSON"),
    ([[]], "must be an object"),
    ([{"error": {"message": "OOM"}}], "streaming error"),
    ([{"choices": None}], "zero or one"),
    ([{"choices": []}], "no usage"),
    ([_choice(index=1)], "choice index"),
    ([{"choices": [{"delta": None}]}], "delta must be an object"),
    ([_choice({"content": ["not text"]})], "content must be text"),
    ([_choice({"reasoning": 1})], "reasoning must be text"),
    ([_choice({"reasoning": "a", "reasoning_content": "b"})], "Conflicting"),
    ([_choice({"tool_calls": [{"id": "a"}]})], "Structured tool calls"),
    ([_choice(reason="stop"), _choice(reason="stop")], "duplicate finish"),
    ([_choice(reason="stop"), _choice({"content": "late"})], "after the finish"),
    ([{"choices": [], "usage": {"completion_tokens": True}}], "Invalid usage"),
    ([{"choices": [], "usage": []}], "Usage must be an object"),
    ([_choice({"content": "DONE"}), _choice(reason="stop")], "without \\[DONE\\]"),
    (["[DONE]"], "missing a finish reason"),
])
def test_protocol_failures_are_not_silent_and_close_response(monkeypatch, values, message):
    stream, response, _ = _request(monkeypatch, values)
    with pytest.raises(ModelStreamError, match=message):
        list(stream)
    assert response.closed


@pytest.mark.parametrize(("lines", "message"), [
    ([b"event: error", b"data: {}", b""], "SSE error event"),
    ([b"<html>503</html>"], "Unexpected SSE field"),
    ([b"data: \xff", b""], "UTF-8"),
    ([b"data: [DONE]"], "inside an event"),
])
def test_malformed_sse_frames_fail_closed(monkeypatch, lines, message):
    stream, response, _ = _request(monkeypatch, [])
    response.lines = lines
    with pytest.raises(ModelStreamError, match=message):
        list(stream)
    assert response.closed


def test_network_error_after_partial_output_closes_response(monkeypatch):
    def lines():
        yield from _frame(_choice({"content": "```bash\necho unsafe\n```"}))
        raise TimeoutError("stream stalled")

    stream, response, _ = _request(monkeypatch, [])
    response.lines = lines()
    assert next(stream)["type"] == "delta"
    with pytest.raises(TimeoutError, match="stream stalled"):
        next(stream)
    assert response.closed


def test_consumer_cancel_closes_owned_response(monkeypatch):
    stream, response, _ = _request(monkeypatch, [_choice({"content": "partial"})])
    assert next(stream)["content"] == "partial"
    assert not response.closed
    stream.close()
    assert response.closed


def test_http_failure_closes_response(monkeypatch):
    stream, response, _ = _request(monkeypatch, [])
    response.status_error = RuntimeError("HTTP failure")
    with pytest.raises(RuntimeError, match="HTTP failure"):
        next(stream)
    assert response.closed


def test_transport_reports_length_instead_of_inventing_stop(monkeypatch):
    stream, response, _ = _request(monkeypatch, [_choice(reason="length"), "[DONE]"])
    assert list(stream) == [{"type": "done", "finish_reason": "length"}]
    assert response.closed


CONTEXT_MESSAGE = (
    "Requested token count exceeds the model's maximum context length of 32768 tokens. "
    "You requested a total of 35528 tokens: 27336 tokens from the input messages and "
    "8192 tokens for the completion. Please reduce the number of tokens in the input "
    "messages or the completion to fit within the limit."
)


@pytest.mark.parametrize("named_event", [False, True])
def test_context_error_body_is_retained_safely_in_exception_and_artifact_string(monkeypatch, named_event):
    value = {"error": {"message": CONTEXT_MESSAGE, "type": "BadRequestError", "code": 400}}
    stream, response, _ = _request(monkeypatch, [value])
    if named_event:
        response.lines.insert(0, b"event: error")
    with pytest.raises(ModelStreamError) as caught:
        list(stream)
    error = caught.value.endpoint_error["error"]
    assert error["message"] == CONTEXT_MESSAGE
    assert error["context_budget"] == {"context_length": 32768, "requested_total_tokens": 35528,
                                       "prompt_tokens": 27336, "requested_completion_tokens": 8192}
    assert CONTEXT_MESSAGE in str(caught.value)
    assert response.closed


@pytest.mark.parametrize("message", [
    "Bearer local-secret; api_key=other-secret; prompt=private task; /private/key",
    CONTEXT_MESSAGE + "\nAuthorization: Bearer local-secret",
    {"authorization": "local-secret"},
    "local-secret" * 2000,
])
def test_unrecognized_endpoint_fields_and_text_cannot_leak_secrets(monkeypatch, message):
    stream, response, _ = _request(monkeypatch, [
        {"error": {"message": message, "type": "local-secret", "code": "other-secret",
                   "param": "/private/key", "request": {"secret": "private task"}}}])
    with pytest.raises(ModelStreamError) as caught:
        list(stream)
    persisted = str(caught.value) + json.dumps(caught.value.endpoint_error)
    for secret in ("local-secret", "other-secret", "private task", "/private/key"):
        assert secret not in persisted
    assert caught.value.endpoint_error["body_redacted"] is True
    assert len(persisted) < 1800
    assert response.closed


def test_http_error_retains_safe_server_body_and_not_exception_url(monkeypatch):
    import requests

    stream, response, _ = _request(monkeypatch, [])
    response.status_error = requests.HTTPError("400 for https://user:local-secret@host/path?token=private")
    response.status_code = 400
    response.iter_content = lambda **kwargs: iter([json.dumps({"error": {"message": CONTEXT_MESSAGE}}).encode()])
    with pytest.raises(ModelStreamError) as caught:
        list(stream)
    assert caught.value.endpoint_error["http_status"] == 400
    assert caught.value.endpoint_error["error"]["category"] == "context_length_exceeded"
    assert "local-secret" not in str(caught.value)
    assert "https://" not in str(caught.value)
    assert response.closed


def _budget_request(monkeypatch, *, prompt_tokens=27336, context_length=32768,
                    reserved_tokens=0, change=None, generation=None, messages=None):
    import copy
    from eval.sglang_budget import PROTOCOL, RENDERER_SHA256, canonical_sha256

    calls = []
    count_response = Response([])
    generation_response = Response([line for value in (generation or [_choice(reason="stop"), "[DONE]"])
                                    for line in _frame(value)])
    def post(url, **kwargs):
        calls.append((url, copy.deepcopy(kwargs)))
        if url.endswith("/sfx/chat-token-count"):
            receipt = {"protocol": PROTOCOL, "renderer_sha256": RENDERER_SHA256,
                       "server_instance_id": "a" * 32, "model": kwargs["json"]["model"],
                       "request_sha256": canonical_sha256(kwargs["json"]),
                       "prompt_sha256": "b" * 64, "prompt_tokens": prompt_tokens,
                       "context_length": context_length, "reserved_tokens": reserved_tokens,
                       "safety_margin_tokens": 1}
            if change:
                change(receipt)
            count_response.json = lambda: receipt
            return count_response
        assert url.endswith("/chat/completions")
        return generation_response
    monkeypatch.setattr("requests.post", post)
    stream = stream_chat_completion("http://127.0.0.1:30000/v1", "test-model", "unused",
                                    messages or [{"role": "user", "content": "task"}],
                                    max_tokens=8192, context_budget="sglang")
    return stream, calls, count_response, generation_response


def test_real_context_regression_caps_request_using_exact_server_count(monkeypatch):
    stream, calls, counted, generated = _budget_request(monkeypatch)
    events = list(stream)
    budget = events[0]["budget"]
    assert events[0]["type"] == "context_budget"
    assert budget["requested_max_tokens"] == 8192 and budget["effective_max_tokens"] == 5431
    assert calls[0][1]["json"]["max_tokens"] == 8192
    assert calls[1][1]["json"]["max_tokens"] == 5431
    assert 27336 + calls[1][1]["json"]["max_tokens"] < 32768
    proof = json.loads(calls[1][1]["headers"]["X-SFX-Context-Budget"])
    assert proof["max_tokens"] == 5431 and proof["prompt_sha256"] == "b" * 64
    assert len(calls) == 2 and counted.closed and generated.closed


def test_context_cap_respects_server_reserved_tokens(monkeypatch):
    stream, calls, _, _ = _budget_request(monkeypatch, prompt_tokens=32760, reserved_tokens=4)
    assert list(stream)[0]["budget"]["effective_max_tokens"] == 3
    assert calls[1][1]["json"]["max_tokens"] == 3


@pytest.mark.parametrize("tokens", [32767, 32768, 40000])
def test_no_room_is_typed_bounded_error_before_generation(monkeypatch, tokens):
    stream, calls, counted, generated = _budget_request(monkeypatch, prompt_tokens=tokens)
    event = next(stream)
    with pytest.raises(ContextBudgetExceeded) as caught:
        next(stream)
    assert caught.value.context_budget == event["budget"]
    assert caught.value.context_budget["effective_max_tokens"] == 0
    assert len(calls) == 1 and counted.closed and not generated.closed


@pytest.mark.parametrize("change", [
    lambda r: r.update(request_sha256="c" * 64),
    lambda r: r.update(model="different"),
    lambda r: r.update(renderer_sha256="c" * 64),
    lambda r: r.update(server_instance_id="not-an-instance"),
    lambda r: r.update(prompt_sha256="wrong"),
    lambda r: r.update(prompt_tokens=True),
    lambda r: r.update(prompt_tokens=-1),
    lambda r: r.update(context_length=0),
    lambda r: r.update(reserved_tokens=0.5),
    lambda r: r.update(reserved_tokens=r["context_length"]),
    lambda r: r.update(safety_margin_tokens=0),
    lambda r: r.update(unexpected="local-secret"),
])
def test_untrustworthy_preflight_never_falls_back_to_inference(monkeypatch, change):
    stream, calls, counted, _ = _budget_request(monkeypatch, change=change)
    with pytest.raises(ModelStreamError) as caught:
        list(stream)
    assert not isinstance(caught.value, ContextBudgetExceeded)
    assert "local-secret" not in str(caught.value)
    assert len(calls) == 1 and counted.closed


def test_budgeting_never_retries_endpoint_error(monkeypatch):
    stream, calls, counted, generated = _budget_request(monkeypatch, generation=[{"error": {"message": "OOM"}}])
    with pytest.raises(ModelStreamError, match="streaming error"):
        list(stream)
    assert len(calls) == 2 and counted.closed and generated.closed


def test_counted_request_is_frozen_against_caller_mutation(monkeypatch):
    messages = [{"role": "user", "content": "original"}]
    stream, calls, _, _ = _budget_request(monkeypatch, messages=messages)
    next(stream)
    messages[0]["content"] = "changed"
    list(stream)
    assert calls[1][1]["json"]["messages"] == [{"role": "user", "content": "original"}]


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "8192"])
def test_invalid_requested_completion_budget_never_posts(monkeypatch, value):
    stream, _, calls = _request(monkeypatch, [], max_tokens=value)
    with pytest.raises(ValueError, match="positive integer"):
        list(stream)
    assert calls == []
