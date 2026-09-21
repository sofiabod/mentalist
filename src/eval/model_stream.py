"""Small, strict SSE transport for a vLLM chat-completions endpoint.

The transport yields content/reasoning deltas, endpoint usage, and a terminal
event. Chunks are not tokens. A finish reason AND the SSE [DONE] marker are
required so an interrupted response cannot silently become an executable tool.

Protocol references (including old/new reasoning field names):
https://docs.vllm.ai/en/v0.12.0/features/reasoning_outputs/
https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/
"""
import copy
import hashlib
import json
import re


class ModelStreamError(RuntimeError):
    """The response is malformed, incomplete, or unsuitable for execution."""

    def __init__(self, message, *, endpoint_error=None):
        self.endpoint_error = endpoint_error
        # Existing trajectory artifacts persist str(exc), not custom attributes.
        # Include only the already-sanitized diagnostic in that stable path.
        if endpoint_error is not None:
            message += "; endpoint_error=" + json.dumps(endpoint_error, sort_keys=True)
        super().__init__(message)


class ContextBudgetExceeded(ModelStreamError):
    """The exact rendered request leaves no room for any completion tokens."""

    def __init__(self, budget):
        self.context_budget = copy.deepcopy(budget)
        super().__init__("Model context budget exhausted; context_budget=" +
                         json.dumps(self.context_budget, sort_keys=True))


_ERROR_TEXT_LIMIT = 16384
_CONTEXT_TOTAL_ERROR = re.compile(
    r"Requested token count exceeds the model's maximum context length of "
    r"(?P<context_length>[0-9]{1,12}) tokens\. You requested a total of "
    r"(?P<requested_total_tokens>[0-9]{1,12}) tokens: "
    r"(?P<prompt_tokens>[0-9]{1,12}) tokens from the input messages and "
    r"(?P<requested_completion_tokens>[0-9]{1,12}) tokens for the completion\. "
    r"Please reduce the number of tokens in the input messages or the completion "
    r"to fit within the limit\."
)
_CONTEXT_INPUT_ERROR = re.compile(
    r"The input \((?P<prompt_tokens>[0-9]{1,12}) tokens\) is longer than the "
    r"model's context length \((?P<context_length>[0-9]{1,12}) tokens\)\."
)
_SAFE_ERROR_VALUES = {
    "type": {"BadRequest", "BadRequestError", "InvalidRequestError", "InternalServerError",
             "AuthenticationError", "PermissionDeniedError", "NotFoundError",
             "RateLimitError", "ServiceUnavailableError", "invalid_request_error",
             "server_error", "authentication_error", "rate_limit_error"},
    "code": {"context_length_exceeded", "model_not_found", "invalid_api_key",
             "rate_limit_exceeded", "invalid_request_error", "server_error"},
    "param": {"messages", "model", "max_tokens", "max_completion_tokens", "stream"},
}
_SAFE_BUDGET_MESSAGES = {
    "SFX budget authorization failed",
    "SFX exact chat budgeting is unavailable for this request",
    "SFX context budget receipt is invalid",
    "SFX context budget receipt no longer matches the serving request",
    "SFX budget endpoint requires a loopback client",
    "SFX budget endpoint requires server API-key authentication",
    "SFX budget counting requires the supported text-only serving configuration",
    "SFX budget server limits are invalid",
}


def _endpoint_error_diagnostic(value, *, http_status=None):
    """Retain a safe error-body projection, never arbitrary endpoint text.

    Endpoint errors can echo prompts, auth headers, file paths, or credentials.
    A regex denylist cannot reliably remove those. Preserve allowlisted protocol
    fields and exact known numeric context errors; fingerprint/redact everything
    else. The original body is deliberately not attached to the exception.
    """
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoded = raw.encode("utf-8", errors="replace")
    diagnostic = {"body_sha256": hashlib.sha256(encoded).hexdigest(),
                  "body_bytes": len(encoded), "body_hash_scope": "utf8_error_data",
                  "body_redacted": True}
    if type(http_status) is int and 100 <= http_status <= 599:
        diagnostic["http_status"] = http_status
    if len(encoded) > _ERROR_TEXT_LIMIT:
        diagnostic["reason"] = "oversized_error_body_redacted"
        return diagnostic
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = {"message": value}
    if isinstance(value, dict) and "error" in value:
        value = value["error"]
    if isinstance(value, str):
        value = {"message": value}
    if not isinstance(value, dict):
        diagnostic["reason"] = "unrecognized_error_body_redacted"
        return diagnostic
    safe = {}
    for field, allowed in _SAFE_ERROR_VALUES.items():
        item = value.get(field)
        if isinstance(item, str) and item in allowed:
            safe[field] = item
        elif field == "code" and type(item) is int and 100 <= item <= 599:
            safe[field] = item
    message = value.get("message")
    match = None
    if isinstance(message, str):
        match = _CONTEXT_TOTAL_ERROR.fullmatch(message) or _CONTEXT_INPUT_ERROR.fullmatch(message)
    if match:
        safe["category"] = "context_length_exceeded"
        safe["context_budget"] = {key: int(number) for key, number in match.groupdict().items()}
        safe["message"] = message  # Full-match grammar permits fixed text and numbers only.
    elif message in ("OOM", "CUDA out of memory", "CUDA out of memory."):
        safe["category"] = "out_of_memory"
        safe["message"] = message
    elif isinstance(message, str) and message in _SAFE_BUDGET_MESSAGES:
        safe["category"] = "context_budget_preflight_error"
        safe["message"] = message
    else:
        safe["message"] = "[unrecognized endpoint message redacted]"
    diagnostic["error"] = safe
    return diagnostic


def _http_status(response):
    """Wrap real HTTP errors without persisting their credential-bearing URL."""
    import requests

    try:
        response.raise_for_status()
    except requests.HTTPError:
        # Read only a bounded error body; never use response.text/content, which
        # could consume an unbounded streaming response into diagnostic memory.
        body = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    body.extend(chunk[:_ERROR_TEXT_LIMIT + 1 - len(body)])
                if len(body) > _ERROR_TEXT_LIMIT:
                    break
        except Exception:
            body = bytearray()
        diagnostic = _endpoint_error_diagnostic(body.decode("utf-8", errors="replace"),
                                                http_status=getattr(response, "status_code", None))
        if len(body) > _ERROR_TEXT_LIMIT:
            diagnostic.update(body_truncated=True, body_hash_scope="captured_utf8_error_data_prefix")
        raise ModelStreamError("Endpoint returned an HTTP error", endpoint_error=diagnostic) from None


def _context_budget_preflight(base_url, headers, payload):
    """Get exact same-server rendering counts, never an estimated local count."""
    import requests
    from eval.sglang_budget import (HEADER, PROTOCOL, RENDERER_SHA256, ROUTE,
                                   _RECEIPT_FIELDS, canonical_sha256)

    endpoint = base_url.rstrip("/").removesuffix("/v1") + ROUTE
    response = requests.post(endpoint, headers=headers, json=payload, timeout=(15, 30))
    try:
        _http_status(response)
        try:
            receipt = response.json()
        except (TypeError, ValueError):
            raise ModelStreamError("Invalid context budget response JSON") from None
    finally:
        response.close()
    if (type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS | {"request_sha256"}
            or receipt["protocol"] != PROTOCOL or receipt["renderer_sha256"] != RENDERER_SHA256
            or receipt["model"] != payload["model"]
            or receipt["request_sha256"] != canonical_sha256(payload)
            or not isinstance(receipt["server_instance_id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", receipt["server_instance_id"]) is None
            or not isinstance(receipt["prompt_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt["prompt_sha256"]) is None):
        raise ModelStreamError("Context budget response does not match the requested model and input")
    for name in ("prompt_tokens", "reserved_tokens", "context_length", "safety_margin_tokens"):
        if type(receipt[name]) is not int or receipt[name] < 0:
            raise ModelStreamError("Context budget response contains invalid token limits")
    if (receipt["context_length"] <= 1 or receipt["safety_margin_tokens"] != 1
            or receipt["reserved_tokens"] >= receipt["context_length"]):
        raise ModelStreamError("Context budget response contains invalid token limits")
    available = receipt["context_length"] - receipt["prompt_tokens"] - receipt["reserved_tokens"] - 1
    effective = min(payload["max_tokens"], max(0, available))
    budget = {"provider": "sglang", **{key: receipt[key] for key in
              ("protocol", "request_sha256", "server_instance_id", "renderer_sha256",
               "prompt_tokens", "reserved_tokens", "context_length", "safety_margin_tokens")},
              "requested_max_tokens": payload["max_tokens"], "effective_max_tokens": effective}
    proof = {key: receipt[key] for key in _RECEIPT_FIELDS}
    proof["max_tokens"] = effective
    return budget, {**headers, HEADER: json.dumps(proof, sort_keys=True, separators=(",", ":"))}


def _sse_data(lines):
    """Read complete SSE frames, allowing comments and standard metadata."""
    data, event_name = [], None
    for line in lines:
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ModelStreamError("SSE response is not valid UTF-8") from exc
        if not isinstance(line, str):
            raise ModelStreamError("SSE line must be text")
        if not line:
            if event_name == "error":
                raise ModelStreamError("Endpoint sent an SSE error event",
                                       endpoint_error=_endpoint_error_diagnostic("\n".join(data)))
            if data:
                yield "\n".join(data)
            data, event_name = [], None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data.append(value)
        elif field == "event":
            event_name = value
        elif field not in ("id", "retry"):
            raise ModelStreamError(f"Unexpected SSE field: {field!r}")
    if data or event_name:
        raise ModelStreamError("SSE response ended inside an event")


def _usage(value):
    if not isinstance(value, dict):
        raise ModelStreamError("Usage must be an object")
    for name in ("completion_tokens", "prompt_tokens", "total_tokens"):
        if name in value and (type(value[name]) is not int or value[name] < 0):
            raise ModelStreamError(f"Invalid usage.{name}")
    return value


def stream_chat_completion(base_url, model, api_key, messages, *,
                           temperature=0.0, seed=0, max_tokens=1024, context_budget=None):
    """Yield delta/usage/done dictionaries; always close the owned HTTP response.

    The caller decides which terminal reasons it accepts. In particular, the
    agent loop rejects ``length`` rather than executing a truncated generation.
    ``context_budget='sglang'`` requires the pinned process-local counting
    extension. It caps completion tokens only; it never truncates prompts or
    retries. None preserves legacy transports without claiming budget safety.
    """
    import requests

    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if context_budget not in (None, "sglang"):
        raise ValueError("context_budget must be None or 'sglang'")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = copy.deepcopy({"model": model, "messages": messages, "temperature": temperature,
                             "seed": seed, "max_tokens": max_tokens, "stream": True,
                             "stream_options": {"include_usage": True}})
    if context_budget == "sglang":
        budget, headers = _context_budget_preflight(base_url, headers, payload)
        yield {"type": "context_budget", "budget": copy.deepcopy(budget)}
        if budget["effective_max_tokens"] == 0:
            raise ContextBudgetExceeded(budget)
        payload["max_tokens"] = budget["effective_max_tokens"]
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers, json=payload,
        stream=True, timeout=(15, 180))
    finish_reason = None
    try:
        _http_status(response)
        # requests' default line buffer can hide short deltas until more bytes
        # arrive. A one-byte read size exposes each complete SSE line promptly.
        lines = response.iter_lines(chunk_size=1, decode_unicode=False)
        for data in _sse_data(lines):
            if data == "[DONE]":
                if finish_reason is None:
                    raise ModelStreamError("SSE [DONE] is missing a finish reason")
                yield {"type": "done", "finish_reason": finish_reason}
                return
            try:
                chunk = json.loads(data)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ModelStreamError("Malformed SSE JSON") from exc
            if not isinstance(chunk, dict):
                raise ModelStreamError("SSE JSON must be an object")
            if "error" in chunk:
                raise ModelStreamError("Endpoint reported a streaming error",
                                       endpoint_error=_endpoint_error_diagnostic(data))
            choices = chunk.get("choices")
            if not isinstance(choices, list) or len(choices) > 1:
                raise ModelStreamError("Expected zero or one streaming choice")
            usage = chunk.get("usage")
            if not choices and usage is None:
                raise ModelStreamError("Empty streaming chunk has no usage")
            if choices:
                choice = choices[0]
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise ModelStreamError("Unexpected streaming choice index")
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise ModelStreamError("Streaming delta must be an object")
                if delta.get("tool_calls") or delta.get("function_call"):
                    raise ModelStreamError("Structured tool calls are not supported by this bash agent")
                content = delta.get("content")
                reasoning = delta.get("reasoning")
                if reasoning is None:
                    reasoning = delta.get("reasoning_content")
                if (delta.get("reasoning") is not None
                        and delta.get("reasoning_content") is not None
                        and delta["reasoning"] != delta["reasoning_content"]):
                    raise ModelStreamError("Conflicting reasoning delta fields")
                for name, value in (("content", content), ("reasoning", reasoning)):
                    if value is not None and not isinstance(value, str):
                        raise ModelStreamError(f"Streaming {name} must be text")
                if finish_reason is not None and (content or reasoning):
                    raise ModelStreamError("Text arrived after the finish reason")
                if content is not None or reasoning is not None:
                    event = {"type": "delta", "content": content or ""}
                    if reasoning is not None:
                        event["reasoning"] = reasoning
                    yield event
                reason = choice.get("finish_reason")
                if reason is not None:
                    if not isinstance(reason, str) or not reason or finish_reason is not None:
                        raise ModelStreamError("Invalid or duplicate finish reason")
                    finish_reason = reason
            if usage is not None:
                yield {"type": "usage", "usage": _usage(usage)}
        raise ModelStreamError("SSE response ended without [DONE]")
    finally:
        response.close()
