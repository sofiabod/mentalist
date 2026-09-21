"""Native function-call transport for the existing single-command agent loop.

SGLang GPT-OSS requires both ``--reasoning-parser gpt-oss`` and
``--tool-call-parser gpt-oss``. Its Harmony tool parser also needs the terminal
handoff marker, so native requests retain stop tokens with ``no_stop_trim``.
This is a protocol adapter, not another agent:
only an explicit execute_bash call can become a command. Reasoning and ordinary
assistant text are never parsed as shell. A complete call, normal native finish,
and SSE [DONE] are required before emitting the canonical action.

Native argument/content chunks remain timestamped metadata, not partial Bash.
Thus first canonical-content time is action-ready time, not native token latency.
Endpoint usage and finish reasons are retained without inventing token counts.
"""
import copy
import json

from eval.model_stream import ModelStreamError, _sse_data, _usage


NATIVE_BASH_TOOLS = [
    {"type": "function", "function": {
        "name": "execute_bash",
        "description": "Execute one Bash command in the task workspace.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The Bash command to execute."}},
            "required": ["command"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Finish when the task and its submission requirements are complete.",
        "parameters": {"type": "object", "properties": {
            "done": {"type": "boolean", "const": True}},
                       "required": ["done"], "additionalProperties": False}}},
]
_BASH_PREFIX, _BASH_SUFFIX = "```bash\n", "\n```"


def canonical_bash(command):
    """Serialize exact command bytes; consume only with parse_native_action."""
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise ModelStreamError("Native Bash command must be nonempty text without NUL")
    return _BASH_PREFIX + command + _BASH_SUFFIX


def parse_native_action(text):
    """Remove only the adapter's exact wrapper, preserving shell whitespace.

    This parser is solely for validated native actions and their attributed replay;
    it must not replace the general fenced-text parser. Embedded Markdown fences
    inside a shell string or heredoc are ordinary command bytes here.
    """
    if text == "DONE":
        return None
    if (not isinstance(text, str) or not text.startswith(_BASH_PREFIX)
            or not text.endswith(_BASH_SUFFIX)):
        raise ModelStreamError("Native action is missing its exact canonical wrapper")
    command = text[len(_BASH_PREFIX):-len(_BASH_SUFFIX)]
    if canonical_bash(command) != text:
        raise ModelStreamError("Native action is not canonical")
    return command


def native_bash_messages(messages):
    """Reconstruct native call/output pairs from run_agent's canonical history.

    IDs are deterministic, request-local aliases; they are not claimed to be the
    endpoint's original IDs. Tool output text (including its Output: wrapper) is
    preserved exactly. No conversation state survives outside the supplied history.
    """
    converted, pending, call_index = [], None, 0
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ModelStreamError("Native transport requires text message history")
        role, content = message.get("role"), message["content"]
        if pending is not None:
            if role != "user" or not content.startswith("Output:\n"):
                raise ModelStreamError("Canonical tool call is missing its output message")
            converted.append({"role": "tool", "tool_call_id": pending, "content": content})
            pending = None
        elif role == "assistant":
            command = parse_native_action(content)
            if command is None:
                raise ModelStreamError("Native history cannot continue after DONE")
            pending = f"sfx_call_{call_index:06d}"
            call_index += 1
            converted.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": pending, "type": "function", "function": {
                    "name": "execute_bash", "arguments": json.dumps(
                        {"command": command}, ensure_ascii=False)}}]})
        elif role in ("system", "user"):
            converted.append(copy.deepcopy(message))
        else:
            raise ModelStreamError("Unexpected role in canonical agent history")
    if pending is not None:
        raise ModelStreamError("Canonical tool call is missing its output message")
    return converted


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelStreamError("Duplicate native tool argument key")
        result[key] = value
    return result


def _canonical_call(call):
    if (not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}
            or any(not isinstance(call[key], str) for key in call)):
        raise ModelStreamError("Native call requires text ID, name, and arguments")
    if not call["id"] or call["name"] not in ("execute_bash", "finish"):
        raise ModelStreamError("Missing native tool ID or unsupported tool name")
    try:
        arguments = json.loads(call["arguments"], object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ModelStreamError("Malformed native tool arguments") from exc
    if not isinstance(arguments, dict):
        raise ModelStreamError("Native tool arguments must be an object")
    if call["name"] == "finish":
        if set(arguments) != {"done"} or arguments["done"] is not True:
            raise ModelStreamError("finish requires exactly done=true")
        return "DONE"
    if set(arguments) != {"command"}:
        raise ModelStreamError("execute_bash requires exactly the command argument")
    return canonical_bash(arguments["command"])


def _extend_call(call, delta):
    if not isinstance(delta, dict) or type(delta.get("index")) is not int or delta["index"] != 0:
        raise ModelStreamError("Expected exactly one native tool call at index zero")
    if delta.get("type") not in (None, "function"):
        raise ModelStreamError("Native tool call must have function type")
    identifier = delta.get("id")
    if identifier is not None:
        if not isinstance(identifier, str) or not identifier:
            raise ModelStreamError("Native tool ID must be nonempty text")
        if call["id"] and call["id"] != identifier:
            raise ModelStreamError("Multiple native tool IDs are not allowed")
        call["id"] = identifier
    function = delta.get("function", {})
    if not isinstance(function, dict):
        raise ModelStreamError("Native function delta must be an object")
    for field in ("name", "arguments"):
        value = function.get(field)
        if value is not None:
            if not isinstance(value, str):
                raise ModelStreamError(f"Native tool {field} delta must be text")
            call[field] += value


def stream_native_bash(base_url, model, api_key, messages, *,
                       temperature=0.0, seed=0, max_tokens=1024):
    """Yield run_agent-compatible events, retaining native ``tool_calls`` finish.

    Ordinary final content exactly DONE is also accepted with a stop finish. A
    truncated/failed stream never emits executable canonical content, even if it
    already contained syntactically complete arguments.
    """
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": native_bash_messages(messages),
              "tools": copy.deepcopy(NATIVE_BASH_TOOLS), "tool_choice": "required",
              "parallel_tool_calls": False, "temperature": temperature,
              "seed": seed, "max_tokens": max_tokens, "stream": True,
              # Trimming <|call|> leaves Harmony tool calls buffered forever.
              "no_stop_trim": True,
              "stream_options": {"include_usage": True}},
        stream=True, timeout=(15, 180))
    finish_reason, ordinary_content, call = None, "", None
    try:
        response.raise_for_status()
        for data in _sse_data(response.iter_lines(chunk_size=1, decode_unicode=False)):
            if data == "[DONE]":
                if finish_reason is None:
                    raise ModelStreamError("SSE [DONE] is missing a finish reason")
                if finish_reason not in ("stop", "tool_calls"):
                    # Let the agent retain the actual abnormal terminal reason.
                    yield {"type": "done", "finish_reason": finish_reason}
                    return
                if call is not None:
                    if finish_reason != "tool_calls":
                        raise ModelStreamError("Native tool call is missing tool_calls finish")
                    action = _canonical_call(call)
                elif finish_reason == "stop" and ordinary_content.strip() == "DONE":
                    action = "DONE"
                else:
                    raise ModelStreamError("Response has no native action or exact DONE final")
                event = {"type": "delta", "content": action,
                         "native_action_ready": True}
                if call is not None:
                    event["native_tool_call"] = dict(call)
                yield event
                yield {"type": "done", "finish_reason": finish_reason}
                return
            try:
                chunk = json.loads(data)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ModelStreamError("Malformed SSE JSON") from exc
            if not isinstance(chunk, dict):
                raise ModelStreamError("SSE JSON must be an object")
            if "error" in chunk:
                raise ModelStreamError("Endpoint reported a streaming error")
            choices, usage = chunk.get("choices"), chunk.get("usage")
            if not isinstance(choices, list) or len(choices) > 1:
                raise ModelStreamError("Expected zero or one streaming choice")
            if not choices and usage is None:
                raise ModelStreamError("Empty streaming chunk has no usage")
            if choices:
                choice = choices[0]
                if (not isinstance(choice, dict) or type(choice.get("index", 0)) is not int
                        or choice.get("index", 0) != 0):
                    raise ModelStreamError("Unexpected streaming choice index")
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise ModelStreamError("Streaming delta must be an object")
                if delta.get("function_call") is not None:
                    raise ModelStreamError("Legacy function_call deltas are not supported")
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
                tool_deltas = delta.get("tool_calls")
                if tool_deltas is not None and (not isinstance(tool_deltas, list)
                                                 or len(tool_deltas) > 1):
                    raise ModelStreamError("Expected exactly one native tool call")
                if finish_reason is not None and (content or reasoning or tool_deltas):
                    raise ModelStreamError("Data arrived after the finish reason")
                event = {"type": "delta", "content": ""}
                if content is not None:
                    ordinary_content += content
                    event["native_content_delta"] = content
                if reasoning is not None:
                    event["reasoning"] = reasoning
                if tool_deltas:
                    if call is None:
                        call = {"id": "", "name": "", "arguments": ""}
                    _extend_call(call, tool_deltas[0])
                    event["native_tool_delta"] = copy.deepcopy(tool_deltas[0])
                if content is not None or reasoning is not None or tool_deltas:
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


stream_native_bash.sfx_stream_provenance = {
    "kind": "native_bash_protocol_adapter", "transport": "native_bash_tools",
    "action_emission": "after_complete_validated_tool_call_and_sse_done",
    "history_call_ids": "deterministic_aliases_from_canonical_history",
    "content_timing": "canonical_action_ready_not_native_token_latency",
    "action_parser": "exact_wrapper_preserving_native_command_bytes",
    "finish_arguments": {"done": True},
    "native_request_options": {"no_stop_trim": True, "tool_choice": "required"},
}
