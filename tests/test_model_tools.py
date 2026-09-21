"""CPU-only native tool protocol fixtures, not recorded model generations."""
import copy
import json

import pytest

from eval.model_stream import ModelStreamError
from eval.model_tools import (NATIVE_BASH_TOOLS, canonical_bash,
                              native_bash_messages, parse_native_action, stream_native_bash)


class Response:
    def __init__(self, values):
        self.values = values
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self, **kwargs):
        assert kwargs == {"chunk_size": 1, "decode_unicode": False}
        for value in self.values:
            if isinstance(value, Exception):
                raise value
            payload = value if isinstance(value, str) else json.dumps(value)
            yield f"data: {payload}".encode()
            yield b""

    def close(self):
        self.closed = True


def choice(delta=None, reason=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": reason}]}


def tool(name="execute_bash", arguments='{"command":"pwd"}', *, index=0, identifier="call_1"):
    return {"index": index, "id": identifier, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def request(monkeypatch, values, **kwargs):
    response, calls = Response(values), []

    def post(url, **request_kwargs):
        calls.append((url, request_kwargs))
        return response

    monkeypatch.setattr("requests.post", post)
    stream = stream_native_bash("http://stub/v1/", "stub", "unused",
                                [{"role": "user", "content": "Task"}], **kwargs)
    return stream, response, calls


def test_fragmented_call_is_only_executable_after_done_and_preserves_usage(monkeypatch):
    fragments = [
        choice({"reasoning_content": "planning"}),
        choice({"content": "I will inspect the workspace."}),
        choice({"tool_calls": [tool("execute_", '{"comm')]}),
        choice({"tool_calls": [{"index": 0, "function": {
            "name": "bash", "arguments": "fixture continuation"}}]}),
        choice(reason="tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 31, "completion_tokens": 19,
                                  "total_tokens": 50}},
        "[DONE]",
    ]
    # Use JSON encoding for the second half, avoiding fixture escape ambiguity.
    full_args = json.dumps({"command": 'printf "café"'}, ensure_ascii=False)
    fragments[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = full_args[:6]
    fragments[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = full_args[6:]
    stream, response, calls = request(monkeypatch, fragments, temperature=0.2, seed=7, max_tokens=90)
    before_done = [next(stream) for _ in range(5)]
    assert all(not event.get("content") for event in before_done)
    assert before_done[0]["reasoning"] == "planning"
    assert before_done[1]["native_content_delta"] == "I will inspect the workspace."
    assert before_done[-1] == {"type": "usage", "usage": {
        "prompt_tokens": 31, "completion_tokens": 19, "total_tokens": 50}}
    action, terminal = list(stream)
    assert action["content"] == '```bash\nprintf "café"\n```'
    assert action["native_action_ready"] is True
    assert action["native_tool_call"] == {
        "id": "call_1", "name": "execute_bash", "arguments": full_args}
    assert terminal == {"type": "done", "finish_reason": "tool_calls"}
    assert response.closed
    assert calls[0][0] == "http://stub/v1/chat/completions"
    body = calls[0][1]["json"]
    assert body["tools"] == NATIVE_BASH_TOOLS
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
    assert body["temperature"] == 0.2 and body["seed"] == 7 and body["max_tokens"] == 90
    assert body["stream_options"] == {"include_usage": True}
    assert body["no_stop_trim"] is True
    assert stream_native_bash.sfx_stream_provenance["native_request_options"] == {
        "no_stop_trim": True, "tool_choice": "required"}


@pytest.mark.parametrize("native", [True, False])
def test_finish_is_done_without_inventing_native_finish_reason(monkeypatch, native):
    delta = {"tool_calls": [tool("finish", '{"done":true}')]} if native else {"content": "DONE"}
    reason = "tool_calls" if native else "stop"
    stream, response, _ = request(monkeypatch, [choice(delta), choice(reason=reason), "[DONE]"])
    events = list(stream)
    assert [event["content"] for event in events if event.get("content")] == ["DONE"]
    assert events[-1] == {"type": "done", "finish_reason": reason}
    assert response.closed
    assert parse_native_action("DONE") is None


def test_history_round_trip_preserves_commands_and_output_and_does_not_mutate():
    output = "Output:\n  café\n```bash\nnot a command\n```\n"
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": canonical_bash("printf 'café'")},
                {"role": "user", "content": output},
                {"role": "assistant", "content": canonical_bash("pwd")},
                {"role": "user", "content": "Output:\n/workspace"}]
    original = copy.deepcopy(messages)
    converted = native_bash_messages(messages)
    assert messages == original and converted[:2] == original[:2]
    for assistant, observation, command in ((converted[2], converted[3], "printf 'café'"),
                                             (converted[4], converted[5], "pwd")):
        native = assistant["tool_calls"][0]
        assert json.loads(native["function"]["arguments"]) == {"command": command}
        assert observation["role"] == "tool" and observation["tool_call_id"] == native["id"]
    assert converted[3]["content"] == output
    assert converted[2]["tool_calls"][0]["id"] != converted[4]["tool_calls"][0]["id"]
    assert converted == native_bash_messages(messages)


@pytest.mark.parametrize("messages", [
    [{"role": "assistant", "content": canonical_bash("pwd")}],
    [{"role": "assistant", "content": "prose ```bash\npwd\n```"}],
    [{"role": "assistant", "content": canonical_bash("pwd")},
     {"role": "user", "content": "new request"}],
    [{"role": "tool", "content": "out"}], [{"role": "user", "content": None}],
])
def test_noncanonical_history_fails_closed(messages):
    with pytest.raises(ModelStreamError):
        native_bash_messages(messages)


@pytest.mark.parametrize("command", ["", " \n ", "printf '\x00'", 1, None])
def test_unrepresentable_commands_fail_closed(command):
    with pytest.raises(ModelStreamError):
        canonical_bash(command)


@pytest.mark.parametrize("command", [" pwd", "pwd\n", " \npwd\n ",
                                    "printf '```bash\\npwd\\n```'",
                                    "cat > notes <<'EOF'\n```bash\ncode\n```\nEOF\n"])
def test_native_command_exact_round_trip_including_whitespace_and_fences(command):
    assert parse_native_action(canonical_bash(command)) == command
    history = native_bash_messages([
        {"role": "assistant", "content": canonical_bash(command)},
        {"role": "user", "content": "Output:\n"}])
    arguments = history[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"command": command}


@pytest.mark.parametrize(("delta", "reason"), [
    ({"reasoning_content": "```bash\npwd\n```"}, "stop"),
    ({"content": "```bash\npwd\n```"}, "stop"),
    ({"content": "DONE but run this too"}, "stop"),
    ({"content": "DONE"}, "tool_calls"),
    ({"tool_calls": [tool()]}, "stop"),
    ({"tool_calls": [tool("other")]}, "tool_calls"),
    ({"tool_calls": [tool(arguments="{")]}, "tool_calls"),
    ({"tool_calls": [tool(arguments="[]")]}, "tool_calls"),
    ({"tool_calls": [tool(arguments='{"command":"pwd","command":"ls"}')]}, "tool_calls"),
    ({"tool_calls": [tool(arguments='{"command":"pwd","extra":true}')]}, "tool_calls"),
    ({"tool_calls": [tool(arguments='{"command":null}')]}, "tool_calls"),
    ({"tool_calls": [tool("finish", '{"command":"pwd"}')]}, "tool_calls"),
    ({"tool_calls": [tool("finish", '{}')]}, "tool_calls"),
    ({"tool_calls": [tool("finish", '{"done":false}')]}, "tool_calls"),
    ({"tool_calls": [tool("finish", '{"done":1}')]}, "tool_calls"),
    ({"tool_calls": [tool("finish", '{"done":true,"command":"pwd"}')]}, "tool_calls"),
    ({"tool_calls": [tool(identifier="")]}, "tool_calls"),
    ({"tool_calls": [tool(index=True)]}, "tool_calls"),
    ({"tool_calls": [tool(index=1)]}, "tool_calls"),
    ({"tool_calls": [tool(), tool(index=1)]}, "tool_calls"),
    ({"tool_calls": {}}, "tool_calls"),
    ({"function_call": {"name": "execute_bash", "arguments": "{}"}}, "stop"),
    ({"reasoning": "one", "reasoning_content": "two"}, "stop"),
])
def test_invalid_actions_never_emit_executable_content(monkeypatch, delta, reason):
    stream, response, _ = request(monkeypatch, [choice(delta), choice(reason=reason), "[DONE]"])
    seen = []
    with pytest.raises(ModelStreamError):
        for event in stream:
            seen.append(event)
    assert not any(event.get("content") for event in seen)
    assert response.closed


@pytest.mark.parametrize("ending", [[], [choice(reason="tool_calls")], [TimeoutError("lost")]])
def test_complete_arguments_without_complete_transport_do_not_execute(monkeypatch, ending):
    stream, response, _ = request(monkeypatch, [choice({"tool_calls": [tool()]}), *ending])
    seen = []
    with pytest.raises((ModelStreamError, TimeoutError)):
        for event in stream:
            seen.append(event)
    assert not any(event.get("content") for event in seen)
    assert response.closed


@pytest.mark.parametrize("reason", ["length", "content_filter", "error"])
def test_abnormal_native_finish_is_preserved_without_action(monkeypatch, reason):
    stream, response, _ = request(monkeypatch, [choice({"tool_calls": [tool()]}),
                                             choice(reason=reason), "[DONE]"])
    events = list(stream)
    assert not any(event.get("content") for event in events)
    assert events[-1] == {"type": "done", "finish_reason": reason}
    assert response.closed


def test_cancel_closes_response_and_does_not_wait_for_action(monkeypatch):
    stream, response, _ = request(monkeypatch, [choice({"tool_calls": [tool()]})])
    assert next(stream)["content"] == ""
    stream.close()
    assert response.closed


@pytest.mark.parametrize("extra", [
    choice({"tool_calls": [tool(identifier="different")]}),
    choice({"tool_calls": [tool(index=1)]}),
    choice({"content": "late"}), choice(reason="tool_calls"),
])
def test_no_data_or_second_finish_after_finish(monkeypatch, extra):
    stream, response, _ = request(monkeypatch, [choice({"tool_calls": [tool()]}),
                                             choice(reason="tool_calls"), extra, "[DONE]"])
    seen = []
    with pytest.raises(ModelStreamError):
        for event in stream:
            seen.append(event)
    assert not any(event.get("content") for event in seen)
    assert response.closed


def test_live_agent_native_calls_preserve_exact_command_and_history(monkeypatch):
    from eval.live_ab import run_agent

    command = " \nprintf '```bash\\nliteral\\n```'\n"
    responses = [Response([choice({"reasoning_content": "private thought"}),
                           choice({"tool_calls": [tool(arguments=json.dumps({"command": command}))]}),
                           choice(reason="tool_calls"),
                           {"choices": [], "usage": {"completion_tokens": 19}}, "[DONE]"]),
                 Response([choice({"tool_calls": [tool("finish", '{"done":true}')]}),
                           choice(reason="tool_calls"),
                           {"choices": [], "usage": {"completion_tokens": 4}}, "[DONE]"])]
    requests, executed = [], []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses[len(requests) - 1]

    monkeypatch.setattr("requests.post", post)

    def execute(actual):
        executed.append(actual)
        return "exact observation\n"

    trajectory = run_agent(execute, "synthetic task", base_url="http://stub/v1",
                           model="stub", api_key="unused", max_steps=3,
                           stream_fn=stream_native_bash)
    assert executed == [command] and trajectory["commands"] == [command]
    assert trajectory["stop_reason"] == "done"
    assert trajectory["transport"] == "native_bash_tools"
    assert trajectory["timing_source"] == "live_stream"
    assert trajectory["token_usage_source"] == "endpoint_usage"
    assert trajectory["completion_tokens"] == 23
    assert [row["finish_reason"] for row in trajectory["timings"]] == ["tool_calls", "tool_calls"]
    history = requests[1]["messages"]
    call = history[2]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"command": command}
    assert history[3] == {"role": "tool", "tool_call_id": call["id"],
                          "content": "Output:\nexact observation\n"}
    assert all(response.closed for response in responses)
