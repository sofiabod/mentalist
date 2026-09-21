"""GET-only skips streamed write speculation, never authoritative write fences."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from eval.sfx_live_agent import SFXLiveAgent


WRITE = "cat > target.py <<'EOF'\nprint('after')\nEOF"
WRITE_ARGS = {"path": "target.py", "contents": "print('after')\n"}


def test_write_speculation_is_enabled_by_default_and_persisted(tmp_path):
    agent = SFXLiveAgent(tmp_path)
    assert agent.speculate_writes is True
    context = SimpleNamespace(metadata={})
    agent._persist(context)
    persisted = json.loads((tmp_path / "sfx-live-OFF.json").read_text())
    assert persisted["speculate_writes"] is True
    assert context.metadata["sfx_live"]["speculate_writes"] is True


@pytest.mark.parametrize(("value", "expected"), [
    (True, True), (False, False), ("true", True), ("false", False),
])
def test_constructor_parses_explicit_boolean_and_cli_values(tmp_path, value, expected):
    agent = SFXLiveAgent(tmp_path, arm="ON", speculate_writes=value)
    assert agent.speculate_writes is expected
    agent._persist(SimpleNamespace(metadata=None))
    assert json.loads((tmp_path / "sfx-live-ON.json").read_text())["speculate_writes"] is expected


@pytest.mark.parametrize("value", [None, 0, 1, "", "0", "1", "yes", "False", [], {}])
def test_constructor_rejects_ambiguous_flag_values(tmp_path, value):
    with pytest.raises(ValueError, match="speculate_writes"):
        SFXLiveAgent(tmp_path, speculate_writes=value)


class Environment:
    default_user = None

    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.events.append(("exec", command))
        self.commands.append(command)
        if self.fail:
            raise TimeoutError("authoritative write failed")
        return SimpleNamespace(stdout="", stderr="", return_code=0)


def stack(tmp_path, *, arm="ON", speculate_writes=True, fail=False):
    agent = SFXLiveAgent(tmp_path, arm=arm, speculate_writes=speculate_writes)
    agent._default_cwd = "/app"
    events = []
    environment = Environment(events, fail=fail)

    async def cli(env, operation, *args):
        payload = json.loads(base64.b64decode(args[-1]))
        events.append((operation, payload))
        if operation == "mutation_begin":
            return {"mutation_id": "write-token"}
        if operation == "mutation_end":
            return {"chain_preserved": arm == "ON" and agent.speculate_writes}
        if operation == "resolve":
            return {"served": True, "output": ["test-ok\n", "", 0]}
        return {"ok": True}

    agent._cli = cli
    return agent, environment, events


def route(agent, environment, command=WRITE, **kwargs):
    arguments = {"cwd": None, "env": None, "timeout_sec": None, "user": None,
                 **kwargs}
    return asyncio.run(agent._route(environment, environment.exec, command, **arguments))


def test_default_on_feeds_edit_and_keeps_verified_commit_acknowledgment(tmp_path):
    agent, environment, events = stack(tmp_path)
    route(agent, environment)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "feed", "exec", "mutation_end", "resolve", "report"]
    assert events[0][1] == {"write_args": WRITE_ARGS}
    assert json.loads(events[1][1]["body"]) == WRITE_ARGS
    assert events[3][1] == {"mutation_id": "write-token", "success": True}
    assert events[4][1] == {"tool": "edit", "args": WRITE_ARGS}
    assert agent._counts["writes_fed"] == 1
    assert agent._counts["authoritative"] == 1
    assert environment.commands == [WRITE]


def test_get_only_keeps_fences_and_post_call_speculation(tmp_path):
    agent, environment, events = stack(tmp_path, speculate_writes=False)
    route(agent, environment)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "exec", "mutation_end", "report"]
    assert events[0][1] == {"write_args": None}
    assert events[2][1] == {"mutation_id": "write-token", "success": True}
    assert events[3][1]["speculate"] is True
    assert events[3][1]["args"] == {"cmd": WRITE}
    assert events[3][1]["tool"] == "edit"
    assert agent._counts["writes_fed"] == 0
    assert agent._counts["authoritative"] == 1


def test_get_only_still_resolves_and_serves_follow_up_get(tmp_path):
    agent, environment, events = stack(tmp_path, speculate_writes=False)
    route(agent, environment)
    result = route(agent, environment, "python test_probe.py")
    assert result.stdout == "test-ok\n"
    assert events[-1] == ("resolve", {"tool": "run", "args": {"cmd": "python test_probe.py"}})
    assert environment.commands == [WRITE]
    assert agent._counts["hits"] == 1
    assert agent._counts["writes_fed"] == 0
    assert agent._raw[-1]["served"] is True


@pytest.mark.parametrize("speculate_writes", [True, False])
def test_off_behavior_is_unchanged_by_write_ablation(tmp_path, speculate_writes):
    agent, environment, events = stack(tmp_path, arm="OFF", speculate_writes=speculate_writes)
    route(agent, environment)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "exec", "mutation_end", "report"]
    assert events[0][1] == {"write_args": None}
    assert agent._counts["authoritative"] == 1
    assert agent._counts["writes_fed"] == 0
    assert environment.commands == [WRITE]


@pytest.mark.parametrize("speculate_writes", [True, False])
def test_write_failure_still_closes_mutation_fence(tmp_path, speculate_writes):
    agent, environment, events = stack(tmp_path, speculate_writes=speculate_writes, fail=True)
    with pytest.raises(TimeoutError, match="authoritative write failed"):
        route(agent, environment)
    assert events[0][0] == "mutation_begin"
    assert events[-1] == ("mutation_end", {"mutation_id": "write-token", "success": False})
    assert not any(operation == "report" for operation, _ in events)
    assert environment.commands == [WRITE]


@pytest.mark.parametrize("speculate_writes", [True, False])
def test_ablation_does_not_expand_nondefault_context_eligibility(tmp_path, speculate_writes):
    agent, environment, events = stack(tmp_path, speculate_writes=speculate_writes)
    route(agent, environment, cwd="/other")
    assert [operation for operation, _ in events] == [
        "mutation_begin", "exec", "mutation_end", "report"]
    assert events[0][1] == {"write_args": None}
    assert events[-1][1]["speculate"] is False
    assert agent._counts["writes_fed"] == 0


def test_parent_component_cwd_never_uses_lexical_equivalence(tmp_path):
    agent, environment, events = stack(tmp_path)
    route(agent, environment, cwd="/app/link/..")
    assert agent._counts["writes_fed"] == 0
    assert events[0] == ("mutation_begin", {"write_args": None})
    assert events[-1][1]["speculate"] is False
    assert environment.commands == [WRITE]
