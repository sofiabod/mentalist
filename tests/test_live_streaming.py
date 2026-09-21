"""Stream preparation may run a fork, never the authoritative edit early."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from eval.sfx_live_agent import _literal_write, _streamed_literal_write
from test_live_write_ablation import WRITE, WRITE_ARGS, stack


PREFIX = f"```bash\n{WRITE}\n"
FULL = PREFIX + "```"


@pytest.mark.parametrize("text", [PREFIX, PREFIX + "`", PREFIX + "``", FULL,
                                  FULL + "\nfinished"])
def test_complete_edit_can_be_recognized_before_generation_finishes(text):
    assert _streamed_literal_write(text) == (WRITE, WRITE_ARGS)


@pytest.mark.parametrize("text", [
    "", "thinking", "```bash\ncat > target.py <<'EOF'\nprint('after')\n",
    PREFIX.rstrip("\n"), PREFIX + "echo changed\n", FULL.replace("target.py", "../escape"),
    FULL.replace("target.py", "/absolute.py"), FULL.replace("'EOF'", "EOF"),
    "```bash\npython probe.py\n```",
])
def test_incomplete_unsafe_or_nonliteral_edits_are_not_prepared(text):
    assert _streamed_literal_write(text) is None


@pytest.mark.parametrize("repo", ["/app", "/testbed", "/workspace/project"])
def test_absolute_edit_maps_to_configured_repo_without_changing_command(repo):
    command = WRITE.replace("target.py", f"{repo}/src/target.py")
    expected = {**WRITE_ARGS, "path": "src/target.py"}
    assert _literal_write(command, repo) == expected
    assert _streamed_literal_write(f"```bash\n{command}\n", repo) == (command, expected)


@pytest.mark.parametrize("path", [
    "/elsewhere/target.py", "/application/target.py", "/app/../escape",
    "/app/sub/../../escape", "/app", "/app/", "/app/target.py/",
    "//app/target.py", "../escape", "sub/../target.py", "target.py/",
])
def test_absolute_edit_rejects_escapes_and_non_file_syntax(path):
    command = WRITE.replace("target.py", path)
    assert _literal_write(command, "/app") is None
    assert _streamed_literal_write(f"```bash\n{command}\n", "/app") is None


@pytest.mark.parametrize("repo", ["/", "app", "/app/../other"])
def test_absolute_edit_requires_an_unambiguous_non_root_repository(repo):
    assert _literal_write(WRITE.replace("target.py", "/app/target.py"), repo) is None


async def emit(agent, environment, event, **fields):
    await agent._stream_event(environment, {
        "event": event, "model_step": 0, "elapsed_s": 0.0, **fields})


def test_stream_prepares_fork_but_executes_real_edit_once_after_confirmation(tmp_path):
    agent, environment, events = stack(tmp_path)

    async def drive():
        await emit(agent, environment, "model_start")
        await emit(agent, environment, "model_delta", text=PREFIX)
        assert environment.commands == []
        assert [name for name, _ in events] == ["mutation_begin", "feed"]
        await emit(agent, environment, "model_delta", text=FULL)
        assert len(events) == 2
        await emit(agent, environment, "model_end", text=FULL, command=WRITE)
        assert environment.commands == []
        await agent._route(environment, environment.exec, WRITE, None, None, None, None)

    asyncio.run(drive())
    assert [name for name, _ in events] == [
        "mutation_begin", "feed", "exec", "mutation_end", "resolve", "report"]
    assert environment.commands == [WRITE]
    assert agent._pending_edit is None
    assert agent._counts["writes_fed"] == 1
    assert [e["event"] for e in agent._early_edit_events] == [
        "ready", "feed_started", "feed_finished", "confirmed",
        "authoritative_start", "authoritative_end"]


@pytest.mark.parametrize("event,fields", [
    ("model_abort", {}), ("model_start", {}),
    ("model_end", {"text": FULL + "extra", "command": WRITE + "\necho extra"}),
    ("model_end", {"text": "DONE", "command": None}),
    ("model_end", {"text": "unparseable", "command": None}),
])
def test_abandoned_stream_releases_fence_without_authoritative_write(tmp_path, event, fields):
    agent, environment, events = stack(tmp_path)

    async def drive():
        await emit(agent, environment, "model_delta", text=PREFIX)
        await emit(agent, environment, event, **fields)

    asyncio.run(drive())
    assert events[-1] == ("mutation_end", {"mutation_id": "write-token", "success": False})
    assert environment.commands == []
    assert agent._pending_edit is None
    assert agent._early_edit_events[-1]["event"] == "discarded"


@pytest.mark.parametrize("arm,writes", [("OFF", True), ("ON", False)])
def test_off_and_get_observe_edit_availability_without_speculating(tmp_path, arm, writes):
    agent, environment, events = stack(tmp_path, arm=arm, speculate_writes=writes)

    async def drive():
        await emit(agent, environment, "model_delta", text=PREFIX)
        await emit(agent, environment, "model_delta", text=FULL)
        await emit(agent, environment, "model_end", text=FULL, command=WRITE)

    asyncio.run(drive())
    assert events == [] and environment.commands == []
    assert [e["event"] for e in agent._early_edit_events] == ["ready"]


def test_authoritative_context_change_discards_early_candidate(tmp_path):
    agent, environment, events = stack(tmp_path)

    async def drive():
        await emit(agent, environment, "model_delta", text=PREFIX)
        await emit(agent, environment, "model_end", text=FULL, command=WRITE)
        await agent._route(environment, environment.exec, WRITE, "/other", None, None, None)

    asyncio.run(drive())
    assert events[2] == ("mutation_end", {"mutation_id": "write-token", "success": False})
    assert environment.commands == [WRITE]
    assert agent._counts["writes_fed"] == 1
    assert agent._pending_edit is None


def test_prepared_authoritative_failure_still_releases_fence(tmp_path):
    agent, environment, events = stack(tmp_path, fail=True)

    async def drive():
        await emit(agent, environment, "model_delta", text=PREFIX)
        await emit(agent, environment, "model_end", text=FULL, command=WRITE)
        with pytest.raises(TimeoutError):
            await agent._route(environment, environment.exec, WRITE, None, None, None, None)

    asyncio.run(drive())
    assert events[-1] == ("mutation_end", {"mutation_id": "write-token", "success": False})
    assert agent._pending_edit is None


def test_stream_hook_markers_are_persisted(tmp_path):
    agent, environment, _ = stack(tmp_path)
    asyncio.run(emit(agent, environment, "model_delta", text=PREFIX))
    asyncio.run(agent._cancel_pending_edit(environment, "test_done"))
    context = SimpleNamespace(metadata={})
    agent._persist(context)
    record = json.loads((tmp_path / "sfx-live-ON.json").read_text())
    assert record["early_edit_events"] == agent._early_edit_events
