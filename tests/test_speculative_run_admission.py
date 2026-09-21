import asyncio

import pytest

from eval import sfx_daemon_run
from mining.normalize import classify
from sfx.resolver import Ctx, resolve
from sfx.script_contracts import ScriptInvocationRejected
from test_live_fenced_integration import LocalSandbox, live_stack, make_agent


@pytest.mark.parametrize("command", [
    "python generated.py", "python3 generated.py", "node generated.js",
    "ruby generated.rb", "go run generated.go",
])
@pytest.mark.parametrize("lane", ["GET", "ON"])
def test_bare_interpreter_classification_does_not_authorize_speculation(
        tmp_path, monkeypatch, command, lane):
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unreviewed launch"))
    assert classify("Bash", command) == ("run", "free")
    with pytest.raises(ScriptInvocationRejected, match="explicit source-pinned read-only contract"):
        if lane == "GET":
            sfx_daemon_run._run("run", {"cmd": command})
        else:
            sfx_daemon_run._run_in_fork(tmp_path, ("run", "free", {"cmd": command}))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("observed", [False, True])
def test_unreviewed_run_prediction_is_declined_before_fork_allocation(tmp_path, monkeypatch, observed):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    context = Ctx(repo=tmp_path, session={"run": "python generated.py"} if observed else {},
                  last_edit_path="" if observed else "generated.py")
    assert resolve("run", context) == {"cmd": "python generated.py"}
    assert sfx_daemon_run._resolve_args("run", context) is None


def test_unreviewed_script_still_executes_once_authoritatively_with_mutation_fence(
        live_stack, tmp_path, monkeypatch):
    repo, scratch, daemon = live_stack
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    daemon.resolve_args = sfx_daemon_run._resolve_args
    agent = make_agent(tmp_path, repo)
    environment = LocalSandbox(repo, "unreviewed-authoritative-fallback")
    body = "from pathlib import Path\nPath('marker.txt').write_text('authoritative')\nprint('DONE')\n"
    command = "python3 greet.py"
    events = []
    daemon.log = events.append

    async def drive():
        await agent._cli(environment, "begin", agent._session, repo, scratch, "0")
        try:
            await agent._route(environment, environment.exec,
                               f"cat > greet.py <<'EOF'\n{body}EOF", None, None, None, None)
            session = daemon.sessions[agent._session]
            for event in list(session.cache.jobs.values()):
                event.done.wait(timeout=5)
            assert not (repo / "marker.txt").exists()
            session.cache.put("read", {"cmd": "cat marker.txt"}, 0, result=("STALE\n", "", 0))
            epoch_before = session.cache.epoch
            result = await agent._route(environment, environment.exec, command, None, None, None, None)
            await asyncio.to_thread(session.executor.drain)
            assert result.stdout == "DONE\n" and result.return_code == 0
            assert session.cache.epoch > epoch_before
            assert not session.cache.jobs
            assert (repo / "marker.txt").read_text() == "authoritative"
            assert environment.commands.count(command) == 1
            assert agent._counts["hits"] == 0
            assert agent._raw[-1]["served"] is False
            assert not any(event.get("ev") == "fork_execution" for event in events)
        finally:
            await agent._cli(environment, "end", agent._session)

    asyncio.run(drive())
