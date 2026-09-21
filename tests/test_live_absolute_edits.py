import asyncio
import os

import pytest

from eval import sfx_live_agent
from sfx.fork import WritePathError
from test_live_fenced_integration import LocalSandbox, live_stack, make_agent, review_greet


async def emit(agent, environment, event, **fields):
    await agent._stream_event(environment, {
        "event": event, "model_step": 0, "elapsed_s": 0.0,
        "_exec_cwd": agent.repo, **fields})


def test_absolute_stream_edit_forks_nondefault_repo_then_serves_exact_followup(
        live_stack, tmp_path, monkeypatch):
    repo, scratch, daemon = live_stack
    agent = make_agent(tmp_path, repo)
    agent._pending_edit = None
    environment = LocalSandbox(repo, "absolute-edit")
    body = "import sys\nprint('NEW')\nsys.stderr.write('ABSOLUTE_STDERR\\n')\n"
    review_greet(monkeypatch, body)
    command = f"cat  > {repo}/greet.py <<'EOF'\n{body}EOF"
    prefix = f"```bash\n{command}\n"

    async def drive():
        await agent._cli(environment, "begin", agent._session, repo, scratch, "0")
        try:
            await emit(agent, environment, "model_delta", text=prefix)
            session = daemon.sessions[agent._session]
            assert session.chain is not None and session.chain.future is not None
            await asyncio.to_thread(session.chain.future.result, 5)
            job = next(job for job in session.cache.jobs.values()
                       if job.spec_id == session.chain_spec_ids[0])
            assert job.done.is_set()
            assert job.result == ("NEW\n", "ABSOLUTE_STDERR\n", 0)
            assert session.chain.write.args == {"path": "greet.py", "contents": body}
            assert (repo / "greet.py").read_text() == "print('OLD')\n"
            assert environment.commands == []
            assert agent._pending_edit["command"] == command
            assert agent._pending_edit["confirmed"] is False

            await emit(agent, environment, "model_end", text=prefix + "```", command=command)
            result = await agent._route(environment, environment.exec, command,
                                        str(repo), None, None, None)
            assert result.return_code == 0
            result = await agent._route(environment, environment.exec, "python3 greet.py",
                                        str(repo), None, None, None)
            assert (result.stdout, result.stderr, result.return_code) == job.result
            assert agent._raw[-1]["served"] is True
            assert agent._receipts[0]["command"] == command
            assert environment.commands == [command]
            assert agent._counts["writes_fed"] == agent._counts["hits"] == 1
            assert agent._counts["authoritative"] == 1
            assert (repo / "greet.py").read_text() == body
            assert agent._early_edit_events[-1]["chain_preserved"] is True
        finally:
            await agent._cancel_pending_edit(environment, "test_end")
            await agent._cli(environment, "end", agent._session)

    asyncio.run(drive())
    assert not daemon._mutations
    assert agent._pending_edit is None
    assert agent._session not in daemon.sessions
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("link_kind", ["leaf_symlink", "ancestor_symlink", "hardlink"])
@pytest.mark.parametrize("destination", ["source", "outside"])
def test_absolute_stream_link_target_never_writes_or_serves(
        live_stack, tmp_path, link_kind, destination):
    repo, scratch, daemon = live_stack
    protected_dir = (repo if destination == "source" else tmp_path) / "protected"
    protected_dir.mkdir()
    protected = protected_dir / "keep.py"
    original = "print('KEEP')\n"
    protected.write_text(original)
    if link_kind == "ancestor_symlink":
        link = repo / "linked"
        link.symlink_to(protected_dir, target_is_directory=True)
        target = link / "keep.py"
    else:
        target = repo / "linked.py"
        if link_kind == "leaf_symlink":
            target.symlink_to(protected)
        else:
            os.link(protected, target)
    agent = make_agent(tmp_path, repo)
    agent._pending_edit = None
    environment = LocalSandbox(repo, "absolute-linked-edit")
    command = f"cat > {target} <<'EOF'\nprint('UNSAFE')\nEOF"

    async def drive():
        await agent._cli(environment, "begin", agent._session, repo, scratch, "0")
        try:
            await emit(agent, environment, "model_delta", text=f"```bash\n{command}\n")
            session = daemon.sessions[agent._session]
            assert session.chain is not None and session.chain.future is not None
            with pytest.raises(WritePathError):
                await asyncio.to_thread(session.chain.future.result, 5)
            assert not session.cache.jobs
            assert protected.read_text() == target.read_text() == original
            assert (repo / "greet.py").read_text() == "print('OLD')\n"
            assert environment.commands == []
            await emit(agent, environment, "model_abort")
            reply = await agent._cli(environment, "resolve", agent._session,
                                     sfx_live_agent._b64({
                                         "tool": "run", "args": {"cmd": "python3 greet.py"}}))
            assert reply.get("served") is False
            assert session.chain is None
            assert agent._counts["hits"] == agent._counts["authoritative"] == 0
            assert environment.commands == []
        finally:
            await agent._cancel_pending_edit(environment, "test_end")
            await agent._cli(environment, "end", agent._session)

    asyncio.run(drive())
    assert protected.read_text() == target.read_text() == original
    assert (repo / "greet.py").read_text() == "print('OLD')\n"
    assert not daemon._mutations
    assert agent._pending_edit is None
    assert not list(scratch.iterdir())
