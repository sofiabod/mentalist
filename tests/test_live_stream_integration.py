"""Real-daemon CPU wiring checks, not measurements of natural model overlap.

The injected stream waits for fork completion at an explicit test barrier. This
proves ordering and byte preservation without an arbitrary sleep or a live GPU.
"""
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import sfx_live_agent
from eval.live_ab import ModelStreamError
from test_live_fenced_integration import LocalSandbox, live_stack, make_agent, review_greet


OLD = "print('OLD')\n"
BODY = "import sys\nprint('NEW')\nsys.stderr.write('FORK_STDERR\\n')\n"
EDIT = f"cat > greet.py <<'EOF'\n{BODY}EOF"
PREFIX = f"```bash\n{EDIT}\n"
FULL = PREFIX + "```"
FOLLOWUP = "python3 greet.py"


def _live_agent(tmp_path, repo, monkeypatch, stream_fn):
    agent = make_agent(tmp_path, repo)
    environment = LocalSandbox(repo, "controlled-stream-fixture")
    agent.wrapped = "eval.sfx_live_agent:build_live_agent"
    agent.artifacts = []
    agent.live_config = {
        "base_url": "unused", "model": "CPU-injected-stream", "api_key": "unused",
        "max_steps": 5, "streaming": True, "stream_model_fn": "test:stream",
    }
    stream_fn.sfx_stream_provenance = {
        "kind": "injected_test", "label": "controlled CPU fork-completion barrier"}
    resolve = sfx_live_agent._resolve_model_fn
    monkeypatch.setattr(sfx_live_agent, "_resolve_model_fn",
                        lambda spec: stream_fn if spec == "test:stream" else resolve(spec))
    return agent, environment, SimpleNamespace(metadata={})


def _completed_early_job(agent, repo, daemon):
    session = daemon.sessions[agent._session]
    chain = session.chain
    assert chain is not None and chain.future is not None
    chain.future.result(timeout=5)
    job = next(job for job in session.cache.jobs.values()
               if job.spec_id == session.chain_spec_ids[0])
    assert job.done.is_set()
    assert job.result == ("NEW\n", "FORK_STDERR\n", 0)
    assert (repo / "greet.py").read_text() == OLD
    assert agent._pending_edit is not None and not agent._pending_edit["confirmed"]
    assert all(event["event"] != "confirmed" for event in agent._early_edit_events)
    return session, job


@pytest.mark.parametrize("different_workdir", [False, True])
def test_real_stream_fork_runs_before_model_end_then_exact_followup_is_served(
        live_stack, tmp_path, monkeypatch, different_workdir):
    repo, _, daemon = live_stack
    review_greet(monkeypatch, BODY)
    observations = []
    run_in_fork = daemon.fs_substrate._run_in_fork

    def observe(fork_path, hop):
        result, status = run_in_fork(fork_path, hop)
        observations.append({"fork_path": Path(fork_path),
                             "real_contents": (repo / "greet.py").read_text(),
                             "fork_contents": (Path(fork_path) / "greet.py").read_text(),
                             "result": result})
        return result, status

    monkeypatch.setattr(daemon.fs_substrate, "_run_in_fork", observe)

    def stream_fn(base_url, model, api_key, messages, **kwargs):
        step = sum(row["role"] == "assistant" for row in messages)
        if step == 0:
            yield {"type": "delta", "content": PREFIX}
            _completed_early_job(agent, repo, daemon)
            assert environment.commands == ["pwd"]
            assert len(observations) == 1
            assert observations[0]["fork_path"] != repo
            assert observations[0]["real_contents"] == OLD
            assert observations[0]["fork_contents"] == BODY
            yield {"type": "delta", "content": "```"}
        elif step == 1:
            assert (repo / "greet.py").read_text() == BODY
            yield {"type": "delta", "content": f"```bash\n{FOLLOWUP}\n```"}
        else:
            yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}

    agent, environment, context = _live_agent(tmp_path, repo, monkeypatch, stream_fn)
    if different_workdir:
        # Harbor's image default need not be the selected repository. Both the
        # streamed edit and later command must still use the configured repo.
        environment.root = tmp_path
    asyncio.run(agent.run("CPU stream integration", environment, context))
    assert agent._default_cwd == str(tmp_path if different_workdir else repo)
    assert environment.commands == ["pwd", EDIT]
    assert agent._counts["authoritative"] == 1
    assert agent._counts["writes_fed"] == agent._counts["hits"] == 1
    assert agent._raw[-1] == {
        "index": 1, "command": FOLLOWUP, "served": True,
        "stdout": "NEW\n", "stderr": "FORK_STDERR\n", "returncode": 0,
    }
    assert agent._raw[-1]["stdout"] == observations[0]["result"][0]
    assert agent._raw[-1]["stderr"] == observations[0]["result"][1]
    assert (repo / "greet.py").read_text() == BODY
    assert not daemon._mutations and agent._pending_edit is None
    assert agent._session not in daemon.sessions
    events = agent._early_edit_events
    assert [event["event"] for event in events] == [
        "ready", "feed_started", "feed_finished", "confirmed",
        "authoritative_start", "authoritative_end",
    ]
    assert events[-1]["chain_preserved"] is True
    trajectory = context.metadata["sfx_live_trajectory"]
    assert trajectory["commands"] == [EDIT, FOLLOWUP]
    assert trajectory["timing_source"] == "injected_stream"
    assert trajectory["stream_provenance"]["kind"] == "injected_test"
    assert trajectory["timings"][0]["model_end_s"] >= events[2]["elapsed_s"]
    assert context.metadata["sfx_live"]["completed"] is True


@pytest.mark.parametrize(("failure", "exception"), [
    ("length", ModelStreamError), ("disconnect", TimeoutError),
])
def test_real_stream_abort_discards_completed_fork_without_touching_repo(
        live_stack, tmp_path, monkeypatch, failure, exception):
    repo, _, daemon = live_stack
    review_greet(monkeypatch, BODY)
    early = []

    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": PREFIX}
        early.append(_completed_early_job(agent, repo, daemon))
        if failure == "disconnect":
            raise TimeoutError("controlled stream disconnect")
        yield {"type": "done", "finish_reason": "length"}

    agent, environment, context = _live_agent(tmp_path, repo, monkeypatch, stream_fn)
    with pytest.raises(exception):
        asyncio.run(agent.run("CPU interrupted stream", environment, context))
    assert len(early) == 1
    session, old_job = early[0]
    assert session.chain is None
    assert old_job.spec_id in session.ledger._terminated
    assert all(job is not old_job for job in session.cache.jobs.values())
    assert (repo / "greet.py").read_text() == OLD
    assert environment.commands == ["pwd"]
    assert agent._counts["authoritative"] == agent._counts["hits"] == 0
    assert not daemon._mutations and agent._pending_edit is None
    assert agent._session not in daemon.sessions
    assert agent._early_edit_events[-1]["event"] == "discarded"
    assert agent._early_edit_events[-1]["reason"] == "model_abort"
    record = json.loads((agent.logs_dir / "sfx-live-ON.json").read_text())
    assert record["completed"] is False
    assert record["initial_fs_hash"] == record["final_fs_hash"]
    assert record["raw"] == record["receipts"] == []
    partial = record["sfx_live_trajectory"]
    assert partial["stop_reason"] == "model_error"
    assert partial["stream_events"][-1]["event"] == "model_abort"
    assert partial["stream_events"][-1]["text"] == PREFIX
    assert partial["mean_generation_latency_s"] is None


def test_real_changed_final_command_cannot_serve_the_old_fork_result(
        live_stack, tmp_path, monkeypatch):
    repo, _, daemon = live_stack
    review_greet(monkeypatch, BODY)
    final_body = "print('FINAL')\n"
    continuation = f"cat > greet.py <<'FINAL_EOF'\n{final_body}FINAL_EOF\n"
    final_command = EDIT + "\n" + continuation.rstrip("\n")
    early = []

    def stream_fn(base_url, model, api_key, messages, **kwargs):
        step = sum(row["role"] == "assistant" for row in messages)
        if step == 0:
            yield {"type": "delta", "content": PREFIX}
            early.append(_completed_early_job(agent, repo, daemon))
            yield {"type": "delta", "content": continuation + "```"}
        elif step == 1:
            assert (repo / "greet.py").read_text() == final_body
            session, old_job = early[0]
            assert old_job.spec_id in session.ledger._terminated
            assert all(job is not old_job for job in session.cache.jobs.values())
            assert agent._pending_edit is None
            yield {"type": "delta", "content": f"```bash\n{FOLLOWUP}\n```"}
        else:
            yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}

    agent, environment, context = _live_agent(tmp_path, repo, monkeypatch, stream_fn)
    asyncio.run(agent.run("CPU final-command mismatch", environment, context))
    assert environment.commands.count(final_command) == 1
    assert EDIT not in environment.commands
    assert agent._raw[-1]["stdout"] == "FINAL\n"
    assert agent._raw[-1]["stderr"] == ""
    assert (repo / "greet.py").read_text() == final_body
    assert not daemon._mutations and agent._pending_edit is None
    assert any(event["event"] == "discarded" and event["reason"] == "final_command_mismatch"
               for event in agent._early_edit_events)
    assert not any(event["event"] == "confirmed" for event in agent._early_edit_events)
    assert context.metadata["sfx_live_trajectory"]["commands"] == [final_command, FOLLOWUP]


def test_disconnect_while_fork_is_running_cannot_resurrect_discarded_result(
        live_stack, tmp_path, monkeypatch):
    repo, _, daemon = live_stack
    review_greet(monkeypatch, BODY)
    entered, release = threading.Event(), threading.Event()
    pending, forks = [], []
    run_in_fork = daemon.fs_substrate._run_in_fork

    def paused_run(fork_path, hop):
        forks.append(Path(fork_path))
        assert (Path(fork_path) / "greet.py").read_text() == BODY
        entered.set()
        assert release.wait(timeout=5), "test did not release the owned fork worker"
        return run_in_fork(fork_path, hop)

    monkeypatch.setattr(daemon.fs_substrate, "_run_in_fork", paused_run)

    def stream_fn(*args, **kwargs):
        yield {"type": "delta", "content": PREFIX}
        assert entered.wait(timeout=5)
        session = daemon.sessions[agent._session]
        assert session.chain is not None and not session.chain.future.done()
        pending.append((session, session.chain.future))
        raise TimeoutError("disconnect during fork execution")

    agent, environment, context = _live_agent(tmp_path, repo, monkeypatch, stream_fn)
    try:
        with pytest.raises(TimeoutError, match="during fork execution"):
            asyncio.run(agent.run("CPU in-flight cancellation", environment, context))
        assert pending and not pending[0][1].done()
        assert pending[0][0].chain is None
        assert not daemon._mutations and agent._pending_edit is None
        assert (repo / "greet.py").read_text() == OLD
        assert environment.commands == ["pwd"]
    finally:
        release.set()
        if pending:
            pending[0][1].result(timeout=5)
    assert not pending[0][0].cache.jobs
    assert (repo / "greet.py").read_text() == OLD
    assert all(not fork.exists() for fork in forks)
