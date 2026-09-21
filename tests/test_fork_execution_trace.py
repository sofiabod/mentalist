import json
import os
import threading
import uuid

import pytest

from sfx.fork import ForkError, WritePathError
from test_live_fenced_integration import live_stack, review_greet


@pytest.mark.parametrize("exit_code", [0, 3])
def test_completed_trace_requires_fork_tool_return_and_correlates_chain(
        live_stack, monkeypatch, exit_code):
    repo, scratch, daemon = live_stack
    events = []
    daemon.log = events.append
    entered, release = threading.Event(), threading.Event()
    run_in_fork = daemon.fs_substrate._run_in_fork
    body = f"print('NEW')\nraise SystemExit({exit_code})\n"
    review_greet(monkeypatch, body)

    def blocked_run(path, hop):
        assert path != repo
        assert (path / "greet.py").read_text() == body
        assert (repo / "greet.py").read_text() == "print('OLD')\n"
        entered.set()
        assert release.wait(5)
        return run_in_fork(path, hop)

    monkeypatch.setattr(daemon.fs_substrate, "_run_in_fork", blocked_run)
    daemon.session_start("trace-success", repo=str(repo), role="main", scratch=str(scratch))
    chain = None
    try:
        chain = daemon.call_stream_delta("trace-success", "edit", "Edit", json.dumps({
            "path": "greet.py", "contents": body}))
        assert chain is not None and chain.future is not None
        assert entered.wait(5)
        execution = [event for event in events if event["ev"] == "fork_execution"]
        assert [event["phase"] for event in execution] == ["started"]
        release.set()
        chain.future.result(timeout=5)
        execution = [event for event in events if event["ev"] == "fork_execution"]
        assert [event["phase"] for event in execution] == ["started", "completed"]
        chain_event = next(event for event in events if event["ev"] == "chain")
        assert uuid.UUID(hex=chain_event["execution_id"]).version == 4
        assert all(event["execution_id"] == chain_event["execution_id"] for event in execution)
        assert all(event["kind"] == "run" and event["elapsed_ms"] >= 0 for event in execution)
        assert execution[-1]["monotonic_ms"] >= execution[0]["monotonic_ms"]
        assert execution[-1]["elapsed_ms"] == (
            execution[-1]["monotonic_ms"] - execution[0]["monotonic_ms"])
        assert set(execution[0]) == {
            "ev", "execution_id", "kind", "phase", "elapsed_ms", "monotonic_ms"}
        assert set(execution[-1]) == set(execution[0]) | {"return_code"}
        assert execution[-1]["return_code"] == exit_code
        session = daemon.sessions["trace-success"]
        job = next(job for job in session.cache.jobs.values()
                   if job.spec_id == session.chain_spec_ids[0])
        assert job.done.is_set() and job.result == ("NEW\n", "", exit_code)
        assert (repo / "greet.py").read_text() == "print('OLD')\n"
    finally:
        release.set()
        if chain is not None and chain.future is not None:
            chain.future.result(timeout=5)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_guard_error_trace_contains_only_exception_class_and_no_completion(
        live_stack, tmp_path, monkeypatch, link_kind):
    repo, scratch, daemon = live_stack
    events = []
    daemon.log = events.append
    protected = tmp_path / "private-output.py"
    protected.write_text("print('PRIVATE')\n")
    target = repo / "linked.py"
    if link_kind == "symlink":
        target.symlink_to(protected)
    else:
        os.link(protected, target)
    monkeypatch.setattr(daemon.fs_substrate, "_run_in_fork",
                        lambda *args: pytest.fail("unsafe fork must not run its tool"))
    daemon.session_start("trace-error", repo=str(repo), role="main", scratch=str(scratch))
    chain = daemon.call_stream_delta("trace-error", "edit", "Edit", json.dumps({
        "path": "linked.py", "contents": "print('UNSAFE')\n"}))
    assert chain is not None and chain.future is not None
    with pytest.raises(WritePathError):
        chain.future.result(timeout=5)
    execution = [event for event in events if event["ev"] == "fork_execution"]
    assert [event["phase"] for event in execution] == ["started", "error"]
    chain_event = next(event for event in events if event["ev"] == "chain")
    assert all(event["execution_id"] == chain_event["execution_id"] for event in execution)
    error = execution[-1]
    assert error["error_type"] == "WritePathError"
    assert error["kind"] == "run" and error["elapsed_ms"] >= 0
    assert error["elapsed_ms"] == error["monotonic_ms"] - execution[0]["monotonic_ms"]
    assert set(error) == {
        "ev", "execution_id", "kind", "phase", "elapsed_ms", "monotonic_ms", "error_type"}
    assert protected.read_text() == target.read_text() == "print('PRIVATE')\n"
    assert not daemon.sessions["trace-error"].cache.jobs
    assert not list(scratch.iterdir())


def test_failed_substrate_probe_does_not_claim_fork_execution(live_stack, monkeypatch):
    repo, scratch, daemon = live_stack
    events = []
    daemon.log = events.append

    def unavailable(*args):
        raise ForkError("unavailable")

    monkeypatch.setattr(daemon.fs_substrate, "probe", unavailable)
    daemon.session_start("trace-no-fork", repo=str(repo), role="main", scratch=str(scratch))
    assert daemon.call_stream_delta("trace-no-fork", "edit", "Edit", json.dumps({
        "path": "greet.py", "contents": "print('NEW')\n"})) is None
    assert not any(event["ev"] == "fork_execution" for event in events)
    chain_event = next(event for event in events if event["ev"] == "chain")
    assert chain_event["discard_reason"] == "no_substrate"
    assert "execution_id" not in chain_event
