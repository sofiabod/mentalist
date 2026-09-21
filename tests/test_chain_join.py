import threading
from types import SimpleNamespace

import pytest

from sfx.chain import Chain, Hop
from sfx.daemon import Daemon


COMMAND = {"cmd": "python check.py inputs --rules rules.json"}
WRITE = {"path": "inputs/value.txt", "contents": "new state"}


@pytest.fixture
def pending(tmp_path):
    clock = SimpleNamespace(now=0.0)
    events = []
    daemon = Daemon(clock=lambda: clock.now, global_table={"k": 1, "table": {}}, k=1,
                    log=events.append)
    session = daemon.session_start("s", repo=tmp_path, role="main")
    key, spec_id = session.cache.reserve("run", COMMAND, launch=clock.now)
    chain = Chain(write=SimpleNamespace(args=WRITE), hops=[Hop("run", "free", COMMAND, None)])
    session.chain = chain
    session.chain_spec_ids = [spec_id]
    yield daemon, session, clock, events, session.cache.jobs[key]
    daemon.shutdown()


def consume_edit(daemon):
    assert daemon.resolve("s", "edit", WRITE) == ("hit_completed", "new state")


def start_reader(daemon, job, monkeypatch):
    selected = threading.Event()
    original = job.done.wait

    def observed_wait(timeout=None):
        selected.set()
        return original(timeout)

    monkeypatch.setattr(job.done, "wait", observed_wait)
    results = []
    thread = threading.Thread(target=lambda: results.append(daemon.resolve("s", "run", COMMAND)))
    thread.start()
    assert selected.wait(2)
    return thread, results


def test_edit_receipt_never_waits_for_pending_followup(pending, monkeypatch):
    daemon, session, _, _, job = pending
    monkeypatch.setattr(job.done, "wait", lambda *a, **kw: pytest.fail("edit receipt waited"))

    consume_edit(daemon)

    assert session.chain_cursor == 1
    assert session.chain_claim is None
    assert session.ledger.total_saved_ms == 0


def test_pending_chain_joins_once_and_uses_original_ask_timestamp(pending, monkeypatch):
    daemon, session, clock, events, job = pending
    consume_edit(daemon)
    clock.now = 100
    reader, result = start_reader(daemon, job, monkeypatch)
    try:
        assert session.chain_cursor == 2
        assert len(session.real_hops) == 1
        clock.now = 250
        session.cache.complete(job.spec_id, ("new result", "", 0), duration=250)
    finally:
        reader.join(2)

    assert not reader.is_alive()
    assert result == [("hit_promoted", ("new result", "", 0))]
    assert session.ledger.total_saved_ms == 100
    assert session.chain is session.chain_claim is None
    assert events[-1]["saved_ms"] == 100
    assert events[-1]["wall_ms"] == 150
    assert daemon.resolve("s", "run", COMMAND) == ("miss", None)


def test_already_completed_chain_retains_completed_semantics(pending):
    daemon, session, clock, _, job = pending
    consume_edit(daemon)
    session.cache.complete(job.spec_id, "ready", duration=50)
    clock.now = 100

    assert daemon.resolve("s", "run", COMMAND) == ("hit_completed", "ready")
    assert session.ledger.total_saved_ms == 50


def test_wrong_command_never_joins_and_discards_original_chain(pending, monkeypatch):
    daemon, session, _, events, job = pending
    consume_edit(daemon)
    monkeypatch.setattr(job.done, "wait", lambda *a, **kw: pytest.fail("mismatch waited"))

    assert daemon.resolve("s", "run", {"cmd": COMMAND["cmd"] + " --other"}) == ("miss", None)
    assert session.chain is None
    assert not session.cache.jobs
    assert any(row.get("discard_reason") == "prefix_break" for row in events)


@pytest.mark.parametrize("second_args", [COMMAND, {"cmd": "python unrelated.py"}])
def test_concurrent_request_cannot_reclaim_or_discard_owned_chain(pending, monkeypatch, second_args):
    daemon, session, clock, _, job = pending
    consume_edit(daemon)
    clock.now = 100
    reader, result = start_reader(daemon, job, monkeypatch)
    claim = session.chain_claim
    try:
        assert daemon.resolve("s", "run", second_args) == ("miss", None)
        assert session.chain_claim is claim
        assert len(session.real_hops) == 1
        assert list(session.cache._serve_counts.values()) == [1]
        clock.now = 200
        session.cache.complete(job.spec_id, "one occurrence", duration=200)
    finally:
        reader.join(2)

    assert result == [("hit_promoted", "one occurrence")]
    assert session.ledger.terminal_counts() == {"hit_promoted": 1}


def test_waiting_chain_does_not_block_mutation_and_cannot_serve_old_epoch(pending, monkeypatch):
    daemon, session, _, _, job = pending
    consume_edit(daemon)
    reader, result = start_reader(daemon, job, monkeypatch)
    tokens = []
    writer = threading.Thread(target=lambda: tokens.append(daemon.mutation_begin("s")))
    writer.start()
    writer.join(2)
    reader.join(2)

    assert not writer.is_alive() and not reader.is_alive()
    assert result == [("miss", None)]
    assert session.chain is session.chain_claim is None
    session.cache.complete(job.spec_id, "late old result", duration=1)
    assert not session.cache.jobs
    assert session.ledger.total_saved_ms == 0
    daemon.mutation_end("s", tokens[0])


def test_completed_job_invalidated_before_delivery_is_still_a_miss(pending, monkeypatch):
    daemon, session, _, _, job = pending
    consume_edit(daemon)
    selected, resume = threading.Event(), threading.Event()
    session.cache.complete(job.spec_id, "completed old state", duration=0)

    def paused_wait(timeout=None):
        assert job.done.is_set()
        selected.set()
        assert resume.wait(2)
        return True

    monkeypatch.setattr(job.done, "wait", paused_wait)
    results = []
    reader = threading.Thread(target=lambda: results.append(daemon.resolve("s", "run", COMMAND)))
    reader.start()
    try:
        assert selected.wait(2)
        token = daemon.mutation_begin("s")
        daemon.mutation_end("s", token)
    finally:
        resume.set()
        reader.join(2)

    assert results == [("miss", None)]
    assert session.ledger.total_saved_ms == 0


@pytest.mark.parametrize("operation", ["session_end", "shutdown"])
def test_teardown_releases_pending_chain_without_publishing(pending, monkeypatch, operation):
    daemon, session, _, _, job = pending
    consume_edit(daemon)
    reader, result = start_reader(daemon, job, monkeypatch)
    if operation == "session_end":
        daemon.session_end("s")
    else:
        daemon.shutdown()
    reader.join(2)

    assert not reader.is_alive()
    assert result == [("miss", None)]
    session.cache.complete(job.spec_id, "late", duration=0)
    assert session.cache.jobs == {}


def test_timeout_discards_reserved_occurrence_without_reusing_it(pending, monkeypatch):
    import sfx.daemon as daemon_module

    daemon, session, clock, events, job = pending
    consume_edit(daemon)
    monkeypatch.setattr(daemon_module, "RESOLVE_JOIN_TIMEOUT_S", 0)
    assert daemon.resolve("s", "run", COMMAND) == ("miss", None)
    assert any(row.get("discard_reason") == "chain_join_timeout" for row in events)
    assert session.chain is session.chain_claim is None
    assert not session.cache.jobs
    assert session.ledger.total_saved_ms == 0
    session.cache.complete(job.spec_id, "abandoned", duration=10)
    _, second = session.cache.put("run", COMMAND, 0, "fresh occurrence")
    assert second != job.spec_id
    clock.now = 1
    assert daemon.resolve("s", "run", COMMAND) == ("hit_completed", "fresh occurrence")


def test_completion_after_timeout_does_not_turn_cutoff_into_hit(pending, monkeypatch):
    daemon, session, _, _, job = pending
    consume_edit(daemon)

    def complete_after_cutoff(timeout=None):
        session.cache.complete(job.spec_id, "too late", duration=0)
        return False

    monkeypatch.setattr(job.done, "wait", complete_after_cutoff)
    assert daemon.resolve("s", "run", COMMAND) == ("miss", None)
    assert not session.cache.jobs
    assert session.ledger.total_saved_ms == 0


def test_failed_worker_wakes_join_and_returns_miss(pending, monkeypatch):
    daemon, session, _, events, original_job = pending
    session.cache.discard_spec(original_job.spec_id)
    started, release = threading.Event(), threading.Event()

    def fail():
        started.set()
        assert release.wait(2)
        raise RuntimeError("worker failed")

    spec_id, future = session.executor.submit_chain("run", COMMAND, fail)
    session.chain_spec_ids = [spec_id]
    session.chain.future = future
    job = next(row for row in session.cache.jobs.values() if row.spec_id == spec_id)
    consume_edit(daemon)
    reader, result = start_reader(daemon, job, monkeypatch)
    try:
        assert started.wait(2)
    finally:
        release.set()
        reader.join(2)

    with pytest.raises(RuntimeError, match="worker failed"):
        future.result(timeout=2)
    assert result == [("miss", None)]
    assert session.chain is session.chain_claim is None
    assert not session.cache.jobs
    assert any(row.get("discard_reason") == "chain_unavailable" for row in events)


def test_replaced_chain_identity_cannot_publish_or_discard_replacement(pending, monkeypatch):
    daemon, session, _, _, job = pending
    consume_edit(daemon)
    reader, result = start_reader(daemon, job, monkeypatch)
    old_claim = session.chain_claim
    replacement = Chain(write=SimpleNamespace(args=WRITE), hops=[])
    with daemon._lock:
        session.chain = replacement
        session.chain_claim = None
        session.cache.complete(job.spec_id, "old chain", duration=0)
    reader.join(2)

    assert result == [("miss", None)]
    assert session.chain is replacement
    assert session.chain_claim is not old_claim
    assert session.ledger.total_saved_ms == 0
    assert not session.cache.jobs
