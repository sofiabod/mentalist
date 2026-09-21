"""Real thread interleavings and shared-workspace mutation safety regressions."""
import json
import os
import tempfile
import threading
import time

import pytest

from adapters.protocol import Client, serve
from sfx.cache import Cache
from sfx.daemon import Daemon
from sfx.ledger import Ledger


def test_epoch_change_between_selection_and_delivery_returns_miss(monkeypatch):
    cache = Cache(lambda: 10, Ledger())
    key, _ = cache.put("read", {"cmd": "cat x"}, 0, result="before")
    job = cache.jobs[key]
    selected, resume = threading.Event(), threading.Event()
    original_wait = job.done.wait

    def paused_wait():
        selected.set()
        assert resume.wait(2)
        return original_wait()

    monkeypatch.setattr(job.done, "wait", paused_wait)
    result = []
    reader = threading.Thread(target=lambda: result.append(
        cache.serve("read", {"cmd": "cat x"}, 10)))
    reader.start()
    try:
        assert selected.wait(2)
        cache.bump_epoch()
    finally:
        resume.set()
        reader.join(2)
    assert not reader.is_alive()
    assert result == [("miss", None, None)]
    assert cache.ledger.terminal_counts() == {"epoch_expired": 1}


def test_concurrent_resolves_claim_distinct_occurrences(monkeypatch):
    cache = Cache(lambda: 10, Ledger())
    barrier = threading.Barrier(2)
    for value in ("first", "second"):
        key, _ = cache.put("read", {"cmd": "cat x"}, 0, result=value)
        done = cache.jobs[key].done
        original_wait = done.wait

        def paused_wait(original_wait=original_wait):
            barrier.wait(timeout=2)
            return original_wait()

        monkeypatch.setattr(done, "wait", paused_wait)
    results = []
    readers = [threading.Thread(target=lambda: results.append(
        cache.serve("read", {"cmd": "cat x"}, 10))) for _ in range(2)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join(3)
        assert not reader.is_alive()
    assert {reply[2] for reply in results} == {"first", "second"}
    assert cache.ledger.terminal_counts() == {"hit_completed": 2}


def test_discarded_reservation_does_not_overwrite_later_inflight_job():
    cache = Cache(lambda: 10, Ledger())
    _, failed = cache.reserve("read", {}, 0)
    second_key, second = cache.reserve("read", {}, 0)
    cache.discard_spec(failed)
    third_key, third = cache.reserve("read", {}, 0)
    assert second_key != third_key
    cache.complete(second, "second", 0)
    cache.complete(third, "third", 0)
    assert cache.serve("read", {}, 10)[2] == "second"
    assert cache.serve("read", {}, 10)[2] == "third"


def test_terminated_speculation_is_not_promoted_back_into_a_hit():
    cache = Cache(lambda: 10, Ledger())
    _, spec_id = cache.put("read", {}, 0, result="discarded")
    cache.ledger.terminal(spec_id, "preempted")
    assert cache.serve("read", {}, 10) == ("miss", None, None)


def test_epoch_change_wakes_an_already_preempted_pending_job():
    cache = Cache(lambda: 10, Ledger())
    key, spec_id = cache.reserve("read", {}, 0)
    job = cache.jobs[key]
    cache.ledger.terminal(spec_id, "preempted")
    cache.bump_epoch()
    assert job.done.is_set()
    assert cache.finish(job, 10) == ("miss", None, None)


@pytest.fixture
def runtime(tmp_path):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "value.txt").write_text("before")
    table = {"k": 1, "min_support": 1, "tau": 0.35, "table": {
        "main|edit|edit:OK": {"support": 20, "p": {"read": 0.99}},
        "main|read|read:OK": {"support": 20, "p": {"read": 0.99}},
    }}
    daemon = Daemon(
        clock=lambda: time.monotonic() * 1000, global_table=table, k=1,
        run=lambda kind, args: (((repo / "value.txt").read_text(), 0), 0),
        resolve_args=lambda kind, ctx: {"cmd": "cat value.txt"},
        apply_write=lambda fp, write: (fp / write.args["path"]).write_text(
            write.args["contents"]),
        run_in_fork=lambda fp, hop: (((fp / "value.txt").read_text(), 0), "OK"))
    daemon.session_start("a", repo, "main", scratch=scratch)
    yield daemon, repo, scratch
    daemon.shutdown()


def test_reported_write_invalidates_sessions_using_same_repo_alias(runtime, tmp_path):
    daemon, repo, scratch = runtime
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    daemon.session_start("b", alias, "main", scratch=scratch)
    cache = daemon.sessions["a"].cache
    cache.put("read", {"cmd": "cat value.txt"}, 0, result="before")
    (repo / "value.txt").write_text("after")
    daemon.call_executed("b", "edit", "fork", "OK", {}, 0, speculate=False)
    assert daemon.resolve("a", "read", {"cmd": "cat value.txt"}) == ("miss", None)
    assert cache.epoch == daemon.sessions["b"].cache.epoch == 1


def test_active_mutations_fence_new_sessions_and_all_speculation(runtime):
    daemon, repo, scratch = runtime
    first = daemon.mutation_begin("a")
    b = daemon.session_start("b", repo, "main", scratch=scratch)
    second = daemon.mutation_begin("b")
    daemon.call_executed("b", "read", "free", "OK", {"cmd": "cat value.txt"}, 100)
    assert b.cache.jobs == {}
    assert daemon.stream_get_delta("b", "r", "read", '{"cmd":"cat value.txt"}') is None
    # Even an accidentally inserted job cannot be served while another writer is active.
    b.cache.put("read", {"cmd": "cat value.txt"}, 0, result="during-write")
    assert daemon.resolve("b", "read", {"cmd": "cat value.txt"}) == ("miss", None)
    daemon.mutation_end("b", second)
    assert daemon.resolve("b", "read", {"cmd": "cat value.txt"}) == ("miss", None)
    daemon.mutation_end("a", first)
    assert b.cache.jobs == {}


def test_waiting_resolve_does_not_block_mutation_begin(runtime, monkeypatch):
    daemon, _, _ = runtime
    cache = daemon.sessions["a"].cache
    key, _ = cache.reserve("read", {"cmd": "cat value.txt"}, daemon.clock())
    job = cache.jobs[key]
    selected = threading.Event()
    original_wait = job.done.wait

    def observed_wait(timeout=None):
        selected.set()
        return original_wait(timeout=timeout)

    monkeypatch.setattr(job.done, "wait", observed_wait)
    result, mutation = [], []
    reader = threading.Thread(target=lambda: result.append(
        daemon.resolve("a", "read", {"cmd": "cat value.txt"})))
    writer = threading.Thread(target=lambda: mutation.append(daemon.mutation_begin("a")))
    reader.start()
    assert selected.wait(2)
    writer.start()
    writer.join(2)
    reader.join(2)
    assert not writer.is_alive() and not reader.is_alive()
    assert result == [("miss", None)]
    daemon.mutation_end("a", mutation[0])


@pytest.mark.parametrize("operation", ["shutdown", "session_end"])
def test_teardown_releases_resolve_waiting_for_cancelled_work(runtime, monkeypatch, operation):
    daemon, _, _ = runtime
    cache = daemon.sessions["a"].cache
    key, spec_id = cache.reserve("read", {}, daemon.clock())
    selected = threading.Event()
    original_wait = cache.jobs[key].done.wait

    def observed_wait(timeout=None):
        selected.set()
        return original_wait(timeout=timeout)

    monkeypatch.setattr(cache.jobs[key].done, "wait", observed_wait)
    result = []
    reader = threading.Thread(target=lambda: result.append(daemon.resolve("a", "read", {})))
    reader.start()
    assert selected.wait(2)
    if operation == "session_end":
        daemon.session_end("a")
    else:
        daemon.shutdown()
    reader.join(2)
    assert not reader.is_alive()
    assert result == [("miss", None)]
    cache.complete(spec_id, "late worker completion", 0)
    assert cache.jobs == {}


def test_pending_join_cutoff_misses_without_reusing_or_overwriting_occurrences(runtime, monkeypatch):
    import sfx.daemon as daemon_module

    daemon, _, _ = runtime
    monkeypatch.setattr(daemon_module, "RESOLVE_JOIN_TIMEOUT_S", 0)
    cache = daemon.sessions["a"].cache
    _, first = cache.reserve("read", {}, daemon.clock())
    second_key, second = cache.reserve("read", {}, daemon.clock())
    assert daemon.resolve("a", "read", {}) == ("miss", None)
    assert cache.ledger.total_saved_ms == 0
    third_key, third = cache.reserve("read", {}, daemon.clock())
    assert second_key != third_key
    cache.complete(first, "abandoned", 0)
    cache.complete(second, "second", 0)
    cache.complete(third, "third", 0)
    assert daemon.resolve("a", "read", {}) == ("hit_completed", "second")
    assert daemon.resolve("a", "read", {}) == ("hit_completed", "third")


def _feed_write(daemon, args, mutation_id):
    return daemon.call_stream_delta("a", "edit-1", "Write", json.dumps(args),
                                    mutation_id=mutation_id)


def test_exclusive_streamed_write_preserves_async_chain_after_verified_commit(runtime):
    daemon, repo, _ = runtime
    entered, finish = threading.Event(), threading.Event()

    def run_in_fork(fp, hop):
        value = (fp / "value.txt").read_text()
        entered.set()
        assert finish.wait(3)
        return (value, 0), "OK"

    daemon.fs_substrate._run_in_fork = run_in_fork
    args = {"path": "value.txt", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    chain = _feed_write(daemon, args, mutation)
    try:
        assert entered.wait(2)
        assert not chain.future.done()
        (repo / "value.txt").write_text("after")
        assert daemon.mutation_end("a", mutation, success=True) is True
    finally:
        finish.set()
    chain.future.result(timeout=3)
    assert daemon.resolve("a", "edit", args) == ("hit_completed", "after")
    daemon.call_executed("a", "edit", "fork", "OK", {"cmd": "write value.txt"},
                         0, mutation_id=mutation)
    assert daemon.sessions["a"].chain is chain
    status, output = daemon.resolve("a", "read", {"cmd": "cat value.txt"})
    assert status == "hit_completed" and output == ("after", 0)


@pytest.mark.parametrize("success,actual", [(False, "after"), (True, "wrong")])
def test_failed_or_mismatched_write_discards_chain(runtime, success, actual):
    daemon, repo, _ = runtime
    args = {"path": "value.txt", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    chain = _feed_write(daemon, args, mutation)
    chain.future.result(timeout=3)
    (repo / "value.txt").write_text(actual)
    assert daemon.mutation_end("a", mutation, success=success) is False
    assert daemon.resolve("a", "read", {"cmd": "cat value.txt"}) == ("miss", None)


def test_missing_success_confirmation_does_not_preserve_matching_bytes(runtime):
    daemon, repo, _ = runtime
    args = {"path": "value.txt", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    chain = _feed_write(daemon, args, mutation)
    chain.future.result(timeout=3)
    (repo / "value.txt").write_text("after")
    assert daemon.mutation_end("a", mutation) is False
    assert daemon.resolve("a", "read", {"cmd": "cat value.txt"}) == ("miss", None)


def test_completed_failed_chain_is_not_preserved(runtime):
    daemon, repo, _ = runtime

    def fail_in_fork(fp, hop):
        raise RuntimeError("speculative command failed to start")

    daemon.fs_substrate._run_in_fork = fail_in_fork
    args = {"path": "value.txt", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    chain = _feed_write(daemon, args, mutation)
    with pytest.raises(RuntimeError, match="failed to start"):
        chain.future.result(timeout=3)
    (repo / "value.txt").write_text("after")
    assert daemon.mutation_end("a", mutation, success=True) is False
    assert daemon.sessions["a"].chain is None
    assert not daemon._mutations


def test_fifo_verification_does_not_block_or_leave_mutation_active(runtime):
    daemon, repo, _ = runtime
    fifo = repo / "pipe"
    os.mkfifo(fifo)
    daemon.depth_cap = 0
    args = {"path": "pipe", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    _feed_write(daemon, args, mutation)
    results = []
    ending = threading.Thread(target=lambda: results.append(
        daemon.mutation_end("a", mutation, success=True)))
    ending.start()
    ending.join(1)
    blocked = ending.is_alive()
    if blocked:
        # Unstick an incorrect blocking reader so a regression fails cleanly.
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(writer, b"after")
        finally:
            os.close(writer)
        ending.join(2)
    assert not blocked
    assert results == [False]
    assert not daemon._mutations


def test_overlapping_mutation_discards_even_a_matching_streamed_write(runtime):
    daemon, repo, scratch = runtime
    daemon.session_start("b", repo, "main", scratch=scratch)
    args = {"path": "value.txt", "contents": "after"}
    first = daemon.mutation_begin("a", write_args=args)
    chain = _feed_write(daemon, args, first)
    chain.future.result(timeout=3)
    second = daemon.mutation_begin("b")
    (repo / "value.txt").write_text("after")
    daemon.mutation_end("b", second)
    assert daemon.mutation_end("a", first) is False
    assert daemon.resolve("a", "read", {"cmd": "cat value.txt"}) == ("miss", None)


def test_stream_requires_exact_owner_token_and_arguments(runtime):
    daemon, _, _ = runtime
    args = {"path": "value.txt", "contents": "after"}
    mutation = daemon.mutation_begin("a", write_args=args)
    assert _feed_write(daemon, args, None) is None
    assert _feed_write(daemon, {**args, "contents": "other"}, mutation) is None
    assert daemon.mutation_end("a", mutation) is False


def test_nondefault_context_report_does_not_seed_or_launch_commands(runtime):
    daemon, _, _ = runtime
    daemon.call_executed("a", "read", "free", "OK", {"cmd": "cat elsewhere"},
                         100, speculate=False)
    session = daemon.sessions["a"]
    assert session.arg_by_kind == {}
    assert session.cache.jobs == {}
    assert len(session.predictor.events) == 1


def test_mutation_protocol_round_trip(runtime, tmp_path):
    daemon, repo, _ = runtime
    socket_dir = tempfile.TemporaryDirectory(prefix="sfx-sock-", dir="/tmp")
    socket_path = socket_dir.name + "/s.sock"
    server = serve(daemon, socket_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(socket_path)
    try:
        args = {"path": "value.txt", "contents": "after"}
        begin = client.mutation_begin("a", write_args=args)
        mutation = begin["mutation_id"]
        assert client.feed("a", "e", "Write", json.dumps(args),
                           mutation_id=mutation)["chain_len"] == 1
        daemon.sessions["a"].chain.future.result(timeout=3)
        (repo / "value.txt").write_text("after")
        assert client.mutation_end("a", mutation, success=True)["chain_preserved"] is True
        client.call_executed("a", "edit", "fork", "OK", {}, 0,
                             mutation_id=mutation, speculate=False)
        assert client.resolve("a", "edit", args)["result"] == "hit_completed"
        assert client.resolve("a", "read", {"cmd": "cat value.txt"})["output"] == ["after", 0]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
        socket_dir.cleanup()
