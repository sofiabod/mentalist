"""Measured cleanup must join workers whose Session has already been removed."""
import threading
import time

from eval.stateful_probe import run_one
from sfx.daemon import Daemon


def test_probe_drains_removed_session_pool_inside_cleanup_clock(tmp_path, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    captured = {}
    original_start = Daemon.session_start

    def start_with_pending_worker(daemon, *args, **kwargs):
        session = original_start(daemon, *args, **kwargs)
        pool = session.executor._pool
        shutdown = pool.shutdown
        captured.update(pool=pool, shutdown=shutdown)

        def pending_worker():
            entered.set()
            assert release.wait(5), "test worker was not released during cleanup"
            # A short blocked worker models discarded speculation, without CPU load.
            time.sleep(0.05)
            captured["worker_finished"] = True

        captured["future"] = pool.submit(pending_worker)
        assert entered.wait(1)

        def observed_shutdown(wait=True, *, cancel_futures=False):
            if wait:
                # The ordinary end call already discarded this Session, so looking
                # only through daemon.sessions would miss its still-running pool.
                assert session not in daemon.sessions.values()
                captured["joined_after_session_removal"] = True
                started = time.monotonic()
                release.set()
                shutdown(wait=True, cancel_futures=cancel_futures)
                captured["join_wall_s"] = time.monotonic() - started
            else:
                shutdown(wait=False, cancel_futures=cancel_futures)

        monkeypatch.setattr(pool, "shutdown", observed_shutdown)
        return session

    monkeypatch.setattr(Daemon, "session_start", start_with_pending_worker)
    try:
        record = run_one(tmp_path, mode="native", work_units=1, gap_s=0,
                         rounds=1, case_id="cleanup-regression")
        assert captured.get("joined_after_session_removal") is True
        assert captured.get("worker_finished") is True
        assert captured["future"].done()
        captured["future"].result(timeout=1)
        assert all(not thread.is_alive() for thread in captured["pool"]._threads)
        assert record["workers_drained"] is True
        assert captured["join_wall_s"] >= 0.04
        assert record["cleanup_wall_s"] >= captured["join_wall_s"]
        assert record["total_wall_s"] >= record["wall_s"] + record["cleanup_wall_s"]
        assert record["after_cleanup_fs_hash"] == record["final_fs_hash"]
    finally:
        # Keep a failing regression from leaving our own controlled worker behind.
        release.set()
        if "shutdown" in captured:
            captured["shutdown"](wait=True, cancel_futures=True)
