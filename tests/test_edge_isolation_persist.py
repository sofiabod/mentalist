"""Adversarial edge tests for per-case .sfx isolation and daemon persistence.

Attacks invariants: PER-CASE ISOLATION, LOSSLESSNESS of the warm table,
FAIL-OPEN cleanup, DETERMINISM (no wall-clock), and atexit-on-dead-socket bound.
Independent source of truth for the warm-table math is computed by hand in each
test, never by re-running the code under test.
"""
import json
import socket
import tempfile
import threading
from collections import Counter
from pathlib import Path

import pytest

from sfx.daemon import _persist_repo_table, _load_repo_table, Daemon
from sfx.predictor import Predictor
from sfx.schema import ToolEvent, Outcome


BENCH = {"k": 2, "table": {}}


def _clock():
    v = {"t": 0.0}

    def c():
        return v["t"]
    return c


def _persist_from_counts(repo, key, counts, state_dir=None):
    """Drive a real Predictor's session_counts then persist, as the daemon does."""
    p = Predictor(BENCH, k=2)
    p.session_counts[key] = Counter(counts)
    _persist_repo_table(repo, p.session_counts, state_dir)
    return p


def test_per_case_state_dir_isolates_but_same_case_warms():
    """PER-CASE ISOLATION + ONLINE LEARNING, both must hold. Each eval case gets
    its own state_dir, so a distinct case on the SAME repo path does NOT inherit
    another case's repo tier; but re-running the SAME case (same state_dir) DOES
    warm from its own persisted tier. Isolation is per-case-workspace, warming is
    per-case-across-runs; the two coexist by construction."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        case_a_dir = repo / "caseA" / ".sfx"
        case_b_dir = repo / "caseB" / ".sfx"

        # Case A learns at key "x": always edit.
        _persist_from_counts(repo, "x", {"edit": 10}, state_dir=case_a_dir)

        # Case B: a DISTINCT case on the same repo path, its own state_dir. Isolated.
        case_b = Predictor(BENCH, k=2, repo_table=_load_repo_table(repo, case_b_dir))
        assert case_b.repo_table.get("x") is None, (
            "PER-CASE ISOLATION VIOLATED: distinct case B inherited case A's repo "
            "tier despite its own state_dir")

        # Re-run of case A (same state_dir): warms from A's own learning.
        rerun_a = Predictor(BENCH, k=2, repo_table=_load_repo_table(repo, case_a_dir))
        assert rerun_a.repo_table.get("x") == {"support": 10, "p": {"edit": 1.0}}, (
            "ONLINE LEARNING BROKEN: re-running the same case did not warm from its "
            "own persisted repo tier")


def test_warm_reload_lossless_single_case():
    """LOSSLESSNESS: for a single case, persist then reload must reproduce the
    exact distribution and support. Independent truth computed by hand here."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _persist_from_counts(repo, "role|read,read|OK", {"grep": 3, "edit": 1})
        loaded = _load_repo_table(repo)["table"]["role|read,read|OK"]
        assert loaded["support"] == 4
        assert loaded["p"] == {"grep": 0.75, "edit": 0.25}


def test_merge_two_persists_sums_supports():
    """LOSSLESSNESS across repeated same-repo runs: merging a second persist into
    an existing table must sum counts, not overwrite. Truth: grep 3+1=4, edit 1,
    read 0+2=2 -> support 7."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _persist_from_counts(repo, "k", {"grep": 3, "edit": 1})
        _persist_from_counts(repo, "k", {"grep": 1, "read": 2})
        merged = _load_repo_table(repo)["table"]["k"]
        assert merged["support"] == 7
        assert merged["p"] == pytest.approx(
            {"grep": 4 / 7, "edit": 1 / 7, "read": 2 / 7})


def test_empty_session_persist_is_noop_not_wipe():
    """A case that learned nothing must not clobber an existing warm table."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _persist_from_counts(repo, "k", {"grep": 5})
        before = _load_repo_table(repo)
        _persist_repo_table(repo, {})
        after = _load_repo_table(repo)
        assert after == before


def test_fresh_case_starts_cold_no_table_file():
    """A fresh case in a repo with no .sfx starts cold: load returns None and the
    predictor has an empty repo tier."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        assert _load_repo_table(repo) is None
        p = Predictor(BENCH, k=2, repo_table=_load_repo_table(repo))
        assert p.repo_table == {}
        p.observe(ToolEvent(t=0.0, kind="read", verb="free", role="main",
                            epoch=0, args={}, outcome=Outcome("read", "OK")))
        assert p.propose() == []


def test_persist_atomic_replace_no_partial_file():
    """Persist writes via a .tmp then os.replace; the live table.json is always
    valid JSON, never a half-written file."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _persist_from_counts(repo, "k", {"grep": 2})
        p = repo / ".sfx" / "table.json"
        json.loads(p.read_text())
        assert not (repo / ".sfx" / "table.json.tmp").exists()


def _mem_daemon():
    return Daemon(clock=_clock(), global_table=BENCH, k=2,
                  run=lambda k, a: (f"{k}-r", 0),
                  resolve_args=lambda k, ctx: {"cmd": k})


def test_session_end_persists_then_deletes_session():
    """DETERMINISM + persistence seam: session_end must write the repo table and
    drop the in-memory session (a stale spec can never serve after teardown)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        d = _mem_daemon()
        s = d.session_start("s1", repo=str(repo), role="main")
        s.predictor.observe(ToolEvent(t=0.0, kind="read", verb="free",
                                      role="main", epoch=0, args={},
                                      outcome=Outcome("read", "OK")))
        s.predictor.observe(ToolEvent(t=0.0, kind="grep", verb="free",
                                      role="main", epoch=0, args={},
                                      outcome=Outcome("grep", "OK")))
        d.session_end("s1")
        assert "s1" not in d.sessions
        assert _load_repo_table(repo) is not None


def test_case_isolation_via_distinct_repo_dirs():
    """The ONLY scoping the code offers is a distinct repo directory. Two cases in
    distinct dirs stay isolated. This documents the required upstream contract:
    the harness MUST give each case its own workspace path, or isolation breaks."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _persist_from_counts(Path(a), "x", {"edit": 9})
        _persist_from_counts(Path(b), "x", {"read": 9})
        pa = Predictor(BENCH, k=2, repo_table=_load_repo_table(Path(a)))
        pb = Predictor(BENCH, k=2, repo_table=_load_repo_table(Path(b)))
        assert pa.repo_table["x"]["p"] == {"edit": 1.0}
        assert pb.repo_table["x"]["p"] == {"read": 1.0}


# --- adapter cleanup / atexit-on-dead-socket ---

def _sfx_env_class():
    from adapters.mini_swe import SfxEnvironment
    return SfxEnvironment


class _FakeClient:
    def __init__(self):
        self.calls = []

    def turn_end(self, sid):
        self.calls.append(("turn_end", sid))

    def session_end(self, sid):
        self.calls.append(("session_end", sid))


def test_cleanup_idempotent():
    """FAIL-OPEN: cleanup() called twice must run teardown exactly once."""
    Env = _sfx_env_class()
    e = Env.__new__(Env)
    client = _FakeClient()
    e._sfx = client
    e._cleaned = False
    e.cleanup()
    e.cleanup()
    assert client.calls == [("turn_end", "mini"), ("session_end", "mini")]


def test_cleanup_fail_open_when_no_daemon():
    """FAIL-OPEN: with no daemon connection cleanup must not raise."""
    Env = _sfx_env_class()
    e = Env.__new__(Env)
    e._sfx = None
    e._cleaned = False
    e.cleanup()
    assert e._cleaned is True


def test_atexit_cleanup_bounded_on_hung_socket():
    """atexit on a dead/hung socket must NOT hang the harness. A server that
    accepts and never replies makes Client.readline() block; cleanup routes
    through it. We bound cleanup in a thread and fail if it does not return."""
    from adapters.protocol import Client
    from adapters.mini_swe import SfxEnvironment

    with tempfile.TemporaryDirectory() as tmp:
        sock_path = str(Path(tmp) / "s.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(1)
        accepted = []

        def _accept_and_hang():
            conn, _ = srv.accept()
            accepted.append(conn)  # never reply, hold the conn open
        t = threading.Thread(target=_accept_and_hang, daemon=True)
        t.start()

        client = Client(sock_path)
        e = SfxEnvironment.__new__(SfxEnvironment)
        e._sfx = client
        e._cleaned = False

        done = threading.Event()
        err = []

        def _run():
            try:
                e.cleanup()
            except BaseException as ex:  # noqa: BLE001
                err.append(ex)
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True).start()
        finished = done.wait(timeout=3.0)

        srv.close()
        for c in accepted:
            c.close()

        assert finished, (
            "atexit cleanup HUNG on a socket whose peer accepted but never "
            "replied: Client._send blocks on readline() with no timeout, so a "
            "wedged daemon stalls harness shutdown indefinitely (FAIL-OPEN "
            "violated). Client should set a socket timeout or cleanup should "
            "not block on the reply.")
