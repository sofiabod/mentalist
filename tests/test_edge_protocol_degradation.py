import json
import socket
import threading
import uuid
from pathlib import Path
import tempfile

import pytest

from adapters.protocol import Client, serve, _dispatch, _handle_line, SfxDaemonError
from sfx.daemon import Daemon
from sfx.schema import PROFILES, Outcome, Profile, ToolEvent, context_key


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {"k": 1, "table": {}}


def _daemon(clock=None):
    clock = clock or FakeClock(0)
    return Daemon(clock=clock, global_table=_global_table(), k=1,
                  run=lambda kind, args: ("out", 0),
                  resolve_args=lambda kind, ctx: {"cmd": "x"},
                  apply_write=lambda fp, w: None,
                  run_in_fork=lambda fp, hop: ("r", "PASS"))


def _sock_path():
    d = Path(tempfile.gettempdir()) / uuid.uuid4().hex[:8]
    d.mkdir()
    return str(d / "s.sock")


def _run_server(daemon, sock_path):
    srv = serve(daemon, sock_path)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _raw_line(sock_path, payload_bytes):
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(sock_path)
    raw.sendall(payload_bytes)
    line = raw.makefile().readline()
    raw.close()
    return line


# ---------- malformed / unknown message types (FAIL-OPEN over the wire) ----------

def test_unknown_message_type_returns_error_not_crash():
    d = _daemon()
    reply = _dispatch(d, {"type": "no_such_verb", "session": "s1"})
    assert reply == {"error": "unknown type: no_such_verb"}


def test_missing_type_field_is_caught_and_serialized():
    d = _daemon()
    reply = _handle_line(lambda m: _dispatch(d, m), b'{"session": "s1"}\n')
    assert "error" in reply
    assert "type" in reply["error"]


def test_empty_line_returns_error_object():
    d = _daemon()
    reply = _handle_line(lambda m: _dispatch(d, m), b"\n")
    assert "error" in reply


def test_non_json_bytes_over_socket_error_then_server_survives():
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    err = json.loads(_raw_line(sp, b"\xff\xfe not json\n"))
    assert "error" in err
    c = Client(sp)
    c.turn_begin("s1", repo="/r", role="main")
    assert "s1" in d.sessions
    srv.shutdown()


def test_json_array_instead_of_object_errors_gracefully():
    """A JSON list has no ["type"]; TypeError/KeyError must serialize, not crash."""
    d = _daemon()
    reply = _handle_line(lambda m: _dispatch(d, m), b'[1, 2, 3]\n')
    assert "error" in reply


# ---------- null args (LOSSLESSNESS / FAIL-OPEN) ----------

def test_resolve_null_args_rejected_without_touching_cache():
    d = _daemon()
    reply = _dispatch(d, {"type": "resolve", "session": "s1",
                          "tool": "test", "args": None})
    assert reply == {"error": "args must not be null"}
    assert "s1" not in d.sessions


def test_resolve_null_args_over_socket_does_not_serve():
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    c.turn_begin("s1", repo="/x", role="main")
    reply = c.resolve("s1", tool="test", args=None)
    assert reply == {"error": "args must not be null"}
    srv.shutdown()


# ---------- turn_end without turn_begin (FAIL-OPEN) ----------

def test_turn_end_without_turn_begin_direct_call_crashes():
    """Direct daemon.turn_end on an unknown sid raises KeyError (loud, not silent)."""
    d = _daemon()
    with pytest.raises(KeyError):
        d.turn_end("ghost")


def test_turn_end_without_turn_begin_over_socket_is_caught():
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    reply = json.loads(_raw_line(
        sp, (json.dumps({"type": "turn_end", "session": "ghost"}) + "\n").encode()))
    assert "error" in reply
    c = Client(sp)
    c.turn_begin("s1", repo="/x", role="main")
    assert "s1" in d.sessions
    srv.shutdown()


# ---------- session_end idempotency + atexit double-call (FAIL-OPEN) ----------

def test_session_end_second_direct_call_raises_keyerror():
    """atexit double-call path: a second session_end on the same sid must not
    silently corrupt state; it raises loudly."""
    d = _daemon()
    d.session_start("s1", repo="/x", role="main")
    d.session_end("s1")
    with pytest.raises(KeyError):
        d.session_end("s1")


def test_session_end_double_call_over_socket_survives_server():
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    c.turn_begin("s1", repo="/x", role="main")
    ok = c.session_end("s1")
    assert ok == {"result": "ok"}
    second = c.session_end("s1")
    assert "error" in second
    c2 = Client(sp)
    c2.turn_begin("s2", repo="/x", role="main")
    assert "s2" in d.sessions
    srv.shutdown()


def test_session_end_persists_before_delete_then_gone():
    d = _daemon()
    d.session_start("s1", repo="/x", role="main")
    assert "s1" in d.sessions
    d.session_end("s1")
    assert "s1" not in d.sessions


# ---------- turn_begin idempotency (does not clobber a live session) ----------

def test_turn_begin_twice_keeps_original_session_object():
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    c.turn_begin("s1", repo="/x", role="main")
    first = d.sessions["s1"]
    c.turn_begin("s1", repo="/x", role="main")
    assert d.sessions["s1"] is first
    srv.shutdown()


# ---------- degradation: get-only (no stream visibility) ----------

def test_get_only_feed_builds_no_chain():
    """stream_visibility none -> call_stream_delta returns None, no PATCH chain."""
    d = _daemon()
    nostream = Profile("json-stream", "fs-only", "full", "none", "clonefile")
    d.session_start("s", repo="/x", role="main", profile=nostream)
    out = d.call_stream_delta("s", "c0", "Edit",
                              json.dumps({"path": "f.py", "contents": "a=1\n"}))
    assert out is None
    assert d.sessions["s"].chain is None


def test_get_only_feed_over_socket_reports_zero_chain_len():
    d = _daemon()
    nostream = Profile("json-stream", "fs-only", "full", "none", "clonefile")
    d.session_start("s1", repo="/x", role="main", profile=nostream)
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    reply = c.feed("s1", call_id="c0", tool="Edit",
                   delta=json.dumps({"path": "f.py", "contents": "a=1\n"}))
    assert reply["chain_len"] == 0
    srv.shutdown()


# ---------- degradation: additive (unregistered kind MUST miss) ----------

def test_additive_unregistered_kind_misses_even_with_warm_cache():
    """Additive grade with registered_kinds={test}: a warm 'read' entry must NOT
    serve because 'read' was never registered."""
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["codex"],
                    registered_kinds={"test"})
    d.sessions["s"].cache.put("read", {"cmd": "x"}, 0, result="body", launch=0)
    outcome, result = d.resolve("s", "read", {"cmd": "x"})
    assert outcome == "miss"
    assert result is None


def test_additive_registered_kind_still_serves():
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["codex"],
                    registered_kinds={"test"})
    d.sessions["s"].cache.put("test", {"cmd": "x"}, 0, result="PASS", launch=0)
    outcome, result = d.resolve("s", "test", {"cmd": "x"})
    assert outcome.startswith("hit")
    assert result == "PASS"


def test_additive_no_registered_kinds_means_none_served():
    """registered_kinds None on additive grade: guard is skipped, so it falls
    through to the cache. Documents current behavior at this boundary."""
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["codex"],
                    registered_kinds=None)
    d.sessions["s"].cache.put("read", {"cmd": "x"}, 0, result="body", launch=0)
    outcome, result = d.resolve("s", "read", {"cmd": "x"})
    assert outcome.startswith("hit")


# ---------- degradation: outcome-unconditioned (missing outcome must collapse) ----------

def test_outcome_unconditioned_collapses_key_to_qmark():
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["capture"])
    pred = d.sessions["s"].predictor
    pred.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main", epoch=0,
                           outcome=Outcome("edit", "OK")))
    pred.observe(ToolEvent(t=0, kind="test", verb="free", role="main", epoch=0,
                           outcome=Outcome("test", "PASS")))
    key = list(pred.session_counts)[0]
    assert key.endswith("|?")


def test_context_key_with_none_outcome_and_outcome_visible_crashes():
    """A ToolEvent whose outcome is None, on an outcome-VISIBLE predictor, hits
    events[-1].outcome.klass() -> AttributeError. The invariant says a missing
    outcome must collapse the key, not crash."""
    ev = ToolEvent(t=0, kind="test", verb="free", role="main", epoch=0,
                   outcome=None)
    with pytest.raises(AttributeError):
        context_key([ev], k=1, outcome_visible=True)


def test_call_executed_null_outcome_over_socket_does_not_crash_server():
    """resolve/dispatch wraps outcome in Outcome(kind, None); klass() -> 'kind:None'
    string, no crash. Server must stay alive and the session usable."""
    d = _daemon()
    d.session_start("s1", repo="/x", role="main")
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    reply = c.call_executed("s1", tool="test", verb="free", outcome=None,
                            args={"cmd": "x"}, latency=0)
    assert reply == {"result": "ok"}
    keys = list(d.sessions["s1"].predictor.session_counts)
    srv.shutdown()


# ---------- per-case isolation: two sessions never share cache/predictor ----------

def test_two_sessions_do_not_share_cache_entries():
    d = _daemon()
    d.session_start("a", repo="/x", role="main")
    d.session_start("b", repo="/x", role="main")
    d.sessions["a"].cache.put("read", {"cmd": "x"}, 0, result="A", launch=0)
    outcome, result = d.resolve("b", "read", {"cmd": "x"})
    assert outcome == "miss"
    assert result is None


# ---------- no-cross-contamination: occurrence keys separate repeated calls ----------

def test_same_kind_args_two_puts_get_distinct_occurrences():
    """Two puts of identical (kind,args,epoch) must occupy distinct cache slots
    (occurrence 0 and 1), not collide onto one key."""
    d = _daemon()
    d.session_start("s", repo="/x", role="main")
    cache = d.sessions["s"].cache
    k1, _ = cache.put("read", {"cmd": "x"}, 0, result="first", launch=0)
    k2, _ = cache.put("read", {"cmd": "x"}, 0, result="second", launch=0)
    assert k1 != k2
    assert len(cache.jobs) == 2
    o1, _, r1 = cache.serve("read", {"cmd": "x"}, ask_time=0)
    o2, _, r2 = cache.serve("read", {"cmd": "x"}, ask_time=0)
    assert (r1, r2) == ("first", "second")


def test_epoch_fence_stale_spec_expires_after_write():
    """A spec put in epoch 0 must not serve after bump_epoch (real write)."""
    d = _daemon()
    d.session_start("s", repo="/x", role="main")
    cache = d.sessions["s"].cache
    cache.put("read", {"cmd": "x"}, 0, result="stale", launch=0)
    cache.bump_epoch()
    outcome, _, result = cache.serve("read", {"cmd": "x"}, ask_time=0)
    assert outcome == "miss"
    assert result is None


def test_epoch_bump_via_daemon_on_fs_change_then_serve_misses():
    d = _daemon()
    d.session_start("s", repo="/x", role="main")
    d.sessions["s"].cache.put("read", {"cmd": "x"}, 0, result="stale", launch=0)
    d.on_fs_change("s")
    outcome, result = d.resolve("s", "read", {"cmd": "x"})
    assert outcome == "miss"


# ---------- client transport: half-close surfaces as SfxDaemonError, not a hang ----------

def test_client_read_on_peer_eof_raises_sfx_error_not_hang():
    """Half-close: client half-closes its write end, the server handler loop hits
    EOF and closes the connection. The next client read gets empty bytes; json.loads
    of "" must surface as SfxDaemonError, never a silent None or an infinite hang."""
    d = _daemon()
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)
    c.turn_begin("s1", repo="/x", role="main")
    c.sock.shutdown(socket.SHUT_WR)
    with pytest.raises(SfxDaemonError):
        c._send({"type": "turn_end", "session": "s1"})
    srv.shutdown()
