import json
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

from adapters.protocol import Client, serve
from sfx.daemon import Daemon


def _clock():
    c = {"t": 0.0}

    def tick():
        c["t"] += 100.0
        return c["t"]
    return tick


def _table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.99}}}}


def _sock():
    d = Path(tempfile.gettempdir()) / uuid.uuid4().hex[:8]
    d.mkdir()
    return str(d / "s.sock")


def _server(run=lambda kind, args: ("out", 3000)):
    d = Daemon(clock=_clock(), global_table=_table(), k=1, run=run,
               resolve_args=lambda kind, ctx: {"cmd": "x"})
    sp = _sock()
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, sp


def _live_worker_threads(before):
    return [t for t in threading.enumerate()
            if t not in before and t.is_alive() and not t.daemon]


def test_open_client_leaks_nondaemon_worker_after_shutdown():
    """A live Client connection keeps a non-daemon server thread alive past
    server.shutdown(); that thread is what blocks interpreter exit (the hang)."""
    before = set(threading.enumerate())
    srv, sp = _server()
    c = Client(sp)
    c.turn_begin("s1", repo="/tmp/x", role="main")
    time.sleep(0.2)
    srv.shutdown()
    time.sleep(0.2)
    leaked = _live_worker_threads(before)
    try:
        assert leaked == [], (
            "server left a live non-daemon worker thread blocked on the open "
            "client socket after shutdown(); this blocks interpreter exit. "
            f"leaked={leaked}")
    finally:
        c.f.close()
        c.sock.close()
        srv.server_close()


def test_client_has_no_close_method():
    """Client owns a socket but exposes no way to release it; callers that hold
    a Client (SfxEnvironment) cannot free the server worker thread."""
    assert hasattr(Client, "close"), (
        "protocol.Client has no close(); an open connection can never be torn "
        "down by the owner, so the server worker thread never exits")


def test_closing_client_releases_worker_thread():
    """Independent-truth control: once the client fd is closed the handler hits
    EOF and the worker thread must die; if it does not, teardown is broken."""
    before = set(threading.enumerate())
    srv, sp = _server()
    c = Client(sp)
    c.turn_begin("s1", repo="/tmp/x", role="main")
    time.sleep(0.2)
    c.f.close()
    c.sock.close()
    time.sleep(0.3)
    srv.shutdown()
    try:
        assert _live_worker_threads(before) == []
    finally:
        srv.server_close()


def test_resolve_null_args_fails_open_no_crash():
    """FAIL-OPEN: a resolve carrying null args must be rejected without killing
    the server; the connection must remain usable afterward."""
    srv, sp = _server()
    c = Client(sp)
    try:
        c.turn_begin("s1", repo="/tmp/x", role="main")
        bad = c.resolve("s1", tool="read", args=None)
        assert "error" in bad
        ok = c.turn_end("s1")
        assert ok == {"result": "ok"}
    finally:
        c.f.close()
        c.sock.close()
        srv.shutdown()
        srv.server_close()
