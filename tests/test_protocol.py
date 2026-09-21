import json
import socket
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from adapters.protocol import Client, serve
from sfx.daemon import Daemon


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null",
                        "GIT_CONFIG_SYSTEM": "/dev/null",
                        "HOME": str(cwd),
                        "PATH": "/usr/bin:/bin:/usr/local/bin"})


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {"main|edit|edit:FAIL": {"support": 10, "p": {"test": 0.9}}},
    }


def _daemon(clock):
    return Daemon(clock=clock, global_table=_global_table(), k=1,
                  run=lambda k, a: ("out", 3000),
                  resolve_args=lambda kind, ctx: {"cmd": "pytest"})


def _sock_path():
    d = Path(tempfile.gettempdir()) / uuid.uuid4().hex[:8]
    d.mkdir()
    return str(d / "s.sock")


def _run_server(daemon, sock_path):
    srv = serve(daemon, sock_path)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def test_round_trip_resolve_hit():
    clock = FakeClock(0)
    d = _daemon(clock)
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)

    c.turn_begin("s1", repo="/r", role="main")
    d.call_executed("s1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)
    clock.t = 5000
    reply = c.resolve("s1", tool="test", args={"cmd": "pytest"})
    c.turn_end("s1")

    assert reply["result"] == "hit_completed"
    srv.shutdown()


def test_round_trip_resolve_miss():
    clock = FakeClock(0)
    d = _daemon(clock)
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)

    c.turn_begin("s1", repo="/r", role="main")
    reply = c.resolve("s1", tool="lint", args={"cmd": "ruff"})

    assert reply["result"] == "miss"
    srv.shutdown()


def test_turn_begin_creates_session():
    clock = FakeClock(0)
    d = _daemon(clock)
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)

    c.turn_begin("s1", repo="/r", role="main")

    assert "s1" in d.sessions
    srv.shutdown()


def test_malformed_message_returns_error_and_server_survives():
    clock = FakeClock(0)
    d = _daemon(clock)
    sp = _sock_path()
    srv = _run_server(d, sp)

    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(sp)
    raw.sendall(b"not json at all\n")
    err = json.loads(raw.makefile().readline())
    raw.close()

    assert "error" in err

    c = Client(sp)
    c.turn_begin("s1", repo="/r", role="main")
    assert "s1" in d.sessions
    srv.shutdown()


def _chain_table():
    return {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.9}}},
    }


def _apply_write(fork_path, write):
    (fork_path / write.args["path"]).write_text(write.args["contents"])


def test_feed_round_trip_streams_write_and_serves_chain_over_socket(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "s@x.com")
    _git(repo, "config", "user.name", "s")
    (repo / "foo.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_chain_table(), k=1,
               resolve_args=lambda kind, ctx: {"cmd": "pytest"},
               apply_write=_apply_write,
               run_in_fork=lambda fp, hop: ("out", "PASS"), depth_cap=1)
    sp = _sock_path()
    srv = _run_server(d, sp)
    c = Client(sp)

    c.turn_begin("s1", repo=str(repo), role="main")
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    body = json.dumps({"path": "foo.py", "contents": "VALUE = 42\n"})
    chain_len = None
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        reply = c.feed("s1", call_id="c1", tool="Write", delta=ch)
        chain_len = reply["chain_len"]

    assert chain_len == 1
    d.sessions["s1"].chain.future.result(timeout=10.0)  # chain now runs async
    clock.t = 1000
    edit = c.resolve("s1", tool="edit",
                     args={"path": "foo.py", "contents": "VALUE = 42\n"})
    hop = c.resolve("s1", tool="test", args={"cmd": "pytest"})
    c.turn_end("s1")

    assert edit["result"] == "hit_completed"
    assert hop["result"] == "hit_completed"
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"
    srv.shutdown()
