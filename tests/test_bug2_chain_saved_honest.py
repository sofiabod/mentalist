import json
import subprocess
import threading
import time

import pytest

from sfx.daemon import Daemon

S = 120.0  # ms the slow chain tool takes


def _ms_clock():
    return time.monotonic() * 1000.0


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null",
                        "GIT_CONFIG_SYSTEM": "/dev/null",
                        "HOME": str(cwd),
                        "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "s@x.com")
    _git(r, "config", "user.name", "s")
    (r / "foo.py").write_text("VALUE = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


def _apply_write(fork_path, write):
    (fork_path / write.args["path"]).write_text(write.args["contents"])


def _table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.9}}}}


def _stream_write(d, sid, path="foo.py", contents="VALUE = 42\n"):
    body = json.dumps({"path": path, "contents": contents})
    chain = None
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        chain = d.call_stream_delta(sid, "c1", tool="Write", delta=ch)
    return chain


def _slow_run_in_fork(fork_path, hop):
    time.sleep(S / 1000.0)
    return ("out", "PASS")


def test_chain_run_does_not_block_stream_thread_for_full_duration(repo, scratch):
    clock = _ms_clock
    d = Daemon(clock=clock, global_table=_table(), k=1,
               resolve_args=lambda kind, ctx: {"cmd": "pytest"},
               apply_write=_apply_write, run_in_fork=_slow_run_in_fork)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    t0 = time.monotonic()
    _stream_write(d, "s1")
    stream_wall = (time.monotonic() - t0) * 1000.0

    # the slow RUN runs on the async executor lane, so parsing the streamed write
    # must NOT block the stream thread for the full run duration.
    assert stream_wall < S, (
        f"stream thread blocked {stream_wall:.0f}ms on the RUN: still synchronous")

    # model's think-gap: the chain finishes on the pool while the model generates.
    d.sessions["s1"].chain.future.result(timeout=5.0)

    assert d.resolve("s1", kind="edit",
                     args={"path": "foo.py", "contents": "VALUE = 42\n"})[0] == "hit_completed"
    d.resolve("s1", kind="test", args={"cmd": "pytest"})
    overlap = (time.monotonic() - t0) * 1000.0

    saved = d.sessions["s1"].ledger.total_saved_ms

    # honesty invariant: the RUN was hidden behind the (real) think-gap, so its cost
    # IS credited as saved wall; but never more than the wall actually overlapped.
    assert saved > 0
    assert saved <= overlap, (
        f"credited {saved:.0f}ms but only {overlap:.0f}ms of wall was overlapped")
    d.shutdown()
