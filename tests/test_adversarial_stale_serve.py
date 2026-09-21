import json
import subprocess
import threading
import time

import pytest

from sfx import resolver
from sfx.daemon import Daemon


class RealClock:
    def __call__(self):
        return time.monotonic() * 1000.0


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                        "HOME": str(cwd), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "s@x")
    _git(r, "config", "user.name", "s")
    (r / "v.txt").write_text("0\n")
    (r / "foo.py").write_text("print('run')\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "i")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


def _edit_run_table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 20, "p": {"run": 0.99}},
                      "main|run|run:OK": {"support": 20, "p": {"edit": 0.99}}}}


def _apply(fp, w):
    (fp / w.args["path"]).write_text(w.args["contents"])


def _stream_edit(d, sid, contents):
    body = json.dumps({"path": "v.txt", "contents": contents})
    for i in range(0, len(body), 6):
        d.call_stream_delta(sid, "e0", "Write", body[i:i + 6])


def _resolve_args(kind, ctx):
    r = resolver.resolve(kind, ctx)
    if r:
        return r
    if kind == "run":
        return {"cmd": "python3 foo.py"}
    return None


def _daemon(repo, run_in_fork, clock=None):
    return Daemon(clock=clock or RealClock(), global_table=_edit_run_table(), k=1,
                  run=lambda k, a: (("x", 0), 10.0),
                  resolve_args=_resolve_args,
                  apply_write=_apply, run_in_fork=run_in_fork,
                  depth_cap=1, log=(lambda r: None))


# adversarial scenario 1 variant: chain A is STILL IN FLIGHT when edit B arrives.
# A then completes LATE with pre-edit-B ("stale") bytes. The post-edit-B run resolve
# must NOT serve A's stale result. The existing suite only tested the case where A
# had already finished before edit B; this covers the racing-in-flight case.
def test_inflight_chain_A_fenced_by_edit_B(repo, scratch):
    hold = threading.Event()
    which = {"contents": None}

    def run_in_fork(fp, hop):
        # read what the fork sees so the result is tied to a specific edit
        seen = (fp / "v.txt").read_text().strip()
        if seen == "AAA":
            hold.wait(5.0)  # stall chain A until edit B has landed
        return ((f"out-{seen}", 0), "OK")

    d = _daemon(repo, run_in_fork)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    _stream_edit(d, "s", contents="AAA\n")          # chain A submitted, stalls in fork
    fut_a = d.sessions["s"].chain.future
    _stream_edit(d, "s", contents="BBB\n")          # edit B: must fence chain A
    hold.set()                                       # let A finish LATE
    fut_a.result(timeout=5.0)                        # A completed (or was discarded)

    d.sessions["s"].chain.future.result(timeout=5.0)  # wait for chain B
    d.resolve("s", kind="edit", args={"path": "v.txt", "contents": "BBB\n"})
    outcome, result = d.resolve("s", kind="run", args={"cmd": "python3 foo.py"})

    # must serve B's bytes (or clean miss), never A's stale "out-AAA"
    assert result != ("out-AAA", 0), f"served STALE pre-edit-B bytes: {result}"
    if outcome != "miss":
        assert result == ("out-BBB", 0), f"served wrong bytes: {result}"
    d.shutdown()
