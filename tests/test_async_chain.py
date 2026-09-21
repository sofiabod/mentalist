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


def _stream_edit(d, sid, path="v.txt", contents="1\n"):
    body = json.dumps({"path": path, "contents": contents})
    for i in range(0, len(body), 6):
        d.call_stream_delta(sid, "e0", "Write", body[i:i + 6])


def _resolve_args(kind, ctx):
    r = resolver.resolve(kind, ctx)
    if r:
        return r
    if kind == "run":
        return {"cmd": "python3 foo.py"}
    return None


def _daemon(repo, run, run_in_fork, log, clock=None):
    return Daemon(clock=clock or RealClock(), global_table=_edit_run_table(), k=1,
                  run=run,
                  resolve_args=_resolve_args,
                  apply_write=_apply, run_in_fork=run_in_fork,
                  depth_cap=1, log=log.append)


# 1. the predicted run runs on a background (non-caller) thread, not inline in the
#    stream-delta call, and its result serves the later real run call.
def test_chain_run_executes_on_background_thread(repo, scratch):
    caller = threading.get_ident()
    ran_on = {}
    started = threading.Event()

    def run_in_fork(fp, hop):
        ran_on["tid"] = threading.get_ident()
        started.set()
        return ("chained-out", 0), "OK"

    log = []
    d = _daemon(repo, lambda k, a: (("x", 0), 10.0), run_in_fork, log)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    t0 = time.monotonic()
    _stream_edit(d, "s")
    submit_ms = (time.monotonic() - t0) * 1000.0

    assert started.wait(5.0)
    assert ran_on["tid"] != caller

    d.sessions["s"].chain.future.result(timeout=5.0)
    d.resolve("s", kind="edit", args={"path": "v.txt", "contents": "1\n"})
    outcome, result = d.resolve("s", kind="run", args={"cmd": "python3 foo.py"})
    assert outcome in ("hit_completed", "hit_promoted")
    assert result == ("chained-out", 0)
    d.shutdown()


def test_chain_not_done_by_join_deadline_is_safe_miss(repo, scratch, monkeypatch):
    import sfx.daemon as daemon_module

    join_bound = 0.02
    monkeypatch.setattr(daemon_module, "RESOLVE_JOIN_TIMEOUT_S", join_bound)
    release = threading.Event()

    def run_in_fork(fp, hop):
        release.wait(5.0)
        return ("late", 0), "OK"

    log = []
    d = _daemon(repo, lambda k, a: (("x", 0), 10.0), run_in_fork, log)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    future = None
    try:
        _stream_edit(d, "s")
        session = d.sessions["s"]
        future = session.chain.future
        d.resolve("s", kind="edit", args={"path": "v.txt", "contents": "1\n"})
        t0 = time.monotonic()
        outcome, result = d.resolve("s", kind="run", args={"cmd": "python3 foo.py"})
        waited = time.monotonic() - t0
        assert (outcome, result) == ("miss", None)
        assert join_bound <= waited < 1.0
        assert session.chain is session.chain_claim is None
        assert session.ledger.total_saved_ms == 0
        assert any(row.get("discard_reason") == "chain_join_timeout" for row in log)
        release.set()
        future.result(timeout=5)
        assert not session.cache.jobs
        assert d.resolve("s", kind="run", args={"cmd": "python3 foo.py"}) == ("miss", None)
    finally:
        release.set()
        if future is not None:
            future.result(timeout=5)
        d.shutdown()


def test_second_edit_fences_stale_chain(repo, scratch):
    def run_in_fork(fp, hop):
        return ((fp / "v.txt").read_text(), 0), "OK"

    log = []
    d = _daemon(repo, lambda k, a: (("x", 0), 10.0), run_in_fork, log)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    try:
        _stream_edit(d, "s", contents="1\n")
        session = d.sessions["s"]
        first_chain = session.chain
        first_spec = session.chain_spec_ids[0]
        first_chain.future.result(timeout=5)
        first_job = next(job for job in session.cache.jobs.values() if job.spec_id == first_spec)
        assert first_job.result == ("1\n", 0)
        _stream_edit(d, "s", contents="2\n")
        assert session.chain is not first_chain
        assert first_spec in session.ledger._terminated
        assert not any(job.spec_id == first_spec for job in session.cache.jobs.values())
        first_terminal = next(row for row in session.ledger.records
                              if row["ev"] == "spec_end" and row["id"] == first_spec)
        assert first_terminal["terminal"] in ("discarded", "epoch_expired")
        d.resolve("s", kind="edit", args={"path": "v.txt", "contents": "2\n"})
        outcome, result = d.resolve("s", kind="run", args={"cmd": "python3 foo.py"})
        assert outcome in ("hit_completed", "hit_promoted")
        assert result == ("2\n", 0)
    finally:
        d.shutdown()


# 4. byte gate: the served chained result is byte-identical to a live run of the
#    same hop against the post-edit filesystem.
def test_served_chain_is_byte_identical_to_live(repo, scratch):
    # a real subprocess so stdout/rc come from the actual command in the fork
    def run_in_fork(fp, hop):
        from mining.normalize import _status
        kind, _verb, args = hop
        p = subprocess.run(args["cmd"], shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, cwd=fp)
        return (p.stdout, p.returncode), _status(p.returncode != 0, kind)

    def run(kind, args):
        p = subprocess.run(args["cmd"], shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, cwd=repo)
        return (p.stdout, p.returncode), 10.0

    (repo / "foo.py").write_text("print(open('v.txt').read().strip())\n")
    log = []
    d = _daemon(repo, run, run_in_fork, log)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    _stream_edit(d, "s", contents="42\n")
    # wait for chain
    for _ in range(100):
        if d.sessions["s"].chain and d.sessions["s"].chain.future.done():
            break
        time.sleep(0.02)
    d.resolve("s", kind="edit", args={"path": "v.txt", "contents": "42\n"})
    outcome, served = d.resolve("s", kind="run", args={"cmd": "python3 foo.py"})
    d.shutdown()
    assert outcome in ("hit_completed", "hit_promoted")

    # independent source of truth: apply the same edit live, run the same cmd
    (repo / "v.txt").write_text("42\n")
    p = subprocess.run("python3 foo.py", shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, cwd=repo)
    live = (p.stdout, p.returncode)
    assert served == live
