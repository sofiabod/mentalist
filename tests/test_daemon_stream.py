import json
import subprocess

import pytest

from sfx.daemon import Daemon


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


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


def _table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.9}}}}


def _apply_write(fork_path, write):
    (fork_path / write.args["path"]).write_text(write.args["contents"])


def test_streamed_write_triggers_patch_chain_in_fork_repo_untouched(repo, scratch):
    clock = FakeClock(0)
    observed = {}

    def run_in_fork(fork_path, hop):
        observed["contents"] = (fork_path / "foo.py").read_text()
        observed["kind"] = hop[0]
        return ("out", "PASS")

    d = Daemon(clock=clock, global_table=_table(), k=1,
               resolve_args=lambda kind, ctx: {"cmd": "pytest"},
               apply_write=_apply_write, run_in_fork=run_in_fork)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    body = json.dumps({"path": "foo.py", "contents": "VALUE = 42\n"})
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        chain = d.call_stream_delta("s1", "c1", tool="Write", delta=ch)

    assert chain is not None
    chain.future.result(timeout=5.0)  # the chain runs async on the executor lane
    assert observed["contents"] == "VALUE = 42\n"
    assert observed["kind"] == "test"
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"
    d.shutdown()


def _chain_table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.9}},
                      "main|test|test:PASS": {"support": 10, "p": {"lint": 0.9}}}}


def _stream_write(d, sid, path="foo.py", contents="VALUE = 42\n"):
    body = json.dumps({"path": path, "contents": contents})
    chain = None
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        chain = d.call_stream_delta(sid, "c1", tool="Write", delta=ch)
    if chain is not None and chain.future is not None:
        chain.future.result(timeout=5.0)  # think-gap: let the async chain land
    return chain


def _chain_daemon(clock, args_by_kind):
    def resolve_args(kind, ctx):
        return args_by_kind[kind]

    d = Daemon(clock=clock, global_table=_chain_table(), k=1,
               resolve_args=resolve_args,
               apply_write=_apply_write,
               run_in_fork=lambda fp, hop: ("out", "PASS"),
               depth_cap=1)
    return d


def test_resolve_serves_cached_chain_hop_after_edit(repo, scratch):
    clock = FakeClock(0)
    args = {"test": {"cmd": "pytest"}, "lint": {"cmd": "ruff"}}
    d = _chain_daemon(clock, args)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    chain = _stream_write(d, "s1")
    assert [h.kind for h in chain.hops] == ["test"]  # single PATCH->RUN hop

    clock.t = 1000
    assert d.resolve("s1", kind="edit",
                     args={"path": "foo.py", "contents": "VALUE = 42\n"})[0] == "hit_completed"
    assert d.resolve("s1", kind="test", args={"cmd": "pytest"})[0] == "hit_completed"
    d.shutdown()


def test_divergent_real_call_misses(repo, scratch):
    clock = FakeClock(0)
    args = {"test": {"cmd": "pytest"}, "lint": {"cmd": "ruff"}}
    d = _chain_daemon(clock, args)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    _stream_write(d, "s1")

    clock.t = 1000
    assert d.resolve("s1", kind="edit",
                     args={"path": "foo.py", "contents": "VALUE = 42\n"})[0] == "hit_completed"
    assert d.resolve("s1", kind="typecheck", args={"cmd": "mypy"})[0] == "miss"

    ledger = d.sessions["s1"].ledger
    assert ledger.terminal_counts()["discarded"] == 1
    d.shutdown()


def test_tier_a_write_with_revised_args_misses(repo, scratch):
    clock = FakeClock(0)
    args = {"test": {"cmd": "pytest"}, "lint": {"cmd": "ruff"}}
    d = _chain_daemon(clock, args)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    _stream_write(d, "s1", contents="VALUE = 42\n")

    clock.t = 1000
    assert d.resolve("s1", kind="edit",
                     args={"path": "foo.py", "contents": "VALUE = 99\n"})[0] == "miss"


def _run_table():
    return {"k": 2, "min_support": 1, "tau": 0.35,
            "table": {"main|read,edit|edit:OK": {"support": 10, "p": {"run": 0.9}},
                      "main|edit,edit|edit:OK": {"support": 10, "p": {"edit": 0.9}}}}


def _stream_write_call(d, sid, call_id, path, contents):
    body = json.dumps({"path": path, "contents": contents})
    chain = None
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        chain = d.call_stream_delta(sid, call_id, tool="Write", delta=ch)
    if chain is not None and chain.future is not None:
        chain.future.result(timeout=5.0)
    return chain


def test_interleaved_edits_carry_run_spec_to_final_fork(repo, scratch):
    clock = FakeClock(0)

    def run_in_fork(fork_path, hop):
        files = sorted(p.name for p in fork_path.iterdir() if p.suffix == ".py")
        return (",".join(files), "PASS")

    d = Daemon(clock=clock, global_table=_run_table(), k=2,
               resolve_args=lambda kind, ctx: {"cmd": "python a.py"},
               apply_write=_apply_write, run_in_fork=run_in_fork, depth_cap=1)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    d.call_executed("s1", kind="read", verb="free", outcome="OK",
                    args={"cmd": "cat foo.py"}, latency=0.0)

    for name in ("a.py", "b.py", "c.py"):
        _stream_write_call(d, "s1", name, name, f"# {name}\n")
        assert d.resolve("s1", kind="edit",
                         args={"path": name, "contents": f"# {name}\n"})[0] == "hit_completed"
        _apply_write(repo, type("W", (), {"args": {"path": name, "contents": f"# {name}\n"}})())

    clock.t = 1000
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    assert outcome == "hit_completed"
    assert result == "a.py,b.py,c.py,foo.py"


def test_no_substrate_fails_open_skips_write_spec(repo, scratch, monkeypatch):
    from sfx import fork as forkmod
    monkeypatch.setattr(forkmod, "_probe_cp", lambda *a: False)
    monkeypatch.setattr(forkmod, "_probe_overlay", lambda: False)
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_table(), k=1,
               resolve_args=lambda kind, ctx: {"cmd": "pytest"},
               apply_write=_apply_write, run_in_fork=lambda fp, hop: ("out", "PASS"))
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    chain = _stream_write(d, "s1")

    assert chain is None
    assert d.sessions["s1"].chain is None
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"


def test_incomplete_stream_triggers_no_chain(repo, scratch):
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_table(), k=1,
               resolve_args=lambda kind, ctx: {"cmd": "pytest"},
               apply_write=_apply_write, run_in_fork=lambda fp, hop: ("out", "PASS"))
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))

    result = d.call_stream_delta("s1", "c1", tool="Write", delta='{"path": "foo.py"')

    assert result is None
