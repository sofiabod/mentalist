import json
import subprocess

import pytest

from adapters import mini_swe
from sfx import resolver
from sfx.daemon import Daemon


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


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
    (r / "foo.py").write_text("V=1\n")
    (r / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts=''\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "i")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


def _table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 20, "p": {"test": 0.9}},
                      "main|read|read:OK": {"support": 20, "p": {"grep": 0.9}},
                      "main|grep|grep:OK": {"support": 20, "p": {"read": 0.9}},
                      "main|test|test:PASS": {"support": 20, "p": {"lint": 0.9}}}}


def _daemon(clock, log, repo):
    return Daemon(clock=clock, global_table=_table(), k=1,
                  run=lambda kind, args: (f"{kind}-out", 3000),
                  resolve_args=lambda kind, ctx: resolver.resolve(
                      kind, resolver.Ctx(repo=repo)),
                  apply_write=lambda fp, w: (fp / w.args["path"]).write_text(w.args["contents"]),
                  run_in_fork=lambda fp, hop: (f"{hop[0]}-out", "PASS"),
                  depth_cap=2, log=log.append)


def _ge(repo, kind, command="x"):
    return mini_swe._get_args(kind, command)


# 1. read -> edit -> test (fact-kind) yields >=1 serve through the adapter arg shapes
def test_live_sequence_produces_a_serve(repo, scratch, monkeypatch):
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    log = []
    clock = FakeClock(0)
    d = _daemon(clock, log, repo)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    d.call_executed("s", kind="read", verb="free", outcome="OK",
                    args=_ge(repo, "read", "cat foo.py"), latency=3000)
    clock.t = 100
    d.resolve("s", kind="grep", args=_ge(repo, "grep", "rg x"))
    d.call_executed("s", kind="edit", verb="fork", outcome="OK",
                    args={"path": "foo.py"}, latency=3000)
    clock.t = 5000
    d.resolve("s", kind="test", args=_ge(repo, "test", "pytest"))

    serves = [r for r in log if r["ev"] == "resolve"]
    assert any(r["outcome"] != "miss" for r in serves)


# 2. legible trace: step has proposal, gate, non-empty id; a miss is recorded
def test_trace_is_legible_and_records_misses(repo, scratch):
    log = []
    clock = FakeClock(0)
    d = _daemon(clock, log, repo)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    d.call_executed("s", kind="read", verb="free", outcome="OK", args=_ge(repo, "read"), latency=3000)
    clock.t = 100
    d.resolve("s", kind="lint", args=_ge(repo, "lint"))  # never speculated -> miss

    steps = [r for r in log if r["ev"] == "step"]
    assert steps
    st = steps[0]
    assert st["id"]
    assert st["proposal"]
    assert st["gate"] in ("execute", "reject")
    assert st["gate_reason"]
    misses = [r for r in log if r["ev"] == "resolve" and r["outcome"] == "miss"]
    assert misses


# 3. compositional fork->free serves on the live path via adapter arg shapes
def test_compositional_patch_get_serves(repo, scratch):
    log = []
    clock = FakeClock(0)
    d = _daemon(clock, log, repo)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    body = json.dumps({"path": "foo.py", "contents": "V=2\n"})
    for i in range(0, len(body), 6):
        d.call_stream_delta("s", "e0", "Write", body[i:i + 6])

    d.sessions["s"].chain.future.result(timeout=10.0)  # chain now runs async
    clock.t = 1000
    edit_out = d.resolve("s", kind="edit", args={"path": "foo.py", "contents": "V=2\n"})
    assert edit_out[0] == "hit_completed"
    test_out = d.resolve("s", kind="test", args=_ge(repo, "test", "pytest"))
    assert test_out[0] == "hit_completed"
    assert test_out[1] == "test-out"


def _run_table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|run|run:OK": {"support": 20, "p": {"edit": 0.9}},
                      "main|edit|edit:OK": {"support": 20, "p": {"run": 0.9}}}}


# 5. a run executed earlier resolves via the session tier so a later predicted
#    run (not config-derivable) admits to execute.
def test_session_tier_admits_predicted_run_to_execute(repo, scratch):
    log = []
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_run_table(), k=1,
               run=lambda kind, args: (f"{kind}-out", 3000),
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               depth_cap=2, log=log.append)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    d.call_executed("s", kind="run", verb="free", outcome="OK",
                    args={"cmd": "python repro.py"}, latency=3000)
    d.call_executed("s", kind="edit", verb="fork", outcome="OK",
                    args={"path": "foo.py"}, latency=3000)

    step = [r for r in log if r["ev"] == "step" and r["kind"] == "run"][-1]
    assert step["args_tier"] == "session"
    assert step["resolved_args"] == {"cmd": "python repro.py"}
    assert step["gate"] == "execute"
    assert step["spec"] is not None


# 6. when edit-churn keeps edit ranked top-1, a run above tau is still pre-launched
#    so the repro re-run does not have to wait for run to become top-1.
def test_run_above_tau_prelaunched_even_when_not_top1(repo, scratch):
    table = {"k": 1, "min_support": 1, "tau": 0.35,
             "table": {"main|edit|edit:OK": {"support": 20,
                                             "p": {"edit": 0.5, "run": 0.4}}}}
    log = []
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=table, k=1,
               run=lambda kind, args: (f"{kind}-out", 3000),
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               depth_cap=2, log=log.append)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    d.call_executed("s", kind="run", verb="free", outcome="OK",
                    args={"cmd": "python repro.py"}, latency=3000)
    d.call_executed("s", kind="edit", verb="fork", outcome="OK",
                    args={"path": "foo.py"}, latency=3000)

    steps = [r for r in log if r["ev"] == "step"]
    run_specs = [r for r in steps if r.get("spec") and r["spec"]["kind"] == "run"]
    assert run_specs, "run above tau should be pre-launched even when edit is top-1"


# 4. get-only sequence with no matching stream never enters a chain
def test_get_only_no_chain(repo, scratch):
    log = []
    clock = FakeClock(0)
    d = _daemon(clock, log, repo)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))
    d.call_executed("s", kind="read", verb="free", outcome="OK", args=_ge(repo, "read"), latency=3000)
    assert d.sessions["s"].chain is None
    assert not any(r["ev"] == "chain" for r in log)
