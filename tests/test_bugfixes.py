import json

import pytest

from adapters import mini_swe
from adapters.protocol import SfxDaemonError
from sfx.cache import Cache
from sfx.chain import WritePathError, run_chain
from sfx.daemon import Daemon
from sfx.executor import Executor
from sfx.ledger import Ledger


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _job(result, duration):
    return lambda: (result, duration)


# 1. cache stores/serves the real result bytes
def test_cache_serves_real_result_bytes():
    clock = FakeClock(0)
    led = Ledger()
    cache = Cache(clock, led)
    ex = Executor(clock=clock, cache=cache, ledger=led, slots=2)

    ex.speculate("grep", {"cmd": "grep foo"}, priority=5.0, job=_job("hit1\nhit2\n", 100))

    clock.t = 500
    outcome, _, result = cache.serve("grep", {"cmd": "grep foo"}, ask_time=500)
    assert outcome == "hit_completed"
    assert result == "hit1\nhit2\n"


# 2. slot leak: completed specs are reclaimed so free_slots recovers
def test_free_slots_reclaims_completed_specs():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)

    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 100))
    ex.speculate("lint", {"cmd": "ruff"}, priority=4.0, job=_job("y", 100))
    assert ex.free_slots() == 0

    ex.drain()  # deterministic: wait for the background lane before checking synthetic reclaim
    clock.t = 200
    assert ex.free_slots() == 2
    assert ex.speculate("build", {"cmd": "make"}, priority=3.0, job=_job("z", 100)) is not None


# 3. streamed write missing 'path' is rejected, not a crash
def test_run_chain_missing_path_raises_writepatherror():
    class W:
        args = {"contents": "x"}

    with pytest.raises(WritePathError):
        run_chain(W(), repo="/r", scratch=None, apply_write=lambda p, w: None,
                  next_hop=lambda o: None, run_get=lambda p, h: "x",
                  depth_cap=1, clock=lambda: 0)


# 4. speculative job exception is a silent no-op
def test_speculate_swallows_job_exception():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)

    def boom():
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")

    ex.speculate("grep", {"cmd": "grep"}, priority=5.0, job=boom)
    ex.drain()  # deterministic: let the background lane run boom to completion

    assert ex.free_slots() == 2                 # failed spec reclaimed its slot
    assert len(ex.running) == 0
    assert ex.cache.serve("grep", {"cmd": "grep"}, ask_time=1)[0] == "miss"  # nothing served


# 5. resolve with null args is rejected at the boundary
def test_protocol_rejects_null_args():
    from adapters import protocol
    d = Daemon(clock=FakeClock(0), global_table={"k": 1, "table": {}}, k=1,
               run=lambda k, a: ("o", 1))
    d.session_start("s", repo="/r", role="main")
    reply = protocol._dispatch(d, {"type": "resolve", "session": "s",
                                   "tool": "grep", "args": None})
    assert "error" in reply


# 6. chain replacement terminalizes the outgoing chain's spec_ids
def test_chain_replacement_discards_previous_chain(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "HOME": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "s@x"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "s"], check=True, env=env)
    (repo / "foo.py").write_text("V=1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "i"], check=True, env=env)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    table = {"k": 1, "min_support": 1, "tau": 0.35,
             "table": {"main|edit|edit:OK": {"support": 20, "p": {"test": 0.9}}}}
    d = Daemon(clock=FakeClock(0), global_table=table, k=1,
               resolve_args=lambda k, ctx: {"cmd": "pytest"},
               apply_write=lambda fp, w: (fp / w.args["path"]).write_text(w.args["contents"]),
               run_in_fork=lambda fp, hop: (f"{hop[0]}-result", "PASS"), depth_cap=1)
    d.session_start("s", repo=str(repo), role="main", scratch=str(scratch))

    def stream(contents):
        body = json.dumps({"path": "foo.py", "contents": contents})
        for i in range(0, len(body), 6):
            d.call_stream_delta("s", "c1", "Edit", body[i:i + 6])

    stream("V=2\n")
    first_ids = list(d.sessions["s"].chain_spec_ids)
    stream("V=3\n")

    led = d.sessions["s"].ledger
    for sid in first_ids:
        assert sid in led._terminated


# 7. bump_epoch keeps running/_specs consistent; preempt skips terminated
def test_preempt_skips_epoch_terminated_specs():
    clock = FakeClock(0)
    led = Ledger()
    cache = Cache(clock, led)
    ex = Executor(clock=clock, cache=cache, ledger=led, slots=2)
    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 10000))
    ex.speculate("lint", {"cmd": "ruff"}, priority=1.0, job=_job("y", 10000))

    clock.t = 100
    cache.bump_epoch()

    ex.authoritative(_job("r", 100), need_slot=True)
    assert led.terminal_counts().get("preempted", 0) == 0


# 8. repo_table wired: predictor blends a third (repo) tier
def test_repo_table_three_tier(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".sfx").mkdir(parents=True)
    global_tbl = {"k": 1, "min_support": 1, "tau": 0.35,
                  "table": {"main|edit|edit:OK": {"support": 1, "p": {"test": 1.0}}}}
    repo_tbl = {"k": 1, "table": {"main|edit|edit:OK": {"support": 100, "p": {"lint": 1.0}}}}
    (repo / ".sfx" / "table.json").write_text(json.dumps(repo_tbl))

    d = Daemon(clock=FakeClock(0), global_table=global_tbl, k=1,
               run=lambda k, a: ("o", 1), resolve_args=lambda k, ctx: {"cmd": "x"})
    s = d.session_start("s", repo=str(repo), role="main")
    from sfx.schema import Outcome, ToolEvent
    s.predictor.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main",
                                  epoch=0, args={}, outcome=Outcome("edit", "OK")))
    ranked = dict(s.predictor.propose())
    assert ranked["lint"] > ranked["test"]


# 9. k derived from the table, not the CLI depth arg
def test_daemon_k_comes_from_table():
    d = Daemon(clock=FakeClock(0), global_table={"k": 3, "table": {}}, k=1,
               run=lambda k, a: ("o", 1))
    assert d.k == 3


# 10. bump_epoch drops expired jobs from cache.jobs
def test_bump_epoch_drops_expired_jobs():
    clock = FakeClock(0)
    cache = Cache(clock, Ledger())
    cache.put("test", {"cmd": "pytest"}, duration=100)
    cache.put("lint", {"cmd": "ruff"}, duration=100)
    assert len(cache.jobs) == 2

    clock.t = 1000
    cache.bump_epoch()
    assert len(cache.jobs) == 0


# 11. fail-open: daemon errors degrade to real execution, never abort
def test_claim_or_none_fails_open_on_daemon_error():
    class DeadClient:
        def resolve(self, *a, **k):
            raise SfxDaemonError("socket dead")

    assert mini_swe.claim_or_none(DeadClient(), "test", "pytest") is None


def test_report_executed_fails_open_on_daemon_error():
    class DeadClient:
        def call_executed(self, *a, **k):
            raise SfxDaemonError("socket dead")

    mini_swe.report_executed(DeadClient(), "test", "free", "pytest", returncode=0, latency=1.0)


def test_handle_line_returns_error_frame_on_any_exception():
    from adapters import protocol

    def boom(msg):
        raise RuntimeError("state corrupt")

    reply = protocol._handle_line(boom, '{"type":"resolve"}')
    assert reply == {"error": "state corrupt"}


# 12. git kind resolves to a valid policy (no KeyError) and is never speculated
def test_git_kind_resolves_without_keyerror_and_never_speculated():
    from sfx.daemon import KIND_POLICY

    assert KIND_POLICY["git"] == "never"

    steps = []
    table = {"k": 1, "min_support": 1, "tau": 0.35,
             "table": {"main|edit|edit:OK": {"support": 10, "p": {"git": 0.9}}}}
    d = Daemon(clock=FakeClock(0), global_table=table, k=1,
               run=lambda k, a: ("o", 1),
               resolve_args=lambda k, ctx: {"cmd": "git commit"},
               log=lambda rec: steps.append(rec) if rec.get("ev") == "step" else None)
    d.session_start("s", repo="/r", role="main")

    d.call_executed("s", kind="edit", verb="fork", outcome="OK",
                    args={"path": "foo.py"}, latency=10)

    step = steps[-1]
    assert step["kind"] == "git" and step["verb"] == "never"
    assert step["gate"] == "reject" and step["gate_reason"] == "not_speculable"
    assert step["spec"] is None
