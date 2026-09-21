import pytest

from sfx.daemon import Daemon


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {
        "k": 1,
        "min_support": 1,
        "tau": 0.35,
        "table": {
            "main|edit|edit:FAIL": {"support": 10, "p": {"test": 0.9}},
        },
    }


def _spec_job(kind, duration):
    return lambda k, args: (f"{kind}-out", duration)


def _new_daemon(clock):
    return Daemon(clock=clock, global_table=_global_table(), k=1,
                  run=_spec_job("x", 3000),
                  resolve_args=lambda kind, ctx: {"cmd": "pytest"})


def test_fs_change_bumps_epoch_without_adapter_event():
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_global_table(), k=1)
    s = d.session_start("sess1", repo="/r", role="main")
    assert s.cache.epoch == 0

    d.on_fs_change("sess1")

    assert s.cache.epoch == 1


def test_idle_timer_fires_without_adapter_event():
    clock = FakeClock(0)
    fired = []
    d = Daemon(clock=clock, global_table=_global_table(), k=1,
               on_idle=lambda sid: fired.append(sid))
    d.session_start("sess1", repo="/r", role="main")

    clock.t = 100
    d.tick(idle_after=50)

    assert fired == ["sess1"]


@pytest.mark.parametrize("teardown", ["session_end", "shutdown"])
def test_teardown_flushes_expiration_cost_after_last_turn(tmp_path, teardown):
    clock = FakeClock(0)
    events = []
    d = Daemon(clock=clock, global_table=_global_table(), k=1,
               log=events.append)
    s = d.session_start("sess1", repo=tmp_path, role="main")
    _, spec_id = s.cache.put("test", {"cmd": "pytest"}, duration=25,
                            result="unused")
    d.turn_end("sess1")
    assert not events
    clock.t = 100

    if teardown == "session_end":
        d.session_end("sess1")
    else:
        d.shutdown()

    assert events == [{"ev": "spec_end", "id": spec_id,
                       "terminal": "epoch_expired", "wasted_cpu_ms": 25}]
    assert s.ledger.total_wasted_ms == 25
    assert not d.sessions
    d.shutdown()
    assert len(events) == 1


def test_edit_event_predicts_admits_and_launches_speculative():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    d.session_start("sess1", repo="/r", role="main")

    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    s = d.sessions["sess1"]
    assert len(s.executor.running) == 1


def test_matching_real_call_serves_cache_hit():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    d.session_start("sess1", repo="/r", role="main")
    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    clock.t = 5000
    result = d.resolve("sess1", kind="test", args={"cmd": "pytest"})

    assert result[0] == "hit_completed"


def test_unpredicted_real_call_is_a_miss():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    d.session_start("sess1", repo="/r", role="main")
    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    result = d.resolve("sess1", kind="lint", args={"cmd": "ruff"})

    assert result[0] == "miss"


def test_hit_promoted_promotes_spec():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    d.session_start("sess1", repo="/r", role="main")
    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    s = d.sessions["sess1"]
    s.executor.slots = 2
    spec_key = next(iter(s.executor.running))
    s.executor.speculate("lint", {"cmd": "ruff"}, priority=99.0,
                         job=lambda: ("lint-out", 3000))

    clock.t = 1000
    assert d.resolve("sess1", kind="test", args={"cmd": "pytest"})[0] == "hit_promoted"
    assert s.executor._specs[spec_key].promoted is True

    s.executor.authoritative(lambda: ("r", 100), need_slot=True)

    assert spec_key in s.executor.running
    assert s.ledger.terminal_counts()["preempted"] == 1


def test_get_breadth_capped_at_two_per_step():
    clock = FakeClock(0)
    table = {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {
            "main|test|test:PASS": {"support": 10,
                "p": {"read": 0.9, "grep": 0.9, "search": 0.9, "lint": 0.9}},
        },
    }
    d = Daemon(clock=clock, global_table=table, k=1,
               run=_spec_job("x", 3000),
               resolve_args=lambda kind, ctx: {"cmd": kind})
    d.session_start("s", repo="/r", role="main")

    d.call_executed("s", kind="test", verb="free", outcome="PASS",
                    args={}, latency=10)

    assert len(d.sessions["s"].executor.running) == 2


def test_one_daemon_holds_distinct_repositories_independently():
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_global_table(), k=1)
    d.session_start("a", repo="/r", role="main")
    d.session_start("b", repo="/other", role="main")

    d.on_fs_change("a")

    assert d.sessions["a"].cache.epoch == 1
    assert d.sessions["b"].cache.epoch == 0


def test_spec_disabled_emits_step_but_launches_no_spec():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    steps = []
    d.log = lambda rec: steps.append(rec)
    d.session_start("sess1", repo="/r", role="main", spec_disabled=True)

    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    s = d.sessions["sess1"]
    assert len(s.executor.running) == 0
    assert any(r["ev"] == "step" for r in steps)


def test_oracle_serves_every_recorded_call():
    clock = FakeClock(0)
    d = _new_daemon(clock)
    traj = [
        {"kind": "edit", "args": {"path": "foo.py"}},
        {"kind": "test", "args": {"cmd": "pytest"}},
    ]
    d.session_start("sess1", repo="/r", role="main", trajectory=traj)

    d.call_executed("sess1", kind="edit", verb="fork", outcome="FAIL",
                    args={"path": "foo.py"}, latency=10)

    clock.t = 5000
    outcome, _ = d.resolve("sess1", kind="test", args={"cmd": "pytest"})
    assert outcome.startswith("hit")
