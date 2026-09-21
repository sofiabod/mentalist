import pytest

from sfx.cache import Cache
from sfx.executor import Executor
from sfx.ledger import Ledger


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _job(result, duration):
    return lambda: (result, duration)


def test_authoritative_always_runs_returns_result():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)

    out = ex.authoritative(_job("r", 100))

    assert out == "r"


def test_speculative_bounded_by_slot_budget():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)

    a = ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 100))
    b = ex.speculate("lint", {"cmd": "ruff"}, priority=4.0, job=_job("y", 100))
    c = ex.speculate("build", {"cmd": "make"}, priority=3.0, job=_job("z", 100))

    assert a is not None and b is not None
    assert c is None
    assert len(ex.running) == 2


def test_speculative_result_lands_in_cache():
    clock = FakeClock(0)
    led = Ledger()
    cache = Cache(clock, led)
    ex = Executor(clock=clock, cache=cache, ledger=led, slots=2)

    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("PASS", 3000))

    clock.t = 5000
    assert cache.serve("test", {"cmd": "pytest"}, ask_time=5000)[0] == "hit_completed"


def test_authoritative_preempts_lowest_ev_speculative_when_full():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)
    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 100))
    low = ex.speculate("lint", {"cmd": "ruff"}, priority=1.0, job=_job("y", 100))

    clock.t = 40
    ex.authoritative(_job("r", 100), need_slot=True)

    counts = led.terminal_counts()
    assert counts["preempted"] == 1
    assert low not in ex.running
    assert led.total_wasted_ms == pytest.approx(40)


def test_promoted_job_not_preempted_even_if_lowest_ev():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=2)
    high = ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 100))
    low = ex.speculate("lint", {"cmd": "ruff"}, priority=1.0, job=_job("y", 100))
    ex.promote(low)

    ex.authoritative(_job("r", 100), need_slot=True)

    assert low in ex.running
    assert high not in ex.running
    assert led.terminal_counts()["preempted"] == 1


def test_all_promoted_no_preemption_authoritative_survives():
    clock = FakeClock(0)
    led = Ledger()
    ex = Executor(clock=clock, cache=Cache(clock, led), ledger=led, slots=1)
    key = ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_job("x", 100))
    ex.promote(key)

    out = ex.authoritative(_job("r", 100), need_slot=True)

    assert out == "r"
    assert key in ex.running
    assert led.terminal_counts()["preempted"] == 0
