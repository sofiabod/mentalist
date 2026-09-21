import pytest

from sfx.cache import Cache, _canon
from sfx.ledger import Ledger


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def test_occurrence_multiplicity_six_independent_entries():
    clock = FakeClock(0)
    cache = Cache(clock=clock, ledger=Ledger())
    keys = [cache.put("sub-LLM", {"prompt": "x"}, duration=100)[0] for _ in range(6)]

    occurrences = [k[2] for k in keys]
    assert occurrences == [0, 1, 2, 3, 4, 5]

    clock.t = 200
    served = [cache.serve("sub-LLM", {"prompt": "x"}, clock.t)[0] for _ in range(6)]
    assert all(o == "hit_completed" for o in served)

    clock.t = 300
    assert cache.serve("sub-LLM", {"prompt": "x"}, clock.t)[0] == "miss"


def test_completed_before_ask_is_hit_completed():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=3000)

    out, _, _ = cache.serve("test", {"cmd": "pytest"}, ask_time=6000)

    assert out == "hit_completed"
    assert led.total_saved_ms == pytest.approx(3000)


def test_promotion_joins_running_job_saved_min_duration_ask_minus_launch():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=8000)

    out, _, _ = cache.serve("test", {"cmd": "pytest"}, ask_time=6000)

    assert out == "hit_promoted"
    assert led.total_saved_ms == pytest.approx(5000)


def test_promoted_job_is_non_preemptible_and_survives_epoch_bump():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=8000)
    cache.serve("test", {"cmd": "pytest"}, ask_time=6000)

    clock.t = 6500
    cache.bump_epoch()

    assert led.terminal_counts()["hit_promoted"] == 1
    assert "epoch_expired" not in led.terminal_counts()


def test_completed_entry_unservable_after_epoch_bump():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=1000)

    clock.t = 3000
    cache.bump_epoch()

    assert cache.serve("test", {"cmd": "pytest"}, ask_time=4000)[0] == "miss"


def test_inflight_job_from_old_epoch_killed_on_bump():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=8000)

    clock.t = 3000
    cache.bump_epoch()

    assert led.terminal_counts()["epoch_expired"] == 1
    assert led.total_wasted_ms == pytest.approx(2000)
    assert cache.serve("test", {"cmd": "pytest"}, ask_time=4000)[0] == "miss"


def test_completed_unserved_wasted_cpu_is_capped_at_duration_on_bump():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("test", {"cmd": "pytest"}, duration=1000)

    clock.t = 3000
    cache.bump_epoch()

    assert led.total_wasted_ms == pytest.approx(1000)


def test_second_serve_of_same_args_gets_second_job_not_a_replay():
    clock = FakeClock(1000)
    led = Ledger()
    cache = Cache(clock=clock, ledger=led)
    cache.put("run", {"cmd": "python r.py"}, duration=8000, result="r0")
    cache.put("run", {"cmd": "python r.py"}, duration=8000, result="r1")

    out0, _, res0 = cache.serve("run", {"cmd": "python r.py"}, ask_time=6000)
    out1, _, res1 = cache.serve("run", {"cmd": "python r.py"}, ask_time=6000)

    assert out0 == "hit_promoted"
    assert res0 == "r0"
    assert res1 == "r1"


def test_failed_spec_serves_miss_not_crash():
    import threading

    clock = FakeClock(1000)
    cache = Cache(clock=clock, ledger=Ledger())
    _, spec_id = cache.reserve("run", {"cmd": "python r.py"}, launch=1000)

    got = []
    t = threading.Thread(
        target=lambda: got.append(cache.serve("run", {"cmd": "python r.py"}, 6000)))
    t.start()
    cache.discard_spec(spec_id)
    t.join(timeout=3.0)

    assert not t.is_alive()
    assert got and got[0][0] == "miss"


def make(clock_val=0.0):
    led = Ledger()
    return Cache(lambda: clock_val, led), led


def last_saved(led):
    return led.records[-1]["saved_ms"]


def test_miss_does_not_drift_occurrence_counter():
    c, led = make()
    assert c.serve("run", {"cmd": "x"}, 5.0) == ("miss", None, None)
    c.put("run", {"cmd": "x"}, duration=10.0, result="R", launch=0.0)
    status, _, result = c.serve("run", {"cmd": "x"}, 5.0)
    assert status != "miss"
    assert result == "R"


def test_saved_never_negative_when_ask_before_launch():
    c, led = make()
    c.put("run", {"cmd": "x"}, duration=10.0, result="R", launch=100.0)
    c.serve("run", {"cmd": "x"}, 50.0)
    assert last_saved(led) >= 0.0


def test_second_ask_of_once_speculated_call_misses_not_replays():
    c, led = make()
    c.put("run", {"cmd": "x"}, duration=10.0, result="R", launch=0.0)
    first = c.serve("run", {"cmd": "x"}, 5.0)
    second = c.serve("run", {"cmd": "x"}, 6.0)
    assert first[0] == "hit_promoted"
    assert first[2] == "R"
    assert second[0] == "miss"


def test_list_arg_does_not_crash_cache():
    c, led = make()
    c.put("run", {"files": ["a", "b"]}, duration=5.0, result="R", launch=0.0)
    c.serve("run", {"files": ["a", "b"]}, 2.0)


def test_epoch_fence_stale_spec_never_serves():
    c, led = make()
    c.put("run", {"cmd": "x"}, duration=10.0, result="OLD", launch=0.0)
    c.bump_epoch()
    assert c.serve("run", {"cmd": "x"}, 5.0) == ("miss", None, None)


def test_ask_equals_launch_saved_is_zero():
    c, led = make()
    c.put("run", {"cmd": "x"}, duration=10.0, result="R", launch=0.0)
    status, _, _ = c.serve("run", {"cmd": "x"}, 0.0)
    assert status == "hit_promoted"
    assert last_saved(led) == 0.0


def test_done_exactly_at_boundary_is_completed():
    c, led = make()
    c.put("run", {"cmd": "y"}, duration=10.0, result="R", launch=0.0)
    status, _, _ = c.serve("run", {"cmd": "y"}, 10.0)
    assert status == "hit_completed"
    assert last_saved(led) == 10.0


def test_saved_capped_at_duration_for_completed():
    c, led = make()
    c.put("run", {"cmd": "y"}, duration=10.0, result="R", launch=0.0)
    c.serve("run", {"cmd": "y"}, 999.0)
    assert last_saved(led) == 10.0


def test_no_cross_contamination_between_kinds():
    c, led = make()
    c.put("run", {"cmd": "x"}, 5.0, result="RUN", launch=0.0)
    c.put("edit", {"cmd": "x"}, 5.0, result="EDIT", launch=0.0)
    assert c.serve("run", {"cmd": "x"}, 2.0)[2] == "RUN"
    assert c.serve("edit", {"cmd": "x"}, 2.0)[2] == "EDIT"


def test_canon_arg_order_independent():
    assert _canon({"a": 1, "b": 2}) == _canon({"b": 2, "a": 1})
    c, led = make()
    c.put("run", {"a": 1, "b": 2}, 5.0, result="R", launch=0.0)
    assert c.serve("run", {"b": 2, "a": 1}, 2.0)[2] == "R"


def test_canon_none_and_empty():
    assert _canon({}) == ()
    assert _canon({"a": None}) == (("a", None),)
    c, led = make()
    c.put("run", {"a": None}, 5.0, result="N", launch=0.0)
    assert c.serve("run", {"a": None}, 2.0)[2] == "N"


def test_bump_epoch_evicts_and_does_not_double_terminate():
    c, led = make()
    _, sid = c.put("run", {"cmd": "x"}, duration=10.0, result="R", launch=0.0)
    c.bump_epoch()
    assert led.terminal_counts() == {"epoch_expired": 1}
    assert c.jobs == {}


def test_discard_then_reput_reuses_occurrence():
    c, led = make()
    _, sid = c.put("run", {"cmd": "z"}, 5.0, result="A", launch=0.0)
    c.discard_spec(sid)
    c.put("run", {"cmd": "z"}, 5.0, result="B", launch=0.0)
    assert c.serve("run", {"cmd": "z"}, 2.0)[2] == "B"
