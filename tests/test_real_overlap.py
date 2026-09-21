import time

from sfx.cache import Cache
from sfx.executor import Executor
from sfx.ledger import Ledger

S = 150.0  # ms the slow tool sleeps


def _ms_clock():
    return time.monotonic() * 1000.0


def _slow_tool(result):
    def job():
        t0 = time.monotonic()
        time.sleep(S / 1000.0)
        return result, (time.monotonic() - t0) * 1000.0
    return job


def _new():
    led = Ledger()
    cache = Cache(_ms_clock, led)
    ex = Executor(clock=_ms_clock, cache=cache, ledger=led, slots=4)
    return led, cache, ex


def test_overlap_is_physical_wall_near_S_not_2S():
    led, cache, ex = _new()

    wall_t0 = time.monotonic()
    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_slow_tool("PASS"))

    # main thread does ~S ms of its own work (model "thinking") while the spec runs
    time.sleep(S / 1000.0)

    outcome, _, result = cache.serve("test", {"cmd": "pytest"}, ask_time=_ms_clock())
    serve_done = time.monotonic()

    total_wall = (serve_done - wall_t0) * 1000.0

    # either completed or promoted is a real spec hit; the knife-edge between them is timing,
    # what matters is the served bytes are the spec's and the wall proves concurrency
    assert outcome in ("hit_completed", "hit_promoted")
    assert result == "PASS"
    # overlapped: spec finished during the main-thread work, so total ~= S, well under 2*S
    assert total_wall < 1.5 * S
    ex.shutdown()


def test_no_overlap_control_wall_near_2S():
    led, cache, ex = _new()

    wall_t0 = time.monotonic()
    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_slow_tool("PASS"))

    # control: drain the lane BEFORE doing the main-thread work, forcing sequential timing
    ex.drain()

    time.sleep(S / 1000.0)

    outcome, _, _ = cache.serve("test", {"cmd": "pytest"}, ask_time=_ms_clock())
    serve_done = time.monotonic()

    total_wall = (serve_done - wall_t0) * 1000.0

    assert outcome == "hit_completed"
    # sequential: spec (S) then work (S) => ~2*S, clearly above the overlapped ceiling
    assert total_wall > 1.7 * S
    ex.shutdown()


def test_saved_ms_is_real_and_equals_hidden_wall():
    led, cache, ex = _new()

    launch = _ms_clock()
    ex.speculate("test", {"cmd": "pytest"}, priority=5.0, job=_slow_tool("PASS"))

    # main-thread work outlasts the tool so the serve deterministically hits a completed spec
    time.sleep(1.5 * S / 1000.0)

    ask_time = _ms_clock()
    outcome, _, _ = cache.serve("test", {"cmd": "pytest"}, ask_time=ask_time)

    assert outcome == "hit_completed"
    saved = led.total_saved_ms
    hidden = ask_time - launch  # wall the agent overlapped between launch and ask

    assert saved > 0
    assert saved <= S * 1.2  # bounded by the real tool duration
    # saved == min(duration, ask_time - launch); since ask happened after the tool finished,
    # saved reflects the real tool duration, and the hidden wall is at least that much
    assert saved <= hidden + 1e-6
    assert abs(saved - min(S, hidden)) < 0.4 * S
    ex.shutdown()
