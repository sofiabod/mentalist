import pytest

from sfx.schema import TERMINAL_STATES
from sfx.ledger import Ledger


def test_all_seven_terminal_states_recordable():
    led = Ledger()
    for state in TERMINAL_STATES:
        led.terminal("spec-" + state, state)
    counts = led.terminal_counts()
    assert set(counts) == TERMINAL_STATES
    assert all(counts[s] == 1 for s in TERMINAL_STATES)


def test_saved_and_wasted_totals_sum():
    led = Ledger()
    led.terminal("s1", "hit_completed", saved_ms=8120)
    led.terminal("s2", "hit_promoted", saved_ms=5000)
    led.terminal("s3", "discarded", wasted_ms=240)
    led.terminal("s4", "preempted", wasted_ms=100)

    assert led.total_saved_ms == pytest.approx(13120)
    assert led.total_wasted_ms == pytest.approx(340)


def test_terminal_recorded_once_per_speculation():
    led = Ledger()
    led.terminal("s1", "miss")
    with pytest.raises(ValueError):
        led.terminal("s1", "hit_completed")
