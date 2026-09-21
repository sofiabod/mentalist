import pytest

from sfx.gate import admit, rank, Candidate, TAU_GET, TAU_FORK


def _free(p=0.9, args=None, hidden_ms=100.0, cost_ms=1.0):
    return Candidate(kind="test", verb="free", p=p,
                     args={} if args is None else args,
                     hidden_ms=hidden_ms, cost_ms=cost_ms)


def test_p_exactly_tau_get_admits():
    a = admit(_free(p=TAU_GET), free_slots=1)
    assert a.action == "execute"


def test_fork_rejected_as_stream_only():
    c = Candidate(kind="edit", verb="fork", p=TAU_FORK, args={"x": 1}, hidden_ms=10, cost_ms=1)
    a = admit(c, free_slots=1)
    assert a.action == "reject"
    assert a.reason == "stream_only"


def test_p_one_ulp_below_tau_get_rejects():
    import math
    just_below = math.nextafter(TAU_GET, 0.0)
    a = admit(_free(p=just_below), free_slots=1)
    assert a.action == "reject"
    assert a.reason == "below_tau"


def test_unresolved_args_beats_confidence():
    a = admit(Candidate(kind="test", verb="free", p=0.999, args=None), free_slots=1)
    assert a.action == "reject"
    assert a.reason == "unresolved_args"


def test_not_speculable_rejected_even_confident_resolved():
    c = Candidate(kind="push", verb="never", p=1.0, args={"cmd": "git push"}, hidden_ms=9999, cost_ms=1)
    a = admit(c, free_slots=5)
    assert a.action == "reject"
    assert a.reason == "not_speculable"


def test_free_slots_zero_rejects():
    a = admit(_free(), free_slots=0)
    assert a.action == "reject"
    assert a.reason == "no_slots"


def test_free_slots_negative_rejects():
    a = admit(_free(), free_slots=-1)
    assert a.action == "reject"
    assert a.reason == "no_slots"


def test_no_slots_precedence_over_not_speculable():
    c = Candidate(kind="push", verb="never", p=1.0, args={"cmd": "x"})
    assert admit(c, free_slots=0).reason == "no_slots"


def test_hidden_ms_zero_yields_zero_priority():
    a = admit(_free(p=0.9, hidden_ms=0.0, cost_ms=1.0), free_slots=1)
    assert a.action == "execute"
    assert a.priority == 0.0


def test_priority_value_tiny_cost():
    a = admit(_free(p=0.5, hidden_ms=100.0, cost_ms=0.001), free_slots=1)
    assert a.priority == pytest.approx(0.5 * 100.0 / 0.001)


def test_priority_value_large_cost():
    a = admit(_free(p=0.5, hidden_ms=100.0, cost_ms=1e9), free_slots=1)
    assert a.priority == pytest.approx(0.5 * 100.0 / 1e9)


def test_cost_ms_zero_fails_open_zero_priority():
    """FAIL-OPEN: cost_ms==0 stays admittable at zero priority, never crashes."""
    a = admit(_free(p=0.9, hidden_ms=100.0, cost_ms=0.0), free_slots=1)
    assert a.action == "execute"
    assert a.priority == 0.0


def test_rank_stable_order_on_equal_priority():
    a = Candidate(kind="A", verb="free", p=0.5, args={}, hidden_ms=100, cost_ms=1)
    b = Candidate(kind="B", verb="free", p=0.5, args={}, hidden_ms=100, cost_ms=1)
    c = Candidate(kind="C", verb="free", p=0.5, args={}, hidden_ms=100, cost_ms=1)
    ranked = rank([a, b, c], free_slots=3)
    assert [x.kind for x in ranked] == ["A", "B", "C"]


def test_rank_empty_candidate_list():
    assert rank([], free_slots=5) == []


def test_rank_drops_rejected_keeps_admitted():
    good = _free(p=0.9, hidden_ms=100, cost_ms=1)
    bad = Candidate(kind="never", verb="never", p=0.9, args={})
    ranked = rank([bad, good], free_slots=2)
    assert [c.verb for c in ranked] == ["free"]


def test_rank_orders_by_priority_descending():
    lo = Candidate(kind="lo", verb="free", p=0.5, args={}, hidden_ms=100, cost_ms=10)
    hi = Candidate(kind="hi", verb="free", p=0.9, args={}, hidden_ms=100, cost_ms=1)
    ranked = rank([lo, hi], free_slots=2)
    assert [c.kind for c in ranked] == ["hi", "lo"]


def test_rank_zero_slots_drops_everything():
    assert rank([_free(), _free()], free_slots=0) == []
