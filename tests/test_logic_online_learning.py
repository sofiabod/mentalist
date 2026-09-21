import pytest

from sfx.schema import ToolEvent, Outcome
from sfx.predictor import Predictor, SUPPORT_CAP


TAU = 0.35


def _ev(kind, verb="free", status="OK", role="main"):
    return ToolEvent(t=0.0, kind=kind, verb=verb, role=role, epoch=0,
                     outcome=Outcome(kind, status))


def _empty_global():
    return {"table": {}}


def _unrelated_global():
    return {"table": {"main|zzz|zzz:OK": {"support": 1000, "p": {"other": 1.0}}}}


def test_o1_session_only_ranks_kind_top1_and_crosses_tau():
    p = Predictor(_empty_global(), k=1)

    p.observe(_ev("x"))
    assert p.propose() == []

    p.observe(_ev("x"))
    ranked = p.propose()
    assert ranked[0] == ("x", pytest.approx(1.0))
    assert ranked[0][1] >= TAU

    for _ in range(5):
        p.observe(_ev("x"))
    ranked = p.propose()
    assert ranked[0] == ("x", pytest.approx(1.0))
    assert ranked[0][1] >= TAU


def test_o1_unrelated_global_never_contaminates_session_key():
    p = Predictor(_unrelated_global(), k=1)

    p.observe(_ev("x"))
    p.observe(_ev("x"))

    ranked = p.propose()
    assert dict(ranked) == pytest.approx({"x": 1.0})
    assert ranked[0][0] == "x"
    assert ranked[0][1] >= TAU


def _confident_global_x():
    return {"table": {"main|x|x:OK": {"support": SUPPORT_CAP, "p": {"a": 1.0}}}}


def test_o2_cold_start_serves_global_before_session():
    p = Predictor(_confident_global_x(), k=1)
    p.observe(_ev("x"))

    ranked = p.propose()
    assert dict(ranked) == pytest.approx({"a": 1.0})
    assert ranked[0][0] == "a"


def test_o2_session_flips_global_at_support_cap():
    p = Predictor(_confident_global_x(), k=1)
    for _ in range(SUPPORT_CAP + 1):
        p.observe(_ev("x"))

    probs = dict(p.propose())
    assert probs == pytest.approx({"a": 1 / 3, "x": 2 / 3})
    assert max(probs, key=probs.get) == "x"


def test_o2_flip_boundary_is_between_12_and_13_session_obs():
    p12 = Predictor(_confident_global_x(), k=1)
    for _ in range(13):
        p12.observe(_ev("x"))
    probs12 = dict(p12.propose())
    assert probs12 == pytest.approx({"a": 12.5 / 24.5, "x": 12 / 24.5})
    assert max(probs12, key=probs12.get) == "a"

    p13 = Predictor(_confident_global_x(), k=1)
    for _ in range(14):
        p13.observe(_ev("x"))
    probs13 = dict(p13.propose())
    assert probs13 == pytest.approx({"a": 400 / 829, "x": 429 / 829})
    assert max(probs13, key=probs13.get) == "x"
