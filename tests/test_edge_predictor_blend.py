import pytest

from sfx.schema import ToolEvent, Outcome
from sfx.predictor import Predictor, SUPPORT_CAP


def _ev(kind, verb="free", status="OK", role="main"):
    return ToolEvent(t=0.0, kind=kind, verb=verb, role=role, epoch=0,
                     outcome=Outcome(kind, status))


def _observe_key_x(p, n):
    """Drive predictor into context key main|x|x:OK with n session observations of x."""
    p.observe(_ev("x"))
    for _ in range(n):
        p.observe(_ev("x"))


def test_empty_predictor_propose_does_not_crash():
    p = Predictor({"table": {}}, k=1)
    assert p.propose() == []


def test_cap_boundary_support_equals_cap_and_cap_plus_one_blend_identically():
    gt_cap = {"table": {"main|x|x:OK": {"support": SUPPORT_CAP, "p": {"a": 1.0}}}}
    gt_over = {"table": {"main|x|x:OK": {"support": SUPPORT_CAP + 1, "p": {"a": 1.0}}}}

    p_cap = Predictor(gt_cap, k=1)
    _observe_key_x(p_cap, SUPPORT_CAP)
    p_over = Predictor(gt_over, k=1)
    _observe_key_x(p_over, SUPPORT_CAP)

    assert dict(p_cap.propose()) == pytest.approx({"a": 1 / 3, "x": 2 / 3})
    assert dict(p_over.propose()) == pytest.approx({"a": 1 / 3, "x": 2 / 3})


def test_confident_global_not_suppressed_when_no_session_signal():
    gt = {"table": {"main|edit|edit:OK": {"support": 1000,
                                          "p": {"run": 0.95, "edit": 0.05}}}}
    p = Predictor(gt, k=1)
    p.observe(_ev("edit"))

    ranked = p.propose()

    assert ranked[0] == ("run", pytest.approx(0.95))
    assert dict(ranked)["edit"] == pytest.approx(0.05)


def test_unknown_context_key_returns_empty_not_crash():
    gt = {"table": {"main|edit|edit:OK": {"support": 10, "p": {"edit": 1.0}}}}
    p = Predictor(gt, k=1)
    p.observe(_ev("grep"))

    assert p.propose() == []


def test_three_tier_equal_support_ties_blend_to_thirds():
    gt = {"table": {"main|x|x:OK": {"support": 10, "p": {"a": 1.0}}}}
    rt = {"table": {"main|x|x:OK": {"support": 10, "p": {"b": 1.0}}}}
    p = Predictor(gt, k=1, repo_table=rt)
    _observe_key_x(p, 10)

    probs = dict(p.propose())

    assert probs == pytest.approx({"a": 2 / 7, "b": 2 / 7, "x": 3 / 7})


def test_session_absent_gives_only_repo_and_global():
    gt = {"table": {"main|edit|edit:OK": {"support": 10, "p": {"a": 1.0}}}}
    rt = {"table": {"main|edit|edit:OK": {"support": 10, "p": {"b": 1.0}}}}
    p = Predictor(gt, k=1, repo_table=rt)
    p.observe(_ev("edit"))

    probs = dict(p.propose())

    assert probs == pytest.approx({"a": 0.5, "b": 0.5})


def test_context_shorter_than_k_uses_available_events():
    gt = {"table": {"main|read|read:OK": {"support": 10, "p": {"a": 1.0}}}}
    p = Predictor(gt, k=3, repo_table=None)
    p.observe(_ev("read"))

    assert dict(p.propose()) == pytest.approx({"a": 1.0})


def test_context_longer_than_k_truncates_to_last_k_kinds():
    gt = {"table": {"main|edit,test|test:OK": {"support": 10, "p": {"a": 1.0}}}}
    p = Predictor(gt, k=2)
    p.observe(_ev("read"))
    p.observe(_ev("edit"))
    p.observe(_ev("test"))

    assert dict(p.propose()) == pytest.approx({"a": 1.0})


def test_no_cross_contamination_between_distinct_context_keys():
    gt = {"table": {
        "main|read|read:OK": {"support": 10, "p": {"a": 1.0}},
        "main|edit|edit:OK": {"support": 10, "p": {"b": 1.0}},
    }}
    p = Predictor(gt, k=1)
    p.observe(_ev("read"))
    assert dict(p.propose()) == pytest.approx({"a": 1.0})
    p.observe(_ev("edit"))
    assert dict(p.propose()) == pytest.approx({"b": 1.0})
