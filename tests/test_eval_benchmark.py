import pytest

from sfx.schema import ToolEvent, Outcome
from mining.eval_benchmark import edit_run_recall


def _ev(kind, status="OK"):
    verb = "fork" if kind == "edit" else "free"
    return ToolEvent(t=0.0, kind=kind, verb=verb, role="main", epoch=0,
                     outcome=Outcome(kind, status))


def _edit_run_session():
    return [_ev("read"), _ev("edit", "OK"), _ev("run", "OK")]


def test_edit_run_top1_is_one_when_pattern_dominant():
    sessions = {f"s{i}": _edit_run_session() for i in range(20)}
    from mining.eval_offline import split_sessions
    train, val = split_sessions(sessions, val_frac=0.3)
    r = edit_run_recall(train, val, k=2, min_support=1)
    assert r["top1"] == pytest.approx(1.0)
    assert r["top3"] == pytest.approx(1.0)
    assert r["n"] > 0


def test_edit_run_ignores_non_edit_predecessors():
    sessions = {f"s{i}": [_ev("read"), _ev("grep"), _ev("run", "OK")]
                for i in range(20)}
    from mining.eval_offline import split_sessions
    train, val = split_sessions(sessions, val_frac=0.3)
    r = edit_run_recall(train, val, k=2, min_support=1)
    assert r["n"] == 0
