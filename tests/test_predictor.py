import pytest

from sfx.schema import ToolEvent, Outcome
from sfx.predictor import Predictor


def _ev(kind, verb, status):
    return ToolEvent(t=0.0, kind=kind, verb=verb, role="main", epoch=0,
                     outcome=Outcome(kind, status))


def _global_table():
    return {
        "k": 1,
        "min_support": 1,
        "tau": 0.35,
        "table": {
            "main|edit|edit:FAIL": {"support": 10, "p": {"edit": 0.7, "read": 0.3}},
        },
    }


def test_global_only_returns_mined_prior_ranked():
    p = Predictor(_global_table(), k=1)
    p.observe(_ev("edit", "fork", "FAIL"))

    ranked = p.propose()

    assert ranked[0] == ("edit", pytest.approx(0.7))
    assert dict(ranked)["read"] == pytest.approx(0.3)
    assert [kind for kind, _ in ranked] == ["edit", "read"]


def test_session_blends_with_global_by_evidence_weight():
    p = Predictor(_global_table(), k=1)
    p.observe(_ev("edit", "fork", "FAIL"))
    p.observe(_ev("edit", "fork", "FAIL"))
    p.observe(_ev("edit", "fork", "FAIL"))
    p.observe(_ev("test", "free", "PASS"))
    p.observe(_ev("edit", "fork", "FAIL"))

    probs = dict(p.propose())

    assert probs["edit"] == pytest.approx(186 / 269)
    assert probs["read"] == pytest.approx(60 / 269)
    assert probs["test"] == pytest.approx(23 / 269)


def test_cap_lets_session_override_large_split_prior():
    global_table = {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {
            "main|edit|edit:OK": {"support": 1000, "p": {"run": 0.6, "edit": 0.4}},
        },
    }
    p = Predictor(global_table, k=1)
    for _ in range(21):
        p.observe(_ev("edit", "fork", "OK"))

    ranked = p.propose()

    assert ranked[0][0] == "edit"


def test_warm_prior_serves_cold_start_then_fades_to_session():
    global_table = {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {
            "main|read|read:OK": {"support": 20, "p": {"edit": 1.0}},
        },
    }
    p = Predictor(global_table, k=1)
    p.observe(_ev("read", "fork", "OK"))

    assert p.propose()[0][0] == "edit"

    for _ in range(20):
        p.observe(_ev("read", "fork", "OK"))

    ranked = dict(p.propose())
    assert max(ranked, key=ranked.get) == "read"
    assert ranked["read"] == pytest.approx(20 / 30)
    assert ranked["edit"] == pytest.approx(10 / 30)


def test_unseen_context_returns_empty():
    p = Predictor(_global_table(), k=1)
    p.observe(_ev("grep", "free", "OK"))

    assert p.propose() == []
