import pytest

from sfx.schema import (
    ToolEvent,
    Outcome,
    context_key,
    POLICIES,
    SPECULABLE,
    STREAM_ONLY,
    TERMINAL_STATES,
)


def test_policy_taxonomy():
    assert POLICIES == {"free", "fork", "never"}


def test_speculable_is_free_and_fork():
    assert SPECULABLE == {"free", "fork"}


def test_stream_only_writes():
    assert STREAM_ONLY == {"fork"}


def test_terminal_states():
    assert TERMINAL_STATES == {
        "hit_completed",
        "hit_promoted",
        "miss",
        "discarded",
        "preempted",
        "epoch_expired",
    }


def test_reject_bad_verb():
    with pytest.raises(ValueError):
        ToolEvent(t=0.0, kind="test", verb="FETCH", role="main", epoch=0)


def test_reject_missing_required_field():
    with pytest.raises(TypeError):
        ToolEvent.from_dict({"t": 0.0, "kind": "test", "verb": "free", "role": "main"})


def test_outcome_class_string():
    assert Outcome("test", "FAIL").klass() == "test:FAIL"
    assert Outcome("test", "PASS").klass() == "test:PASS"


def _ev(kind, verb, status=None):
    return ToolEvent(
        t=0.0,
        kind=kind,
        verb=verb,
        role="main",
        epoch=0,
        outcome=Outcome(kind, status) if status else None,
    )


def test_context_key_role_lastk_kinds_last_outcome():
    seq = [_ev("read", "free"), _ev("edit", "fork"), _ev("test", "free", "FAIL")]
    assert context_key(seq, k=2) == ("main", ("edit", "test"), "test:FAIL")


def test_context_key_pads_when_fewer_than_k():
    seq = [_ev("read", "free", "PASS")]
    assert context_key(seq, k=3) == ("main", ("read",), "read:PASS")


def test_event_round_trip():
    ev = ToolEvent(
        t=1727000000.0,
        kind="test",
        verb="free",
        role="main",
        args={"cmd": "pytest"},
        epoch=4,
        outcome=Outcome("test", "FAIL"),
    )
    assert ToolEvent.from_dict(ev.to_dict()) == ev
