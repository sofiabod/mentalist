import json

import pytest

from sfx.daemon import Daemon


@pytest.fixture
def daemon(tmp_path):
    events = []
    instance = Daemon(clock=lambda: 0, global_table={"k": 1, "table": {}}, k=1,
                      log=events.append)
    instance.session_start("s", repo=str(tmp_path), role="main")
    yield instance, instance.sessions["s"], events
    instance.shutdown()


@pytest.mark.parametrize("unsupported", ["edit", "unknown", "not-a-known-kind"])
def test_next_admissible_ranked_category_used(daemon, monkeypatch, unsupported):
    instance, session, events = daemon
    session.arg_by_kind["run"] = "python cli.py repo --rules rules.json"
    monkeypatch.setattr(session.predictor, "propose", lambda: [(unsupported, .6), ("run", .4)])

    assert instance._next_hop(session)([]) == (
        "run", "free", {"cmd": "python cli.py repo --rules rules.json"})
    decisions = [event for event in events if event["ev"] == "chain_prediction"]
    assert [event["gate"] for event in decisions] == ["reject", "execute"]
    assert decisions[-1]["probability"] == .4
    assert decisions[-1]["resolved_args"] == {"cmd": "python cli.py repo --rules rules.json"}


def test_skipping_top_category_does_not_renormalize_low_probability_candidate(
        daemon, monkeypatch):
    instance, session, events = daemon
    session.arg_by_kind["run"] = "python cli.py --known"
    monkeypatch.setattr(session.predictor, "propose", lambda: [("edit", .8), ("run", .2)])

    assert instance._next_hop(session)([]) is None
    assert events[-1]["gate_reason"] == "below_tau"
    assert events[-1]["probability"] == .2


def test_unresolved_candidate_allows_next_resolved_candidate(daemon, monkeypatch):
    instance, session, events = daemon
    session.arg_by_kind["run"] = "python cli.py --known"
    monkeypatch.setattr(session.predictor, "propose", lambda: [("test", .6), ("run", .4)])

    assert instance._next_hop(session)([])[0] == "run"
    assert events[0]["gate_reason"] == "unresolved_args"


def test_cold_required_args_rejection_uses_pending_edit_and_is_logged(daemon, monkeypatch):
    instance, session, events = daemon
    session.last_edit_path = "cli.py"
    monkeypatch.setattr(session.predictor, "propose", lambda: [("run", .9)])
    write = {"path": "cli.py", "contents": "parser.add_argument('--rules', required=True)"}

    assert instance._next_hop(session, write_args=write)([]) is None
    assert events[-1]["gate_reason"] == "required_python_arguments"


def test_stream_dispatch_passes_pending_edit_source_to_prediction(daemon, monkeypatch):
    instance, session, events = daemon
    (session.repo / "cli.py").write_text("print('OLD')\n")
    monkeypatch.setattr(session.predictor, "propose", lambda: [("run", .9)])

    chain = instance.call_stream_delta("s", "edit", "Write", json.dumps({
        "path": "cli.py", "contents": "parser.add_argument('--rules', required=True)"}))

    assert chain.hops == []
    assert any(event.get("gate_reason") == "required_python_arguments" for event in events)


def test_no_free_slots_does_not_launch_fork(daemon, monkeypatch):
    instance, session, events = daemon
    session.arg_by_kind["run"] = "python cli.py --known"
    monkeypatch.setattr(session.predictor, "propose", lambda: [("run", .9)])
    monkeypatch.setattr(session.executor, "free_slots", lambda: 0)

    assert instance._next_hop(session)([]) is None
    assert events[-1]["gate_reason"] == "no_slots"
