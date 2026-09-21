import copy
import tempfile
from pathlib import Path

from sfx.daemon import _persist_repo_table, _load_repo_table
from sfx.predictor import Predictor
from sfx.schema import ToolEvent, Outcome


BENCH = {"k": 2, "table": {}}


def _observe(p, kind):
    p.observe(ToolEvent(t=0.0, kind=kind, verb="free", role="main", epoch=0,
                        args={}, outcome=Outcome(kind, "OK")))


def test_i1_distinct_state_dir_does_not_see_other_case_key():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        dir_a = repo / "caseA" / ".sfx"
        dir_b = repo / "caseB" / ".sfx"

        a = Predictor(BENCH, k=2)
        a.session_counts["x"].update({"edit": 10})
        _persist_repo_table(repo, a.session_counts, state_dir=dir_a)

        b = Predictor(BENCH, k=2, repo_table=_load_repo_table(repo, dir_b))
        assert b.repo_table == {}
        assert b.repo_table.get("x") is None


def test_i2_reload_same_state_dir_warms_exact_support_and_p():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        dir_a = repo / "caseA" / ".sfx"

        a = Predictor(BENCH, k=2)
        a.session_counts["x"].update({"grep": 3, "edit": 1})
        _persist_repo_table(repo, a.session_counts, state_dir=dir_a)

        reload_a = Predictor(BENCH, k=2, repo_table=_load_repo_table(repo, dir_a))
        assert reload_a.repo_table["x"] == {
            "support": 4, "p": {"grep": 0.75, "edit": 0.25}}


def test_i3_observe_does_not_mutate_global_table():
    global_table = {"k": 2, "table": {
        "main|read,read|read:OK": {"support": 8, "p": {"grep": 0.5, "edit": 0.5}}}}
    before = copy.deepcopy(global_table)

    p = Predictor(global_table, k=2)
    _observe(p, "read")
    _observe(p, "read")
    _observe(p, "grep")
    p.propose()

    assert global_table == before


def test_i4_empty_persist_leaves_warm_table_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        dir_a = repo / "caseA" / ".sfx"

        a = Predictor(BENCH, k=2)
        a.session_counts["k"].update({"grep": 5})
        _persist_repo_table(repo, a.session_counts, state_dir=dir_a)
        before = _load_repo_table(repo, dir_a)
        assert before["table"]["k"] == {"support": 5, "p": {"grep": 1.0}}

        _persist_repo_table(repo, {}, state_dir=dir_a)
        after = _load_repo_table(repo, dir_a)
        assert after == before
