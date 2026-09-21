import json

from sfx.daemon import Daemon, _load_repo_table
from sfx.schema import Outcome, ToolEvent


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {"k": 1, "min_support": 1, "tau": 0.35, "table": {}}


def _new_daemon(clock):
    return Daemon(clock=clock, global_table=_global_table(), k=1,
                  run=lambda kind, args: ("out", 0),
                  resolve_args=lambda kind, ctx: {"cmd": "x"})


def _observe_edit_test(d, sid, n):
    for _ in range(n):
        d.call_executed(sid, "edit", "fork", "FAIL", {}, latency=0)
        d.call_executed(sid, "test", "free", "PASS", {}, latency=0)


def test_session_end_writes_loadable_repo_table(tmp_path):
    d = _new_daemon(FakeClock(0))
    d.session_start("s1", repo=str(tmp_path), role="main")
    _observe_edit_test(d, "s1", 3)
    d.session_end("s1")

    p = tmp_path / ".sfx" / "table.json"
    assert p.exists()
    loaded = _load_repo_table(tmp_path)
    key = "main|edit|edit:FAIL"
    assert loaded["table"][key]["p"]["test"] > 0
    assert loaded["table"][key]["support"] >= 3


def test_fresh_session_loads_persisted_counts_into_propose(tmp_path):
    d = _new_daemon(FakeClock(0))
    d.session_start("s1", repo=str(tmp_path), role="main")
    _observe_edit_test(d, "s1", 3)
    d.session_end("s1")

    d2 = _new_daemon(FakeClock(0))
    s = d2.session_start("s2", repo=str(tmp_path), role="main")
    from sfx.schema import ToolEvent, Outcome
    s.predictor.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main",
                                  epoch=0, outcome=Outcome("edit", "FAIL")))
    ranked = s.predictor.propose()
    assert ranked and ranked[0][0] == "test"


def test_second_session_merges_not_overwrites(tmp_path):
    d = _new_daemon(FakeClock(0))
    d.session_start("s1", repo=str(tmp_path), role="main")
    _observe_edit_test(d, "s1", 2)
    d.session_end("s1")
    first = _load_repo_table(tmp_path)["table"]["main|edit|edit:FAIL"]["support"]

    d2 = _new_daemon(FakeClock(0))
    d2.session_start("s2", repo=str(tmp_path), role="main")
    _observe_edit_test(d2, "s2", 2)
    d2.session_end("s2")
    second = _load_repo_table(tmp_path)["table"]["main|edit|edit:FAIL"]["support"]

    assert second > first


def test_write_stays_inside_repo(tmp_path):
    d = _new_daemon(FakeClock(0))
    d.session_start("s1", repo=str(tmp_path), role="main")
    _observe_edit_test(d, "s1", 2)
    d.session_end("s1")

    written = list(tmp_path.rglob("*"))
    for w in written:
        assert tmp_path in w.resolve().parents or w.resolve() == tmp_path
    assert (tmp_path / ".sfx" / "table.json") in written


# --- per-task isolation (step 4) ---

def _observe_edit(d, sid, next_kind, n):
    for _ in range(n):
        d.call_executed(sid, "edit", "fork", "FAIL", {}, latency=0)
        d.call_executed(sid, next_kind, "free", "PASS", {}, latency=0)


def _predict_after_edit(d, sid):
    s = d.sessions[sid]
    s.predictor.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main",
                                  epoch=0, outcome=Outcome("edit", "FAIL")))
    return s.predictor.propose()


def test_task_b_does_not_inherit_task_a_repo_tier(tmp_path):
    # each task owns its workspace; NO .sfx carryover across distinct tasks.
    task_a = tmp_path / "task_a"
    task_b = tmp_path / "task_b"
    task_a.mkdir()
    task_b.mkdir()

    d = _new_daemon(FakeClock(0))
    d.session_start("a", repo=str(task_a), role="main")
    _observe_edit(d, "a", "test", 5)
    d.session_end("a")

    d2 = _new_daemon(FakeClock(0))
    d2.session_start("b", repo=str(task_b), role="main")
    ranked = _predict_after_edit(d2, "b")
    # task B's global table is empty and its own repo tier is empty, so it must
    # NOT know task A's edit->test pattern.
    assert all(kind != "test" for kind, _ in ranked) or not ranked
    assert not (task_b / ".sfx" / "table.json").exists() \
        or _load_repo_table(task_b) is None or "main|edit|edit:FAIL" \
        not in _load_repo_table(task_b)["table"]


def test_warm_repo_tier_changes_prediction_on_repeated_same_repo_run(tmp_path):
    # step 6 regression: session_end now fires, so a REPEATED run of the SAME
    # repo starts warm and predicts differently than a cold first run.
    repo = tmp_path / "repo"
    repo.mkdir()

    cold = _new_daemon(FakeClock(0))
    cold.session_start("run1", repo=str(repo), role="main")
    ranked_cold = _predict_after_edit(cold, "run1")
    assert not ranked_cold  # empty global + cold repo tier -> no signal
    _observe_edit(cold, "run1", "test", 5)
    cold.session_end("run1")

    warm = _new_daemon(FakeClock(0))
    warm.session_start("run2", repo=str(repo), role="main")
    ranked_warm = _predict_after_edit(warm, "run2")
    assert ranked_warm and ranked_warm[0][0] == "test"


def test_session_tier_learns_online_within_task(tmp_path):
    # session tier is always-on online adaptation, independent of persistence.
    repo = tmp_path / "repo"
    repo.mkdir()
    d = _new_daemon(FakeClock(0))
    d.session_start("s", repo=str(repo), role="main")
    before = _predict_after_edit(d, "s")
    assert not before
    _observe_edit(d, "s", "lint", 4)
    after = _predict_after_edit(d, "s")
    assert after and after[0][0] == "lint"
