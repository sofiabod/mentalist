from sfx.daemon import Daemon
from sfx.schema import Outcome, ToolEvent


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {"k": 1, "min_support": 1, "tau": 0.35, "table": {}}


def _new_daemon():
    return Daemon(clock=FakeClock(0), global_table=_global_table(), k=1,
                  run=lambda kind, args: ("out", 0),
                  resolve_args=lambda kind, ctx: {"cmd": "x"})


def _observe_edit_test(d, sid, n):
    for _ in range(n):
        d.call_executed(sid, "edit", "fork", "FAIL", {}, latency=0)
        d.call_executed(sid, "test", "free", "PASS", {}, latency=0)


def _predict_after_edit(d, sid):
    s = d.sessions[sid]
    s.predictor.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main",
                                  epoch=0, outcome=Outcome("edit", "FAIL")))
    return s.predictor.propose()


def test_shared_state_dir_carries_across_different_repos(tmp_path):
    shared = tmp_path / "shared"
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    d1 = _new_daemon()
    d1.session_start("case1", repo=str(repo_a), role="main", state_dir=str(shared))
    _observe_edit_test(d1, "case1", 5)
    d1.session_end("case1")

    d2 = _new_daemon()
    d2.session_start("case2", repo=str(repo_b), role="main", state_dir=str(shared))
    ranked = _predict_after_edit(d2, "case2")
    assert ranked and ranked[0][0] == "test"
    assert not (repo_b / ".sfx" / "table.json").exists()


def test_distinct_state_dirs_stay_isolated(tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    d1 = _new_daemon()
    d1.session_start("case1", repo=str(repo_a), role="main",
                     state_dir=str(tmp_path / "state_a"))
    _observe_edit_test(d1, "case1", 5)
    d1.session_end("case1")

    d2 = _new_daemon()
    d2.session_start("case2", repo=str(repo_b), role="main",
                     state_dir=str(tmp_path / "state_b"))
    ranked = _predict_after_edit(d2, "case2")
    assert all(kind != "test" for kind, _ in ranked) or not ranked
