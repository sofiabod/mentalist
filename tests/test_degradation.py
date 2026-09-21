import json

from sfx.daemon import Daemon
from sfx.schema import PROFILES, Outcome, Profile, ToolEvent

FULL_FILESYSTEM = Profile("json-stream", "fs-only", "full", "full", "clonefile")


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _global_table():
    return {"k": 1, "table": {}}


def _daemon(clock=None):
    clock = clock or FakeClock(0)
    return Daemon(clock=clock, global_table=_global_table(), k=1,
                  run=lambda kind, args: ("out", 0),
                  resolve_args=lambda kind, ctx: {"cmd": "x"},
                  apply_write=lambda fp, w: None,
                  run_in_fork=lambda fp, hop: ("r", "PASS"))


def test_profiles_map_to_declared_grades():
    assert FULL_FILESYSTEM.modes() == \
        ["patch-chains", "intercept", "outcome-conditioned"]
    assert PROFILES["claude-code"].interception_grade == "envelope"
    assert PROFILES["codex"].interception_grade == "additive"
    assert PROFILES["codex"].modes()[1] == "additive-tool"
    assert PROFILES["capture"].modes() == \
        ["get-only", "additive-tool", "outcome-unconditioned"]


def test_bad_grade_crashes_loud():
    import pytest
    with pytest.raises(ValueError):
        Profile("json", "fs-only", "bogus", "full", "clonefile")


def test_no_stream_visibility_means_no_patch_chain():
    # ladder rung (a): no stream deltas -> GET-only, no chain built
    d = _daemon()
    nostream = Profile("json-stream", "fs-only", "full", "none", "clonefile")
    d.session_start("s", repo="/x", role="main", profile=nostream)
    chain = d.call_stream_delta("s", "c0", "Edit",
                                json.dumps({"path": "f.py", "contents": "a=1\n"}))
    assert chain is None
    assert d.sessions["s"].chain is None


def test_stream_visibility_full_still_builds_chain(tmp_path):
    d = _daemon()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    d.session_start("s", repo=str(tmp_path), role="main", scratch=str(scratch),
                    profile=FULL_FILESYSTEM)
    chain = d.call_stream_delta("s", "c0", "Write",
                                json.dumps({"path": "f.py", "contents": "a=1\n"}))
    assert chain is not None


def test_additive_grade_serves_only_registered_kinds():
    # ladder rung (b): no interception -> additive tool serves only wrapped kinds
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["codex"],
                    registered_kinds={"test"})
    d.sessions["s"].cache.put("test", {"cmd": "x"}, 0, result="PASS", launch=0)
    d.sessions["s"].cache.put("read", {"cmd": "x"}, 0, result="body", launch=0)
    served_reg, _ = d.resolve("s", "test", {"cmd": "x"})
    served_unreg, _ = d.resolve("s", "read", {"cmd": "x"})
    assert served_reg.startswith("hit")
    assert served_unreg == "miss"


def test_full_grade_serves_any_kind():
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=FULL_FILESYSTEM,
                    registered_kinds={"test"})
    d.sessions["s"].cache.put("read", {"cmd": "x"}, 0, result="body", launch=0)
    outcome, _ = d.resolve("s", "read", {"cmd": "x"})
    assert outcome.startswith("hit")


def test_outcome_unconditioned_keys_collapse_status():
    # ladder rung (c): no outcome visibility -> keys ignore last_outcome
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=PROFILES["capture"])
    pred = d.sessions["s"].predictor
    pred.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main", epoch=0,
                           outcome=Outcome("edit", "OK")))
    pred.observe(ToolEvent(t=0, kind="test", verb="free", role="main", epoch=0,
                           outcome=Outcome("test", "PASS")))
    key = list(pred.session_counts)[0]
    assert key.endswith("|?")


def test_outcome_conditioned_keys_keep_status():
    d = _daemon()
    d.session_start("s", repo="/x", role="main", profile=FULL_FILESYSTEM)
    pred = d.sessions["s"].predictor
    pred.observe(ToolEvent(t=0, kind="edit", verb="fork", role="main", epoch=0,
                           outcome=Outcome("edit", "OK")))
    pred.observe(ToolEvent(t=0, kind="test", verb="free", role="main", epoch=0,
                           outcome=Outcome("test", "PASS")))
    key = list(pred.session_counts)[0]
    assert key.endswith("|edit:OK")


def test_string_profile_name_resolves():
    d = _daemon()
    s = d.session_start("s", repo="/x", role="main", profile="claude-code")
    assert s.profile.interception_grade == "envelope"
