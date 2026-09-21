"""Unsupported execution models must not silently use filesystem semantics."""
import json

import pytest

from adapters import protocol
from sfx.daemon import Daemon
from sfx.schema import PROFILES, Profile


@pytest.fixture
def daemon():
    d = Daemon(clock=lambda: 0, global_table={"k": 1, "table": {}}, k=1)
    yield d
    d.shutdown()


@pytest.mark.parametrize("state_model", ["repl", "unknown"])
@pytest.mark.parametrize("profile", [None, "claude-code"])
def test_session_rejects_unsupported_state_model(daemon, tmp_path, state_model, profile):
    with pytest.raises(ValueError, match="unsupported state model"):
        daemon.session_start("s", repo=tmp_path, role="main",
                             state_model=state_model, profile=profile)
    assert daemon.sessions == {}


@pytest.mark.parametrize("profile", ["prime-agent", "unknown"])
def test_session_rejects_removed_or_unknown_profile(daemon, tmp_path, profile):
    with pytest.raises(ValueError, match="unknown profile"):
        daemon.session_start("s", repo=tmp_path, role="main", profile=profile)
    assert daemon.sessions == {}


@pytest.mark.parametrize("state_model", ["repl", "unknown"])
def test_custom_profile_rejects_unsupported_state_model(state_model):
    with pytest.raises(ValueError, match="unsupported state model"):
        Profile("json-stream", state_model, "full", "full", "clonefile")


def test_custom_profile_rejects_removed_substrate():
    with pytest.raises(ValueError, match="unsupported fork substrate"):
        Profile("json-stream", "fs-only", "full", "full", "shadow")


def test_removed_namespace_argument_is_rejected(daemon, tmp_path):
    with pytest.raises(TypeError, match="repl_ns"):
        daemon.session_start("s", repo=tmp_path, role="main", repl_ns={})
    assert daemon.sessions == {}


@pytest.mark.parametrize("fields", [{"state_model": "repl"}, {"profile": "prime-agent"}])
def test_protocol_returns_error_without_creating_unsupported_session(daemon, tmp_path, fields):
    message = {"type": "turn_begin", "session": "s", "repo": str(tmp_path),
               "role": "main", **fields}
    reply = protocol._handle_line(lambda msg: protocol._dispatch(daemon, msg),
                                  json.dumps(message))
    assert "error" in reply
    assert daemon.sessions == {}


@pytest.mark.parametrize("profile", [None, "capture"])
def test_filesystem_and_capture_profiles_remain_available(daemon, tmp_path, profile):
    session = daemon.session_start("s", repo=tmp_path, role="main", profile=profile)
    if profile is None:
        assert session.profile.state_model == "fs-only"
        assert session.profile.streams
    else:
        assert session.profile == PROFILES["capture"]
        assert not session.profile.streams
