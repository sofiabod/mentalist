"""Conformance matrix: every adapter ships a valid Profile, satisfies the minimal
event-feed contract, and the daemon selects the right degradation mode from each
Profile (GOAL adapter contract, lines 221-228)."""
import importlib
import subprocess

import pytest

from sfx.schema import INTERCEPTION_GRADES, PROFILES, Profile

ADAPTERS = {"mini_swe": "adapters.mini_swe"}

# task mapping: grade -> degradation mode, plus no-stream -> GET-only
EXPECTED_MODE = {"full": "intercept", "envelope": "intercept+verify",
                 "additive": "fallback", "capture": "replay"}


def _profile(name):
    return importlib.import_module(ADAPTERS[name]).PROFILE


def test_every_adapter_exposes_a_valid_profile():
    for name in ADAPTERS:
        p = _profile(name)
        assert isinstance(p, Profile)
        assert p.interception_grade in INTERCEPTION_GRADES
        assert p.emission_format and p.state_model and p.fork_substrate


def test_every_adapter_satisfies_minimal_event_feed_contract():
    # minimal event feed (GOAL line 225): a call-lifecycle seam. filesystem
    # adapters attach a live socket Client (turn_begin/feed/resolve/call_executed);
    # trace adapters map captured lifecycle lines into the same call shape.
    for name in ADAPTERS:
        mod = importlib.import_module(ADAPTERS[name])
        live = hasattr(mod, "connect")
        feed = any(hasattr(mod, f) for f in
                   ("feed_from_row", "feed_from_capture"))
        assert live or feed, f"{name} exposes no event-feed seam"


@pytest.mark.parametrize("adapter,grade", [("mini_swe", "full")])
def test_daemon_selects_right_mode_from_each_adapter_profile(adapter, grade):
    p = _profile(adapter)
    assert p.interception_grade == grade
    assert p.degradation_mode() == EXPECTED_MODE[grade]


def test_capture_profile_selects_replay_mode():
    assert PROFILES["capture"].interception_grade == "capture"
    assert PROFILES["capture"].degradation_mode() == "replay"


def test_no_stream_visibility_forces_get_only_regardless_of_grade():
    # capture grade is offline replay (its own mode); the ladder rung applies to
    # a live profile that merely lacks stream deltas.
    for grade in INTERCEPTION_GRADES - {"capture"}:
        nostream = Profile("json-stream", "fs-only", grade, "none", "clonefile")
        assert nostream.degradation_mode() == "get-only"


def test_mode_map_covers_every_declared_grade():
    assert set(EXPECTED_MODE) == INTERCEPTION_GRADES


def test_daemon_core_has_no_harness_specific_branch():
    # GOAL line 139: the daemon is harness-agnostic. The only harness tokens
    # allowed in src/sfx are the static PROFILES registry (a keyed lookup, not
    # dispatch) and comments; there must be no `if harness == ...` branching.
    out = subprocess.run(
        ["grep", "-rniE", "--include=*.py",
         r"claude|codex|\bmini\b|harbor|openhands", "src/sfx/"],
        capture_output=True, text=True).stdout
    offending = []
    for line in out.splitlines():
        path, _, body = line.partition(":")
        body = body.split(":", 1)[1] if ":" in body else body
        stripped = body.strip()
        if stripped.startswith("#"):
            continue
        if "PROFILES" in line or '"claude-code"' in line \
                or '"codex"' in line:
            continue  # static declared-profile registry, not dispatch
        offending.append(line)
    assert offending == [], f"harness-specific logic in daemon core:\n" + "\n".join(offending)
