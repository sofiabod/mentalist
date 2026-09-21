import pytest

from mining.normalize import classify, normalize_row


@pytest.mark.parametrize(
    "tool_name,skeleton,expected",
    [
        ("Read", None, ("read", "free")),
        ("Grep", None, ("grep", "free")),
        ("Glob", None, ("grep", "free")),
        ("Edit", None, ("edit", "fork")),
        ("Write", None, ("edit", "fork")),
        ("apply_patch", None, ("edit", "fork")),
        ("Agent", None, ("sub-LLM", "free")),
        ("Task", None, ("sub-LLM", "free")),
        ("WebFetch", None, ("read", "free")),
        ("WebSearch", None, ("grep", "free")),
        ("Bash", "rg", ("grep", "free")),
        ("Bash", "cat", ("read", "free")),
        ("Bash", "ls", ("read", "free")),
        ("Bash", "sed -i 's/a/b/' f.py", ("edit", "fork")),
        ("Bash", "sed -n '1,5p' f.py", ("read", "free")),
        ("Bash", "pytest", ("test", "free")),
        ("Bash", "python reproduce_error.py", ("run", "free")),
        ("Bash", "python3 repro.py", ("run", "free")),
        ("Bash", "node app.js", ("run", "free")),
        ("Bash", "go run main.go", ("run", "free")),
        ("Bash", "ruby s.rb", ("run", "free")),
        ("Bash", "python -m pytest", ("test", "free")),
        ("Bash", "python manage.py test", ("unknown", "never")),
        ("Bash", "python", ("read", "free")),
        ("Bash", "cat x.py", ("read", "free")),
        ("Bash", "ruff", ("lint", "free")),
        ("Bash", "mypy", ("typecheck", "free")),
        ("Bash", "make", ("build", "free")),
        ("Bash", "pip install", ("install", "never")),
        ("Bash", "git status", ("git", "never")),
        ("Bash", "git commit", ("git", "never")),
        ("Bash", "git push", ("git", "never")),
        ("shell_command", "git push origin main", ("git", "never")),
        ("Bash", "echo hi && ls", ("read", "free")),
    ],
)
def test_classify(tool_name, skeleton, expected):
    assert classify(tool_name, skeleton) == expected


def test_normalize_row_maps_timing_events_to_sfx_events():
    row = {
        "provider": "claude",
        "session_id": "claude:sess1",
        "round_index": 0,
        "user": "user_a",
        "timing_events": [
            {"event_type": "user_message", "timestamp": "2026-05-31T14:18:50.524Z", "content_chars": 3},
            {"event_type": "text", "timestamp": "2026-05-31T14:18:52.841Z", "content_chars": 87},
            {"event_type": "tool_call", "timestamp": "2026-05-31T14:18:53.000Z", "tool_call_id": "c1", "tool_name": "Read"},
            {"event_type": "tool_result", "timestamp": "2026-05-31T14:18:53.100Z", "tool_call_id": "c1", "is_error": False},
        ],
        "tools": [
            {"tool_name": "Read", "tool_call_id": "c1", "is_error": False,
             "tool_internal_latency_ms": None, "tool_wall_latency_ms": 60},
        ],
    }
    events = normalize_row(row)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "read"
    assert ev.verb == "free"
    assert ev.role == "main"
    assert ev.outcome.klass() == "read:OK"
    assert ev.args == {"latency_ms": 60}


def test_normalize_row_tool_error_status():
    row = {
        "provider": "claude",
        "session_id": "s",
        "round_index": 0,
        "timing_events": [
            {"event_type": "tool_call", "tool_call_id": "c1", "tool_name": "Bash"},
            {"event_type": "tool_result", "tool_call_id": "c1", "is_error": True},
        ],
        "tools": [
            {"tool_name": "Bash", "tool_call_id": "c1", "is_error": True,
             "command_skeleton": "pytest", "tool_internal_latency_ms": 40, "tool_wall_latency_ms": 50},
        ],
    }
    events = normalize_row(row)
    assert len(events) == 1
    assert events[0].kind == "test"
    assert events[0].outcome.klass() == "test:FAIL"
    assert events[0].args == {"latency_ms": 40}


def test_normalize_row_subagent_role():
    row = {
        "provider": "claude",
        "session_id": "s",
        "round_index": 0,
        "first_input_event_type": "tool_result",
        "timing_events": [
            {"event_type": "tool_call", "tool_call_id": "c1", "tool_name": "Agent"},
            {"event_type": "tool_result", "tool_call_id": "c1", "is_error": False},
        ],
        "tools": [
            {"tool_name": "Agent", "tool_call_id": "c1", "is_error": False,
             "tool_internal_latency_ms": 100, "tool_wall_latency_ms": 120},
        ],
    }
    events = normalize_row(row)
    assert events[0].kind == "sub-LLM"
    assert events[0].verb == "free"
    assert events[0].role == "main"
