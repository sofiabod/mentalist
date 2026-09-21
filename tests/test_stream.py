import json

from adapters.stream import StreamParser


def _chunks(s, n):
    return [s[i:i + n] for i in range(0, len(s), n)]


def test_write_captured_verbatim_once_args_complete():
    parser = StreamParser()
    args = {"path": "foo.py", "old_string": "a", "new_string": "b"}
    body = json.dumps(args)
    captured = None
    for chunk in _chunks(body, 5):
        captured = parser.feed("c1", tool="Edit", delta=chunk)
    assert captured is not None
    assert captured.verb == "fork"
    assert captured.tool == "Edit"
    assert captured.args == args


def test_incomplete_stream_captures_nothing():
    parser = StreamParser()
    body = json.dumps({"path": "foo.py", "old_string": "a", "new_string": "b"})
    partial = body[: len(body) // 2]
    result = None
    for chunk in _chunks(partial, 5):
        result = parser.feed("c1", tool="Edit", delta=chunk)
    assert result is None


def test_garbled_delta_does_not_crash_and_captures_nothing():
    parser = StreamParser()
    result = parser.feed("c1", tool="Edit", delta='{"path": "foo.py", nonsense')
    assert result is None


def test_non_write_call_is_not_captured():
    parser = StreamParser()
    body = json.dumps({"pattern": "def foo"})
    captured = None
    for chunk in _chunks(body, 4):
        captured = parser.feed("c2", tool="Grep", delta=chunk)
    assert captured is None
