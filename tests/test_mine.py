import gzip
import json

import pytest

from sfx.schema import ToolEvent, Outcome
from mining.mine import DEFAULT_CORPUS, K, MIN_SUPPORT, TAU, build_table, load_sessions, main, mine, sessions_from_rows


def _ev(kind, status="OK"):
    verb = "fork" if kind == "edit" else "free"
    return ToolEvent(t=0.0, kind=kind, verb=verb, role="main", epoch=0,
                     outcome=Outcome(kind, status))


def _sessions():
    return {
        "s1": [_ev("read"), _ev("edit"), _ev("test", "FAIL"), _ev("edit"), _ev("test", "PASS")],
        "s2": [_ev("read"), _ev("edit"), _ev("test", "PASS")],
        "s3": [_ev("read"), _ev("grep"), _ev("edit"), _ev("test", "FAIL"), _ev("edit")],
    }


def test_build_table_probabilities_k1_min_support_2():
    table = build_table(_sessions(), k=1, min_support=2)

    read_ctx = "main|read|read:OK"
    assert table[read_ctx]["support"] == 3
    assert table[read_ctx]["p"]["edit"] == pytest.approx(2 / 3)
    assert table[read_ctx]["p"]["grep"] == pytest.approx(1 / 3)

    edit_ctx = "main|edit|edit:OK"
    assert table[edit_ctx]["support"] == 4
    assert table[edit_ctx]["p"]["test"] == pytest.approx(1.0)

    fail_ctx = "main|test|test:FAIL"
    assert table[fail_ctx]["support"] == 2
    assert table[fail_ctx]["p"]["edit"] == pytest.approx(1.0)


def test_min_support_drops_low_evidence_contexts():
    table = build_table(_sessions(), k=1, min_support=2)
    assert "main|grep|grep:OK" not in table


def test_min_support_1_keeps_all():
    table = build_table(_sessions(), k=1, min_support=1)
    assert table["main|grep|grep:OK"]["p"]["edit"] == pytest.approx(1.0)


def test_k2_context_uses_last_two_kinds():
    table = build_table(_sessions(), k=2, min_support=1)
    ctx = "main|read,edit|edit:OK"
    assert table[ctx]["p"]["test"] == pytest.approx(1.0)
    assert table[ctx]["support"] == 2


def test_raw_row_mining_keeps_original_api():
    rows = [{
        "session_id": "a", "round_index": 0,
        "timing_events": [
            {"event_type": "tool_call", "tool_call_id": "r", "tool_name": "Read"},
            {"event_type": "tool_call", "tool_call_id": "e", "tool_name": "Edit"},
        ],
    }]
    sessions = sessions_from_rows(rows)

    assert [event.kind for event in sessions["a"]] == ["read", "edit"]
    assert mine(rows, k=1, min_support=1) == build_table(sessions, k=1, min_support=1)


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_load_normalized_sessions_preserves_order_and_outcomes(tmp_path, suffix):
    events = [("a", _ev("read")), ("b", _ev("edit")), ("a", _ev("test", "FAIL"))]
    rows = "\n".join(json.dumps({"session_id": sid, **event.to_dict()}) for sid, event in events)
    corpus = tmp_path / f"corpus{suffix}"
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(corpus, "wt", encoding="utf-8") as output:
        output.write("\n" + rows + "\n\n")

    assert load_sessions(corpus) == {"a": [events[0][1], events[2][1]], "b": [events[1][1]]}


@pytest.mark.parametrize("legacy,current", [
    ("GET", "free"), ("EXEC", "free"), ("PUT", "fork"), ("PATCH", "fork"),
])
def test_load_legacy_policy_labels_without_reclassifying_events(tmp_path, legacy, current):
    event = _ev("edit" if current == "fork" else "read")
    corpus = tmp_path / "legacy.jsonl"
    corpus.write_text(json.dumps({"session_id": "a", **event.to_dict(), "verb": legacy}))

    assert load_sessions(corpus) == {"a": [event]}


def test_load_unknown_policy_fails(tmp_path):
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(json.dumps({"session_id": "a", **_ev("read").to_dict(), "verb": "invalid"}))

    with pytest.raises(ValueError, match="unknown policy"):
        load_sessions(corpus)


def test_cli_writes_requested_output_with_default_metadata(tmp_path, monkeypatch):
    sessions = {str(i): [_ev("read"), _ev("edit")] for i in range(MIN_SUPPORT)}
    corpus = tmp_path / "normalized.jsonl"
    corpus.write_text("\n".join(
        json.dumps({"session_id": sid, **event.to_dict()})
        for sid, events in sessions.items() for event in events
    ), encoding="utf-8")
    output = tmp_path / "table.json"
    monkeypatch.chdir(tmp_path)

    main(["--corpus", str(corpus), "--output", str(output)])

    assert json.loads(output.read_text()) == {
        "k": K, "min_support": MIN_SUPPORT, "tau": TAU, "table": build_table(sessions),
    }
    assert {path.name for path in tmp_path.iterdir()} == {"normalized.jsonl", "table.json"}


def test_cli_requires_explicit_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_cli_defaults_to_packaged_corpus(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("mining.mine.load_sessions", lambda path: seen.append(path) or {})
    output = tmp_path / "table.json"

    main(["--output", str(output)])

    assert seen == [DEFAULT_CORPUS]
    assert json.loads(output.read_text())["table"] == {}
