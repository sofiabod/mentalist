import gzip
import json

import pytest

from sfx.schema import ToolEvent, Outcome
from mining.eval_offline import main, recall_per_kind, split_sessions


def _ev(kind, status="OK"):
    verb = "fork" if kind == "edit" else "free"
    return ToolEvent(t=0.0, kind=kind, verb=verb, role="main", epoch=0,
                     outcome=Outcome(kind, status))


def test_recall_per_kind_top1_and_topk():
    train = {
        "t1": [_ev("read"), _ev("edit"), _ev("test", "PASS")],
        "t2": [_ev("read"), _ev("edit"), _ev("test", "PASS")],
        "t3": [_ev("read"), _ev("grep")],
    }
    val = {
        "v1": [_ev("read"), _ev("edit")],
        "v2": [_ev("read"), _ev("grep")],
        "v3": [_ev("edit"), _ev("test", "PASS")],
    }
    r = recall_per_kind(train, val, k=1, top=1, min_support=1)

    assert r["edit"]["top1"] == pytest.approx(1.0)
    assert r["grep"]["top1"] == pytest.approx(0.0)
    assert r["test"]["top1"] == pytest.approx(1.0)

    r2 = recall_per_kind(train, val, k=1, top=2, min_support=1)
    assert r2["edit"]["topk"] == pytest.approx(1.0)
    assert r2["grep"]["topk"] == pytest.approx(1.0)


def test_split_sessions_deterministic_by_id_hash():
    ids = {f"s{i}": [] for i in range(100)}
    train, val = split_sessions(ids, val_frac=0.2)
    again_train, again_val = split_sessions(ids, val_frac=0.2)
    assert set(train) == set(again_train)
    assert set(val) == set(again_val)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(ids)
    assert 10 <= len(val) <= 30


def test_cli_evaluates_supplied_normalized_corpus_without_raw_sources(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "normalized.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as output:
        for index in range(100):
            for event in (_ev("read"), _ev("edit"), _ev("test", "PASS")):
                output.write(json.dumps({"session_id": f"s{index}", **event.to_dict()}) + "\n")
    before = corpus.read_bytes()
    monkeypatch.chdir(tmp_path)

    main(["--corpus", str(corpus)])

    printed = capsys.readouterr().out
    assert "corpus=normalized.jsonl.gz train=" in printed
    assert "top1=1.000 top3=1.000" in printed
    assert corpus.read_bytes() == before
    assert list(tmp_path.iterdir()) == [corpus]
