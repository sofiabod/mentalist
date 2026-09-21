import gzip
import json
from pathlib import Path

from mining.mine import build_table, load_sessions
from sfx.schema import POLICIES

CORPUS = Path(__file__).resolve().parents[1] / "data/corpus/normalized.jsonl.gz"
KINDS = {"read", "grep", "edit", "sub-LLM", "test", "lint", "typecheck", "build", "install", "git"}


def test_corpus_smoke():
    assert CORPUS.exists()
    with gzip.open(CORPUS, "rt", encoding="utf-8") as rows:
        kinds = {json.loads(line)["kind"] for line in rows if line.strip()}
    assert kinds
    assert kinds <= KINDS


def test_historical_corpus_loads_and_builds_with_current_schema():
    sessions = load_sessions(CORPUS)

    assert sessions
    assert {event.verb for events in sessions.values() for event in events} <= POLICIES
    assert {event.kind for events in sessions.values() for event in events} <= KINDS
    table = build_table(sessions)
    assert table
    assert all(row["support"] >= 5 for row in table.values())
