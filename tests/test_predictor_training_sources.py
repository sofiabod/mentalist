"""Training entrypoints are offline unless their download CLI is invoked."""

import gzip
import json
from pathlib import Path
import subprocess
import sys

from mining import build_corpus


def test_raw_corpus_builder_uses_tiny_local_trace(tmp_path, monkeypatch):
    source = tmp_path / "sources/tracelab-repo/trace/syfi_coding_trace.jsonl.gz"
    source.parent.mkdir(parents=True)
    row = {
        "session_id": "fixture", "round_index": 0,
        "tools": [{"tool_call_id": "read-1", "tool_name": "Read", "is_error": False}],
        "timing_events": [{"event_type": "tool_call", "tool_call_id": "read-1"}],
    }
    with gzip.open(source, "wt", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")
    monkeypatch.setattr(build_corpus, "__file__", str(tmp_path / "src/mining/build_corpus.py"))

    output, count = build_corpus.build()

    assert count == 1
    assert output == tmp_path / "data/corpus/normalized.jsonl"
    event = json.loads(output.read_text())
    assert event["session_id"] == "fixture"
    assert event["kind"] == "read"
    assert event["verb"] == "free"
    assert event["outcome"] == {"kind": "read", "status": "OK"}


def test_training_module_imports_need_no_downloads_or_mining_extra(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc
import socket
import sys

class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'pyarrow', 'huggingface_hub'}:
            raise AssertionError('optional training dependency imported eagerly')

def reject_network(*args, **kwargs):
    raise AssertionError('training import attempted network access')

sys.meta_path.insert(0, RejectOptionalImports())
socket.socket = reject_network
sys.path.insert(0, sys.argv[1])
import mining.ingest
import mining.build_corpus
import mining.ingest_openhands
import mining.eval_offline
import mining.eval_benchmark
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
