import gzip
import json

from mining.ingest import read_rows


def test_read_rows_skips_malformed_lines(tmp_path):
    p = tmp_path / "trace.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"session_id": "a"}) + "\n")
        f.write("{not valid json\n")
        f.write("\n")
        f.write(json.dumps({"session_id": "b"}) + "\n")
    rows = list(read_rows(str(p)))
    assert [r["session_id"] for r in rows] == ["a", "b"]
