"""Offline raw-trace normalization; rebuilding is a new corpus generation.

The historical bundled corpus and shipped global prior are distinct artifacts.
Re-running today's normalizer does not promise to reproduce either exactly.
"""

import gzip
import json
from pathlib import Path

from mining.ingest import read_rows
from mining.mine import sessions_from_rows

SIZE_LIMIT = 10 * 1024 * 1024


def build():
    root = Path(__file__).resolve().parents[2]
    asset = root / "sources/tracelab-repo/trace/syfi_coding_trace.jsonl.gz"
    out_dir = root / "data/corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = sessions_from_rows(read_rows(asset))
    lines = []
    for session_id, events in sessions.items():
        for e in events:
            lines.append(json.dumps({"session_id": session_id, **e.to_dict()}))
    body = "\n".join(lines) + "\n"

    plain = out_dir / "normalized.jsonl"
    if len(body.encode()) > SIZE_LIMIT:
        out = out_dir / "normalized.jsonl.gz"
        with gzip.open(out, "wt") as f:
            f.write(body)
    else:
        out = plain
        out.write_text(body)
    return out, len(lines)


def main():
    out, n = build()
    print(f"wrote {n} events to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
