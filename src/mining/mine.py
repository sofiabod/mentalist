import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from sfx.schema import ToolEvent, context_key

K = 2
MIN_SUPPORT = 5
TAU = 0.35
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data/corpus/normalized.jsonl.gz"
_LEGACY_POLICIES = {"GET": "free", "EXEC": "free", "PUT": "fork", "PATCH": "fork"}


def _key_str(role, kinds, last_outcome):
    return f"{role}|{','.join(kinds)}|{last_outcome}"


def build_table(sessions, k=K, min_support=MIN_SUPPORT):
    counts = defaultdict(Counter)
    for events in sessions.values():
        for i in range(1, len(events)):
            role, kinds, last_outcome = context_key(events[:i], k)
            key = _key_str(role, kinds, last_outcome)
            counts[key][events[i].kind] += 1

    table = {}
    for key, ctr in counts.items():
        support = sum(ctr.values())
        if support < min_support:
            continue
        table[key] = {
            "support": support,
            "p": {kind: c / support for kind, c in ctr.items()},
        }
    return table


def sessions_from_rows(rows):
    from mining.normalize import normalize_row

    sessions = defaultdict(list)
    for row in rows:
        sessions[row["session_id"]].extend(normalize_row(row))
    return sessions


def mine(rows, k=K, min_support=MIN_SUPPORT):
    return build_table(sessions_from_rows(rows), k, min_support)


def load_sessions(path=DEFAULT_CORPUS):
    """Load ordered, normalized ToolEvents without raw-trace dependencies."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    sessions = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            session_id = row.pop("session_id")
            row["verb"] = _LEGACY_POLICIES.get(row["verb"], row["verb"])
            sessions[session_id].append(ToolEvent.from_dict(row))
    return sessions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build SFX priors from a historical normalized corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    table = build_table(load_sessions(args.corpus))
    payload = {"k": K, "min_support": MIN_SUPPORT, "tau": TAU, "table": table}
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    print(f"wrote {len(table)} contexts to {args.output}")


if __name__ == "__main__":
    main()
