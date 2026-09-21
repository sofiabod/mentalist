"""Held-out kind prediction on normalized traces, not a runtime speed benchmark.

The bundled corpus is historical; it is not the exact input for the shipped
global prior. This evaluator trains its own table on the training split.
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from sfx.schema import context_key
from mining.mine import DEFAULT_CORPUS, K, MIN_SUPPORT, build_table, load_sessions, _key_str


def split_sessions(sessions, val_frac=0.2):
    train, val = {}, {}
    threshold = int(val_frac * 1000)
    for sid, events in sessions.items():
        h = int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 1000
        (val if h < threshold else train)[sid] = events
    return train, val


def recall_per_kind(train, val, k=2, top=1, min_support=5):
    table = build_table(train, k=k, min_support=min_support)
    hits = defaultdict(int)
    total = defaultdict(int)
    for events in val.values():
        for i in range(1, len(events)):
            role, kinds, last_outcome = context_key(events[:i], k)
            key = _key_str(role, kinds, last_outcome)
            true = events[i].kind
            total[true] += 1
            entry = table.get(key)
            if not entry:
                continue
            ranked = sorted(entry["p"], key=lambda x: entry["p"][x], reverse=True)
            if true in ranked[:top]:
                hits[true] += 1

    result = {}
    for kind, n in total.items():
        result[kind] = {("topk" if top > 1 else "top1"): hits[kind] / n}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate held-out prediction on a normalized corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args(argv)
    sessions = load_sessions(args.corpus)
    train, val = split_sessions(sessions)
    r1 = recall_per_kind(train, val, k=K, top=1, min_support=MIN_SUPPORT)
    r3 = recall_per_kind(train, val, k=K, top=3, min_support=MIN_SUPPORT)
    print(f"corpus={args.corpus.name} train={len(train)} val={len(val)} sessions")
    for kind in sorted(r1):
        print(f"  {kind:10s} top1={r1[kind]['top1']:.3f} top3={r3[kind]['topk']:.3f}")


if __name__ == "__main__":
    main()
