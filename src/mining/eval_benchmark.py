import json
from pathlib import Path

from sfx.schema import context_key
from mining.mine import build_table, _key_str, K, MIN_SUPPORT
from mining.eval_offline import split_sessions, recall_per_kind

PARQUET = "data/train.success.oss-00000-of-00001.parquet"


def edit_run_recall(train, val, k=K, min_support=MIN_SUPPORT):
    table = build_table(train, k=k, min_support=min_support)
    hit1 = hit3 = n = 0
    for events in val.values():
        for i in range(1, len(events)):
            if events[i].kind != "run" or events[i - 1].kind != "edit":
                continue
            n += 1
            role, kinds, last_outcome = context_key(events[:i], k)
            entry = table.get(_key_str(role, kinds, last_outcome))
            if not entry:
                continue
            ranked = sorted(entry["p"], key=lambda x: entry["p"][x], reverse=True)
            hit1 += "run" in ranked[:1]
            hit3 += "run" in ranked[:3]
    return {"top1": hit1 / n if n else 0.0, "top3": hit3 / n if n else 0.0, "n": n}


def load_blocklist(root):
    return set(json.loads((root / "data/tables/deepswe_repos.json").read_text())["owner_name"])


def main():
    from huggingface_hub import hf_hub_download
    from mining.ingest_openhands import ingest_trajectories
    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    blocklist = load_blocklist(root)
    path = hf_hub_download(repo_id="SWE-Gym/OpenHands-SFT-Trajectories",
                           repo_type="dataset", filename=PARQUET)
    df = pd.read_parquet(path)
    trajectories = [list(row) for row in df["messages"]]
    sessions, kept, dropped = ingest_trajectories(trajectories, blocklist)
    train, val = split_sessions(sessions)
    r1 = recall_per_kind(train, val, k=K, top=1, min_support=MIN_SUPPORT)
    r3 = recall_per_kind(train, val, k=K, top=3, min_support=MIN_SUPPORT)
    er = edit_run_recall(train, val, k=K, min_support=MIN_SUPPORT)

    print(f"kept={kept} dropped={dropped} train={len(train)} val={len(val)}")
    for kind in sorted(r1):
        print(f"  {kind:10s} top1={r1[kind]['top1']:.3f} top3={r3[kind]['topk']:.3f}")
    print(f"edit->run n={er['n']} top1={er['top1']:.3f} top3={er['top3']:.3f}")


if __name__ == "__main__":
    main()
