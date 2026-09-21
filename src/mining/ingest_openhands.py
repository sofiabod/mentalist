"""Train the benchmark prior from OpenHands trajectories with repo exclusions.

The historical table records only a generic provenance label, not a dataset
revision or content hash. Keep source snapshots when reproducing a generation;
the download CLI alone cannot establish identity with that historical input.
"""

import json
import re
from pathlib import Path

from mining.normalize import classify
from mining.mine import build_table, K, MIN_SUPPORT, TAU
from sfx.schema import ToolEvent, Outcome

_UPLOADED = re.compile(r"<uploaded_files>\s*(.*?)\s*</uploaded_files>", re.S)
_FUNC = re.compile(r"<function=([a-zA-Z0-9_\-]+)>(.*?)</function>", re.S)
_PARAM = re.compile(r"<parameter=([a-zA-Z0-9_\-]+)>(.*?)</parameter>", re.S)
_EXIT = re.compile(r"exit code (\d+)")

EDITOR_READ_CMDS = {"view"}


def trajectory_repo(messages):
    for m in messages:
        if m["role"] != "user":
            continue
        mm = _UPLOADED.search(m["content"])
        if mm:
            parts = mm.group(1).strip().split("/")[-1].split("__")
            return f"{parts[0]}/{parts[1]}"
    return None


def _params(body):
    return {k: v.strip() for k, v in _PARAM.findall(body)}


def _tool_calls(messages):
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for name, body in _FUNC.findall(m["content"]):
            obs = messages[i + 1] if i + 1 < len(messages) else None
            yield name, _params(body), obs


def _classify_call(name, params):
    if name == "str_replace_editor":
        cmd = params.get("command", "")
        if cmd in EDITOR_READ_CMDS:
            return ("read", "free")
        return ("edit", "fork")
    if name == "execute_bash":
        return classify("Bash", params.get("command", ""))
    return None


def _is_error(name, obs):
    if obs is None:
        return False
    content = obs["content"]
    if name == "execute_bash":
        mm = _EXIT.search(content)
        return mm is not None and mm.group(1) != "0"
    return content.lstrip().startswith("ERROR")


def _status(is_error, kind):
    if kind in ("test", "lint", "typecheck", "build"):
        return "FAIL" if is_error else "PASS"
    return "ERR" if is_error else "OK"


def messages_to_events(messages):
    events = []
    t = 0
    for name, params, obs in _tool_calls(messages):
        kv = _classify_call(name, params)
        if kv is None:
            continue
        kind, verb = kv
        is_error = _is_error(name, obs)
        events.append(
            ToolEvent(
                t=float(t),
                kind=kind,
                verb=verb,
                role="main",
                epoch=0,
                args={"provenance": "benchmark"},
                outcome=Outcome(kind, _status(is_error, kind)),
            )
        )
        t += 1
    return events


def ingest_trajectories(trajectories, blocklist, drop_blocked=True):
    sessions = {}
    kept = 0
    dropped = 0
    for idx, messages in enumerate(trajectories):
        repo = trajectory_repo(messages)
        if drop_blocked and repo in blocklist:
            dropped += 1
            continue
        sessions[f"openhands:{idx}"] = messages_to_events(messages)
        kept += 1
    ingested_repos = {trajectory_repo(m) for i, m in enumerate(trajectories) if f"openhands:{i}" in sessions}
    assert not (ingested_repos & blocklist), f"blocklist leak: {ingested_repos & blocklist}"
    return sessions, kept, dropped


def build_benchmark_table(parquet_path, blocklist):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    trajectories = [list(row) for row in df["messages"]]
    sessions, kept, dropped = ingest_trajectories(trajectories, blocklist)
    table = build_table(sessions, K, MIN_SUPPORT)
    return table, sessions, kept, dropped


def main():
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parents[2]
    blocklist = set(json.loads((root / "data/tables/deepswe_repos.json").read_text())["owner_name"])
    path = hf_hub_download(
        repo_id="SWE-Gym/OpenHands-SFT-Trajectories",
        repo_type="dataset",
        filename="data/train.success.oss-00000-of-00001.parquet",
    )
    table, sessions, kept, dropped = build_benchmark_table(path, blocklist)
    n_events = sum(len(v) for v in sessions.values())
    out = root / "data/tables/benchmark.json"
    with open(out, "w") as f:
        json.dump(
            {"k": K, "min_support": MIN_SUPPORT, "tau": TAU, "provenance": "benchmark", "table": table},
            f, indent=2, sort_keys=True,
        )
    print(f"kept={kept} dropped={dropped} sessions={len(sessions)} events={n_events} contexts={len(table)} -> {out}")


if __name__ == "__main__":
    main()
