import json
import os
import subprocess
from pathlib import Path

import pytest


def _init_repo(d):
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    (d / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True)


def _make_env(cwd):
    from adapters.mini_swe import SfxEnvironment
    return SfxEnvironment(cwd=str(cwd))


def _read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def test_capture_writes_per_call_records(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    caps = tmp_path / "caps"
    monkeypatch.setenv("SFX_CAPTURE", "1")
    monkeypatch.setenv("SFX_CAPTURE_DIR", str(caps))
    monkeypatch.setenv("SFX_CAPTURE_TASK", "demo")
    monkeypatch.delenv("SFX_SOCKET", raising=False)

    env = _make_env(repo)
    env.execute({"command": "cat a.txt"}, cwd=str(repo))
    env.execute({"command": "false"}, cwd=str(repo))
    env.cleanup()

    lines = _read_jsonl(caps / "demo.jsonl")
    snap0, r1, r2, snapf = lines[0], lines[1], lines[2], lines[-1]

    assert snap0["ev"] == "clean_snapshot"
    assert snap0["git_rev"] and snap0["tree_hash"]
    assert snapf["ev"] == "final_snapshot"

    assert r1["ev"] == "call" and r1["seq"] == 0
    assert r1["returncode"] == 0
    assert "hello" in r1["stdout"]
    assert r1["latency_ms"] >= 0
    assert r1["think_gap_ms"] == 0.0
    assert r1["stdout_sha"] and r1["fs_hash"]
    assert r1["kind"] and r1["verb"]

    assert r2["ev"] == "call" and r2["seq"] == 1
    assert r2["returncode"] != 0
    assert r2["think_gap_ms"] >= 0.0
    assert r2["signal"] == 0


def test_capture_records_write_body_and_fs_change(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    caps = tmp_path / "caps"
    monkeypatch.setenv("SFX_CAPTURE", "1")
    monkeypatch.setenv("SFX_CAPTURE_DIR", str(caps))
    monkeypatch.setenv("SFX_CAPTURE_TASK", "w")
    monkeypatch.delenv("SFX_SOCKET", raising=False)

    env = _make_env(repo)
    env.execute({"command": "echo 'newcontent' > a.txt"}, cwd=str(repo))
    env.cleanup()

    lines = _read_jsonl(caps / "w.jsonl")
    calls = [l for l in lines if l["ev"] == "call"]
    assert len(calls) == 1
    assert calls[0]["write_body"] == "newcontent\n"
    clean = lines[0]["tree_hash"]
    assert calls[0]["fs_hash"] != clean


def test_capturer_starts_fresh_tape_no_concatenation(tmp_path, monkeypatch):
    # BUG 4: _append opened the jsonl in append mode and _path never truncated, so a
    # second capture run on the same task CONCATENATED trajectories (garbage ceiling).
    from eval.capture import Capturer
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    caps = tmp_path / "caps"
    monkeypatch.setenv("SFX_CAPTURE", "1")
    monkeypatch.setenv("SFX_CAPTURE_DIR", str(caps))
    monkeypatch.setenv("SFX_CAPTURE_TASK", "t")
    monkeypatch.delenv("SFX_SOCKET", raising=False)

    def one_run():
        cap = Capturer(repo)
        cap.record(kind="run", verb="free", args={"cmd": "x"}, think_gap_ms=0.0,
                   latency_ms=1.0, result={"output": "o", "returncode": 0},
                   write_body=None, now=1.0)
        cap.final()

    one_run()
    one_run()

    lines = _read_jsonl(caps / "t.jsonl")
    calls = [l for l in lines if l["ev"] == "call"]
    snaps = [l for l in lines if l["ev"] == "clean_snapshot"]
    assert len(calls) == 1, f"tape concatenated: {len(calls)} calls"
    assert len(snaps) == 1


def test_load_capture_and_fs_hash_helpers(tmp_path):
    from eval.capture import fs_hash, load_capture
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    h1 = fs_hash(repo)
    (repo / "a.txt").write_text("changed\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    h2 = fs_hash(repo)
    assert h1 != h2

    cap = tmp_path / "x.jsonl"
    cap.write_text('{"ev":"call","seq":0}\n\n{"ev":"call","seq":1}\n')
    recs = load_capture(str(cap))
    assert [r["seq"] for r in recs] == [0, 1]


def test_fs_hash_sees_untracked_ignored_and_deleted_files(tmp_path):
    from eval.capture import fs_hash
    _init_repo(tmp_path)
    initial = fs_hash(tmp_path)
    (tmp_path / "new.py").write_text("new source\n")
    assert fs_hash(tmp_path) != initial
    (tmp_path / ".gitignore").write_text("artifact.bin\n")
    before_artifact = fs_hash(tmp_path)
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    assert fs_hash(tmp_path) != before_artifact
    before_delete = fs_hash(tmp_path)
    (tmp_path / "a.txt").unlink()
    assert fs_hash(tmp_path) != before_delete


def test_fs_hash_no_git_tracks_files_modes_and_empty_directories(tmp_path):
    from eval.capture import fs_hash
    empty = fs_hash(tmp_path)
    p = tmp_path / "script"
    p.write_text("hello\n")
    regular = fs_hash(tmp_path)
    assert regular != empty
    p.chmod(0o755)
    executable = fs_hash(tmp_path)
    assert executable != regular
    (tmp_path / "empty").mkdir()
    assert fs_hash(tmp_path) != executable


def test_fs_hash_symlink_targets_and_hardlink_topology(tmp_path):
    from eval.capture import fs_hash
    (tmp_path / "a").write_text("same")
    (tmp_path / "b").write_text("same")
    link = tmp_path / "link"
    link.symlink_to("a")
    first = fs_hash(tmp_path)
    link.unlink()
    link.symlink_to("b")
    assert fs_hash(tmp_path) != first
    separate = fs_hash(tmp_path)
    (tmp_path / "b").unlink()
    os.link(tmp_path / "a", tmp_path / "b")
    assert fs_hash(tmp_path) != separate
    link.unlink()
    link.symlink_to("missing")
    assert fs_hash(tmp_path)  # hashes a dangling link without following it


def test_fs_hash_only_excludes_git_metadata(tmp_path):
    from eval.capture import fs_hash
    before = fs_hash(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("metadata")
    assert fs_hash(tmp_path) == before
    (tmp_path / ".sfx").mkdir()
    (tmp_path / ".sfx" / "table.json").write_text("{}")
    assert fs_hash(tmp_path) != before


def test_fs_hash_rejects_special_files(tmp_path):
    from eval.capture import fs_hash
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="unsupported workspace entry"):
        fs_hash(tmp_path)
