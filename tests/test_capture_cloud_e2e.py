import os
import subprocess

from eval.capture import load_capture


def _init_repo(repo):
    repo.mkdir(parents=True, exist_ok=True)
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], env=env,
                       check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "sfx@example.invalid")
    git("config", "user.name", "SFX fixture")
    (repo / "foo.py").write_text(
        "import time; time.sleep(0.08); print(open('v.txt').read().strip())\n")
    (repo / "v.txt").write_text("0\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")


def _capture_via_env(repo, caps_dir, task, steps, monkeypatch, stop_after=None):
    """Drive a real observed command sequence through SfxEnvironment capture path.

    Writes to caps_dir (the surfaced /logs/agent stand-in). stop_after simulates a
    self-kill: cleanup() (final_snapshot) is skipped, mimicking `kill -9` mid-run.
    """
    from adapters.mini_swe import SfxEnvironment
    monkeypatch.setenv("SFX_CAPTURE", "1")
    monkeypatch.setenv("SFX_CAPTURE_DIR", str(caps_dir))
    monkeypatch.setenv("SFX_CAPTURE_TASK", task)
    monkeypatch.delenv("SFX_SOCKET", raising=False)
    env = SfxEnvironment(cwd=str(repo))
    import atexit
    import time
    for i, cmd in enumerate(steps):
        if i:
            time.sleep(0.02)
        env.execute({"command": cmd}, cwd=str(repo))
        if stop_after is not None and i == stop_after:
            atexit.unregister(env.cleanup)  # kill -9 never runs atexit; no final_snapshot
            return
    env.cleanup()


_STEPS = [
    "cat > v.txt <<'EOF'\n42\nEOF",
    "cat v.txt",
    "cat > v.txt <<'EOF'\n99\nEOF",
    "cat v.txt",
]


def test_capture_emit_full_shape_on_surfaced_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    caps = tmp_path / "logs" / "agent"  # surfaced /logs/agent stand-in
    _capture_via_env(repo, caps, "t1", _STEPS, monkeypatch)

    cap = load_capture(caps / "t1.jsonl")
    assert cap[0]["ev"] == "clean_snapshot"
    assert cap[0]["git_rev"] and cap[0]["tree_hash"]
    assert cap[-1]["ev"] == "final_snapshot"

    calls = [c for c in cap if c["ev"] == "call"]
    assert len(calls) == 4
    assert calls[0]["write_body"] == "42\n"
    assert calls[2]["write_body"] == "99\n"
    assert "42" in calls[1]["stdout"]
    assert calls[1]["think_gap_ms"] > 0.0
    for c in calls:
        assert "returncode" in c and "latency_ms" in c and "fs_hash" in c


def test_capture_partial_survives_kill(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    caps = tmp_path / "logs" / "agent"
    _capture_via_env(repo, caps, "k1", _STEPS, monkeypatch, stop_after=1)

    cap = load_capture(caps / "k1.jsonl")
    assert cap[0]["ev"] == "clean_snapshot"
    calls = [c for c in cap if c["ev"] == "call"]
    assert len(calls) == 2  # only the two before the kill
    assert not any(c["ev"] == "final_snapshot" for c in cap)
