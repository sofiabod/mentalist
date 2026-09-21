"""Background drain: a tool call that backgrounds work with '&' must be fully
settled before we read/hash the workspace. run_drained() runs a shell command in
its own process group and waits for the WHOLE group (not just the shell) to exit,
so a stray '&' never leaves the fs mid-write when we hash fork-vs-authoritative or
OFF-vs-ON.
"""
import time
import subprocess
from pathlib import Path

from eval.drain import run_drained
import pytest


def test_backgrounded_write_is_complete_after_drain(tmp_path):
    target = tmp_path / "out.txt"
    # backgrounds a writer that sleeps then writes; without draining the group the
    # shell returns immediately and the file is absent/partial when we hash.
    cmd = f"(sleep 0.4; echo DONE > {target}) & echo shell-returned"
    out, rc = run_drained(cmd, cwd=str(tmp_path))
    assert rc == 0
    assert "shell-returned" in out
    assert target.exists(), "drain must wait for the backgrounded writer"
    assert target.read_text().strip() == "DONE"


def test_drain_actually_waits_wall_time(tmp_path):
    cmd = "(sleep 0.5; true) & true"
    t0 = time.monotonic()
    run_drained(cmd, cwd=str(tmp_path))
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.45, f"drain returned before background job finished: {elapsed:.2f}s"


def test_foreground_command_result_unaffected(tmp_path):
    out, rc = run_drained("echo hello", cwd=str(tmp_path))
    assert rc == 0 and out.strip() == "hello"


def test_separate_stderr_preserves_both_channels_and_exit_code(tmp_path):
    result = run_drained("printf out; printf err >&2; exit 3", cwd=str(tmp_path),
                         separate_stderr=True)
    assert result == ("out", "err", 3)


def test_timeout_terminates_spawned_process_group(monkeypatch):
    processes = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        proc = original(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(subprocess.TimeoutExpired):
        run_drained("sleep 30", timeout=0.05)
    assert processes[0].poll() is not None
