"""Run a shell command and drain any background ('&') children before returning,
so the filesystem is never mid-write when the caller hashes it.

The command runs in a fresh session (its own process group). After the foreground
shell exits we wait for the rest of the group to finish: a detached '&' writer keeps
running under the same group leader, so polling the group to empty guarantees every
started process has exited before we return.
"""
import os
import signal
import subprocess
import time


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_drained(cmd, cwd=None, env=None, timeout=None, drain_timeout=30.0,
                separate_stderr=False, executable=None):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env,
                         executable=executable,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE if separate_stderr else subprocess.STDOUT,
                         text=True, start_new_session=True)
    pgid = os.getpgid(p.pid)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        raise
    rc = p.returncode
    deadline = time.monotonic() + drain_timeout
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise subprocess.TimeoutExpired(cmd, drain_timeout, output=out, stderr=err)
    if separate_stderr:
        return out, err or "", rc
    return out, rc
