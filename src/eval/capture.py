import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

CAPTURE_ENV = "SFX_CAPTURE"
CAPTURE_DIR_ENV = "SFX_CAPTURE_DIR"
CAPTURE_TASK_ENV = "SFX_CAPTURE_TASK"


def _sha(obj):
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_capture(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def fs_hash(workdir):
    """Hash workspace contents, entry types, permission bits and symlink targets.

    Git control metadata is the sole exclusion. Untracked and ignored files,
    empty directories, and hardlink relationships are included, also outside a
    Git repo. Timestamps and ownership are not portable across fresh sandboxes
    and are not part of this content/permissions comparison. Symlinks are never
    followed; unreadable or unsupported entries fail the measurement explicitly.
    """
    workdir = Path(workdir).resolve(strict=True)
    if not workdir.is_dir():
        raise ValueError(f"workspace is not a directory: {workdir}")
    parts = []
    hardlinks = {}

    def visit(directory):
        for p in sorted(directory.iterdir(), key=lambda p: p.name):
            if p.name == ".git":
                continue
            rel = p.relative_to(workdir).as_posix()
            info = p.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                parts.append((rel, "symlink", mode, os.readlink(p)))
            elif stat.S_ISDIR(info.st_mode):
                parts.append((rel, "directory", mode))
                visit(p)
            elif stat.S_ISREG(info.st_mode):
                inode = (info.st_dev, info.st_ino)
                linked_to = hardlinks.setdefault(inode, rel)
                digest = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
                parts.append((rel, "file", mode, digest.hexdigest(), linked_to))
            else:
                raise ValueError(f"unsupported workspace entry: {rel}")

    visit(workdir)
    return _sha(parts)


def git_rev(workdir):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=workdir,
                          capture_output=True, text=True).stdout.strip()


def capture_enabled():
    return bool(os.environ.get(CAPTURE_ENV))


def _path():
    d = Path(os.environ.get(CAPTURE_DIR_ENV, "data/captures"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{os.environ.get(CAPTURE_TASK_ENV, 'task')}.jsonl"


def _append(rec):
    with open(_path(), "a") as f:
        f.write(json.dumps(rec) + "\n")


class Capturer:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.seq = 0
        self.prev_return_t = None
        _path().write_text("")  # fresh tape per run; append mode would concatenate runs
        _append({"ev": "clean_snapshot", "git_rev": git_rev(self.workdir),
                 "tree_hash": fs_hash(self.workdir)})

    def start_gap(self, now):
        return 0.0 if self.prev_return_t is None else max((now - self.prev_return_t) * 1000.0, 0.0)

    def record(self, *, kind, verb, args, think_gap_ms, latency_ms, result,
               write_body, now):
        stdout = result.get("output", "")
        stderr = ""
        rc = result.get("returncode", 0)
        _append({
            "ev": "call", "seq": self.seq, "kind": kind, "verb": verb,
            "args": args, "think_gap_ms": think_gap_ms, "latency_ms": latency_ms,
            "stdout": stdout, "stderr": stderr, "returncode": rc,
            "signal": -rc if rc < 0 else 0,
            "write_body": write_body,
            "stdout_sha": _sha(stdout), "stderr_sha": _sha(stderr),
            "fs_hash": fs_hash(self.workdir),
        })
        self.seq += 1
        self.prev_return_t = now

    def final(self):
        _append({"ev": "final_snapshot", "git_rev": git_rev(self.workdir),
                 "tree_hash": fs_hash(self.workdir)})
