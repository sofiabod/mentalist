import ctypes
import errno
import os
import shutil
import stat
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath


class ForkError(Exception):
    pass


class WritePathError(ForkError):
    pass


def validate_write_path(path, root=None):
    """Reject lexical escapes and existing links anywhere in a write target.

    Symlinks copied into a fork can still point into the authoritative tree.
    This guard also protects custom write callbacks; production writes should
    use write_text_in_fork for descriptor-relative, atomic replacement.
    """
    if not isinstance(path, str) or "\x00" in path:
        raise WritePathError(f"invalid streamed write path: {path!r}")
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WritePathError(f"streamed write path escapes fork: {path!r}")
    if root is None:
        return relative
    current = Path(root)
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise WritePathError(f"symlink in streamed write path: {current}")
        final = index == len(relative.parts) - 1
        if not final and not stat.S_ISDIR(info.st_mode):
            raise WritePathError(f"non-directory write ancestor: {current}")
        if final and (not stat.S_ISREG(info.st_mode) or info.st_nlink > 1):
            raise WritePathError(f"non-private regular write target: {current}")
    return current


def write_text_in_fork(root, path, contents):
    """Write one private file without following leaf or ancestor symlinks.

    Keep directories open while traversing, then replace the destination rather
    than truncating its inode. A raced leaf link cannot redirect the write, and
    an existing hard link cannot modify its other aliases.
    """
    relative = validate_write_path(path)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = None
    temporary = None
    try:
        parent_fd = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, dir_fd=parent_fd)
            except FileExistsError:
                pass
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        leaf = relative.name
        try:
            info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise WritePathError(f"non-regular streamed write target: {path!r}")
        temporary = f".sfx-write-{uuid.uuid4().hex}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o666, dir_fd=parent_fd)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            if info is not None:
                os.fchmod(output.fileno(), stat.S_IMODE(info.st_mode))
            output.write(contents)
        os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary = None
    except OSError as exc:
        raise WritePathError(f"cannot safely write {path!r} in fork: {exc}") from exc
    finally:
        if temporary is not None and parent_fd is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if parent_fd is not None:
            os.close(parent_fd)


@dataclass
class ForkHandle:
    path: Path
    scratch_root: Path
    substrate: str
    repo: Path


LARGE_REPO_BYTES = 512 * 1024 * 1024
BACKEND_TIMEOUT_S = 30
PROBE_TIMEOUT_S = 5


@lru_cache(maxsize=1)
def _native_clonefile():
    try:
        clone = ctypes.CDLL(None, use_errno=True).clonefile
    except AttributeError as exc:
        raise OSError(errno.ENOTSUP, "native clonefile is unavailable") from exc
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    return clone


def _clone_file(src, dst):
    """Strict native clone; unlike macOS cp -c, never copy file contents."""
    # CLONE_NOFOLLOW | CLONE_ACL: preserve ACLs, never clone a link's referent.
    if _native_clonefile()(os.fsencode(src), os.fsencode(dst), 0x0001 | 0x0004) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(src))
    return dst


def _repo_probe_source(repo):
    """Use a real nonempty source file, without creating files in the repo."""
    for root, _dirs, files in os.walk(repo, followlinks=False):
        for name in files:
            candidate = Path(root) / name
            info = candidate.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_size:
                return candidate
    return None


def _probe_cp(scratch_root, flag, repo=None):
    """Probe strict cloning across the actual source/destination filesystems.

    The historical -c selector uses clonefile directly, not cp's fallback mode.
    An empty repo can use a scratch-local sample only on the same filesystem.
    Every file in the eventual fork still has to pass the strict clone operation.
    """
    src = None
    owns_source = False
    dst = scratch_root / f".sfx-probe-{uuid.uuid4().hex}"
    try:
        if repo is not None:
            src = _repo_probe_source(repo)
            if src is None and repo.stat().st_dev != scratch_root.stat().st_dev:
                return False
        if src is None:
            src = scratch_root / f".sfx-probe-{uuid.uuid4().hex}"
            owns_source = True
            src.write_bytes(b"x")
        if flag == "-c":
            _clone_file(src, dst)
        else:
            subprocess.run(["cp", flag, str(src), str(dst)],
                           check=True, capture_output=True, timeout=PROBE_TIMEOUT_S)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if owns_source:
            src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def _probe_overlay():
    if shutil.which("fuse-overlayfs"):
        return True
    return Path("/sys/module/overlay").exists() or _proc_supports_overlay()


def _proc_supports_overlay():
    try:
        return "overlay\n" in Path("/proc/filesystems").read_text().replace("\t", "")
    except OSError:
        return False


def _decide_substrate(caps):
    if caps["clonefile"]:
        return "clonefile"
    if caps["large_repo"] and caps["overlay"]:
        return "overlayfs"
    if caps["reflink"]:
        return "reflink"
    if caps["overlay"]:
        return "overlayfs"
    raise ForkError("no COW/overlay substrate available; cannot fork")


_repo_bytes_cache = {}


def _repo_bytes(repo):
    resolved = Path(repo).resolve()
    key = (resolved, resolved.stat().st_mtime_ns)
    if key not in _repo_bytes_cache:
        total = 0
        for p in resolved.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        _repo_bytes_cache[key] = total
    return _repo_bytes_cache[key]


def choose_substrate(scratch_root, repo=None):
    try:
        caps = {
            "clonefile": _probe_cp(scratch_root, "-c", repo),
            "reflink": _probe_cp(scratch_root, "--reflink=always", repo),
            "overlay": _probe_overlay(),
            "large_repo": repo is not None and _repo_bytes(repo) >= LARGE_REPO_BYTES,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForkError(f"cannot probe COW substrate: {exc}") from exc
    return _decide_substrate(caps)


def _canonicalize_existing(path):
    missing = []
    current = path
    while True:
        try:
            canonical = current.resolve(strict=True)
        except (OSError, RuntimeError):
            missing.append(current.name)
            current = current.parent
            continue
        for name in reversed(missing):
            canonical = canonical / name
        return canonical


def _validate_target(target, root):
    parts = target.relative_to(root).parts
    for part in parts:
        if part in ("", ".", ".."):
            raise ForkError(f"non-normal path component: {part!r}")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ForkError(f"symlink in fork path: {current}")


def _verify_canonical(path, expected):
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise ForkError(f"canonical mismatch: {resolved} != {expected}")


def _clonefile_fork(repo, target):
    # clonefile's directory operation is discouraged by its API; copytree only
    # creates directories/symlinks, while each regular file must clone natively.
    aliases = {}

    def clone_entry(src, dst):
        info = os.stat(src, follow_symlinks=False)
        linked = stat.S_ISREG(info.st_mode) and info.st_nlink > 1
        identity = (info.st_dev, info.st_ino)
        if linked and identity in aliases:
            # Link only to a prior PRIVATE clone, never an authoritative inode.
            os.link(aliases[identity], dst, follow_symlinks=False)
        else:
            _clone_file(src, dst)
            if linked:
                aliases[identity] = dst
        return dst

    shutil.copytree(repo, target, symlinks=True, copy_function=clone_entry)


def _reflink_fork(repo, target):
    subprocess.run(["cp", "--reflink=always", "-R", "--preserve=links", str(repo), str(target)],
                   check=True, capture_output=True, timeout=BACKEND_TIMEOUT_S)


def _overlay_dirs(target):
    return target / "merged", target / "upper", target / "work"


def _overlayfs_fork(repo, target):
    merged, upper, work = _overlay_dirs(target)
    for d in (target, merged, upper, work):
        d.mkdir()
    opts = f"lowerdir={repo},upperdir={upper},workdir={work}"
    failures = []
    try:
        plain = subprocess.run(
            ["mount", "-t", "overlay", "overlay", "-o", opts, str(merged)],
            capture_output=True, timeout=BACKEND_TIMEOUT_S)
        if plain.returncode == 0:
            return
        failures.append(plain.stderr.decode(errors="replace"))
    except (OSError, subprocess.SubprocessError) as exc:
        failures.append(str(exc))
    if shutil.which("fuse-overlayfs"):
        try:
            fuse = subprocess.run(
                ["fuse-overlayfs", "-o", opts, str(merged)],
                capture_output=True, timeout=BACKEND_TIMEOUT_S)
            if fuse.returncode == 0:
                return
            failures.append(fuse.stderr.decode(errors="replace"))
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(str(exc))
    raise ForkError(f"overlayfs unavailable: {'; '.join(failures)}")


def _overlay_umount(target):
    merged, _, _ = _overlay_dirs(target)
    if not merged.is_mount():
        return
    r = subprocess.run(["umount", str(merged)], capture_output=True,
                       timeout=BACKEND_TIMEOUT_S)
    if r.returncode != 0 and shutil.which("fusermount"):
        r = subprocess.run(["fusermount", "-u", str(merged)], capture_output=True,
                           timeout=BACKEND_TIMEOUT_S)
    if merged.is_mount():
        raise ForkError(f"overlay umount failed: {r.stderr.decode(errors='replace')}")


def _in_scratch(scratch_root, path):
    try:
        root = scratch_root.resolve(strict=True)
    except OSError:
        return False
    resolved = _canonicalize_existing(path)
    return root != resolved and root in resolved.parents


def _teardown(handle):
    try:
        if handle.substrate == "overlayfs":
            root = handle.path.parent
            _overlay_umount(root)
            if root.exists():
                shutil.rmtree(root)
            return
        if handle.path.exists():
            shutil.rmtree(handle.path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForkError(f"cannot clean up {handle.substrate} fork: {exc}") from exc


def _teardown_root(handle):
    if handle.substrate == "overlayfs":
        return handle.path.parent
    return handle.path


def discard(handle):
    removed = _teardown_root(handle)
    if not _in_scratch(handle.scratch_root, removed):
        raise ForkError(
            f"refusing to discard path outside scratch root: {removed}")
    _teardown(handle)


@contextmanager
def fork(repo_dir, scratch_root, substrate=None):
    try:
        repo = Path(repo_dir).resolve(strict=True)
        scratch_root = Path(scratch_root).resolve(strict=True)
        substrate = substrate or choose_substrate(scratch_root, repo)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ForkError(f"cannot prepare COW fork: {exc}") from exc
    if substrate not in ("clonefile", "reflink", "overlayfs"):
        raise ForkError(f"unknown COW substrate: {substrate!r}")
    name = uuid.uuid4().hex
    target = scratch_root / name
    _validate_target(target, scratch_root)
    path = target / "merged" if substrate == "overlayfs" else target
    handle = ForkHandle(path=path, scratch_root=scratch_root,
                        substrate=substrate, repo=repo)
    try:
        if substrate == "clonefile":
            _clonefile_fork(repo, target)
        elif substrate == "reflink":
            _reflink_fork(repo, target)
        elif substrate == "overlayfs":
            _overlayfs_fork(repo, target)
        _verify_canonical(path, path)
    except Exception as exc:
        _teardown(handle)
        if isinstance(exc, (OSError, subprocess.SubprocessError, shutil.Error)):
            raise ForkError(f"{substrate} fork failed without copy fallback: {exc}") from exc
        raise
    try:
        yield handle
    finally:
        _teardown(handle)
