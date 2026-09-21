"""COW backend selection must not mistake a full copy for a successful clone."""

import ctypes
import errno
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import sfx.fork as F


@pytest.fixture
def roots(tmp_path):
    repo = tmp_path / "repo"
    scratch = tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "source.py").write_text("print('authoritative')\n")
    return repo, scratch


def test_reflink_probe_uses_actual_repo_source_and_strict_flag(roots, monkeypatch):
    repo, scratch = roots
    source = repo / "source.py"
    before = source.read_bytes()
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert Path(argv[-2]) == source
        assert Path(argv[-1]).parent == scratch
        Path(argv[-1]).write_bytes(b"probe artifact")

    monkeypatch.setattr(F.subprocess, "run", run)
    assert F._probe_cp(scratch, "--reflink=always", repo)
    argv, kwargs = calls[0]
    assert argv[:2] == ["cp", "--reflink=always"]
    assert kwargs == dict(check=True, capture_output=True, timeout=F.PROBE_TIMEOUT_S)
    assert source.read_bytes() == before
    assert sorted(p.name for p in repo.iterdir()) == ["source.py"]
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", [
    FileNotFoundError("cp missing"),
    subprocess.CalledProcessError(1, ["cp"], stderr=b"unsupported filesystem"),
    subprocess.TimeoutExpired(["cp"], 5),
])
def test_failed_reflink_probe_preserves_source_and_removes_partial_copy(
        roots, monkeypatch, failure):
    repo, scratch = roots
    original = (repo / "source.py").read_bytes()

    def fail(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(F.subprocess, "run", fail)
    assert not F._probe_cp(scratch, "--reflink=always", repo)
    assert list(scratch.iterdir()) == []
    assert (repo / "source.py").read_bytes() == original


def test_choose_substrate_passes_repo_to_both_strict_probes(roots, monkeypatch):
    repo, scratch = roots
    calls = []

    def probe(destination, flag, source):
        calls.append((destination, flag, source))
        return flag == "--reflink=always"

    monkeypatch.setattr(F, "_probe_cp", probe)
    monkeypatch.setattr(F, "_probe_overlay", lambda: False)
    assert F.choose_substrate(scratch, repo) == "reflink"
    assert calls == [(scratch, "-c", repo), (scratch, "--reflink=always", repo)]


def test_reflink_fork_requires_clone_and_reports_backend(roots, monkeypatch):
    repo, scratch = roots
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        # Simulate the backend target so the path/teardown contract can run on macOS.
        Path(argv[-1]).mkdir()

    monkeypatch.setattr(F.subprocess, "run", run)
    with F.fork(repo, scratch, substrate="reflink") as handle:
        assert handle.substrate == "reflink"
        assert handle.path.is_dir()
    argv, kwargs = calls[0]
    assert argv[:3] == ["cp", "--reflink=always", "-R"]
    assert "--preserve=links" in argv
    assert kwargs == dict(check=True, capture_output=True, timeout=F.BACKEND_TIMEOUT_S)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", [
    FileNotFoundError("cp missing"),
    subprocess.CalledProcessError(1, ["cp"], stderr=b"not supported"),
    subprocess.TimeoutExpired(["cp"], 30),
])
def test_reflink_failure_is_forkerror_and_removes_partial_tree(
        roots, monkeypatch, failure):
    repo, scratch = roots
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        destination = Path(argv[-1])
        destination.mkdir()
        (destination / "partial").write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(F.subprocess, "run", fail)
    with pytest.raises(F.ForkError, match="without copy fallback"):
        with F.fork(repo, scratch, substrate="reflink"):
            pytest.fail("a failed clone must never yield a handle")
    assert len(calls) == 1
    assert "--reflink=always" in calls[0]
    assert list(scratch.iterdir()) == []
    assert (repo / "source.py").read_text() == "print('authoritative')\n"


def test_native_clone_propagates_errno_without_copying(roots, monkeypatch):
    repo, scratch = roots
    source = repo / "source.py"
    destination = scratch / "clone"
    calls = []

    def clone(src, dst, flags):
        calls.append((src, dst, flags))
        ctypes.set_errno(errno.ENOTSUP)
        return -1

    monkeypatch.setattr(F, "_native_clonefile", lambda: clone)
    with pytest.raises(OSError) as exc:
        F._clone_file(source, destination)
    assert exc.value.errno == errno.ENOTSUP
    assert calls == [(bytes(source), bytes(destination), 0x0001 | 0x0004)]
    assert not destination.exists()


def test_clonefile_probe_uses_native_call_not_cp_c(roots, monkeypatch):
    repo, scratch = roots
    calls = []

    def clone(src, dst):
        calls.append((src, dst))
        dst.write_bytes(b"probe artifact")

    def forbidden(*args, **kwargs):
        pytest.fail("macOS cp -c may copy, so must not establish COW support")

    monkeypatch.setattr(F, "_clone_file", clone)
    monkeypatch.setattr(F.subprocess, "run", forbidden)
    assert F._probe_cp(scratch, "-c", repo)
    assert calls[0][0] == repo / "source.py"
    assert list(scratch.iterdir()) == []


def test_clonefile_unsupported_fails_without_cp_or_copyfile(roots, monkeypatch):
    repo, scratch = roots

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "clone unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("strict clone must never fall back to ordinary file copying")

    monkeypatch.setattr(F, "_clone_file", unsupported)
    monkeypatch.setattr(F.subprocess, "run", forbidden)
    monkeypatch.setattr(F.shutil, "copyfile", forbidden)
    assert not F._probe_cp(scratch, "-c", repo)
    with pytest.raises(F.ForkError, match="without copy fallback"):
        with F.fork(repo, scratch, substrate="clonefile"):
            pytest.fail("a failed clone must never yield a handle")
    assert list(scratch.iterdir()) == []


def test_native_clone_preserves_links_modes_and_private_writes(roots):
    repo, scratch = roots
    source = repo / "source.py"
    source.chmod(0o751)
    (repo / "link").symlink_to("source.py")
    (repo / "empty").mkdir()
    if not F._probe_cp(scratch, "-c", repo):
        pytest.skip("native clonefile unavailable on this filesystem")
    with F.fork(repo, scratch, substrate="clonefile") as handle:
        assert handle.substrate == "clonefile"
        assert (handle.path / "link").is_symlink()
        assert (handle.path / "link").readlink() == Path("source.py")
        assert (handle.path / "empty").is_dir()
        clone = handle.path / "source.py"
        assert stat.S_IMODE(clone.stat().st_mode) == 0o751
        assert clone.stat().st_ino != source.stat().st_ino
        clone.write_text("private modification")
        assert source.read_text() == "print('authoritative')\n"
    assert list(scratch.iterdir()) == []


def test_overlay_mount_is_writable_and_bounded(roots, monkeypatch):
    repo, scratch = roots
    calls = []

    def mount(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(F.subprocess, "run", mount)
    with F.fork(repo, scratch, substrate="overlayfs") as handle:
        assert handle.substrate == "overlayfs"
    argv, kwargs = calls[0]
    assert argv[:5] == ["mount", "-t", "overlay", "overlay", "-o"]
    options = argv[5].split(",")
    assert "ro" not in options
    assert f"lowerdir={repo.resolve()}" in options
    assert any(option.startswith("upperdir=") for option in options)
    assert any(option.startswith("workdir=") for option in options)
    assert kwargs["timeout"] == F.BACKEND_TIMEOUT_S
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", [
    FileNotFoundError("mount missing"),
    subprocess.TimeoutExpired(["mount"], 30),
])
def test_overlay_missing_or_timed_out_is_forkerror(roots, monkeypatch, failure):
    repo, scratch = roots

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(F.subprocess, "run", fail)
    monkeypatch.setattr(F.shutil, "which", lambda name: None)
    with pytest.raises(F.ForkError, match="overlayfs unavailable"):
        with F.fork(repo, scratch, substrate="overlayfs"):
            pytest.fail("an unavailable overlay must never yield a handle")
    assert list(scratch.iterdir()) == []


def test_unknown_backend_is_explicit_forkerror(roots):
    repo, scratch = roots
    with pytest.raises(F.ForkError, match="unknown COW substrate"):
        with F.fork(repo, scratch, substrate="copy"):
            pytest.fail("ordinary copying is not a COW backend")
    assert list(scratch.iterdir()) == []


def test_missing_repo_is_forkerror(roots):
    repo, scratch = roots
    with pytest.raises(F.ForkError, match="cannot prepare COW fork"):
        with F.fork(repo / "missing", scratch):
            pytest.fail("missing source must fail before launching speculation")


def test_empty_repo_on_other_filesystem_is_not_probed_in_scratch(tmp_path, monkeypatch):
    repo = tmp_path / "empty"
    scratch = tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    real_stat = Path.stat

    def stat_with_other_device(path, *args, **kwargs):
        if path == repo:
            return SimpleNamespace(st_dev=scratch.stat().st_dev + 1)
        return real_stat(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("scratch-local cloning does not prove cross-filesystem support")

    monkeypatch.setattr(Path, "stat", stat_with_other_device)
    monkeypatch.setattr(F, "_clone_file", forbidden)
    monkeypatch.setattr(F.subprocess, "run", forbidden)
    assert not F._probe_cp(scratch, "-c", repo)
    assert not F._probe_cp(scratch, "--reflink=always", repo)
    assert list(repo.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_empty_repo_same_filesystem_probe_leaves_both_trees_unchanged(
        tmp_path, monkeypatch):
    repo = tmp_path / "empty"
    scratch = tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()

    def clone(src, dst):
        assert src.parent == scratch
        assert src.read_bytes() == b"x"
        dst.write_bytes(b"probe artifact")

    monkeypatch.setattr(F, "_clone_file", clone)
    assert F._probe_cp(scratch, "-c", repo)
    assert list(repo.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_overlay_missing_unmount_command_does_not_delete_mounted_tree(
        roots, monkeypatch):
    repo, scratch = roots
    target = scratch / "fork"
    merged = target / "merged"
    merged.mkdir(parents=True)
    (merged / "keep").write_text("still mounted")
    handle = F.ForkHandle(merged, scratch.resolve(), "overlayfs", repo)

    def fail(*args, **kwargs):
        raise FileNotFoundError("umount missing")

    monkeypatch.setattr(Path, "is_mount", lambda path: path == merged)
    monkeypatch.setattr(F.subprocess, "run", fail)
    with pytest.raises(F.ForkError, match="cannot clean up overlayfs"):
        F.discard(handle)
    assert (merged / "keep").read_text() == "still mounted"
