import subprocess
from pathlib import Path

import pytest

from sfx.fork import (fork, discard, ForkError, ForkHandle, _decide_substrate,
                      _validate_target, choose_substrate)


def _caps(clonefile=False, reflink=False, overlay=False, large_repo=False):
    return {"clonefile": clonefile, "reflink": reflink,
            "overlay": overlay, "large_repo": large_repo}


def test_apfs_prefers_clonefile():
    assert _decide_substrate(_caps(clonefile=True, reflink=True, overlay=True)) == "clonefile"


def test_linux_reflink_when_no_clonefile():
    assert _decide_substrate(_caps(reflink=True, overlay=True)) == "reflink"


def test_large_repo_no_reflink_uses_overlay():
    assert _decide_substrate(_caps(overlay=True, large_repo=True)) == "overlayfs"


def test_no_reflink_uses_overlay_when_available():
    assert _decide_substrate(_caps(overlay=True)) == "overlayfs"


def test_no_substrate_fails_open():
    with pytest.raises(ForkError):
        _decide_substrate(_caps())


def test_overlay_unavailable_fails_open():
    with pytest.raises(ForkError):
        _decide_substrate(_caps(large_repo=True))


def test_large_repo_prefers_overlay_over_reflink():
    assert _decide_substrate(_caps(reflink=True, overlay=True, large_repo=True)) == "overlayfs"


def test_small_repo_prefers_reflink_over_overlay():
    assert _decide_substrate(_caps(reflink=True, overlay=True, large_repo=False)) == "reflink"


@pytest.mark.parametrize("substrate", ["reflink", "overlayfs"])
def test_discard_outside_scratch_refuses(tmp_path, substrate):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    handle = ForkHandle(path=outside, scratch_root=scratch.resolve(),
                        substrate=substrate, repo=tmp_path)
    with pytest.raises(ForkError):
        discard(handle)
    assert (outside / "keep.txt").read_text() == "keep"


def test_overlay_discard_refuses_when_merged_outside_scratch(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside" / "merged"
    outside.mkdir(parents=True)
    handle = ForkHandle(path=outside, scratch_root=scratch.resolve(),
                        substrate="overlayfs", repo=tmp_path)
    with pytest.raises(ForkError):
        discard(handle)
    assert outside.exists()


def test_validate_rejects_dotdot(tmp_path):
    root = (tmp_path / "scratch")
    root.mkdir()
    root = root.resolve()
    with pytest.raises(ForkError):
        _validate_target(root / ".." / "escape", root)


def test_validate_rejects_symlink_component(tmp_path):
    root = (tmp_path / "scratch")
    root.mkdir()
    root = root.resolve()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(ForkError):
        _validate_target(link / "name", root)


def _reflink_supported(scratch):
    src = scratch / "s"
    dst = scratch / "d"
    src.write_bytes(b"x")
    try:
        subprocess.run(["cp", "--reflink=always", str(src), str(dst)],
                       check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.txt").write_text("hello")
    (r / "pkg").mkdir()
    (r / "pkg" / "b.py").write_text("print(1)")
    return r


def test_repo_bytes_walk_cached_across_forks(repo, tmp_path, monkeypatch):
    from sfx import fork as forkmod
    forkmod._repo_bytes_cache.clear()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    walks = {"n": 0}
    real_rglob = Path.rglob

    def counting_rglob(self, pattern):
        if self == repo.resolve():
            walks["n"] += 1
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", counting_rglob)
    choose_substrate(scratch, repo)
    choose_substrate(scratch, repo)
    choose_substrate(scratch, repo)
    assert walks["n"] == 1


def test_reflink_fork_produces_isolated_copy(repo, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if not _reflink_supported(scratch):
        pytest.skip("no reflink-capable fs (btrfs/XFS) here")
    with fork(repo, scratch, substrate="reflink") as h:
        assert (h.path / "a.txt").read_text() == "hello"
        (h.path / "a.txt").write_text("changed")
    assert (repo / "a.txt").read_text() == "hello"


def test_overlay_teardown_raises_when_umount_fails(tmp_path, monkeypatch):
    from sfx import fork as forkmod
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = scratch / "fork"
    merged = target / "merged"
    merged.mkdir(parents=True)
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    monkeypatch.setattr(forkmod.shutil, "which", lambda name: None)
    monkeypatch.setattr(forkmod.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"busy"))
    handle = ForkHandle(path=merged, scratch_root=scratch.resolve(),
                        substrate="overlayfs", repo=tmp_path)
    with pytest.raises(ForkError):
        discard(handle)
    assert merged.exists()


def _overlay_supported():
    import shutil
    return bool(shutil.which("fuse-overlayfs")) or Path("/sys/module/overlay").exists()


def test_overlayfs_fork_merges_and_isolates(repo, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if not _overlay_supported():
        pytest.skip("no overlay/fuse-overlayfs here (needs Linux)")
    try:
        with fork(repo, scratch, substrate="overlayfs") as h:
            assert (h.path / "a.txt").read_text() == "hello"
            (h.path / "a.txt").write_text("changed")
        assert (repo / "a.txt").read_text() == "hello"
    except ForkError:
        pytest.skip("overlay mount not permitted here")
