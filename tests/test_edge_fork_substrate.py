import subprocess
from pathlib import Path

import pytest

import sfx.fork as F
from sfx.fork import (fork, discard, ForkError, ForkHandle, _decide_substrate,
                      _repo_bytes, _in_scratch, _overlay_umount)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.txt").write_text("hello")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


@pytest.fixture(autouse=True)
def _clear_repo_bytes_cache():
    F._repo_bytes_cache.clear()
    yield
    F._repo_bytes_cache.clear()


# ---- DEFECT candidates (assert the invariant; fail on current code) ----

def test_discard_of_overlay_handle_whose_parent_is_scratch_root_must_not_wipe_scratch(
        tmp_path):
    """discard guards handle.path but rmtrees handle.path.parent for overlayfs."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sibling = scratch / "other-case-fork"
    sibling.mkdir()
    (sibling / "data.txt").write_text("other case data")
    merged = scratch / "merged"
    merged.mkdir()
    handle = ForkHandle(path=merged, scratch_root=scratch.resolve(),
                        substrate="overlayfs", repo=tmp_path)
    try:
        discard(handle)
    except ForkError:
        pass
    assert sibling.exists(), "sibling case fork was wiped by discard teardown"
    assert scratch.exists(), "entire scratch root wiped by discard teardown"


def test_repo_bytes_reflects_growth_after_first_call(tmp_path):
    """large_repo decision uses a permanently cached repo size."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a").write_bytes(b"x" * 10)
    assert _repo_bytes(repo) == 10
    (repo / "big").write_bytes(b"y" * 1000)
    assert _repo_bytes(repo) == 1010, "repo size is stale after growth"


def test_repo_bytes_does_not_carry_across_distinct_repo_at_same_path(tmp_path):
    """distinct repos reusing a resolved path collide in the size cache."""
    p = tmp_path / "reused"
    p.mkdir()
    (p / "small").write_bytes(b"x" * 5)
    assert _repo_bytes(p) == 5
    import shutil
    shutil.rmtree(p)
    p.mkdir()
    (p / "huge").write_bytes(b"z" * 2000)
    assert _repo_bytes(p) == 2000, "stale size from a prior repo at same path"


# ---- HARDENING tests (assert current-correct behavior; expected to pass) ----

def test_decide_substrate_all_caps_false_raises_forkerror():
    with pytest.raises(ForkError):
        _decide_substrate({"clonefile": False, "reflink": False,
                           "overlay": False, "large_repo": False})


def test_decide_substrate_large_repo_without_overlay_falls_to_reflink():
    got = _decide_substrate({"clonefile": False, "reflink": True,
                             "overlay": False, "large_repo": True})
    assert got == "reflink"


def test_decide_substrate_large_repo_with_overlay_prefers_overlayfs():
    got = _decide_substrate({"clonefile": False, "reflink": True,
                             "overlay": True, "large_repo": True})
    assert got == "overlayfs"


def test_discard_symlink_inside_scratch_pointing_outside_is_refused(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("precious")
    link = scratch / "fork"
    link.symlink_to(outside)
    handle = ForkHandle(path=link, scratch_root=scratch.resolve(),
                        substrate="clonefile", repo=tmp_path)
    with pytest.raises(ForkError):
        discard(handle)
    assert (outside / "keep.txt").read_text() == "precious"


def test_overlay_umount_still_mounted_raises(tmp_path, monkeypatch):
    """A failed umount that leaves the mount up must raise, never silently rmtree."""
    target = tmp_path / "t"
    (target / "merged").mkdir(parents=True)

    class R:
        returncode = 1
        stderr = b"busy"
    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(Path, "is_mount", lambda self: str(self).endswith("merged"))
    monkeypatch.setattr(F.shutil, "which", lambda x: None)
    with pytest.raises(ForkError):
        _overlay_umount(target)


def test_failed_overlay_fork_command_tears_down_and_raises(repo, scratch,
                                                           monkeypatch):
    """No-substrate/failed mount fail-open: raise ForkError, leave scratch clean."""
    class R:
        returncode = 1
        stderr = b"no overlay"
        stdout = b""
    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(F.shutil, "which", lambda x: None)
    with pytest.raises(ForkError):
        with fork(repo, scratch, substrate="overlayfs"):
            pass
    assert list(scratch.iterdir()) == []


def test_creation_failure_via_clonefile_leaves_no_leftover(repo, scratch,
                                                           monkeypatch):
    real = F._clonefile_fork
    seen = {}

    def boom(repo_arg, target):
        seen["t"] = target
        real(repo_arg, target)
        raise RuntimeError("mid-creation")
    monkeypatch.setattr(F, "_clonefile_fork", boom)
    with pytest.raises(RuntimeError):
        with fork(repo, scratch, substrate="clonefile"):
            pass
    assert not seen["t"].exists()
    assert list(scratch.iterdir()) == []


def test_in_scratch_false_when_scratch_root_missing(tmp_path):
    scratch = tmp_path / "gone"
    p = scratch / "fork"
    assert _in_scratch(scratch, p) is False


def test_fork_clonefile_teardown_target_is_the_uuid_dir_not_scratch(repo, scratch):
    """clonefile tears down handle.path (the uuid dir), never scratch itself."""
    with fork(repo, scratch, substrate="clonefile") as h:
        assert h.path.parent.resolve() == scratch.resolve()
        assert scratch.resolve() in h.path.resolve().parents
    assert scratch.exists()
    assert list(scratch.iterdir()) == []
