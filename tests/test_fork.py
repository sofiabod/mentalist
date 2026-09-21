import subprocess
from pathlib import Path

import pytest

from sfx.fork import (fork, discard, ForkError, choose_substrate, ForkHandle,
                      _validate_target, _verify_canonical, _in_scratch)


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True, env={"GIT_CONFIG_GLOBAL": "/dev/null",
                                             "GIT_CONFIG_SYSTEM": "/dev/null",
                                             "HOME": str(cwd),
                                             "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "s@x.com")
    _git(r, "config", "user.name", "s")
    (r / "a.txt").write_text("hello")
    (r / "pkg").mkdir()
    (r / "pkg" / "b.py").write_text("print(1)")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


def test_fork_yields_isolated_copy_matching_source(repo, scratch):
    with fork(repo, scratch) as h:
        assert h.path.exists()
        assert (h.path / "a.txt").read_text() == "hello"
        assert (h.path / "pkg" / "b.py").read_text() == "print(1)"


def test_write_in_fork_does_not_touch_source(repo, scratch):
    with fork(repo, scratch) as h:
        (h.path / "a.txt").write_text("changed")
        (h.path / "new.txt").write_text("new")
    assert (repo / "a.txt").read_text() == "hello"
    assert not (repo / "new.txt").exists()


def test_fork_path_is_under_scratch_root(repo, scratch):
    with fork(repo, scratch) as h:
        assert scratch.resolve() in h.path.resolve().parents


def test_context_exit_removes_the_fork(repo, scratch):
    with fork(repo, scratch) as h:
        p = h.path
        assert p.exists()
    assert not p.exists()


def test_validate_rejects_dotdot_component(scratch):
    root = scratch.resolve()
    with pytest.raises(ForkError):
        _validate_target(root / ".." / "escape", root)


def test_validate_rejects_symlink_component_pointing_outside(tmp_path, scratch):
    root = scratch.resolve()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(ForkError):
        _validate_target(link / "name", root)


def test_canonical_mismatch_via_symlinked_target_raises(tmp_path, scratch):
    outside = tmp_path / "real"
    outside.mkdir()
    link = scratch.resolve() / "name"
    link.symlink_to(outside)
    with pytest.raises(ForkError):
        _verify_canonical(link, scratch.resolve() / "name")


def test_fork_through_symlinked_scratch_stays_canonical(repo, tmp_path):
    real = tmp_path / "real-scratch"
    real.mkdir()
    linked = tmp_path / "linked-scratch"
    linked.symlink_to(real)
    with fork(repo, linked, substrate="clonefile") as h:
        assert h.scratch_root == real.resolve()
        assert h.path.resolve() == h.path
        assert real.resolve() in h.path.parents


def test_discard_outside_scratch_refuses_and_preserves(tmp_path, scratch):
    outside = tmp_path / "outside-fork"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    handle = ForkHandle(path=outside, scratch_root=scratch.resolve(),
                        substrate="clonefile", repo=tmp_path)
    with pytest.raises(ForkError):
        discard(handle)
    assert outside.exists()
    assert (outside / "keep.txt").read_text() == "keep"


def test_discard_already_removed_path_does_not_crash(repo, scratch):
    with fork(repo, scratch, substrate="clonefile") as h:
        handle = h
    assert not handle.path.exists()
    assert _in_scratch(handle.scratch_root, handle.path)
    discard(handle)
    assert not handle.path.exists()


def test_teardown_runs_when_creation_raises(repo, scratch, monkeypatch):
    import sfx.fork as fork_mod

    created = {}

    real_clone = fork_mod._clonefile_fork

    def boom(repo_arg, target):
        created["target"] = target
        real_clone(repo_arg, target)
        raise RuntimeError("mid-creation failure")

    monkeypatch.setattr(fork_mod, "_clonefile_fork", boom)
    with pytest.raises(RuntimeError):
        with fork(repo, scratch, substrate="clonefile"):
            pass
    assert not created["target"].exists()
