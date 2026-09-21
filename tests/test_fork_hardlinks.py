"""A speculative replacement must not hide authoritative hardlink alias writes."""

import os
from types import SimpleNamespace

import pytest

import sfx.fork as F
from sfx.substrate import FilesystemSubstrate


@pytest.fixture
def roots(tmp_path):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "target.py").write_text("before")
    if not F._probe_cp(scratch, "-c", repo):
        pytest.skip("native clonefile unavailable on this filesystem")
    return repo, scratch


@pytest.mark.parametrize("external_alias", [False, True])
def test_hardlinked_source_write_is_rejected_before_custom_callback(roots, tmp_path, external_alias):
    repo, scratch = roots
    alias = (tmp_path if external_alias else repo) / "alias.py"
    os.link(repo / "target.py", alias)
    called = []

    def apply(path, write):
        called.append(True)
        F.write_text_in_fork(path, write.args["path"], write.args["contents"])

    substrate = FilesystemSubstrate(apply, None)
    with F.fork(repo, scratch, substrate="clonefile") as handle:
        write = SimpleNamespace(args={"path": "target.py", "contents": "after"})
        with pytest.raises(F.WritePathError, match="non-private regular write target"):
            substrate.apply(handle, write)
        assert called == []
        assert (handle.path / "target.py").read_text() == "before"
        assert alias.read_text() == "before"
        assert (repo / "target.py").read_text() == "before"
    assert list(scratch.iterdir()) == []


def test_native_clone_preserves_internal_hardlink_topology_but_not_source_inodes(roots):
    repo, scratch = roots
    (repo / "nested").mkdir()
    source = repo / "target.py"
    os.link(source, repo / "nested" / "alias.py")
    with F.fork(repo, scratch, substrate="clonefile") as handle:
        first, alias = handle.path / "target.py", handle.path / "nested" / "alias.py"
        assert first.stat().st_ino == alias.stat().st_ino
        assert first.stat().st_nlink == alias.stat().st_nlink == 2
        assert first.stat().st_ino != source.stat().st_ino
        # Ordinary writes in the fork see the same alias relationship as source.
        first.write_text("fork only")
        assert alias.read_text() == "fork only"
        assert source.read_text() == (repo / "nested" / "alias.py").read_text() == "before"


def test_missing_source_root_is_rejected_before_speculative_write(tmp_path):
    called = []
    substrate = FilesystemSubstrate(lambda *args: called.append(True), None)
    with pytest.raises(F.WritePathError, match="source root"):
        substrate.apply(SimpleNamespace(path=tmp_path),
                        SimpleNamespace(args={"path": "new.py", "contents": "after"}))
    assert called == []


@pytest.mark.parametrize("backend", ["clonefile", "reflink", "overlayfs"])
def test_source_hardlink_guard_does_not_depend_on_backend_or_fork_link_count(tmp_path, backend):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    source = repo / "target.py"
    source.write_text("before")
    os.link(source, tmp_path / "outside-alias.py")
    private = scratch / "private"
    private.mkdir()
    (private / "target.py").write_text("before")
    assert (private / "target.py").stat().st_nlink == 1
    called = []
    substrate = FilesystemSubstrate(lambda *args: called.append(True), None)
    with pytest.raises(F.WritePathError, match="non-private regular write target"):
        substrate.apply(F.ForkHandle(private, scratch, backend, repo),
                        SimpleNamespace(args={"path": "target.py", "contents": "after"}))
    assert called == []


def test_source_alias_created_after_fork_is_checked_at_write_time(roots, tmp_path):
    repo, scratch = roots
    called = []
    substrate = FilesystemSubstrate(lambda *args: called.append(True), None)
    with F.fork(repo, scratch, substrate="clonefile") as handle:
        os.link(repo / "target.py", tmp_path / "new-external-alias.py")
        assert (handle.path / "target.py").stat().st_nlink == 1
        with pytest.raises(F.WritePathError, match="non-private regular write target"):
            substrate.apply(handle, SimpleNamespace(args={"path": "target.py", "contents": "after"}))
        assert called == []
