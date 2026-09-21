import os
from types import SimpleNamespace

import pytest

from sfx.fork import WritePathError, write_text_in_fork
from sfx.substrate import FilesystemSubstrate


@pytest.mark.parametrize("link_kind", ["leaf", "ancestor", "dangling"])
def test_substrate_rejects_links_before_custom_write(tmp_path, link_kind):
    root = tmp_path / "fork"
    root.mkdir()
    outside = tmp_path / "authoritative"
    outside.mkdir()
    original = outside / "keep.txt"
    original.write_text("original")
    if link_kind == "ancestor":
        (root / "link").symlink_to(outside, target_is_directory=True)
        path = "link/keep.txt"
    else:
        target = original if link_kind == "leaf" else outside / "new.txt"
        (root / "link").symlink_to(target)
        path = "link"
    called = []
    def callback(fp, write):
        called.append(True)
        (fp / write.args["path"]).write_text("changed")
    substrate = FilesystemSubstrate(callback, None)
    with pytest.raises(WritePathError):
        substrate.apply(SimpleNamespace(path=root), SimpleNamespace(args={"path": path}))
    assert called == []
    assert original.read_text() == "original"
    assert not (outside / "new.txt").exists()


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape", "", ".", "a/../../escape"])
def test_secure_write_rejects_lexical_escapes(tmp_path, path):
    with pytest.raises(WritePathError):
        write_text_in_fork(tmp_path, path, "changed")


@pytest.mark.parametrize("ancestor", [False, True])
def test_secure_write_does_not_follow_links(tmp_path, ancestor):
    root = tmp_path / "fork"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "file"
    victim.write_text("original")
    (root / "link").symlink_to(outside if ancestor else victim)
    with pytest.raises(WritePathError):
        write_text_in_fork(root, "link/file" if ancestor else "link", "changed")
    assert victim.read_text() == "original"


def test_secure_write_preserves_external_hardlink_and_file_mode(tmp_path):
    root = tmp_path / "fork"
    root.mkdir()
    original = tmp_path / "original"
    original.write_text("original")
    original.chmod(0o751)
    os.link(original, root / "file")
    write_text_in_fork(root, "file", "changed")
    assert original.read_text() == "original"
    assert (root / "file").read_text() == "changed"
    assert (root / "file").stat().st_mode & 0o777 == 0o751


def test_secure_write_creates_nested_file_and_cleans_temporary(tmp_path):
    write_text_in_fork(tmp_path, "new/nested/file.txt", "héllo\n")
    assert (tmp_path / "new/nested/file.txt").read_text() == "héllo\n"
    assert list(tmp_path.rglob(".sfx-write-*")) == []


def test_raced_leaf_symlink_is_replaced_without_touching_target(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.write_text("original")
    root = tmp_path / "fork"
    root.mkdir()
    real_replace = os.replace
    def raced_replace(source, destination, **kwargs):
        (root / destination).symlink_to(original)
        return real_replace(source, destination, **kwargs)
    monkeypatch.setattr(os, "replace", raced_replace)
    write_text_in_fork(root, "file", "changed")
    assert original.read_text() == "original"
    assert (root / "file").read_text() == "changed"
    assert not (root / "file").is_symlink()
