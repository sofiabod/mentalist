import asyncio
import shlex
import shutil
import sys
from types import SimpleNamespace

import pytest

from eval import sfx_daemon_run
from eval.capture import fs_hash


@pytest.fixture
def trees(tmp_path, monkeypatch):
    repo, fork = tmp_path / "repo", tmp_path / "fork"
    repo.mkdir()
    fork.mkdir()
    (repo / "source.txt").write_text("authoritative before edit\n")
    (fork / "source.txt").write_text("speculative after edit\n")
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "proot")
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    return repo, fork


def test_general_proot_preserves_exact_absolute_command(trees, monkeypatch):
    repo, fork = trees
    command = f"cat  {shlex.quote(str(repo / 'source.txt'))}"
    launches = []
    monkeypatch.setattr(sfx_daemon_run.shutil, "which", lambda _: "/usr/bin/proot")

    def execute(command, **kwargs):
        launches.append((shlex.split(command), kwargs))
        return "speculative after edit\n", "", 0

    monkeypatch.setattr(sfx_daemon_run, "run_drained", execute)
    assert sfx_daemon_run._run_in_fork(fork, ("read", "free", {"cmd": command})) == (
        ("speculative after edit\n", "", 0), "OK")
    assert launches == [([
        "/usr/bin/proot", "-r", "/", "-b", f"{fork}:{repo}", "-w", str(repo),
        "/bin/sh", "-c", command,
    ], {"cwd": fork, "separate_stderr": True})]


@pytest.mark.parametrize("argument", [
    "/etc/passwd", "/app-adjacent/secret", "../outside", "/app/../outside",
])
def test_absolute_external_or_parent_paths_never_launch(trees, monkeypatch, argument):
    _, fork = trees
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unsafe launch"))
    with pytest.raises(ValueError, match="leaves the working tree"):
        sfx_daemon_run._run_in_fork(fork, ("read", "free", {"cmd": f"cat {argument}"}))


def test_absolute_symlink_alias_does_not_escape_fork(trees, monkeypatch):
    repo, fork = trees
    (fork / "alias.txt").symlink_to(repo / "source.txt")
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unsafe launch"))
    with pytest.raises(ValueError, match="symlink"):
        sfx_daemon_run._run_in_fork(fork, ("read", "free", {"cmd": f"cat {repo}/alias.txt"}))


@pytest.mark.parametrize("view", ["cwd", "proot"])
@pytest.mark.parametrize("prefix,kind", [("cat", "read"), ("grep needle", "grep")])
def test_double_colon_does_not_hide_path_traversal(trees, monkeypatch, view, prefix, kind):
    repo, fork = trees
    (fork / "dir::").mkdir()
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", view)
    argument = "dir::/../../outside"
    if view == "proot":
        argument = f"{repo}/{argument}"
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unsafe launch"))
    with pytest.raises(ValueError, match="leaves the working tree"):
        sfx_daemon_run._run_in_fork(fork, (kind, "free", {"cmd": f"{prefix} {argument}"}))


@pytest.mark.parametrize("view", ["cwd", "proot"])
def test_double_colon_does_not_hide_nested_symlink(trees, monkeypatch, view):
    repo, fork = trees
    (fork / "dir::").mkdir()
    (fork / "dir::" / "alias").symlink_to(repo / "source.txt")
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", view)
    argument = f"{repo}/dir::/alias" if view == "proot" else "dir::/alias"
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unsafe launch"))
    with pytest.raises(ValueError, match="symlink"):
        sfx_daemon_run._run_in_fork(fork, ("read", "free", {"cmd": f"cat {argument}"}))


@pytest.mark.parametrize("prefix", ["pytest", "python -m pytest", "python3 -m pytest"])
def test_actual_pytest_node_ids_validate_the_underlying_file(trees, prefix):
    repo, fork = trees
    sfx_daemon_run._validate_fork_command(
        f"{prefix} {repo}/test_example.py::TestExample::test_one", fork, repo=str(repo))
    (fork / "test_example.py").symlink_to(repo / "source.txt")
    with pytest.raises(ValueError, match="symlink"):
        sfx_daemon_run._validate_fork_command(
            f"{prefix} {repo}/test_example.py::TestExample::test_one", fork, repo=str(repo))


@pytest.mark.parametrize("selector", ["../outside", "group/../../outside", "..", ""])
def test_pytest_node_ids_reject_ambiguous_path_selectors(trees, monkeypatch, selector):
    repo, fork = trees
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unsafe launch"))
    command = f"pytest {repo}/test_example.py::{selector}"
    with pytest.raises(ValueError, match="unsupported pytest selector"):
        sfx_daemon_run._validate_fork_command(command, fork, repo=str(repo))


def test_double_colon_is_a_literal_path_for_ordinary_reads(trees):
    repo, fork = trees
    (fork / "dir::").mkdir()
    (fork / "dir::" / "source.txt").write_text("literal name\n")
    sfx_daemon_run._validate_fork_command(
        f"cat {repo}/dir::/source.txt", fork, repo=str(repo))


def test_missing_proot_is_not_an_authoritative_fallback(trees, monkeypatch):
    repo, fork = trees
    monkeypatch.setattr(sfx_daemon_run.shutil, "which", lambda _: None)
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("fallback launch"))
    with pytest.raises(ValueError, match="unavailable"):
        sfx_daemon_run._run_in_fork(fork, ("read", "free", {"cmd": f"cat {repo}/source.txt"}))


@pytest.mark.parametrize("view", ["cwd", "proot"])
def test_live_install_requires_the_requested_path_view_dependency(tmp_path, monkeypatch, view):
    from eval.sfx_live_agent import SFXLiveAgent

    agent = SFXLiveAgent(tmp_path / "logs", fork_path_view=view)
    dependencies = []
    path_view_dependencies = []

    async def require(environment, programs):
        dependencies.append(programs)

    async def noop(*args, **kwargs):
        return None

    async def require_proot(environment):
        path_view_dependencies.append(environment)

    monkeypatch.setattr(agent, "ensure_system_dependencies", require)
    monkeypatch.setattr(agent, "_ensure_proot", require_proot)
    monkeypatch.setattr(agent, "_launch_daemon", noop)
    environment = SimpleNamespace(upload_dir=noop, upload_file=noop, exec=noop)
    asyncio.run(agent.install(environment))
    assert dependencies == [("git", "python3")]
    assert path_view_dependencies == ([environment] if view == "proot" else [])


@pytest.mark.parametrize("prefix,kind", [("ls -i", "read"), ("ls -la", "read"),
                                          ("stat", "read"), ("find", "grep")])
@pytest.mark.parametrize("absolute", [True, False])
def test_fork_metadata_cannot_be_served_as_authoritative_metadata(trees, monkeypatch, prefix, kind, absolute):
    repo, fork = trees
    argument = str(repo / "source.txt") if absolute else "source.txt"
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("metadata launch"))
    with pytest.raises(ValueError, match="filesystem metadata"):
        sfx_daemon_run._validate_fork_command(f"{prefix} {argument}", fork, repo=str(repo))
    reason = "supported speculative contract" if prefix == "find" else "filesystem metadata"
    with pytest.raises(ValueError, match=reason):
        sfx_daemon_run._run_in_fork(fork, (kind, "free", {"cmd": f"{prefix} {argument}"}))


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("proot") is None,
                    reason="requires Linux and the real PRoot binary")
@pytest.mark.parametrize("kind,prefix", [("read", "cat"), ("grep", "grep speculative")])
def test_real_general_proot_reads_modified_fork_at_authoritative_path(trees, kind, prefix):
    repo, fork = trees
    command = f"{prefix} {shlex.quote(str(repo / 'source.txt'))}"
    before = fs_hash(repo), fs_hash(fork)
    speculative, _status = sfx_daemon_run._run_in_fork(fork, (kind, "free", {"cmd": command}))
    assert speculative == ("speculative after edit\n", "", 0)
    assert (fs_hash(repo), fs_hash(fork)) == before
    assert (repo / "source.txt").read_text() == "authoritative before edit\n"
    (repo / "source.txt").write_text("speculative after edit\n")
    native, _duration = sfx_daemon_run._run(kind, {"cmd": command})
    assert native == speculative
