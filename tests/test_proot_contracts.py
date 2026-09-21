import hashlib
import json
import os
import shlex
import shutil
import sys

import pytest

from eval import sfx_daemon_run
from eval.capture import fs_hash


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("proot") is None,
    reason="requires Linux with the real PRoot binary",
)


CONTRACT = {"script": "check.py", "positionals": 1,
            "path_options": ["--config"], "value_options": [],
            "flags": [], "required": ["--config"]}


def review_check(monkeypatch, source):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([{
        **CONTRACT, "source_sha256": {"check.py": hashlib.sha256(source.encode()).hexdigest()},
    }]))


def program(revision, return_code=0):
    return (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"revision = {revision!r}\n"
        "print(json.dumps({'revision': revision, 'cwd': os.getcwd(), '__file__': __file__, "
        "'argv': sys.argv, 'input': Path(sys.argv[1]).read_text(), "
        "'config': Path(sys.argv[3]).read_text()}, sort_keys=True))\n"
        "sys.stderr.write('checked:' + __file__ + ':' + revision + '\\n')\n"
        f"sys.exit({return_code})\n"
    )


@pytest.fixture
def path_view(tmp_path, monkeypatch):
    repo, fork = tmp_path / "repo", tmp_path / "private-fork"
    repo.mkdir()
    (repo / "check.py").write_text(program("OLD"))
    (repo / "input.txt").write_text("OLD input\n")
    (repo / "config.json").write_text('{"revision":"OLD"}\n')
    shutil.copytree(repo, fork)
    monkeypatch.setenv("SFX_REPO", str(repo))
    review_check(monkeypatch, program("OLD"))
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "proot")
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    command = (f"python3  {shlex.quote(str(repo / 'check.py'))} "
               f"{shlex.quote(str(repo / 'input.txt'))} --config "
               f"{shlex.quote(str(repo / 'config.json'))}")
    return repo, fork, command


@pytest.mark.parametrize("return_code", [0, 7])
def test_real_proot_keeps_absolute_paths_and_matches_native_after_commit(
        path_view, monkeypatch, return_code):
    repo, fork, command = path_view
    (fork / "check.py").write_text(program("NEW", return_code))
    review_check(monkeypatch, program("NEW", return_code))
    (fork / "input.txt").write_text("NEW input\n")
    (fork / "config.json").write_text('{"revision":"NEW"}\n')
    repo_before, fork_before = fs_hash(repo), fs_hash(fork)
    actual_runner, launches = sfx_daemon_run.run_drained, []

    def record_launch(command, **kwargs):
        launches.append(command)
        return actual_runner(command, **kwargs)

    monkeypatch.setattr(sfx_daemon_run, "run_drained", record_launch)
    result, status = sfx_daemon_run._run_in_fork(fork, ("run", "free", {"cmd": command}))
    stdout, stderr, actual_return_code = result

    assert shlex.split(launches[0])[-3:] == ["/bin/sh", "-c", command]
    observed = json.loads(stdout)
    assert observed == {
        "revision": "NEW", "cwd": str(repo), "__file__": str(repo / "check.py"),
        "argv": [str(repo / "check.py"), str(repo / "input.txt"), "--config", str(repo / "config.json")],
        "input": "NEW input\n", "config": '{"revision":"NEW"}\n',
    }
    assert stderr == f"checked:{repo / 'check.py'}:NEW\n"
    assert actual_return_code == return_code
    assert status == ("OK" if return_code == 0 else "ERR")
    assert fs_hash(repo) == repo_before and fs_hash(fork) == fork_before
    assert (repo / "check.py").read_text() == program("OLD")
    assert (repo / "input.txt").read_text() == "OLD input\n"

    for name in ("check.py", "input.txt", "config.json"):
        shutil.copy2(fork / name, repo / name)
    native = actual_runner(command, cwd=repo, separate_stderr=True)
    assert native == result
    assert str(fork) not in stdout + stderr


def test_real_proot_readonly_violation_has_no_publishable_result(path_view, monkeypatch):
    repo, fork, command = path_view
    dishonest_source = (
        "from pathlib import Path\n"
        "Path(__file__).with_name('unexpected.txt').write_text('speculative write')\n"
        "print('THIS MUST NOT BECOME A CACHE RESULT')\n"
    )
    (fork / "check.py").write_text(dishonest_source)
    review_check(monkeypatch, dishonest_source)
    repo_before = fs_hash(repo)
    with pytest.raises(ValueError, match="declared read-only command changed its fork"):
        sfx_daemon_run._run_in_fork(fork, ("run", "free", {"cmd": command}))

    assert (fork / "unexpected.txt").read_text() == "speculative write"
    assert not (repo / "unexpected.txt").exists()
    assert fs_hash(repo) == repo_before


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory_symlink"])
def test_real_proot_rejects_fork_aliases_before_launch(path_view, monkeypatch, alias):
    repo, fork, command = path_view
    if alias == "symlink":
        (fork / "alias").symlink_to(repo / "input.txt")
    elif alias == "hardlink":
        os.link(repo / "input.txt", fork / "alias")
    else:
        (fork / "alias").symlink_to(repo, target_is_directory=True)
    actual_runner, launches = sfx_daemon_run.run_drained, []

    def record_launch(command, **kwargs):
        launches.append(command)
        return actual_runner(command, **kwargs)

    monkeypatch.setattr(sfx_daemon_run, "run_drained", record_launch)
    with pytest.raises(ValueError, match="contract fork contains aliases"):
        sfx_daemon_run._run_in_fork(fork, ("run", "free", {"cmd": command}))

    assert launches == []
    assert (repo / "input.txt").read_text() == "OLD input\n"
