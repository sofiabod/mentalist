import json

import pytest

from eval import sfx_daemon_run


@pytest.mark.parametrize("kind,command", [
    ("read", "cat /app/value.txt"),
    ("read", "cat ../value.txt"),
    ("read", "cat nested/../value.txt"),
    ("read", "cat --file=/app/value.txt"),
    ("read", "cat --file=../value.txt"),
    ("read", "cat -I/app"),
    ("read", "cat -I.."),
    ("read", 'cat "$PWD/value.txt"'),
    ("read", "cat ~/value.txt"),
    ("read", "cat value*.txt"),
    ("read", "cat value.txt && cat value.txt"),
    ("read", "cat value.txt | head -n 1"),
    ("read", "cat value.txt\ncat value.txt"),
    ("read", "pwd"),
    ("run", "python /app/program.py"),
    ("grep", "find -L ."),
    ("grep", "find . -follow"),
    ("grep", "grep -R pattern ."),
    ("grep", "grep -nR pattern ."),
    ("grep", "grep --dereference-recursive pattern ."),
    ("grep", "rg --follow pattern ."),
])
def test_fork_rejections_never_start_runner(tmp_path, monkeypatch, kind, command):
    calls = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError):
        sfx_daemon_run._run_in_fork(tmp_path, (kind, "free", {"cmd": command}))
    assert calls == []


@pytest.mark.parametrize("link_kind", ["leaf", "ancestor", "dangling"])
def test_fork_argument_symlinks_never_start_runner(tmp_path, monkeypatch, link_kind):
    source, private = tmp_path / "source", tmp_path / "fork"
    source.mkdir()
    private.mkdir()
    (source / "value.txt").write_text("OLD")
    (private / "value.txt").write_text("NEW")
    target = source if link_kind == "ancestor" else source / "value.txt"
    if link_kind == "dangling":
        target = source / "missing.txt"
    (private / "alias").symlink_to(target)
    argument = "alias/value.txt" if link_kind == "ancestor" else "alias"
    calls = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="symlink"):
        sfx_daemon_run._run_in_fork(private, ("read", "free", {"cmd": f"cat {argument}"}))
    assert calls == []
    assert (source / "value.txt").read_text() == "OLD"


def test_relative_fork_command_and_result_are_not_rewritten(tmp_path, monkeypatch):
    command = "cat  './nested/value.txt'"
    output = ("NEW\n", "separate stderr", 0)
    calls = []

    def runner(actual, **kwargs):
        calls.append((actual, kwargs))
        return output

    monkeypatch.setattr(sfx_daemon_run, "run_drained", runner)
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    result = sfx_daemon_run._run_in_fork(tmp_path, ("read", "free", {"cmd": command}))
    assert result == (output, "OK")
    assert calls == [(command, {"cwd": tmp_path, "separate_stderr": True})]


@pytest.mark.parametrize("absolute", [False, True])
def test_get_runner_preserves_exact_inside_repository_command(tmp_path, monkeypatch, absolute):
    command = f"cat  '{tmp_path}/value.txt'" if absolute else "cat  './value.txt'"
    calls = []
    output = ("authoritative view", 0)

    def runner(actual, **kwargs):
        calls.append((actual, kwargs))
        return output

    monkeypatch.setattr(sfx_daemon_run, "run_drained", runner)
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    assert sfx_daemon_run._run("read", {"cmd": command})[0] == output
    assert calls[0][0] == command and calls[0][1]["cwd"] == str(tmp_path)


@pytest.mark.parametrize("command", [
    "pwd", "ls -la", "stat value.txt", "cat /proc/uptime", "cat /tmp/outside.txt",
    "cat ../value.txt", "cat nested/../value.txt", "cat /app/value.txt",
])
def test_get_external_traversal_and_metadata_reads_never_execute(tmp_path, monkeypatch, command):
    from sfx.resolver import Ctx
    from sfx.script_contracts import ScriptInvocationRejected

    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: pytest.fail("unsafe GET"))
    with pytest.raises(ScriptInvocationRejected):
        sfx_daemon_run._run("read", {"cmd": command})
    assert sfx_daemon_run._resolve_args("read", Ctx(repo=tmp_path, session={"read": command})) is None


@pytest.mark.parametrize("link_kind", ["leaf", "ancestor", "dangling"])
def test_get_explicit_path_aliases_never_execute(tmp_path, monkeypatch, link_kind):
    from sfx.resolver import Ctx
    from sfx.script_contracts import ScriptInvocationRejected

    repo, outside = tmp_path / "repo", tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (outside / "value.txt").write_text("outside\n")
    target = outside if link_kind == "ancestor" else outside / "value.txt"
    if link_kind == "dangling":
        target = outside / "missing"
    (repo / "alias").symlink_to(target)
    argument = "alias/value.txt" if link_kind == "ancestor" else "alias"
    command = f"cat {argument}"
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: pytest.fail("aliased GET"))
    with pytest.raises(ScriptInvocationRejected, match="symlink"):
        sfx_daemon_run._run("read", {"cmd": command})
    assert sfx_daemon_run._resolve_args("read", Ctx(repo=repo, session={"read": command})) is None


@pytest.mark.parametrize("option", ["-r", "-R", "-nr", "--recursive", "--rec",
                                   "--dereference-recursive", "-d recurse", "--directories=recurse"])
@pytest.mark.parametrize("forked", [False, True])
def test_recursive_grep_is_not_a_repository_local_read(tmp_path, monkeypatch, option, forked):
    from sfx.script_contracts import ScriptInvocationRejected

    (tmp_path / "alias").symlink_to(tmp_path.parent, target_is_directory=True)
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: pytest.fail("recursive grep"))
    args = {"cmd": f"grep {option} pattern ."}
    with pytest.raises(ScriptInvocationRejected, match="search option"):
        if forked:
            sfx_daemon_run._run_in_fork(tmp_path, ("grep", "free", args))
        else:
            sfx_daemon_run._run("grep", args)


@pytest.mark.parametrize("command", ["grep -n word value.txt", "rg -n word .", "rg --files ."])
def test_nonfollowing_search_remains_admissible_for_get(tmp_path, monkeypatch, command):
    calls = []
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.delenv("RIPGREP_CONFIG_PATH", raising=False)
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda actual, **kw: (
        calls.append(actual) or ("result\n", 0)))
    assert sfx_daemon_run._run("grep", {"cmd": command})[0] == ("result\n", 0)
    assert calls == [command]


@pytest.mark.parametrize("absolute", [False, True])
def test_old_tree_read_prediction_is_discarded_after_authoritative_commit(
        tmp_path, monkeypatch, absolute):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    target = repo / "value.txt"
    target.write_text("OLD")
    (repo / "alias").symlink_to(target)
    command = f"cat {target}" if absolute else "cat alias"
    table = {"k": 1, "min_support": 1, "tau": .35,
             "table": {"main|edit|edit:OK": {"support": 20, "p": {"read": .99}}}}
    daemon = sfx_daemon_run.build_daemon(1, table=table)
    daemon.resolve_args = lambda kind, ctx: {"cmd": command}
    calls = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: calls.append(a))
    daemon.session_start("audit", repo, "main", scratch=scratch)
    write = {"path": "value.txt", "contents": "NEW"}
    try:
        token = daemon.mutation_begin("audit", write)
        chain = daemon.call_stream_delta("audit", "edit", "Edit", json.dumps(write),
                                         mutation_id=token)
        assert chain is not None and chain.future is not None
        with pytest.raises(ValueError):
            chain.future.result(timeout=5)
        assert target.read_text() == "OLD"
        target.write_text("NEW")
        assert daemon.mutation_end("audit", token, success=True) is False
        assert daemon.resolve("audit", "read", {"cmd": command}) == ("miss", None)
        assert calls == []
    finally:
        daemon.shutdown()
