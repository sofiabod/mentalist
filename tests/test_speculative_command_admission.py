"""Production admission: familiar runner names are not read-only contracts."""
import hashlib
import json

import pytest

from eval import sfx_daemon_run as runtime
from mining.normalize import classify
from sfx.fork import ForkError
from sfx.resolver import Ctx
from sfx.script_contracts import ScriptInvocationRejected


CODE_COMMANDS = [
    ("run", "python3 generated.py"),
    ("test", "pytest"),
    ("test", "python3 -m pytest"),
    ("test", "cargo test"),
    ("test", "npm test"),
    ("lint", "ruff check ."),
    ("lint", "eslint ."),
    ("typecheck", "mypy ."),
    ("typecheck", "tsc"),
    ("build", "make"),
    ("build", "cargo build"),
    ("build", "npm run build"),
]


@pytest.mark.parametrize("kind,command", CODE_COMMANDS)
@pytest.mark.parametrize("runner", ["ordinary", "fork"])
def test_unreviewed_code_commands_decline_before_any_execution(tmp_path, monkeypatch, kind, command, runner):
    # These real programs would create an artifact if the guard let them run.
    (tmp_path / "Makefile").write_text("all:\n\t@printf changed > artifact.txt\n")
    (tmp_path / "test_artifact.py").write_text(
        "from pathlib import Path\ndef test_artifact():\n"
        "    Path('artifact.txt').write_text('changed')\n")
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    launches = []
    monkeypatch.setattr(runtime, "run_drained", lambda *args, **kwargs: launches.append(args))
    assert classify("Bash", command) == (kind, "free")
    with pytest.raises(ScriptInvocationRejected, match="source-pinned"):
        if runner == "ordinary":
            runtime._run(kind, {"cmd": command})
        else:
            runtime._run_in_fork(tmp_path, (kind, "free", {"cmd": command}))
    assert launches == [] and not (tmp_path / "artifact.txt").exists()
    assert runtime._resolve_args(kind, Ctx(repo=tmp_path, session={kind: command})) is None


def _definition(source, *, pinned=True):
    result = {"script": "reviewed.py", "positionals": 0, "path_options": [],
              "value_options": [], "flags": [], "required": []}
    if pinned:
        result["source_sha256"] = {"reviewed.py": hashlib.sha256(source.encode()).hexdigest()}
    return result


@pytest.mark.parametrize("runner", ["ordinary", "fork"])
def test_unpinned_contract_does_not_bypass_code_admission(tmp_path, monkeypatch, runner):
    source = "from pathlib import Path\nPath('artifact.txt').write_text('changed')\n"
    (tmp_path / "reviewed.py").write_text(source)
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([_definition(source, pinned=False)]))
    monkeypatch.setattr(runtime, "run_drained", lambda *a, **kw: pytest.fail("unpinned execution"))
    with pytest.raises(ScriptInvocationRejected, match="source-pinned"):
        if runner == "ordinary":
            runtime._run("run", {"cmd": "python3 reviewed.py"})
        else:
            runtime._run_in_fork(tmp_path, ("run", "free", {"cmd": "python3 reviewed.py"}))
    assert not (tmp_path / "artifact.txt").exists()
    assert runtime._resolve_args("run", Ctx(repo=tmp_path, session={"run": "python3 reviewed.py"})) is None


def _reviewed_repo(tmp_path, monkeypatch, source):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "input.txt").write_text("before\n")
    (repo / "reviewed.py").write_text(source)
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "cwd")
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([_definition(source)]))
    try:
        backend = runtime.choose_substrate(scratch, repo)
    except ForkError:
        pytest.skip("real clonefile/reflink backend unavailable")
    if backend not in ("clonefile", "reflink"):
        pytest.skip("real clonefile/reflink backend unavailable")
    return repo, scratch


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("writes", [False, True])
def test_pinned_readonly_result_can_serve_but_fork_writes_cannot(tmp_path, monkeypatch, streamed, writes):
    source = ("from pathlib import Path\nPath('artifact.txt').write_text('changed')\n"
              if writes else "") + "print('reviewed output')\n"
    repo, scratch = _reviewed_repo(tmp_path, monkeypatch, source)
    table = {"k": 1, "min_support": 1, "tau": 0.35, "table": {
        "main|edit|edit:OK": {"support": 20, "p": {"run": 1.0}},
        "main|read|read:OK": {"support": 20, "p": {"run": 1.0}},
    }}
    log = []
    daemon = runtime.build_daemon(1, table=table, log=log.append)
    session = daemon.session_start("test", str(repo), "main", scratch=str(scratch))
    session.arg_by_kind["run"] = "python3 reviewed.py"
    snapshots = []
    real_fork = runtime.fork
    from contextlib import contextmanager

    @contextmanager
    def observed_fork(*args, **kwargs):
        with real_fork(*args, **kwargs) as handle:
            snapshots.append(handle.path)
            yield handle

    if not streamed:
        monkeypatch.setattr(runtime, "fork", observed_fork)
    try:
        if streamed:
            write = {"path": "input.txt", "contents": "after\n"}
            token = daemon.mutation_begin("test", write_args=write)
            chain = daemon.call_stream_delta("test", "edit", "Edit", json.dumps(write), mutation_id=token)
            assert chain is not None and chain.future is not None
            if writes:
                with pytest.raises(ValueError, match="changed its fork"):
                    chain.future.result(timeout=10)
            else:
                chain.future.result(timeout=10)
            (repo / "input.txt").write_text(write["contents"])
            assert daemon.mutation_end("test", token, success=True) is (not writes)
            daemon.resolve("test", "edit", write)
        else:
            daemon.call_executed("test", "read", "free", "OK", {"cmd": "cat input.txt"}, 1)
            session.executor.drain()
            assert snapshots and all(not path.exists() for path in snapshots)
        outcome, result = daemon.resolve("test", "run", {"cmd": "python3 reviewed.py"})
        assert not (repo / "artifact.txt").exists()
        if writes:
            assert (outcome, result) == ("miss", None)
        else:
            assert outcome == "hit_completed" and result == ("reviewed output\n", "", 0)
    finally:
        daemon.shutdown()


@pytest.mark.parametrize("command", [
    "cat data && make", "cat data | make", "cat data\npytest",
    "sed -e 'e touch marker' data", "less data", "pdftotext document.pdf", "find .",
    "cat {a,b}.txt", "rg --pre=program pattern .", "rg --pr=program pattern .",
    "rg -z pattern .", "rg --search-zip pattern .", "rg --hostname-bin=program pattern .",
])
@pytest.mark.parametrize("runner", ["ordinary", "fork"])
def test_read_labels_cannot_hide_execution_or_shell_expansion(tmp_path, monkeypatch, command, runner):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.setattr(runtime, "run_drained", lambda *a, **kw: pytest.fail("unsafe read execution"))
    kind, _verb = classify("Bash", command)
    with pytest.raises(ValueError):
        if runner == "ordinary":
            runtime._run(kind, {"cmd": command})
        else:
            runtime._run_in_fork(tmp_path, (kind, "free", {"cmd": command}))


@pytest.mark.parametrize("command", ["cat data", "head -n 2 data", "grep -n word data", "rg -n word .", "rg --no-config -n word ."])
def test_supported_literal_reads_still_execute(tmp_path, monkeypatch, command):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.delenv("RIPGREP_CONFIG_PATH", raising=False)
    launches = []
    monkeypatch.setattr(runtime, "run_drained", lambda *args, **kwargs: (launches.append(args) or ("read\n", 0)))
    kind, verb = classify("Bash", command)
    assert verb == "free"
    assert runtime._run(kind, {"cmd": command})[0] == ("read\n", 0)
    assert launches == [(command,)]


def test_ripgrep_config_guard_uses_actual_task_environment(monkeypatch):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.delenv("RIPGREP_CONFIG_PATH", raising=False)
    monkeypatch.setattr(runtime, "run_drained", lambda *a, **kw: ("read\n", 0))
    with pytest.raises(ScriptInvocationRejected, match="environment config"):
        runtime._run("grep", {"cmd": "rg -n word ."}, tool_env={"RIPGREP_CONFIG_PATH": "/task/config"})
    assert runtime._run("grep", {"cmd": "rg --no-config -n word ."},
                        tool_env={"RIPGREP_CONFIG_PATH": "/task/config"})[0] == ("read\n", 0)
    # A flag-shaped positional after '--' does not actually disable config.
    with pytest.raises(ScriptInvocationRejected, match="environment config"):
        runtime._run("grep", {"cmd": "rg -- --no-config data"}, tool_env={"RIPGREP_CONFIG_PATH": "/task/config"})
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", "/controller/config")
    assert runtime._run("grep", {"cmd": "rg -n word ."}, tool_env={})[0] == ("read\n", 0)


@pytest.mark.parametrize("environment", [
    {"BASH_ENV": "/task/startup.sh"},
    {"ENV": "/task/startup.sh"},
    {"BASH_FUNC_cat%%": "() { printf changed; }"},
    {"BASH_FUNC_cat%%": ""},
])
@pytest.mark.parametrize("runner", ["ordinary", "fork"])
def test_shell_hooks_decline_before_literal_read_execution(tmp_path, monkeypatch, environment, runner):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    monkeypatch.setattr(runtime, "run_drained", lambda *a, **kw: pytest.fail("shell hook executed"))
    original = dict(environment)
    with pytest.raises(ScriptInvocationRejected, match="startup or function hooks"):
        if runner == "ordinary":
            runtime._run("read", {"cmd": "cat data"}, tool_env=environment, tool_shell="/bin/bash")
        else:
            runtime._run_in_fork(tmp_path, ("read", "free", {"cmd": "cat data"}),
                                 tool_env=environment, tool_shell="/bin/bash")
    assert environment == original


def test_shell_hook_guard_uses_actual_environment_without_sanitizing_it(monkeypatch):
    monkeypatch.setenv("BASH_ENV", "/controller/startup.sh")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", "[]")
    calls = []
    monkeypatch.setattr(runtime, "run_drained", lambda *args, **kwargs: (
        calls.append(kwargs) or ("read\n", 0)))
    environment = {"BASH_ENV": "", "ENV": "", "PATH": "/trusted/bin"}
    assert runtime._run("read", {"cmd": "cat data"}, tool_env=environment)[0] == ("read\n", 0)
    assert calls[0]["env"] is environment
    with pytest.raises(ScriptInvocationRejected, match="startup or function hooks"):
        runtime._run("read", {"cmd": "cat data"})
    assert len(calls) == 1
