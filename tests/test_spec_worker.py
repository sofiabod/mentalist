import base64
import hashlib
import json
import os
from pathlib import Path
import shlex

import pytest

from eval import spec_worker


@pytest.fixture
def configuration(tmp_path):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    return {"repo": str(repo), "scratch": str(scratch), "script_contracts": [], "fork_path_view": "cwd"}


def request(configuration, **operation):
    return {"settings": configuration, "request": operation}


def test_worker_runs_real_read_and_preserves_full_output(configuration):
    from pathlib import Path

    (Path(configuration["repo"]) / "data").write_text("λ\nsecond line\n")
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat data"}))
    assert response["ok"] is True
    assert response["result"]["output"] == ("λ\nsecond line\n", "", 0)
    assert response["result"]["duration_ms"] >= 0


def test_worker_runs_real_fork_read_without_changing_authoritative_state(configuration):
    from pathlib import Path

    repo = Path(configuration["repo"])
    fork = Path(configuration["scratch"]) / "job"
    fork.mkdir()
    (repo / "data").write_text("OLD\n")
    (fork / "data").write_text("NEW\n")
    response = spec_worker.process(request(
        configuration, op="run_in_fork", path=str(fork), hop=["read", "free", {"cmd": "cat data"}]))
    assert response == {"ok": True, "result": {"output": ("NEW\n", "", 0), "status": "OK"}}
    assert (repo / "data").read_text() == "OLD\n"


@pytest.mark.parametrize("operation", [
    {"op": "shell", "command": "touch escaped"},
    {"op": "run", "kind": "read", "args": {"cmd": "cat data"}, "settings": {}},
    {"op": "run", "kind": "read", "args": {"cmd": "cat data", "cwd": "/"}},
    {"op": "run", "kind": "read", "args": {"cmd": "cat\x00data"}},
    {"op": "run", "kind": "edit", "args": {"cmd": "touch data"}},
    {"op": "run_in_fork", "path": "/outside", "hop": ["read", "free", {"cmd": "cat data"}]},
])
def test_worker_rejects_untyped_or_escaping_requests_before_execution(configuration, monkeypatch, operation):
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run", lambda *a: pytest.fail("unexpected execution"))
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run_in_fork", lambda *a: pytest.fail("unexpected execution"))
    assert spec_worker.process(request(configuration, **operation)) == {
        "ok": False, "error": {"category": "protocol", "type": "ProtocolViolation"}}


def test_worker_rejects_fork_alias(configuration, tmp_path, monkeypatch):
    from pathlib import Path

    alias = Path(configuration["scratch"]) / "alias"
    alias.symlink_to(tmp_path)
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run_in_fork", lambda *a: pytest.fail("unexpected execution"))
    response = spec_worker.process(request(
        configuration, op="run_in_fork", path=str(alias), hop=["read", "free", {"cmd": "cat data"}]))
    assert response["error"]["category"] == "protocol"


def test_worker_preserves_default_deny_and_safe_decline(configuration):
    response = spec_worker.process(request(
        configuration, op="run", kind="run", args={"cmd": "python generated.py"}))
    assert response == {"ok": False, "error": {"category": "admission", "type": "ScriptInvocationRejected"}}


@pytest.mark.parametrize("command", ["cat /proc/uptime", "cat ../outside", "ls -la", "pwd"])
def test_worker_external_or_metadata_get_is_a_typed_decline(configuration, monkeypatch, command):
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "run_drained", lambda *a, **kw: pytest.fail("unsafe GET"))
    assert spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": command})) == {
        "ok": False, "error": {"category": "admission", "type": "ScriptInvocationRejected"}}


def test_worker_checks_pinned_fork_source_before_execution(configuration, monkeypatch):
    from pathlib import Path

    fork = Path(configuration["scratch"]) / "job"
    fork.mkdir()
    (fork / "reviewed.py").write_text("print('unreviewed')\n")
    configuration["script_contracts"] = [{
        "script": "reviewed.py", "positionals": 0, "path_options": [], "value_options": [],
        "flags": [], "required": [],
        "source_sha256": {"reviewed.py": hashlib.sha256(b"print('reviewed')\n").hexdigest()},
    }]
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "run_drained", lambda *a, **k: pytest.fail("unreviewed execution"))
    response = spec_worker.process(request(
        configuration, op="run_in_fork", path=str(fork), hop=["run", "free", {"cmd": "python reviewed.py"}]))
    assert response == {"ok": False, "error": {"category": "admission", "type": "ScriptSourceMismatch"}}


def test_worker_does_not_demote_postexecution_failure_to_admission(configuration, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("declared read-only command changed its fork")

    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run", fail)
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat data"}))
    assert response == {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}


def test_worker_error_never_echoes_exception_details(configuration, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("SECRET private path /private/lab")

    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run", fail)
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat data"}))
    assert "SECRET" not in json.dumps(response) and "/private/lab" not in json.dumps(response)
    assert response["error"]["category"] == "worker"


def test_worker_main_prints_one_json_envelope(configuration, monkeypatch, capsys):
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run", lambda *a, **kw: (("out\n", "", -9), 2.0))
    encoded = base64.b64encode(json.dumps(request(
        configuration, op="run", kind="read", args={"cmd": "cat data"})).encode()).decode()
    spec_worker.main([encoded])
    output = capsys.readouterr()
    assert output.err == "" and len(output.out.splitlines()) == 1
    assert json.loads(output.out)["result"]["output"] == ["out\n", "", -9]


@pytest.mark.parametrize("data", ['{"op":"run","op":"shell"}', '{"value":NaN}', b'\xff'])
def test_worker_json_decoder_rejects_ambiguity(data):
    with pytest.raises(spec_worker.ProtocolViolation):
        spec_worker.strict_loads(data)


@pytest.mark.parametrize("repo,scratch", [("/app", "/app/scratch"), ("/scratch/app", "/scratch"), ("/app", "/app")])
def test_worker_requires_disjoint_repository_and_scratch(repo, scratch):
    with pytest.raises(spec_worker.ProtocolViolation):
        spec_worker.validate_settings({
            "repo": repo, "scratch": scratch, "script_contracts": [], "fork_path_view": "cwd"})


@pytest.mark.parametrize("operation", ["run", "run_in_fork"])
def test_worker_preserves_callers_environment_without_leaking_private_settings(configuration, monkeypatch, operation):
    keys = ["PYTHONPATH", "SFX_REPO", "SFX_SCRATCH", "SFX_SCRIPT_CONTRACTS",
            "SFX_FORK_PATH_VIEW", "SFX_SEPARATE_STDERR"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/caller-original-pythonpath")
    monkeypatch.setenv("SFX_REPO", "/caller-original-repo")
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "caller-original-view")
    expected = {key: os.environ.get(key) for key in keys}
    source = "import json, os\nprint(json.dumps({key: os.environ.get(key) for key in " + repr(keys) + "}))\n"
    definition = {"script": "probe.py", "positionals": 0, "path_options": [],
                  "value_options": [], "flags": [], "required": [],
                  "source_sha256": {"probe.py": hashlib.sha256(source.encode()).hexdigest()}}
    repo, scratch = Path(configuration["repo"]), Path(configuration["scratch"])
    (repo / "probe.py").write_text(source)
    if operation == "run":
        from sfx.fork import ForkError, choose_substrate

        try:
            backend = choose_substrate(scratch, repo)
        except ForkError:
            pytest.skip("real clonefile/reflink unavailable")
        if backend not in ("clonefile", "reflink"):
            pytest.skip("real clonefile/reflink unavailable")
    configuration["script_contracts"] = [definition]
    args = {"cmd": "python3 probe.py"}
    if operation == "run_in_fork":
        fork = scratch / "job"
        fork.mkdir()
        (fork / "probe.py").write_text(source)
        payload = request(configuration, op=operation, path=str(fork), hop=["run", "free", args])
    else:
        payload = request(configuration, op="run", kind="run", args=args)
    response = spec_worker.process(payload)
    assert response["ok"] is True
    output, stderr, code = response["result"]["output"]
    assert code == 0 and stderr == ""
    assert json.loads(output) == expected
    assert {key: os.environ.get(key) for key in keys} == expected


def test_worker_uses_harbor_bash_semantics_for_admitted_get(configuration, monkeypatch):
    repo = Path(configuration["repo"])
    (repo / "a.txt").write_text("first\n")
    (repo / "b.txt").write_text("second\n")
    actual = spec_worker.sfx_daemon_run.run_drained
    calls = []

    def observe(command, **kwargs):
        calls.append((command, kwargs["executable"]))
        return actual(command, **kwargs)

    monkeypatch.setattr(spec_worker.sfx_daemon_run, "run_drained", observe)
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat a.txt b.txt"}))
    assert response["ok"] is True
    assert response["result"]["output"] == ("first\nsecond\n", "", 0)
    assert calls == [("cat a.txt b.txt", "/bin/bash")]


def test_worker_declines_shell_expansion_before_execution(configuration, monkeypatch):
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "run_drained",
                        lambda *a, **k: pytest.fail("shell expansion must not execute"))
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat {a,b}.txt"}))
    assert response == {"ok": False, "error": {"category": "admission", "type": "ScriptInvocationRejected"}}


def test_worker_proot_inner_and_outer_shell_are_bash_without_environment_override(configuration, monkeypatch):
    configuration["fork_path_view"] = "proot"
    fork = Path(configuration["scratch"]) / "job"
    fork.mkdir()
    command = f"cat {configuration['repo']}/data"
    monkeypatch.setenv("PYTHONPATH", "/caller-original-pythonpath")
    original = dict(os.environ)
    calls = []

    def run(command, **kwargs):
        calls.append((shlex.split(command), kwargs))
        return "contents\n", "", 0

    monkeypatch.setattr(spec_worker.sfx_daemon_run.shutil, "which", lambda _: "/usr/bin/proot")
    monkeypatch.setattr(spec_worker.sfx_daemon_run, "run_drained", run)
    response = spec_worker.process(request(
        configuration, op="run_in_fork", path=str(fork), hop=["read", "free", {"cmd": command}]))
    assert response["ok"] is True
    assert calls[0][0][-3:] == ["/bin/bash", "-c", command]
    assert calls[0][1]["executable"] == "/bin/bash"
    assert calls[0][1]["env"] == original
    assert calls[0][1]["separate_stderr"] is True
    assert dict(os.environ) == original


@pytest.mark.parametrize("operation", ["run", "run_in_fork"])
def test_worker_mixed_stderr_is_not_published_after_completed_execution(configuration, operation):
    repo = Path(configuration["repo"])
    (repo / "existing").write_text("existing-output\n")
    command = "cat existing missing existing"
    if operation == "run":
        payload = request(configuration, op="run", kind="read", args={"cmd": command})
    else:
        fork = Path(configuration["scratch"]) / "job"
        fork.mkdir()
        (fork / "existing").write_text("fork-output\n")
        payload = request(configuration, op="run_in_fork", path=str(fork),
                          hop=["read", "free", {"cmd": command}])
    assert spec_worker.process(payload) == {
        "ok": False, "error": {"category": "non_reusable", "type": "NonReusableToolOutput"}}
    assert (repo / "existing").read_text() == "existing-output\n"


@pytest.mark.parametrize("command,kind,expected", [
    ("grep absent existing", "grep", ("", "", 1)),
    ("python3 fail.py", "run", ("expected failure\n", "", 3)),
])
def test_worker_nonzero_exit_without_stderr_remains_reusable(configuration, command, kind, expected):
    repo = Path(configuration["repo"])
    (repo / "existing").write_text("not a match\n")
    (repo / "fail.py").write_text("print('expected failure')\nraise SystemExit(3)\n")
    configuration["script_contracts"] = [{
        "script": "fail.py", "positionals": 0, "path_options": [],
        "value_options": [], "flags": [], "required": [],
        "source_sha256": {"fail.py": hashlib.sha256((repo / "fail.py").read_bytes()).hexdigest()},
    }]
    if kind == "run":
        from sfx.fork import ForkError, choose_substrate
        try:
            backend = choose_substrate(Path(configuration["scratch"]), repo)
        except ForkError:
            pytest.skip("real clonefile/reflink unavailable")
        if backend not in ("clonefile", "reflink"):
            pytest.skip("real clonefile/reflink unavailable")
    response = spec_worker.process(request(
        configuration, op="run", kind=kind, args={"cmd": command}))
    assert response["ok"] is True
    assert response["result"]["output"] == expected


def test_worker_does_not_classify_failed_drain_as_non_reusable(configuration, monkeypatch):
    import subprocess

    def interrupted(*args, **kwargs):
        raise subprocess.TimeoutExpired("cat existing missing existing", 1, output="partial", stderr="diagnostic")

    monkeypatch.setattr(spec_worker.sfx_daemon_run, "_run", interrupted)
    response = spec_worker.process(request(configuration, op="run", kind="read", args={"cmd": "cat data"}))
    assert response == {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}
