import copy
import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from eval import sfx_daemon_run
from sfx.script_contracts import (
    PinnedScriptRequiresFork, ScriptContracts, ScriptInvocationRejected, ScriptSourceMismatch,
)


DEFINITION = {"script": "code_search.py", "positionals": 1,
              "path_options": ["--rules"], "value_options": ["--encoding"],
              "flags": ["--dry-run", "--help"], "required": ["--rules"]}


@pytest.mark.parametrize("command", [
    "python code_search.py repo --rules rules.json",
    "python3 /app/code_search.py /app/repo --rules=/app/rules.json --dry-run",
    "python3 code_search.py --help",
])
def test_declared_read_only_forms_match_without_rewriting(command):
    policy = ScriptContracts([DEFINITION])
    assert policy.match(command, "/app").script == "code_search.py"
    assert not ScriptContracts().match(command, "/app")


@pytest.mark.parametrize("command", [
    "python code_search.py",
    "python code_search.py repo --rules rules.json --apply-fixes",
    "python code_search.py repo --rules rules.json --output x",
    "python code_search.py repo --rules rules.json --unknown",
    "python code_search.py repo --rules rules.json; touch x",
    "python code_search.py repo --rules $(echo x)",
    "python code_search.py repo --rules=../rules.json",
    "python code_search.py /outside --rules rules.json",
    "python code_search.py repo --rules=/application/rules.json",
    "python code_search.py repo --rules=//app/rules.json",
    "python code_search.py repo --rules=a --rules=b",
    "python code_search.py repo --rules rules.json > output",
    "python -m code_search repo --rules rules.json",
    "python other.py repo --rules rules.json",
    "python code_search.py repo --rules",
    "python code_search.py repo --rules=",
    "python code_search.py repo --rules=x --dry-run=yes",
    "python code_search.py repo --rules=x --help",
])
def test_mutations_unknown_syntax_paths_and_incomplete_argv_rejected(command):
    assert ScriptContracts([DEFINITION]).match(command, "/app") is None


@pytest.mark.parametrize("field,value", [
    ("script", "../run.py"), ("script", "/app/code_search.py"),
    ("positionals", True), ("required", ["--not-declared"]),
    ("flags", ["--rules"]), ("path_options", "--rules"),
])
def test_invalid_contracts_rejected(field, value):
    definition = copy.deepcopy(DEFINITION)
    definition[field] = value
    with pytest.raises(ValueError):
        ScriptContracts([definition])


def test_registered_flags_do_not_globally_change_classifier(tmp_path, monkeypatch):
    from mining.normalize import classify

    command = "python code_search.py repo --rules rules.json"
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    (tmp_path / "code_search.py").write_text("print('reviewed')\n")
    assert classify("Bash", command) == ("unknown", "never")
    with pytest.raises(ValueError):
        sfx_daemon_run._validate_speculative_command("run", {"cmd": command})
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))
    sfx_daemon_run._validate_speculative_command("run", {"cmd": command})
    with pytest.raises(ValueError):
        sfx_daemon_run._validate_speculative_command("run", {"cmd": command + " --apply-fixes"})


def test_missing_path_backend_never_falls_back_to_authoritative_tree(tmp_path, monkeypatch):
    (tmp_path / "code_search.py").write_text("print('reviewed')\n")
    monkeypatch.setenv("SFX_REPO", "/app")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "proot")
    monkeypatch.setattr(sfx_daemon_run.shutil, "which", lambda _: None)
    called = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *a, **kw: called.append(a))
    with pytest.raises(ValueError, match="unavailable"):
        sfx_daemon_run._run_in_fork(tmp_path, ("run", "free", {
            "cmd": "python3 /app/code_search.py /app/repo --rules /app/rules.json"}))
    assert not called


def test_declared_read_only_fork_cannot_publish_filesystem_changes(tmp_path, monkeypatch):
    (tmp_path / "code_search.py").write_text("print('reviewed')\n")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))

    def run(command, **kwargs):
        (tmp_path / "unexpected").write_text("changed")
        return "not safe to serve", 0

    monkeypatch.setattr(sfx_daemon_run, "run_drained", run)
    with pytest.raises(ValueError, match="changed its fork"):
        sfx_daemon_run._run_in_fork(tmp_path, ("run", "free", {
            "cmd": "python code_search.py repo --rules rules.json"}))


def test_contract_forks_keep_exact_command_inside_path_view(tmp_path, monkeypatch):
    (tmp_path / "code_search.py").write_text("print('reviewed')\n")
    monkeypatch.setenv("SFX_REPO", "/app")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "proot")
    monkeypatch.setattr(sfx_daemon_run.shutil, "which", lambda _: "/usr/bin/proot")
    seen = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained",
                        lambda command, **kwargs: (seen.append(command) or ("same", 0)))
    command = "python3 /app/code_search.py /app/repo --rules=/app/rules.json"
    assert sfx_daemon_run._run_in_fork(tmp_path, ("run", "free", {"cmd": command}))[0] == ("same", 0)
    import shlex
    argv = shlex.split(seen[0])
    assert argv[-3:] == ["/bin/sh", "-c", command]
    assert f"{tmp_path}:/app" in argv


def pinned_definition(source="print('reviewed')\n"):
    return {**DEFINITION, "source_sha256": {
        "code_search.py": hashlib.sha256(source.encode()).hexdigest()}}


@pytest.mark.parametrize("pins", [
    {}, {"other.py": "0" * 64}, {"code_search.py": "not-a-digest"},
    {"code_search.py": "A" * 64}, {"code_search.py": None},
    {"code_search.py": "0" * 64, "../external.py": "0" * 64},
    {"code_search.py": "0" * 64, "/app/external.py": "0" * 64},
    {"code_search.py": "0" * 64, ".": "0" * 64},
])
def test_invalid_source_pins_rejected(pins):
    with pytest.raises(ValueError):
        ScriptContracts([{**DEFINITION, "source_sha256": pins}])


@pytest.mark.parametrize("option", ["--apply-fixes", "--fix", "--write", "--output", "--in-place"])
def test_readonly_contract_cannot_override_known_mutation_flags(option):
    with pytest.raises(ValueError, match="mutation options"):
        ScriptContracts([{**DEFINITION, "flags": [option]}])


def test_source_pins_are_copied_and_checked_without_normalizing_the_call(tmp_path):
    source = "print('reviewed')\n"
    (tmp_path / "code_search.py").write_text(source)
    definition = pinned_definition(source)
    policy = ScriptContracts([definition])
    command = "python3 ./code_search.py 'input root' --rules=rules.json"
    invocation = policy.match(command, str(tmp_path))
    assert invocation is not None and policy.verify_sources(invocation, tmp_path)
    definition["source_sha256"]["code_search.py"] = "0" * 64
    assert policy.verify_sources(invocation, tmp_path)
    (tmp_path / "code_search.py").write_text("print('unreviewed')\n")
    assert not policy.verify_sources(invocation, tmp_path)


def test_declared_dependency_change_rejects_source_identity(tmp_path):
    (tmp_path / "code_search.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    definition = pinned_definition("import helper\n")
    definition["source_sha256"]["helper.py"] = hashlib.sha256(b"VALUE = 1\n").hexdigest()
    policy = ScriptContracts([definition])
    invocation = policy.match("python code_search.py input --rules rules.json", str(tmp_path))
    assert policy.verify_sources(invocation, tmp_path)
    (tmp_path / "helper.py").write_text("VALUE = 2\n")
    assert not policy.verify_sources(invocation, tmp_path)


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "fifo", "directory", "parent_symlink"])
def test_pinned_sources_reject_aliases_and_nonregular_files(tmp_path, alias):
    external = tmp_path / "external.py"
    external.write_text("print('reviewed')\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = "code_search.py"
    target = repo / script
    if alias == "symlink":
        target.symlink_to(external)
    elif alias == "hardlink":
        os.link(external, target)
    elif alias == "fifo":
        os.mkfifo(target)
    elif alias == "directory":
        target.mkdir()
    else:
        (repo / "package").symlink_to(tmp_path, target_is_directory=True)
        script = "package/external.py"
    definition = {**DEFINITION, "script": script, "source_sha256": {
        script: hashlib.sha256(external.read_bytes()).hexdigest()}}
    policy = ScriptContracts([definition])
    invocation = policy.match(f"python {script} input --rules rules.json", str(repo))
    assert invocation is not None and not policy.verify_sources(invocation, repo)


def test_declared_script_cannot_bypass_required_argv_as_generic_bare_python(tmp_path, monkeypatch):
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([DEFINITION]))
    with pytest.raises(ScriptInvocationRejected):
        sfx_daemon_run._validate_speculative_command("run", {"cmd": "python code_search.py"})
    from sfx.resolver import Ctx
    assert sfx_daemon_run._resolve_args("run", Ctx(repo=tmp_path, last_edit_path="code_search.py")) is None


@pytest.mark.parametrize("where", ["authoritative", "fork"])
def test_unreviewed_rewrite_cannot_execute_speculative_side_effect(tmp_path, monkeypatch, where):
    repo, fork = tmp_path / "repo", tmp_path / "fork"
    repo.mkdir()
    fork.mkdir()
    sentinel = tmp_path / "outside-must-not-exist"
    reviewed = "print('reviewed')\n"
    unreviewed = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('escaped')\n"
    (repo / "code_search.py").write_text(unreviewed if where == "authoritative" else reviewed)
    (fork / "code_search.py").write_text(unreviewed)
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition(reviewed)]))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    monkeypatch.setattr(sfx_daemon_run, "choose_substrate", lambda *args: "clonefile")

    @contextmanager
    def private_fork(*args, **kwargs):
        yield SimpleNamespace(path=fork, substrate="clonefile")

    monkeypatch.setattr(sfx_daemon_run, "fork", private_fork)
    launches = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *args, **kwargs: launches.append(args))
    command = "python code_search.py input --rules rules.json"
    with pytest.raises(ScriptSourceMismatch):
        if where == "fork":
            sfx_daemon_run._run_in_fork(fork, ("run", "free", {"cmd": command}))
        else:
            sfx_daemon_run._run("run", {"cmd": command})
    assert launches == [] and not sentinel.exists()


def test_fork_verifies_new_reviewed_revision_not_old_authoritative_source(tmp_path, monkeypatch):
    repo, fork = tmp_path / "repo", tmp_path / "fork"
    repo.mkdir()
    fork.mkdir()
    (repo / "code_search.py").write_text("print('old')\n")
    reviewed = "print('reviewed new')\n"
    (fork / "code_search.py").write_text(reviewed)
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition(reviewed)]))
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *args, **kwargs: ("reviewed new\n", 0))
    command = "python code_search.py input --rules rules.json"
    assert sfx_daemon_run._run_in_fork(fork, ("run", "free", {"cmd": command}))[0] == ("reviewed new\n", 0)


def test_pinned_command_declines_without_private_scratch(tmp_path, monkeypatch):
    source = "print('reviewed')\n"
    (tmp_path / "code_search.py").write_text(source)
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition(source)]))
    monkeypatch.delenv("SFX_SCRATCH", raising=False)
    launches = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *args, **kwargs: launches.append(args))
    command = "python code_search.py input --rules rules.json"
    with pytest.raises(PinnedScriptRequiresFork):
        sfx_daemon_run._run("run", {"cmd": command})
    assert launches == []


@pytest.mark.parametrize("backend", ["overlayfs", "unavailable"])
def test_pinned_get_refuses_non_snapshot_backends_before_execution(tmp_path, monkeypatch, backend):
    from sfx.fork import ForkError

    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "code_search.py").write_text("print('reviewed')\n")
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))

    def choose(*args):
        if backend == "unavailable":
            raise ForkError("no COW backend")
        return backend

    monkeypatch.setattr(sfx_daemon_run, "choose_substrate", choose)
    launches = []
    monkeypatch.setattr(sfx_daemon_run, "run_drained", lambda *args, **kwargs: launches.append(args))
    with pytest.raises(PinnedScriptRequiresFork if backend == "overlayfs" else ForkError):
        sfx_daemon_run._run("run", {"cmd": "python code_search.py input --rules rules.json"})
    assert launches == []


def test_pinned_stateful_chain_refuses_overlay_before_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition()]))
    daemon = sfx_daemon_run.build_daemon(1)
    calls = []
    monkeypatch.setattr(daemon.fs_substrate, "_run_in_fork", lambda *args: calls.append(args))
    handle = SimpleNamespace(path=tmp_path, substrate="overlayfs")
    with pytest.raises(PinnedScriptRequiresFork):
        daemon.fs_substrate.run_get(handle, ("run", "free", {
            "cmd": "python code_search.py input --rules rules.json"}))
    assert calls == []
    daemon.shutdown()


def test_real_pinned_get_uses_snapshot_even_if_authoritative_source_changes(tmp_path, monkeypatch):
    from sfx.fork import ForkError

    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    source = "print('reviewed')\n"
    (repo / "code_search.py").write_text(source)
    try:
        backend = sfx_daemon_run.choose_substrate(scratch, repo)
    except ForkError:
        pytest.skip("real clonefile/reflink backend unavailable")
    if backend not in ("clonefile", "reflink"):
        pytest.skip("real clonefile/reflink backend unavailable")
    sentinel = tmp_path / "outside-must-not-exist"
    unreviewed = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('escaped')\n"
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([pinned_definition(source)]))
    actual_fork = sfx_daemon_run.fork
    snapshots = []

    @contextmanager
    def fork_then_change_authoritative(*args, **kwargs):
        with actual_fork(*args, **kwargs) as handle:
            snapshots.append(handle.path)
            (repo / "code_search.py").write_text(unreviewed)
            yield handle

    monkeypatch.setattr(sfx_daemon_run, "fork", fork_then_change_authoritative)
    result, elapsed = sfx_daemon_run._run("run", {
        "cmd": "python code_search.py input --rules rules.json"})
    assert result == ("reviewed\n", "", 0)
    assert elapsed > 0 and snapshots and all(not path.exists() for path in snapshots)
    assert not sentinel.exists()
    assert (repo / "code_search.py").read_text() == unreviewed
