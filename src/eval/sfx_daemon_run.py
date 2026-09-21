import json
import shlex
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

import os

from adapters.protocol import serve
from eval.drain import run_drained
from eval.table import load_table
from sfx import resolver
from sfx.daemon import Daemon
from sfx.fork import choose_substrate, fork
from sfx.script_contracts import (
    PinnedScriptRequiresFork, ScriptContracts, ScriptInvocationRejected, ScriptSourceMismatch,
)


# Native test/build/lint/typecheck commands run project code and commonly write
# artifacts. Their semantic kind is not a read-only execution contract. The
# supported reviewed-code path is a source-pinned Python wrapper, never an
# automatic inference that a familiar runner is pure.
CODE_EXECUTING_KINDS = frozenset({"run", "test", "lint", "typecheck", "build"})
LITERAL_READ_COMMANDS = frozenset({"cat", "head", "tail", "nl", "grep", "rg", "ls", "stat", "pwd"})
RG_FLAGS = frozenset({
    "--line-number", "--no-line-number", "--with-filename", "--no-filename",
    "--ignore-case", "--case-sensitive", "--smart-case", "--word-regexp", "--line-regexp",
    "--fixed-strings", "--files-with-matches", "--files-without-match", "--count",
    "--count-matches", "--files", "--hidden", "--no-ignore", "--no-ignore-vcs",
    "--text", "--invert-match", "--quiet", "--no-messages", "--null", "--null-data",
    "--no-config", "--heading", "--no-heading", "--multiline", "--pcre2",
})
RG_VALUE_OPTIONS = frozenset({
    "--regexp", "--file", "--glob", "--iglob", "--type", "--type-not", "--max-count",
    "--max-depth", "--after-context", "--before-context", "--context", "--color",
    "--replace", "-e", "-f", "-g", "-t", "-T", "-m", "-A", "-B", "-C", "-r",
})
GREP_FLAGS = frozenset({
    "--extended-regexp", "--fixed-strings", "--basic-regexp", "--perl-regexp",
    "--ignore-case", "--no-ignore-case", "--word-regexp", "--line-regexp",
    "--null-data", "--no-messages", "--invert-match", "--text", "--line-number",
    "--byte-offset", "--with-filename", "--no-filename", "--only-matching",
    "--quiet", "--silent", "--files-without-match", "--files-with-matches",
    "--count", "--null", "--line-buffered",
})
GREP_VALUE_OPTIONS = frozenset({
    "--regexp", "--file", "--max-count", "--after-context", "--before-context", "--context",
    "-e", "-f", "-m", "-A", "-B", "-C",
})


def _validate_tool_environment(tool_env=None):
    environment = os.environ if tool_env is None else tool_env
    if (environment.get("BASH_ENV") or environment.get("ENV")
            or any(name.startswith("BASH_FUNC_") for name in environment)):
        raise ScriptInvocationRejected("speculative shell environment contains unsupported startup or function hooks")


def _validate_literal_read(command, *, tool_env=None):
    if (not isinstance(command, str)
            or any(char in command for char in "$`~\n\r\x00;&|<>(){}*?[]\\")):
        raise ScriptInvocationRejected("speculative reads require one literal supported invocation")
    try:
        words = shlex.split(command)
    except ValueError:
        raise ScriptInvocationRejected("speculative read is not a literal invocation") from None
    if not words or words[0] not in LITERAL_READ_COMMANDS:
        raise ScriptInvocationRejected("read executable has no supported speculative contract")
    if words[0] not in {"rg", "grep"}:
        return
    # ripgrep can invoke external preprocessors/decompressors. Only explicit
    # read-only flags are accepted; abbreviations cannot opt into new behavior.
    name = words[0]
    flags = RG_FLAGS if name == "rg" else GREP_FLAGS
    value_options = RG_VALUE_OPTIONS if name == "rg" else GREP_VALUE_OPTIONS
    short_flags = "nNHIiSsSwxFvlcq0aUuP" if name == "rg" else "EFGPinHhwxvqsclLboZzaIUuy"
    no_config = False
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            break
        if word.startswith("-") and word != "-":
            option, separator, _value = word.partition("=")
            if option in value_options:
                if not separator:
                    index += 1
                    if index == len(words):
                        raise ScriptInvocationRejected("search option is missing its value")
            elif word in flags:
                no_config |= word == "--no-config"
            elif not word.startswith("--") and all(char in short_flags for char in word[1:]):
                pass
            else:
                raise ScriptInvocationRejected("unsupported speculative search option")
        index += 1
    environment = os.environ if tool_env is None else tool_env
    if name == "rg" and environment.get("RIPGREP_CONFIG_PATH") and not no_config:
        raise ScriptInvocationRejected("speculative ripgrep cannot load an unreviewed environment config")


def _clock():
    return time.monotonic() * 1000.0


def _validate_speculative_command(kind, args, *, source_root=None, verify_sources=True, tool_env=None):
    from mining.normalize import classify
    _validate_tool_environment(tool_env)
    command = args.get("cmd")
    repo = os.environ.get("SFX_REPO", "/app")
    policy = _contracts()
    contract = policy.match(command, repo)
    if policy.declares(command, repo) and contract is None:
        raise ScriptInvocationRejected("registered script invocation violates its read-only contract")
    if kind in CODE_EXECUTING_KINDS or contract is not None:
        if contract is None or not contract.source_sha256:
            raise ScriptInvocationRejected("code speculation requires an explicit source-pinned read-only contract")
    elif kind in {"read", "grep", "search"}:
        _validate_literal_read(command, tool_env=tool_env)
    if (not isinstance(command, str)
            or classify("Bash", command) != (kind, "free") and not (kind == "run" and contract)):
        raise ValueError("resolved command is not eligible for speculative execution")
    if verify_sources and contract is not None and not policy.verify_sources(contract, source_root or repo):
        raise ScriptSourceMismatch("reviewed script source identity does not match")
    return contract


def _contracts():
    return ScriptContracts(os.environ.get("SFX_SCRIPT_CONTRACTS", "[]"))


def _run(kind, args, *, tool_env=None, tool_shell=None):
    contract = _validate_speculative_command(kind, args, verify_sources=False, tool_env=tool_env)
    t0 = time.monotonic()
    if contract is not None:
        repo = Path(os.environ.get("SFX_REPO", "/app")).resolve(strict=True)
        scratch_value = os.environ.get("SFX_SCRATCH")
        scratch = Path(scratch_value) if scratch_value else None
        if (scratch is None or not scratch.is_absolute() or scratch.is_symlink()
                or not scratch.is_dir()):
            raise PinnedScriptRequiresFork("pinned script requires an existing private scratch directory")
        scratch = scratch.resolve(strict=True)
        if scratch == repo or scratch.is_relative_to(repo):
            raise PinnedScriptRequiresFork("pinned script scratch must be outside the repository")
        substrate = choose_substrate(scratch, repo)
        if substrate not in ("clonefile", "reflink"):
            raise PinnedScriptRequiresFork("pinned script requires clonefile or reflink, not live overlay lowerdirs")
        with fork(repo, scratch, substrate=substrate) as handle:
            hop = (kind, "free", args)
            _validate_fork_handle(handle, hop)
            result, _status = _run_in_fork(handle.path, hop, tool_env=tool_env, tool_shell=tool_shell)
    else:
        repo = os.environ.get("SFX_REPO", "/app")
        try:
            _validate_fork_command(args["cmd"], repo, repo=repo)
        except ValueError as exc:
            raise ScriptInvocationRejected(str(exc)) from exc
        options = {} if tool_env is None else {"env": tool_env}
        if tool_shell is not None:
            options["executable"] = tool_shell
        result = run_drained(args["cmd"], cwd=repo,
                             separate_stderr=os.environ.get("SFX_SEPARATE_STDERR") == "1", **options)
    return (result, (time.monotonic() - t0) * 1000.0)


def _validate_fork_handle(handle, hop):
    command = hop[2].get("cmd")
    contract = _contracts().match(command, os.environ.get("SFX_REPO", "/app"))
    if contract is not None and contract.source_sha256 and handle.substrate not in ("clonefile", "reflink"):
        raise PinnedScriptRequiresFork("pinned script requires clonefile or reflink, not live overlay lowerdirs")


def _validate_fork_command(command, fork_path, *, repo=None):
    if any(char in command for char in "$`~\n\r;&|<>(){}*?[]\\"):
        raise ValueError("fork command contains unsupported shell syntax")
    try:
        words = shlex.split(command)
    except ValueError as exc:
        raise ValueError("fork command is not a literal invocation") from exc
    if not words or words[0] == "pwd":
        raise ValueError("fork command exposes its working directory")
    if words[0] in {"stat", "ls", "find"}:
        raise ValueError("fork command can depend on filesystem metadata not preserved by a copy")
    pytest_args = (1 if words[0] == "pytest" else 3
                   if words[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"])
                   else None)
    for index, word in enumerate(words):
        option = word.split("=", 1)[0]
        if (option in {"-follow", "--follow", "--dereference", "--dereference-command-line",
                       "--dereference-recursive"}
                or word.startswith("-") and not word.startswith("--")
                and any(char in word for char in "LR")):
            raise ValueError("fork command may follow symlinks")
        values = (word, word.split("=", 1)[-1])
        if word.startswith("-") and ("/" in word or ".." in word):
            raise ValueError("fork command contains an unsupported path option")
        for value in values:
            if pytest_args is not None and index >= pytest_args and "::" in value:
                value, *selectors = value.split("::")
                if any(not selector or "/" in selector or selector in (".", "..")
                       for selector in selectors):
                    raise ValueError("fork command contains an unsupported pytest selector")
            relative = PurePosixPath(value)
            if ".." in relative.parts:
                raise ValueError("fork command path leaves the working tree")
            if relative.is_absolute():
                if repo is None:
                    raise ValueError("fork command path leaves the working tree")
                try:
                    relative = relative.relative_to(repo)
                except ValueError:
                    raise ValueError("fork command path leaves the working tree") from None
            current = Path(fork_path)
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("fork command path contains a symlink")


def _run_in_fork(fp, hop, *, tool_env=None, tool_shell=None):
    from mining.normalize import _status
    kind, _verb, args = hop
    contract = _validate_speculative_command(kind, args, source_root=fp, tool_env=tool_env)
    command = args["cmd"]
    repo = os.environ.get("SFX_REPO", "/app")
    before = None
    if contract is not None:
        from eval.capture import fs_hash
        for path in Path(fp).rglob("*"):
            if path.is_symlink() or path.is_file() and path.stat().st_nlink > 1:
                raise ValueError("contract fork contains aliases")
        before = fs_hash(fp)
    if os.environ.get("SFX_FORK_PATH_VIEW", "cwd") == "proot":
        if contract is None:
            _validate_fork_command(command, fp, repo=repo)
        binary = shutil.which("proot")
        if binary is None:
            raise ValueError("declared proot path view is unavailable")
        if any(c in str(fp) + repo for c in ":!\n\r") or not Path(repo).is_absolute():
            raise ValueError("invalid fork bind path")
        command = shlex.join([binary, "-r", "/", "-b", f"{fp}:{repo}",
                              "-w", repo, tool_shell or "/bin/sh", "-c", command])
    else:
        _validate_fork_command(command, fp)
    # drain background children before the fork's fs is hashed/served as authoritative
    options = {} if tool_env is None else {"env": tool_env}
    if tool_shell is not None:
        options["executable"] = tool_shell
    result = run_drained(command, cwd=fp,
                         separate_stderr=os.environ.get("SFX_SEPARATE_STDERR") == "1", **options)
    if before is not None and fs_hash(fp) != before:
        raise ValueError("declared read-only command changed its fork")
    return result, _status(result[-1] != 0, kind)


def _resolve_args(kind, ctx):
    args = resolver.resolve(kind, ctx)
    if args is not None:
        command = args.get("cmd")
        policy = _contracts()
        if kind in CODE_EXECUTING_KINDS or policy.declares(command, str(ctx.repo)):
            contract = policy.match(command, str(ctx.repo))
            if contract is None or not contract.source_sha256:
                return None
        elif kind in {"read", "grep", "search"}:
            try:
                _validate_literal_read(command)
                _validate_fork_command(command, ctx.repo, repo=str(ctx.repo))
            except ValueError:
                return None
    return args


def _apply_write(fork_path, write):
    from sfx.fork import write_text_in_fork
    write_text_in_fork(fork_path, write.args["path"], write.args["contents"])


def _file_log(path):
    def log(rec):
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return log


def build_daemon(depth, table=None, log=None):
    table = table if table is not None else load_table()
    daemon = Daemon(clock=_clock, global_table=table, k=depth, run=_run,
                    resolve_args=_resolve_args, apply_write=_apply_write,
                    run_in_fork=_run_in_fork,
                    depth_cap=depth, log=log)
    daemon.fs_substrate.validate_handle = _validate_fork_handle
    return daemon


def main(argv):
    sock_path = argv[0]
    depth = int(argv[1]) if len(argv) > 1 else 1
    trace_path = os.environ.get("SFX_TRACE")
    log = _file_log(trace_path) if trace_path else None
    daemon = build_daemon(depth, log=log)
    server = serve(daemon, sock_path)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
