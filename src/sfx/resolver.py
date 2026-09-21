import ast
import json
import os
import re
import shlex
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_NODE_RE = re.compile(r"^FAILED\s+(\S+::\S+)", re.MULTILINE)
_TB_FILE_RE = re.compile(r'File "([^"]+\.py)", line \d+')
_PATH_LINE_RE = re.compile(r"^\s*(\S+\.\w+)\s*$", re.MULTILINE)

_SAFE_PATH_RE = re.compile(r"^[\w./-]+$")
_SAFE_NODE_RE = re.compile(r"^[\w./-]+(::[\w./-]+)+$")


def _safe_path(p):
    return p if _SAFE_PATH_RE.match(p) else None


def _safe_node(n):
    return n if _SAFE_NODE_RE.match(n) else None


@dataclass
class Ctx:
    repo: Path
    last_observation: str = ""
    session: dict = field(default_factory=dict)
    last_edit_path: str = ""
    last_edit_contents: str | None = None


def canonical_command(kind, ctx):
    cmd = _from_config(kind, ctx.repo)
    if cmd:
        return cmd
    return ctx.session.get(kind)


def _from_edit(kind, ctx):
    if kind == "run" and ctx.last_edit_path.endswith(".py"):
        if _requires_python_arguments(ctx):
            return None
        return f"python {shlex.quote(ctx.last_edit_path)}"
    return None


def _python_call(command):
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(words[0]).name):
        return None
    for index, word in enumerate(words[1:], 1):
        if word in ("-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S"):
            continue
        return (word, words[index + 1:]) if not word.startswith("-") else None
    return None


def _repo_relative(value, repo):
    path = Path(value)
    if ".." in path.parts:
        return None
    if path.is_absolute():
        try:
            path = path.relative_to(repo)
        except ValueError:
            return None
    return path


def _uses_edit(command, ctx):
    call = _python_call(command)
    if not call:
        return False
    script, arguments = call
    edited = Path(ctx.last_edit_path)
    if _repo_relative(script, ctx.repo) == edited:
        return True
    for argument in arguments:
        if argument.startswith("--") and "=" in argument:
            argument = argument.split("=", 1)[1]
        if not argument or argument.startswith("-"):
            continue
        path = _repo_relative(argument, ctx.repo)
        if path is not None and (path == edited or
                ((ctx.repo / path).is_dir() and edited.is_relative_to(path))):
            return True
    return False


def _read_edited_source(ctx):
    path = Path(ctx.last_edit_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    parent = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = os.open(ctx.repo, flags)
        for part in path.parts[:-1]:
            child = os.open(part, flags, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if stat.S_ISREG(info.st_mode) and info.st_size <= 1_048_576:
                return source.read(1_048_577).decode("utf-8")
    except (OSError, UnicodeError):
        return None
    finally:
        if parent is not None:
            os.close(parent)
    return None


def _requires_python_arguments(ctx):
    source = ctx.last_edit_contents
    if source is None:
        source = _read_edited_source(ctx)
    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return False
    for node in ast.walk(tree):
        if _rejects_empty_argv(node):
            return True
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("add_argument", "add_subparsers"):
            continue
        options = {keyword.arg: keyword.value for keyword in node.keywords}
        required = options.get("required")
        if isinstance(required, ast.Constant) and required.value is True:
            return True
        if node.func.attr == "add_argument" and node.args:
            name = node.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str) and not name.value.startswith("-"):
                nargs = options.get("nargs")
                if not (isinstance(nargs, ast.Constant) and nargs.value in ("?", "*", 0)):
                    return True
    return False


def _rejects_empty_argv(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    call = test.left
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "len" or len(call.args) != 1 or call.keywords):
        return False
    argument = call.args[0]
    if (not isinstance(argument, ast.Attribute) or argument.attr != "argv"
            or not isinstance(argument.value, ast.Name) or argument.value.id != "sys"):
        return False
    bound = test.comparators[0]
    if not isinstance(bound, ast.Constant) or type(bound.value) is not int:
        return False
    comparisons = {
        ast.Eq: 1 == bound.value, ast.NotEq: 1 != bound.value,
        ast.Lt: 1 < bound.value, ast.LtE: 1 <= bound.value,
        ast.Gt: 1 > bound.value, ast.GtE: 1 >= bound.value,
    }
    if not comparisons.get(type(test.ops[0]), False):
        return False
    for statement in node.body:
        if isinstance(statement, ast.Raise):
            return True
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if (isinstance(call.func, ast.Attribute) and call.func.attr == "exit"
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "sys"
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and call.args[0].value not in (None, 0)):
            return True
    return False


def _from_observation(kind, ctx):
    obs = ctx.last_observation
    if not obs:
        return None
    if kind == "test":
        m = _NODE_RE.search(obs)
        if m and _safe_node(m.group(1)):
            return f"pytest {m.group(1)}"
    if kind == "read":
        m = _TB_FILE_RE.search(obs)
        if m and _safe_path(m.group(1)):
            return f"cat {m.group(1)}"
        m = _PATH_LINE_RE.search(obs)
        if m and _safe_path(m.group(1)):
            return f"cat {m.group(1)}"
    return None


def resolve_details(kind, ctx):
    command = _from_config(kind, ctx.repo)
    if command:
        return {"cmd": command}, "config", "resolved"
    previous = ctx.session.get(kind)
    if kind == "run" and ctx.last_edit_path.endswith(".py"):
        if previous and _uses_edit(previous, ctx):
            _script, arguments = _python_call(previous)
            if arguments or not _requires_python_arguments(ctx):
                return {"cmd": previous}, "session", "resolved"
        command = _from_edit(kind, ctx)
        if command:
            return {"cmd": command}, "edit", "resolved"
        return None, "none", "required_python_arguments"
    command = _from_observation(kind, ctx)
    if command:
        return {"cmd": command}, "observation", "resolved"
    if previous:
        return {"cmd": previous}, "session", "resolved"
    return None, "none", "unresolved_args"


def resolve(kind, ctx):
    return resolve_details(kind, ctx)[0]


def canonical_args(kind, command, ctx):
    """Key a real request by its exact command, never a guessed config command.

    Config and session history ground predictions in resolve(); substituting them
    for the actual request would alias different test selections or programs.
    """
    return {"cmd": command}


def resolve_tier(kind, ctx):
    return resolve_details(kind, ctx)[1]


def _from_config(kind, repo):
    pkg = repo / "package.json"
    if pkg.exists():
        scripts = json.loads(pkg.read_text()).get("scripts", {})
        if kind in scripts:
            return scripts[kind]
    cargo = repo / "Cargo.toml"
    if cargo.exists() and kind in ("test", "build"):
        return "cargo test" if kind == "test" else "cargo build"
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        cfg = tomllib.loads(pyproject.read_text())
        if kind == "test" and "pytest" in cfg.get("tool", {}):
            return "pytest"
    return None
