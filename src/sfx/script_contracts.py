"""Explicit trusted read-only CLI contracts; never infer purity from an interpreter.

Registration is a caller assertion about a reviewed program, not a sandbox or a
proof about newly generated code. The schema permits optional source_sha256
pins, but production speculative execution requires them. Pins detect changes to
declared reviewed files; they do not prove dependency completeness or contain
external effects. A changed source revision requires review and a new pin.
Undeclared flags and shell syntax fail closed.
"""
import hashlib
import json
import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import PurePosixPath


class ScriptInvocationRejected(ValueError):
    pass


class ScriptSourceMismatch(ValueError):
    pass


class PinnedScriptRequiresFork(ValueError):
    pass


def _relative(path, repo):
    value, root = PurePosixPath(path), PurePosixPath(repo)
    if not path or "\x00" in path or ".." in value.parts or path.startswith("//"):
        return None
    if value.is_absolute():
        try:
            value = value.relative_to(root)
        except ValueError:
            return None
    return value.as_posix()


@dataclass(frozen=True)
class Invocation:
    script: str
    paths: tuple[str, ...]
    source_sha256: tuple[tuple[str, str], ...] = ()


class ScriptContracts:
    def __init__(self, definitions=None):
        if isinstance(definitions, str):
            definitions = json.loads(definitions)
        definitions = [] if definitions is None else definitions
        if not isinstance(definitions, list):
            raise ValueError("script_contracts must be a list")
        self.definitions = []
        seen = set()
        for value in definitions:
            fields = {"script", "positionals", "path_options", "value_options", "flags", "required"}
            if (not isinstance(value, dict) or not fields <= set(value)
                    or set(value) - fields - {"source_sha256"}):
                raise ValueError("invalid read-only script contract fields")
            script = value["script"]
            if (not isinstance(script, str) or _relative(script, "/app") != script
                    or PurePosixPath(script).is_absolute() or not script.endswith(".py")
                    or not re.fullmatch(r"[\w./-]+", script) or script in seen):
                raise ValueError("contract script must be a unique relative Python path")
            if type(value["positionals"]) is not int or not 0 <= value["positionals"] <= 16:
                raise ValueError("invalid positional path count")
            options = []
            for field in ("path_options", "value_options", "flags", "required"):
                items = value[field]
                if (not isinstance(items, list)
                        or any(not isinstance(x, str) or not re.fullmatch(r"--[a-z][a-z0-9-]*", x)
                               for x in items)
                        or len(items) != len(set(items))):
                    raise ValueError("contract options must be distinct literal long flags")
                if field != "required":
                    options.extend(items)
            if len(options) != len(set(options)) or not set(value["required"]) <= set(options):
                raise ValueError("ambiguous or missing contract options")
            if set(options) & {"--apply-fixes", "--fix", "--fix-only", "--unsafe-fixes", "--write",
                               "--output", "--output-file", "--output-dir", "--out", "--in-place"}:
                raise ValueError("read-only contracts cannot declare mutation options")
            if "source_sha256" in value:
                pins = value["source_sha256"]
                if not isinstance(pins, dict) or script not in pins:
                    raise ValueError("source_sha256 must include the reviewed script")
                for path, digest in pins.items():
                    if (not isinstance(path, str) or _relative(path, "/app") != path
                            or PurePosixPath(path).is_absolute() or not re.fullmatch(r"[\w./-]+", path)
                            or path == "." or not isinstance(digest, str)
                            or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                        raise ValueError("invalid reviewed source identity")
            seen.add(script)
            self.definitions.append({k: list(v) if isinstance(v, list) else dict(v)
                                     if isinstance(v, dict) else v for k, v in value.items()})

    def declares(self, command, repo):
        if not isinstance(command, str):
            return False
        try:
            words = shlex.split(command)
        except ValueError:
            return False
        return (len(words) >= 2 and words[0] in ("python", "python3")
                and any(d["script"] == _relative(words[1], repo) for d in self.definitions))

    def match(self, command, repo):
        if (not isinstance(command, str) or any(c in command for c in "$`\n\r;&|<>(){}*?[]\\~")):
            return None
        try:
            words = shlex.split(command)
        except ValueError:
            return None
        if len(words) < 2 or words[0] not in ("python", "python3"):
            return None
        script = _relative(words[1], repo)
        definition = next((d for d in self.definitions if d["script"] == script), None)
        if definition is None:
            return None
        paths, positionals, supplied = [script], 0, set()
        args, index = words[2:], 0
        while index < len(args):
            word = args[index]
            if word.startswith("-"):
                option, separator, value = word.partition("=")
                if option in supplied:
                    return None
                supplied.add(option)
                if option in definition["flags"]:
                    if separator:
                        return None
                elif option in definition["path_options"] + definition["value_options"]:
                    if not separator:
                        index += 1
                        if index == len(args):
                            return None
                        value = args[index]
                    if not value or value.startswith("-"):
                        return None
                    if option in definition["path_options"]:
                        relative = _relative(value, repo)
                        if relative is None:
                            return None
                        paths.append(relative)
                else:
                    return None
            else:
                relative = _relative(word, repo)
                if relative is None:
                    return None
                paths.append(relative)
                positionals += 1
            index += 1
        help_only = supplied == {"--help"} and positionals == 0
        if "--help" in supplied and not help_only:
            return None
        if not help_only and (positionals != definition["positionals"]
                              or not set(definition["required"]) <= supplied):
            return None
        return Invocation(script, tuple(paths), tuple(sorted(definition.get("source_sha256", {}).items())))

    def verify_sources(self, invocation, source_root):
        if not invocation.source_sha256:
            return True
        for path, expected in invocation.source_sha256:
            parent = None
            try:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                parent = os.open(source_root, flags)
                parts = PurePosixPath(path).parts
                for part in parts[:-1]:
                    child = os.open(part, flags, dir_fd=parent)
                    os.close(parent)
                    parent = child
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(fd, "rb") as source:
                    before = os.fstat(source.fileno())
                    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                            or before.st_size > 16 * 1024 * 1024):
                        return False
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                    after = os.fstat(source.fileno())
                    changed = any(getattr(before, field) != getattr(after, field) for field in
                                  ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                    if digest != expected or changed:
                        return False
            except (OSError, ValueError):
                return False
            finally:
                if parent is not None:
                    os.close(parent)
        return True
