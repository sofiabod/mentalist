"""One speculative job inside the original task container; no controller state."""
import base64
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys

from eval import sfx_daemon_run
from sfx.script_contracts import (
    PinnedScriptRequiresFork, ScriptContracts, ScriptInvocationRejected, ScriptSourceMismatch,
)


MAX_MESSAGE_BYTES = 8 * 1024 * 1024
KINDS = {"read", "grep", "search", "test", "lint", "typecheck", "build", "run"}
ADMISSION_TYPES = {
    cls.__name__: cls for cls in
    (PinnedScriptRequiresFork, ScriptInvocationRejected, ScriptSourceMismatch)
}
PREEXEC_DECLINES = {
    "resolved command is not eligible for speculative execution",
    "fork command contains unsupported shell syntax",
    "fork command is not a literal invocation",
    "fork command exposes its working directory",
    "fork command can depend on filesystem metadata not preserved by a copy",
    "fork command may follow symlinks",
    "fork command contains an unsupported path option",
    "fork command contains an unsupported pytest selector",
    "fork command path leaves the working tree",
    "fork command path contains a symlink",
    "contract fork contains aliases",
    "declared proot path view is unavailable",
    "invalid fork bind path",
}


class ProtocolViolation(ValueError):
    pass


class NonReusableToolOutput(ValueError):
    pass


def absolute_path(value):
    if (not isinstance(value, str) or not value or "\x00" in value
            or not PurePosixPath(value).is_absolute() or value.startswith("//")
            or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise ProtocolViolation("invalid absolute path")
    return value


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ProtocolViolation("invalid message fields")


def _invocation(kind, args):
    _fields(args, {"cmd"})
    if (not isinstance(kind, str) or kind not in KINDS
            or not isinstance(args["cmd"], str) or not args["cmd"] or "\x00" in args["cmd"]
            or len(args["cmd"].encode("utf-8")) > MAX_MESSAGE_BYTES // 4):
        raise ProtocolViolation("invalid invocation")


def validate_request(request, scratch):
    if not isinstance(request, dict):
        raise ProtocolViolation("invalid request")
    if request.get("op") == "run":
        _fields(request, {"op", "kind", "args"})
        _invocation(request["kind"], request["args"])
    elif request.get("op") == "run_in_fork":
        _fields(request, {"op", "path", "hop"})
        path = PurePosixPath(absolute_path(request["path"]))
        root = PurePosixPath(absolute_path(scratch))
        if path == root or not path.is_relative_to(root):
            raise ProtocolViolation("fork path is outside owned scratch")
        hop = request["hop"]
        if not isinstance(hop, (list, tuple)) or len(hop) != 3 or hop[1] != "free":
            raise ProtocolViolation("invalid hop")
        _invocation(hop[0], hop[2])
    else:
        raise ProtocolViolation("unknown operation")
    return request


def validate_settings(settings):
    _fields(settings, {"repo", "scratch", "script_contracts", "fork_path_view"})
    repo = PurePosixPath(absolute_path(settings["repo"]))
    scratch = PurePosixPath(absolute_path(settings["scratch"]))
    if (repo == PurePosixPath("/") or scratch == PurePosixPath("/")
            or scratch.is_relative_to(repo) or repo.is_relative_to(scratch)):
        raise ProtocolViolation("scratch must be outside the repository")
    if settings["fork_path_view"] not in ("cwd", "proot"):
        raise ProtocolViolation("invalid path view")
    ScriptContracts(settings["script_contracts"])
    return settings


def strict_loads(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolViolation("duplicate JSON field")
            result[key] = value
        return result

    def constant(_):
        raise ProtocolViolation("nonfinite JSON number")

    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolViolation("message too large")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise ProtocolViolation("invalid JSON") from exc


def validate_response(response, operation):
    if not isinstance(response, dict) or type(response.get("ok")) is not bool:
        raise ProtocolViolation("invalid response")
    if not response["ok"]:
        _fields(response, {"ok", "error"})
        error = response["error"]
        _fields(error, {"category", "type"})
        allowed = set(ADMISSION_TYPES) | {"SpeculationDeclined"}
        if (error["category"] == "admission" and error["type"] in allowed
                or error["category"] == "non_reusable" and error["type"] == "NonReusableToolOutput"
                or error["category"] == "worker" and error["type"] == "WorkerFailure"
                or error["category"] == "protocol" and error["type"] == "ProtocolViolation"):
            return response
        raise ProtocolViolation("invalid error response")
    _fields(response, {"ok", "result"})
    result = response["result"]
    _fields(result, {"output", "duration_ms"} if operation == "run" else {"output", "status"})
    output = result["output"]
    if (not isinstance(output, (tuple, list)) or len(output) != 3
            or not isinstance(output[0], str) or not isinstance(output[1], str)
            or output[1] != "" or type(output[2]) is not int):
        raise ProtocolViolation("invalid tool result")
    if operation == "run":
        duration = result["duration_ms"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ProtocolViolation("invalid duration")
    elif result["status"] not in {"OK", "ERR", "PASS", "FAIL"}:
        raise ProtocolViolation("invalid status")
    return response


def _private_fork_path(value, scratch):
    root = Path(scratch)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolViolation("invalid scratch directory")
    root = root.resolve(strict=True)
    path = Path(value)
    current = Path(scratch)
    for part in path.relative_to(scratch).parts:
        current = current / part
        if current.is_symlink():
            raise ProtocolViolation("fork path contains an alias")
    resolved = path.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ProtocolViolation("fork path is not an owned directory")
    return path


@contextmanager
def _configured(settings):
    values = {
        "SFX_REPO": settings["repo"], "SFX_SCRATCH": settings["scratch"],
        "SFX_SCRIPT_CONTRACTS": json.dumps(settings["script_contracts"]),
        "SFX_FORK_PATH_VIEW": settings["fork_path_view"], "SFX_SEPARATE_STDERR": "1",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def process(payload):
    tool_env = dict(os.environ)
    try:
        _fields(payload, {"request", "settings"})
        settings = validate_settings(payload["settings"])
        request = validate_request(payload["request"], settings["scratch"])
        with _configured(settings):
            if request["op"] == "run":
                output, duration = sfx_daemon_run._run(
                    request["kind"], request["args"], tool_env=tool_env, tool_shell="/bin/bash")
                result = {"output": _harbor_output(output), "duration_ms": duration}
            else:
                path = _private_fork_path(request["path"], settings["scratch"])
                output, status = sfx_daemon_run._run_in_fork(
                    path, tuple(request["hop"]), tool_env=tool_env, tool_shell="/bin/bash")
                result = {"output": _harbor_output(output), "status": status}
        return validate_response({"ok": True, "result": result}, request["op"])
    except ProtocolViolation:
        return {"ok": False, "error": {"category": "protocol", "type": "ProtocolViolation"}}
    except NonReusableToolOutput:
        return {"ok": False, "error": {"category": "non_reusable", "type": "NonReusableToolOutput"}}
    except tuple(ADMISSION_TYPES.values()) as exc:
        return {"ok": False, "error": {"category": "admission", "type": type(exc).__name__}}
    except ValueError as exc:
        if str(exc) in PREEXEC_DECLINES:
            return {"ok": False, "error": {"category": "admission", "type": "SpeculationDeclined"}}
        return {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}
    except Exception:
        return {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}


def _harbor_output(output):
    if (not isinstance(output, (tuple, list)) or len(output) != 3
            or not isinstance(output[0], str) or not isinstance(output[1], str)
            or type(output[2]) is not int):
        raise ProtocolViolation("invalid separate tool result")
    if output[1]:
        raise NonReusableToolOutput("mixed output stream ordering is not reproducible through Harbor Docker")
    return tuple(output)


def main(argv):
    try:
        if len(argv) != 1 or len(argv[0]) > MAX_MESSAGE_BYTES * 2:
            raise ProtocolViolation("invalid worker invocation")
        payload = strict_loads(base64.b64decode(argv[0], validate=True))
        response = process(payload)
    except Exception:
        response = {"ok": False, "error": {"category": "protocol", "type": "ProtocolViolation"}}
    encoded = json.dumps(response, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode()) > MAX_MESSAGE_BYTES:
        encoded = '{"ok":false,"error":{"category":"worker","type":"WorkerFailure"}}'
    print(encoded)


if __name__ == "__main__":
    main(sys.argv[1:])
