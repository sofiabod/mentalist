import json
import os
import re
import time
from pathlib import Path

from adapters.protocol import Client, SfxDaemonError
from eval.capture import Capturer, capture_enabled
from mining.normalize import _status, classify
from sfx import resolver
from sfx.schema import STREAM_ONLY, Profile

VERB_TOOL = {"fork": "Edit"}

PROFILE = Profile(emission_format="json-tool-loop", state_model="fs-only",
                  interception_grade="full", stream_visibility="parsed-action",
                  fork_substrate="clonefile")

SOCKET_ENV = "SFX_SOCKET"
SESSION_ENV = "SFX_SESSION"
REPO_ENV = "SFX_REPO"
SCRATCH_ENV = "SFX_SCRATCH"
STATE_DIR_ENV = "SFX_STATE_DIR"

ENVIRONMENT_CLASS = "adapters.mini_swe.SfxEnvironment"


def _session():
    return os.environ.get(SESSION_ENV, "mini")


def connect():
    sock = os.environ.get(SOCKET_ENV)
    if not sock:
        return None
    try:
        client = Client(sock)
        client.turn_begin(_session(), repo=os.environ.get(REPO_ENV, os.getcwd()),
                          role="main", scratch=os.environ.get(SCRATCH_ENV),
                          state_dir=os.environ.get(STATE_DIR_ENV))
        return client
    except (SfxDaemonError, OSError):
        return None


def _ctx():
    return resolver.Ctx(repo=Path(os.environ.get(REPO_ENV, os.getcwd())))


def _get_args(kind, command):
    return resolver.canonical_args(kind, command, _ctx())


def claim_or_none(client, kind, command):
    if client is None:
        return None
    try:
        reply = client.resolve(_session(), tool=kind, args=_get_args(kind, command))
    except SfxDaemonError:
        return None
    served = reply["result"]
    if isinstance(served, str) and served.startswith("hit"):
        output, returncode = reply["output"]
        return {"output": output, "returncode": returncode, "exception_info": ""}
    return None


OBS_CAP = 4096


def report_executed(client, kind, verb, command, returncode, latency, output=""):
    if client is None:
        return
    outcome = _status(returncode != 0, kind)
    try:
        client.call_executed(_session(), tool=kind, verb=verb, outcome=outcome,
                             args=_get_args(kind, command), latency=latency,
                             observation=output[:OBS_CAP])
    except SfxDaemonError:
        return


_HEREDOC_RE = re.compile(
    r"(?:cat\s+>|tee\s+(?:-a\s+)?>?)\s*(?P<path>\S+)\s*<<-?\s*"
    r"['\"]?(?P<tok>\w+)['\"]?\s*\n(?P<body>.*)\n(?P=tok)\s*$",
    re.DOTALL)
_ADDFILE_RE = re.compile(
    r"\*\*\* Add File:\s*(?P<path>\S+)\s*\n(?P<body>(?:\+.*\n?)*)")
_ECHO_RE = re.compile(
    r"^\s*(?P<cmd>echo|printf)\s+(?P<arg>'[^']*'|\"[^\"$`\\]*\"|[^\s'\"$`>|&\\]+)"
    r"\s+>\s*(?P<path>\S+)\s*$")


def _extract_write(command):
    if command.count("<<") > 1:
        return None
    m = _HEREDOC_RE.search(command)
    if m:
        return m.group("path"), m.group("body") + "\n"
    if "apply_patch" in command and command.count("*** Add File:") == 1 \
            and "*** Update File:" not in command and "*** Delete File:" not in command:
        m = _ADDFILE_RE.search(command)
        if m:
            body = "".join(line[1:] + "\n" for line in m.group("body").splitlines())
            return m.group("path"), body
    m = _ECHO_RE.match(command)
    if m:
        arg, cmd = m.group("arg"), m.group("cmd")
        text = arg[1:-1] if arg[0] in "'\"" else arg
        if cmd == "printf":
            text = text.replace("\\n", "\n").replace("\\t", "\t")
        else:
            text += "\n"
        return m.group("path"), text
    return None


def mini_config_yaml():
    return f"environment:\n  environment_class: {ENVIRONMENT_CLASS}\n"


try:
    from minisweagent.environments.local import LocalEnvironment

    class SfxEnvironment(LocalEnvironment):
        def __init__(self, **kwargs):
            import atexit
            super().__init__(**kwargs)
            self._sfx = connect()
            self._n = 0
            self._cap = None
            self._cleaned = False
            atexit.register(self.cleanup)

        def execute(self, action, cwd="", *, timeout=None):
            command = action.get("command", "")
            kind, verb = classify("Bash", command)
            if capture_enabled():
                return self._capture_execute(action, cwd, timeout, kind, verb, command)
            if kind == "edit" and verb in STREAM_ONLY:
                pc = _extract_write(command)
                if pc is not None:
                    return self._write_and_chain(command, kind, verb, pc,
                                                 action, cwd, timeout)
            hit = claim_or_none(self._sfx, kind, command)
            if hit is not None:
                return hit
            start = time.monotonic()
            result = super().execute(action, cwd, timeout=timeout)
            latency = (time.monotonic() - start) * 1000.0
            report_executed(self._sfx, kind, verb, command, result["returncode"], latency,
                             output=result.get("output", ""))
            return result

        def _capture_execute(self, action, cwd, timeout, kind, verb, command):
            workdir = cwd or self.config.cwd or os.getcwd()
            if self._cap is None:
                self._cap = Capturer(workdir)
            gap = self._cap.start_gap(time.monotonic())
            start = time.monotonic()
            result = super().execute(action, cwd, timeout=timeout)
            latency = (time.monotonic() - start) * 1000.0
            returned = time.monotonic()
            write_body = ""
            if kind == "edit" and verb in STREAM_ONLY:
                pc = _extract_write(command)
                if pc is not None:
                    write_body = pc[1]
            self._cap.record(kind=kind, verb=verb, args=_get_args(kind, command),
                             think_gap_ms=gap, latency_ms=latency, result=result,
                             write_body=write_body, now=returned)
            return result

        def _plain(self, action, cwd, timeout, kind, verb, command):
            start = time.monotonic()
            result = super().execute(action, cwd, timeout=timeout)
            latency = (time.monotonic() - start) * 1000.0
            report_executed(self._sfx, kind, verb, command, result["returncode"], latency,
                             output=result.get("output", ""))
            return result

        def _write_and_chain(self, command, kind, verb, pc, action, cwd, timeout):
            if self._sfx is None:
                return self._plain(action, cwd, timeout, kind, verb, command)
            path, contents = pc
            body = json.dumps({"path": path, "contents": contents})
            tool = VERB_TOOL[verb]
            cid = f"e{self._n}"
            self._n += 1
            try:
                self._sfx.feed(_session(), cid, tool, body)
            except SfxDaemonError:
                return self._plain(action, cwd, timeout, kind, verb, command)
            result = self._plain(action, cwd, timeout, kind, verb, command)
            try:
                self._sfx.resolve(_session(), tool=kind,
                                  args={"path": path, "contents": contents})
            except SfxDaemonError:
                pass
            return result

        def cleanup(self):
            if self._cleaned:
                return
            self._cleaned = True
            if getattr(self, "_cap", None) is not None:
                self._cap.final()
            if self._sfx is not None:
                try:
                    self._sfx.turn_end(_session())
                    self._sfx.session_end(_session())
                except Exception:
                    pass
            up = getattr(super(), "cleanup", None)
            if up is not None:
                up()
except ImportError:
    pass
