"""SFXLiveAgent: run sfx inside a Harbor sandbox around the fenced-Bash model loop.

The default wrapped model drives the agent; its REAL tool calls are
intercepted at environment.exec and routed through an sfx daemon RUNNING INSIDE THE
SAME SANDBOX with SFX_REPO=/app (the actual workspace). ON forks the real /app in
the sandbox, pre-runs the predicted next call in the fork, and serves the cached
(stdout, stderr, rc) iff the agent's next real call matches. OFF disables
speculation but retains wrapper bookkeeping. Explicit injected-model and captured-
stream factories support CPU evaluation; neither is labeled live inference.

Interception seam: install() launches the daemon + uploads a tiny in-sandbox client
CLI (eval.sfx_client_cli); run() monkeypatches environment.exec so every command the
wrapped agent issues first asks the in-sandbox daemon (resolve), returns the cached
result on a hit, else executes authoritatively and reports the outcome (train +
speculate next). Everything sfx touches lives in the sandbox on /app.
"""
import asyncio
import base64
import hashlib
import importlib
import json
import os
import posixpath
import re
import shlex
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import ExecResult

from mining.normalize import _status, classify
from sfx.schema import STREAM_ONLY

APP = "/app"
SFX_SRC = "/sfx-live-src"
SOCKET = "/tmp/sfx.sock"
SCRATCH = "/tmp/sfx-scratch"
TRACE = "/logs/agent/sfx-trace.jsonl"
SESSION = "live"
OBS_CAP = 4096
PRIVATE_OUTPUT_CAP = 8192
DAEMON_LOG_CAP = 65536
DAEMON_LOG = "/tmp/sfx-daemon.log"
VERB_TOOL = {"fork": "Edit"}
_LITERAL_HEREDOC = re.compile(
    r"\s*cat\s+>\s*(?P<path>[\w./-]+)\s*<<\s*(?P<quote>['\"])"
    r"(?P<token>\w+)(?P=quote)\s*\n(?P<body>.*?)\n(?P=token)\s*\Z", re.S)
_BASH_OPEN = re.compile(r"```bash\s*\n")
_HEREDOC_HEADER = re.compile(
    r"\s*cat\s+>\s*[\w./-]+\s*<<\s*(?P<quote>['\"])(?P<token>\w+)(?P=quote)\s*")


def _literal_write(command, repo=APP):
    match = _LITERAL_HEREDOC.fullmatch(command)
    if match is None:
        return None
    if match.group("token") in match.group("body").splitlines():
        return None
    raw_path = match.group("path")
    path = PurePosixPath(raw_path)
    if not path.parts or ".." in path.parts or raw_path.endswith("/"):
        return None
    if path.is_absolute():
        root = PurePosixPath(repo)
        if not root.is_absolute() or root == PurePosixPath(root.anchor) or ".." in root.parts:
            return None
        try:
            path = path.relative_to(root)
        except ValueError:
            return None
        if not path.parts:
            return None
        raw_path = str(path)
    return {"path": raw_path, "contents": match.group("body") + "\n"}


def _streamed_literal_write(text, repo=APP):
    """Recognize a complete literal edit, never guess missing file contents.

    A newline-terminated heredoc delimiter is sufficient to know the edit before
    the closing Markdown fence arrives. Later compound commands invalidate this
    candidate; the final authoritative command must match it exactly.
    """
    opening = _BASH_OPEN.search(text)
    if opening is None:
        return None
    body = text[opening.end():]
    fence = body.find("```")
    if fence >= 0:
        command = body[:fence].strip()
    else:
        header, newline, _ = body.partition("\n")
        match = _HEREDOC_HEADER.fullmatch(header) if newline else None
        if match is None:
            return None
        ending = re.search(r"\n" + re.escape(match.group("token")) + r"\n", body)
        if ending is None or body[ending.end():].strip() not in ("", "`", "``"):
            return None
        command = body[:ending.end()].strip()
    write = _literal_write(command, repo)
    return (command, write) if write is not None else None


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _diagnostic_text(text):
    """Count/hash untrusted diagnostics; never echo arbitrary error prose."""
    data = (text or "").encode("utf-8", errors="replace")
    result = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    match = re.search(r"\[Errno (\d+)\]", text or "")
    names = {2: "not_found", 13: "permission_denied", 32: "broken_pipe",
             61: "connection_refused", 104: "connection_reset", 110: "timed_out",
             111: "connection_refused"}
    if match and int(match.group(1)) in names:
        result.update(errno=int(match.group(1)), classification=names[int(match.group(1))])
    return result


def _diagnostic_error(exc):
    return {"type": type(exc).__name__, **_diagnostic_text(str(exc))}


def _diagnostic_result(result, *, private_preview=False):
    stdout, stderr = result.stdout or "", result.stderr or ""
    record = {"returncode": result.return_code,
              "stdout": _diagnostic_text(stdout), "stderr": _diagnostic_text(stderr)}
    if private_preview:
        # This is concatenation, not a claim to reconstruct stream interleaving.
        streams = [text.encode("utf-8", errors="replace") for text in (stdout, stderr)]
        per_stream = PRIVATE_OUTPUT_CAP // 2
        preview = "".join(data[:per_stream].decode("utf-8", errors="ignore") for data in streams)
        record.update(stdout_then_stderr_preview=preview,
                      preview_limit_bytes=PRIVATE_OUTPUT_CAP,
                      preview_layout="stdout_prefix_then_stderr_prefix",
                      preview_truncated=any(len(data) > per_stream for data in streams))
    return record


def _write_private(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(contents)


def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


def _repo_root():
    import eval as _e
    return Path(_e.__file__).resolve().parent.parent.parent


def build_stub_agent(logs_dir, environment, artifacts=()):
    """Wrapped-agent factory for CPU dry-runs: emits a FIXED trajectory (no model).

    Drives environment.exec (monkeypatched by SFXLiveAgent to route through the
    in-sandbox daemon) with a heredoc write to sol.py then `python sol.py`, so both
    the STREAM_ONLY write-chain path and the resolve/report RUN path are exercised
    with zero model dependency. `artifacts` are declared task output paths the stub
    touches so Harbor's post-run artifact download does not fail the trial. The
    trajectory does not solve any real task; parity just needs both arms to reach
    the same verdict.
    """
    return _StubAgent(artifacts)


class _StubAgent:
    def __init__(self, artifacts=()):
        self._artifacts = artifacts

    async def run(self, instruction, environment, context):
        contents = "print('sfx cpu dry-run stub')\n"
        heredoc = f"cat > sol.py <<'SFX_EOF'\n{contents}SFX_EOF"
        await environment.exec(command=heredoc, user=environment.default_user)
        await environment.exec(command="python sol.py", user=environment.default_user)
        for path in self._artifacts:
            await environment.exec(command=f"touch {path}", user=environment.default_user)


def build_live_agent(logs_dir, environment, artifacts=(), *, config=None):
    """Wrapped-agent factory that drives the REAL model loop (eval.live_ab.run_agent
    with the task-submission-compatible repro scaffold) through SFXLiveAgent's monkeypatched
    environment.exec.

    config keys (from SFXLiveAgent kwargs / --ak, env fallback in _live_config):
      base_url, model, api_key   -- the OpenAI-compatible endpoint the loop hits
      max_steps                  -- loop budget
      model_fn                   -- optional "module:attr" import path returning a
                                    model_fn (or a "module:attr(arg)" builder). Passed
                                    explicitly only for the CPU smoke; production leaves
                                    it unset for live inference.
      streaming                  -- enabled by default for live inference
      stream_model_fn            -- optional injected/recorded stream factory
      temperature, seed, max_tokens -- sampling configuration, recorded in the trace
      scaffold                   -- "task", "general", or "native" tool calling
      context_budget             -- optional exact server-side token budgeting
    """
    return _LiveAgent(config or {})


class _LiveAgent:
    def __init__(self, config):
        self._config = config

    async def run(self, instruction, environment, context):
        from eval.live_ab import SYSTEM, _TASK_SUBMISSION_SYSTEM, _model_stream, run_agent
        from eval.model_tools import stream_native_bash

        cfg = self._config
        scaffold = cfg.get("scaffold", "task")
        if scaffold not in ("task", "general", "native"):
            raise ValueError("scaffold must be 'task', 'general', or 'native'")
        system = (_TASK_SUBMISSION_SYSTEM if scaffold == "task" else SYSTEM + (
            " Follow the task requirements, implement the requested program, inspect the "
            "workspace, and use checks you find appropriate. Honor all task submission "
            "requirements before replying DONE."
            " There are no native tools or function calls in this environment. An external "
            "runner executes ONLY the single bash command in your assistant FINAL response. "
            "Put that command in the required bash code block in the final channel. "
            "Do not issue native tool calls or tool handoffs."))
        if scaffold == "native":
            system = (
                "You are a coding agent working in the current repository. Follow the task "
                "requirements, implement the requested program, inspect the workspace, and "
                "use checks you find appropriate. Use execute_bash with one shell command "
                "per turn to interact with the environment. Call finish with done=true "
                "only after completing the task and its submission requirements.")
        loop = asyncio.get_running_loop()
        stopped = threading.Event()
        calls_lock = threading.Lock()
        pending_calls = set()

        def from_model_thread(make_coroutine):
            # Cancelling asyncio.to_thread does not stop its worker. A model
            # response arriving after a Harbor checkpoint timeout must never
            # execute a command or start speculation in a later checkpoint.
            with calls_lock:
                if stopped.is_set():
                    raise asyncio.CancelledError("model worker is no longer active")
                future = asyncio.run_coroutine_threadsafe(make_coroutine(), loop)
                pending_calls.add(future)
            try:
                return future.result()
            finally:
                with calls_lock:
                    pending_calls.discard(future)

        def sync_exec(cmd):
            r = from_model_thread(lambda: environment.exec(
                command=cmd, cwd=cfg.get("repo"), user=environment.default_user))
            return json.dumps({"returncode": r.return_code,
                               "stderr": r.stderr or "", "stdout": r.stdout or ""})

        model_fn = _resolve_model_fn(cfg.get("model_fn"))
        kw = {"model_fn": model_fn} if model_fn is not None else {}
        stream_spec = cfg.get("stream_model_fn")
        stream_fn = _resolve_model_fn(stream_spec) if stream_spec else None
        if cfg.get("streaming", True) and (stream_fn is not None or model_fn is None):
            kw["stream_fn"] = stream_fn or (stream_native_bash if scaffold == "native" else _model_stream)
            handler = cfg.get("_on_stream_event")
            if handler is not None:
                def on_stream_event(event):
                    return from_model_thread(lambda: handler(event))
                kw["on_stream_event"] = on_stream_event
        failure = {}

        def drive():
            try:
                return run_agent(
                    sync_exec, instruction,
                    base_url=cfg["base_url"], model=cfg["model"], api_key=cfg["api_key"],
                    max_steps=int(cfg.get("max_steps", 30)), system=system,
                    temperature=float(cfg.get("temperature", 0.0)), seed=int(cfg.get("seed", 0)),
                    max_tokens=int(cfg.get("max_tokens", 1024)),
                    context_budget=cfg.get("context_budget"), **kw)
            except BaseException as exc:
                # Python 3.12 asyncio rebuilds TimeoutError across this thread
                # boundary, losing custom attributes. Preserve evidence first.
                failure["trajectory"] = getattr(exc, "sfx_live_trajectory", None)
                raise

        try:
            traj = await asyncio.to_thread(drive)
        except BaseException as exc:
            partial = failure.get("trajectory", getattr(exc, "sfx_live_trajectory", None))
            if partial is not None:
                partial["scaffold"] = scaffold
                context.metadata = {**(context.metadata or {}), "sfx_live_trajectory": partial}
            raise
        finally:
            with calls_lock:
                stopped.set()
                for future in pending_calls:
                    future.cancel()
        traj["scaffold"] = scaffold
        context.metadata = {**(context.metadata or {}),
                            "sfx_live_trajectory": traj}


def _resolve_trajectory(spec):
    if not spec:
        return None
    mod_name, attr = spec.split(":")
    return getattr(importlib.import_module(mod_name), attr)


def _resolve_model_fn(spec):
    """Turn a "module:attr" (or "module:attr(literal)") spec into a model_fn callable.

    Bare "module:attr" imports the attribute as-is. The "(literal)" form calls it with
    one string arg (e.g. the stub builder stub_model_fn(contents) that returns a
    model_fn). No spec -> None -> the live _model_action default is used.
    """
    if not spec:
        return None
    mod_name, rest = spec.split(":", 1)
    if rest.endswith(")") and "(" in rest:
        attr, arg = rest[:-1].split("(", 1)
        return getattr(importlib.import_module(mod_name), attr)(arg)
    return getattr(importlib.import_module(mod_name), rest)


def _parse_speculate_writes(value):
    return _bool_flag(value, "speculate_writes")


def _bool_flag(value, name):
    if type(value) is bool:
        return value
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise ValueError(f"{name} must be a boolean or 'true'/'false'")


class SFXLiveAgent(BaseInstalledAgent):
    """Wrap a standard installed agent; route its live tool calls through in-sandbox sfx.

    arm="OFF": speculation disabled (plain exec, still routed for parity accounting).
    arm="ON":  speculation enabled; predicted next call is pre-run in a fork of /app.
    speculate_writes=False: GET-only ablation; retain authoritative write fences
    and post-call speculation, but do not feed literal edits into fork chains.
    Harbor CLI also accepts --ak speculate_writes=true|false.
    """

    SUPPORTS_CONFIG = False
    SUPPORTS_RESUME = False
    speculate_writes = True
    _pending_edit = None
    repo = APP
    source_dir = SFX_SRC
    socket_path = SOCKET
    scratch_dir = SCRATCH
    trace_path = TRACE
    python_executable = "python3"
    _control_exec = None
    control_transport = "exec"
    _control_channel = None
    script_contracts = None

    def __init__(self, logs_dir, *, wrapped="eval.sfx_live_agent:build_stub_agent",
                 arm="OFF", depth=1, table="benchmark", artifacts="",
                 base_url=None, model=None, api_key=None, max_steps=30, model_fn=None,
                 trajectory=None, speculate_writes=True, streaming=True,
                 stream_model_fn=None, temperature=0.0, seed=0, max_tokens=1024,
                 scaffold="task", context_budget=None,
                 repo=None, source_dir=None, socket_path=None, scratch_dir=None,
                 trace_path=None, python_executable="python3", control_exec=None,
                 control_transport="exec", script_contracts=None, fork_path_view="cwd", **kwargs):
        from sfx.script_contracts import ScriptContracts

        if control_transport not in ("exec", "in_sandbox"):
            raise ValueError("invalid control transport")
        if fork_path_view not in ("cwd", "proot"):
            raise ValueError("invalid fork path view")
        self.control_transport = control_transport
        self._control_channel = None
        self.script_contracts = ScriptContracts(script_contracts)
        self.fork_path_view = fork_path_view
        self.wrapped = wrapped
        self.arm = arm
        self.speculate_writes = _parse_speculate_writes(speculate_writes)
        if scaffold not in ("task", "general", "native"):
            raise ValueError("scaffold must be 'task', 'general', or 'native'")
        if context_budget not in (None, "sglang"):
            raise ValueError("context_budget must be None or 'sglang'")
        if context_budget is not None and (scaffold == "native" or not _bool_flag(streaming, "streaming")
                                           or stream_model_fn or model_fn or os.environ.get("SFX_MODEL_FN")):
            raise ValueError("context budgeting requires live fenced streaming without injected models")
        self.depth = depth
        self.table = table
        # Hosted ACP runs inside the task container; allow per-run paths without
        # changing process-global defaults used by the existing Harbor wrapper.
        self.repo = str(repo) if repo is not None else APP
        self.source_dir = str(source_dir) if source_dir is not None else SFX_SRC
        self.socket_path = str(socket_path) if socket_path is not None else SOCKET
        self.scratch_dir = str(scratch_dir) if scratch_dir is not None else SCRATCH
        self.trace_path = str(trace_path) if trace_path is not None else TRACE
        self.python_executable = str(python_executable)
        self._control_exec = control_exec
        self.trajectory = _resolve_trajectory(trajectory)
        self.artifacts = [a for a in artifacts.split(",") if a] if artifacts else []
        self.live_config = {
            "base_url": base_url or os.environ.get("OPENAI_BASE_URL"),
            "model": model or os.environ.get("SFX_MODEL"),
            "api_key": api_key or os.environ.get("OPENAI_API_KEY", "sfx"),
            "max_steps": max_steps,
            "model_fn": model_fn or os.environ.get("SFX_MODEL_FN"),
            "streaming": _bool_flag(streaming, "streaming"),
            "stream_model_fn": stream_model_fn,
            "temperature": temperature, "seed": seed, "max_tokens": max_tokens,
            "scaffold": scaffold, "context_budget": context_budget,
        }
        self._counts = {"resolves": 0, "hits": 0, "misses": 0, "never_routed": 0,
                        "authoritative": 0, "writes_fed": 0}
        self._session = f"live-{uuid.uuid4().hex}"
        self._receipts = []
        self._raw = []
        self._last_served = None
        self._route_lock = asyncio.Lock()
        self._default_cwd = None
        self._early_edit_events = []
        self._early_edit_seen = set()
        self._stream_origin = None
        self._run_index = 0
        super().__init__(logs_dir, **kwargs)

    @staticmethod
    def name():
        return "sfx-live-agent"

    def version(self):
        return "1.0"

    async def install(self, environment):
        await self.ensure_system_dependencies(environment, ("git", "python3"))
        if getattr(self, "fork_path_view", "cwd") == "proot":
            await self._ensure_proot(environment)
        root = _repo_root()
        await environment.upload_dir(str(root / "src"), self.source_dir)
        table_file = self.table.removesuffix(".json") + ".json"
        await environment.exec(command="mkdir -p /root/data/tables", user="root")
        # load_table (eval.table) resolves SFX_TABLE against /root/data/tables in-image
        await environment.upload_file(str(root / "data" / "tables" / table_file),
                                      f"/root/data/tables/{table_file}")
        await self._launch_daemon(environment, table_file)

    async def _ensure_proot(self, environment):
        probe = "command -v proot >/dev/null 2>&1"
        result = await environment.exec(command=probe, user=environment.default_user, timeout_sec=10)
        if result.return_code == 0:
            return
        manager = await environment.exec(command="command -v apt-get >/dev/null 2>&1",
                                         user="root", timeout_sec=10)
        if manager.return_code != 0:
            raise RuntimeError("PRoot path view requires a preinstalled proot executable or an apt-get task image")
        result = await environment.exec(
            command="apt-get update && apt-get install -y --no-install-recommends proot",
            user="root", env={"DEBIAN_FRONTEND": "noninteractive"}, timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(f"PRoot installation failed with exit code {result.return_code}")
        result = await environment.exec(command=probe, user=environment.default_user, timeout_sec=10)
        if result.return_code != 0:
            raise RuntimeError("PRoot installation completed without an executable available to the agent")

    async def _launch_daemon(self, environment, table_file):
        scratch, source, repo, trace, sock, python = map(shlex.quote, (
            self.scratch_dir, self.source_dir, self.repo, self.trace_path,
            self.socket_path, self.python_executable))
        contracts = self.script_contracts.definitions if self.script_contracts else []
        policy = shlex.quote(json.dumps(contracts))
        view = shlex.quote(getattr(self, "fork_path_view", "cwd"))
        cmd = (
            f"mkdir -p {scratch} /logs/agent /root/data/tables; "
            f"PYTHONPATH={source} SFX_REPO={repo} SFX_SCRATCH={scratch} SFX_TRACE={trace} "
            f"SFX_TABLE={shlex.quote(table_file)} SFX_SEPARATE_STDERR=1 "
            f"SFX_SCRIPT_CONTRACTS={policy} SFX_FORK_PATH_VIEW={view} "
            f"setsid nohup {python} -m eval.sfx_daemon_run {sock} {self.depth} "
            f">/tmp/sfx-daemon.log 2>&1 & disown; "
            f"for i in $(seq 1 100); do test -S {sock} && echo STARTED && break; "
            f"sleep 0.1; done"
        )
        r = await environment.exec(command=cmd, user=environment.default_user,
                                   env={"PYTHONPATH": self.source_dir})
        if "STARTED" not in (r.stdout or ""):
            raise RuntimeError(
                f"sfx daemon did not bind {self.socket_path}: {r.stdout} {r.stderr}")

    async def _cli(self, environment, *args):
        self._tool_phase("control:" + args[0])
        r = None
        try:
            if self.control_transport == "in_sandbox":
                from eval.control_transport import InSandboxControlTransport

                if self._control_channel is None:
                    self._control_channel = InSandboxControlTransport(self.socket_path)
                return await self._control_channel.request(*args)
            parts = " ".join(shlex.quote(str(a)) for a in args)
            # Control calls bypass interception, so they cannot recurse into _route.
            exec_fn = self._control_exec or self._raw_exec or environment.exec
            r = await exec_fn(
                command=f"PYTHONPATH={shlex.quote(self.source_dir)} "
                        f"SFX_SOCKET={shlex.quote(self.socket_path)} "
                        f"{shlex.quote(self.python_executable)} -m eval.sfx_client_cli {parts}",
                user=environment.default_user)
            if r.return_code != 0:
                raise RuntimeError(f"sfx control call {args[0]} failed (exit code {r.return_code})")
            line = (r.stdout or "").strip().splitlines()
            reply = json.loads(line[-1]) if line else None
            if not isinstance(reply, dict) or reply.get("error"):
                raise RuntimeError(f"invalid sfx control reply for {args[0]}")
            return reply
        except BaseException as exc:
            tool = getattr(self, "_active_tool", None)
            self._diagnostic_state()["control_failures"].append({
                "operation": args[0], "tool_index": tool["index"] if tool else None,
                "error": _diagnostic_error(exc),
                "result": _diagnostic_result(r) if r is not None else None})
            raise

    _raw_exec = None

    async def run(self, instruction, environment, context):
        # Harbor reuses this installed agent across multi-step checkpoints, but
        # each run starts a fresh conversation over the preserved workspace.
        # Replacing these containers also keeps earlier AgentContext records
        # from changing when the next checkpoint adds receipts or counters.
        self._run_index = getattr(self, "_run_index", 0) + 1
        self._session = f"live-{uuid.uuid4().hex}"
        self._counts = {"resolves": 0, "hits": 0, "misses": 0, "never_routed": 0,
                        "authoritative": 0, "writes_fed": 0}
        self._receipts, self._raw = [], []
        self._last_served = None
        self._snapshot = {}
        self._initial_fs_hash = None
        self._wall_s = None
        self._completed = False
        self._failure = None
        self._cleanup_errors = []
        self._diagnostics = {"tools": [], "control_failures": []}
        self._diagnostic_origin = time.monotonic()
        self._active_tool = None
        self._instruction_sha256 = _sha(instruction)
        context.metadata = {key: value for key, value in (context.metadata or {}).items()
                            if key not in ("sfx_live", "sfx_live_trajectory")}
        original_exec = environment.exec
        self._raw_exec = original_exec
        self._pending_edit = None
        self._early_edit_events = []
        self._early_edit_seen = set()
        self._stream_origin = None
        self._env_id = getattr(environment, "session_id", None)
        self._default_cwd = None
        trace_offset = None
        begin_attempted = False
        run_error = None
        lifecycle_started = time.monotonic()
        started = None

        async def intercepted_exec(command, cwd=None, env=None, timeout_sec=None,
                                   user=None):
            return await self._route(environment, original_exec, command,
                                     cwd, env, timeout_sec, user)

        try:
            working_dir = await original_exec(command="pwd", user=environment.default_user)
            self._default_cwd = ((working_dir.stdout or "").strip()
                                 if working_dir.return_code == 0 else None)
            initial = await self._cli(environment, "snapshot", self._session,
                                      self.repo, self.trace_path)
            self._initial_fs_hash = initial["final_fs_hash"]
            trace_offset = len(initial.get("trace", []))
            begin_args = [self._session, self.repo, self.scratch_dir,
                          "0" if self.arm == "ON" else "1"]
            if self.trajectory is not None:
                begin_args.append(_b64(self.trajectory))
            begin_attempted = True
            await self._cli(environment, "begin", *begin_args)
            environment.exec = intercepted_exec
            started = time.monotonic()
            agent = self._load_wrapped(environment)
            await agent.run(instruction, environment, context)
            self._completed = True
        except BaseException as exc:
            run_error = exc
            self._failure = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            environment.exec = original_exec
            if started is not None:
                self._wall_s = time.monotonic() - started
            cleanup_error = None
            operations = [("cancel_pending_edit", lambda: self._cancel_pending_edit(
                environment, "run_end"))]
            if begin_attempted:
                operations.append(("end", lambda: self._cli(environment, "end", self._session)))
            operations.append(("snapshot", lambda: self._cli(
                environment, "snapshot", self._session, self.repo, self.trace_path)))
            for phase, operation in operations:
                try:
                    result = await operation()
                    if phase == "snapshot":
                        self._snapshot = {**result, "trace": (
                            result.get("trace", [])[trace_offset:]
                            if trace_offset is not None else [])}
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    self._cleanup_errors.append({"phase": phase,
                                                 "type": type(exc).__name__, "message": str(exc)})
            if self._control_channel is not None:
                try:
                    await self._control_channel.aclose()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    self._cleanup_errors.append({"phase": "control_close",
                                                 "type": type(exc).__name__, "message": str(exc)})
                finally:
                    self._control_channel = None
            try:
                await self._archive_daemon_log(environment, original_exec)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
                self._cleanup_errors.append({"phase": "daemon_log",
                                             "type": type(exc).__name__, "message": str(exc)})
            if cleanup_error is not None:
                self._completed = False
                if run_error is None:
                    self._failure = {"type": type(cleanup_error).__name__,
                                     "message": str(cleanup_error)}
            self._lifecycle_wall_s = time.monotonic() - lifecycle_started
            self._raw_exec = None
            self._persist(context)
            if run_error is None and cleanup_error is not None:
                raise cleanup_error

    def _edit_event(self, event, model_step, **fields):
        self._early_edit_events.append({
            "event": event, "model_step": model_step,
            "elapsed_s": time.monotonic() - self._stream_origin, **fields})

    async def _cancel_pending_edit(self, environment, reason):
        pending = self._pending_edit
        if pending is None:
            return
        self._pending_edit = None
        await self._cli(environment, "mutation_end", self._session,
                        _b64({"mutation_id": pending["token"], "success": False}))
        self._edit_event("discarded", pending["model_step"], reason=reason)

    async def _stream_event(self, environment, event):
        """A speculative reservation never executes the authoritative edit."""
        if self._stream_origin is None:
            origin = event.get("_monotonic_origin")
            self._stream_origin = (origin if origin is not None
                                   else time.monotonic() - event["elapsed_s"])
        kind, step = event["event"], event["model_step"]
        async with self._route_lock:
            if kind in ("model_start", "model_abort"):
                await self._cancel_pending_edit(environment, kind)
                return
            if kind == "model_end":
                pending = self._pending_edit
                if pending is not None:
                    if (pending["model_step"] != step or event.get("command") != pending["command"]
                            or event.get("text", "").strip().startswith("DONE")):
                        await self._cancel_pending_edit(environment, "final_command_mismatch")
                    else:
                        pending["confirmed"] = True
                        self._edit_event("confirmed", step, command=pending["command"])
                return
            if kind != "model_delta" or self._pending_edit is not None:
                return
            raw = event.get("raw_stream_event", {})
            if raw.get("native_action_ready"):
                from eval.model_tools import parse_native_action
                native_command = parse_native_action(event.get("text", ""))
                native_write = _literal_write(native_command, self.repo) if native_command is not None else None
                candidate = (native_command, native_write) if native_write is not None else None
            else:
                candidate = _streamed_literal_write(event.get("text", ""), self.repo)
            if candidate is None:
                return
            command, write = candidate
            if step in self._early_edit_seen:
                return
            self._early_edit_seen.add(step)
            self._edit_event("ready", step, command=command)
            if self.arm != "ON" or not self.speculate_writes:
                return
            if not self._eligible_context(environment, event.get("_exec_cwd"), None,
                                          None, environment.default_user):
                return
            begin = await self._cli(environment, "mutation_begin", self._session,
                                    _b64({"write_args": write}))
            token = begin.get("mutation_id")
            if not token:
                raise RuntimeError("sfx did not fence streamed speculation")
            self._pending_edit = {"token": token, "write": write, "command": command,
                                  "model_step": step, "confirmed": False}
            try:
                self._edit_event("feed_started", step)
                await self._cli(environment, "feed", self._session,
                                _b64({"call_id": f"stream-{step}", "tool": "Edit",
                                      "body": json.dumps(write), "mutation_id": token}))
                self._counts["writes_fed"] += 1
                self._edit_event("feed_finished", step)
            except BaseException:
                await self._cancel_pending_edit(environment, "feed_failed")
                raise

    async def _route(self, environment, original_exec, command, cwd, env,
                     timeout_sec, user):
        async with self._route_lock:
            diagnostics = self._diagnostic_state()
            tool = {"index": len(diagnostics["tools"]), "command": command,
                    "tool_start_s": time.monotonic() - self._diagnostic_origin,
                    "phase": "routing", "result": None}
            diagnostics["tools"].append(tool)
            self._active_tool = tool
            try:
                self._last_served = None
                pending = self._pending_edit
                if pending is not None and (
                        not pending["confirmed"] or pending["command"] != command
                        or not self._eligible_context(environment, cwd, env, timeout_sec, user)):
                    await self._cancel_pending_edit(environment, "authoritative_request_mismatch")
                result = await self._route_one(environment, original_exec, command,
                                               cwd, env, timeout_sec, user)
                tool.update(phase="completed", served=bool(self._last_served),
                            tool_end_s=time.monotonic() - self._diagnostic_origin,
                            result=_diagnostic_result(result, private_preview=True))
            except BaseException as exc:
                tool.update(phase=tool.get("primary_error_phase", tool["phase"]),
                            error=_diagnostic_error(exc),
                            tool_abort_s=time.monotonic() - self._diagnostic_origin)
                raise
            finally:
                self._active_tool = None
            self._raw.append({
                "index": len(self._receipts),
                "command": command,
                "served": bool(self._last_served),
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "returncode": result.return_code,
            })
            self._receipts.append({
                "command": command,
                "context": {"cwd": cwd or self._default_cwd,
                            "env_sha": _sha(json.dumps(env or {}, sort_keys=True)),
                            "timeout_sec": timeout_sec,
                            "user": user or environment.default_user},
                "stdout_sha": _sha(result.stdout or ""),
                "stderr_sha": _sha(result.stderr or ""),
                "returncode": result.return_code,
                "signal": -result.return_code if result.return_code < 0 else 0,
            })
            return result

    def _eligible_context(self, environment, cwd, env, timeout_sec, user):
        working_dir = cwd or self._default_cwd
        return (bool(working_dir) and ".." not in PurePosixPath(working_dir).parts
                and ".." not in PurePosixPath(self.repo).parts
                and posixpath.normpath(working_dir) == self.repo
                and not env and timeout_sec is None
                and user in (None, environment.default_user))

    async def _route_one(self, environment, original_exec, command, cwd, env,
                         timeout_sec, user):
        kind, verb = (("edit", "fork") if _literal_write(command, self.repo) is not None
                      else classify("Bash", command))
        if (kind == "unknown" and self.script_contracts is not None
                and self.script_contracts.match(command, self.repo) is not None):
            kind, verb = "run", "free"
        args = {"cmd": command}
        eligible = self._eligible_context(environment, cwd, env, timeout_sec, user)
        if verb in STREAM_ONLY:
            return await self._write_and_chain(environment, original_exec, command,
                                               kind, verb, cwd, env, timeout_sec, user)
        if verb == "never" or not eligible or self.arm != "ON":
            self._counts["never_routed"] += 1
            return await self._authoritative(environment, original_exec, command, kind,
                                             verb, cwd, env, timeout_sec, user,
                                             speculate=eligible)

        self._counts["resolves"] += 1
        reply = await self._cli(environment, "resolve", self._session,
                                _b64({"tool": kind, "args": args}))
        if reply.get("served"):
            output = reply.get("output")
            if (isinstance(output, (list, tuple)) and len(output) == 3
                    and isinstance(output[0], str) and isinstance(output[1], str)
                    and type(output[2]) is int):
                self._counts["hits"] += 1
                self._last_served = True
                stdout, stderr, rc = output
                return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)
        self._counts["misses"] += 1
        return await self._authoritative(environment, original_exec, command, kind, verb,
                                         cwd, env, timeout_sec, user)

    async def _authoritative(self, environment, original_exec, command, kind, verb,
                             cwd, env, timeout_sec, user, *, write_args=None,
                             speculate=True, prepared=None):
        if prepared is None:
            begin = await self._cli(environment, "mutation_begin", self._session,
                                    _b64({"write_args": write_args}))
            token = begin.get("mutation_id")
        else:
            token = prepared["token"]
        if not token:
            raise RuntimeError("sfx did not fence authoritative execution")
        self._counts["authoritative"] += 1
        t0 = time.monotonic()
        result = None
        primary_error = None
        try:
            if write_args is not None and prepared is None:
                await self._cli(environment, "feed", self._session,
                                _b64({"call_id": f"e{self._counts['writes_fed']}",
                                      "tool": "Edit", "body": json.dumps(write_args),
                                      "mutation_id": token}))
                self._counts["writes_fed"] += 1
            if prepared is not None:
                self._edit_event("authoritative_start", prepared["model_step"], command=command)
            self._tool_phase("authoritative_execution")
            tool = getattr(self, "_active_tool", None)
            if tool is not None:
                tool["authoritative_start_s"] = time.monotonic() - self._diagnostic_origin
            result = await original_exec(command=command, cwd=cwd, env=env,
                                         timeout_sec=timeout_sec, user=user)
            tool = getattr(self, "_active_tool", None)
            if tool is not None:
                tool["result"] = _diagnostic_result(result, private_preview=True)
                tool["authoritative_end_s"] = time.monotonic() - self._diagnostic_origin
        except BaseException as exc:
            primary_error = exc
            tool = getattr(self, "_active_tool", None)
            if tool is not None:
                tool["primary_error_phase"] = tool["phase"]
                if "authoritative_start_s" in tool:
                    tool["execution_error"] = _diagnostic_error(exc)
                    tool["authoritative_abort_s"] = time.monotonic() - self._diagnostic_origin
            raise
        finally:
            try:
                ended = await self._cli(environment, "mutation_end", self._session,
                                        _b64({"mutation_id": token,
                                              "success": result is not None and result.return_code == 0}))
                if prepared is not None:
                    self._edit_event("authoritative_end", prepared["model_step"],
                                     chain_preserved=bool(ended.get("chain_preserved")))
            except BaseException as exc:
                tool = getattr(self, "_active_tool", None)
                if tool is not None:
                    tool["mutation_end_error"] = _diagnostic_error(exc)
                if primary_error is None:
                    raise
        latency = (time.monotonic() - t0) * 1000.0
        if write_args is not None and ended.get("chain_preserved"):
            await self._cli(environment, "resolve", self._session,
                            _b64({"tool": kind, "args": write_args}))
        outcome = _status(result.return_code != 0, kind)
        await self._cli(environment, "report", self._session, _b64({
            "tool": kind, "verb": verb, "outcome": outcome,
            "args": {"cmd": command} if speculate else {}, "latency": latency,
            "observation": (result.stdout or "")[:OBS_CAP],
            "mutation_id": token, "speculate": speculate}))
        return result

    async def _write_and_chain(self, environment, original_exec, command, kind, verb,
                               cwd, env, timeout_sec, user):
        eligible = self._eligible_context(environment, cwd, env, timeout_sec, user)
        write_args = (_literal_write(command, self.repo)
                      if eligible and self.arm == "ON" and self.speculate_writes else None)
        prepared = self._pending_edit
        if prepared is not None:
            if prepared["confirmed"] and prepared["command"] == command and prepared["write"] == write_args:
                self._pending_edit = None
            else:
                await self._cancel_pending_edit(environment, "write_mismatch")
                prepared = None
        return await self._authoritative(environment, original_exec, command, kind,
                                         verb, cwd, env, timeout_sec, user,
                                         write_args=write_args, speculate=eligible, prepared=prepared)

    def _load_wrapped(self, environment):
        mod_name, attr = self.wrapped.split(":")
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, attr)
        if attr == "build_live_agent":
            async def on_stream_event(event):
                # Streamed candidates and the eventual authoritative command
                # use the same explicit workspace, including SWE's /testbed.
                await self._stream_event(environment, {**event, "_exec_cwd": self.repo})
            return factory(self.logs_dir, environment, artifacts=self.artifacts,
                           config={**self.live_config, "repo": self.repo,
                                   "_on_stream_event": on_stream_event})
        return factory(self.logs_dir, environment, artifacts=self.artifacts)

    def _diagnostic_state(self):
        if not hasattr(self, "_diagnostics"):
            self._diagnostics = {"tools": [], "control_failures": []}
            self._diagnostic_origin = time.monotonic()
        return self._diagnostics

    def _tool_phase(self, phase):
        tool = getattr(self, "_active_tool", None)
        if tool is not None:
            tool["phase"] = phase

    def _private_artifact(self, kind, extension):
        index = getattr(self, "_run_index", 1)
        suffix = f"-run-{index:04d}" if index > 1 else ""
        return self.logs_dir / f"sfx-private-{kind}-{self.arm}{suffix}.{extension}"

    async def _archive_daemon_log(self, environment, original_exec):
        diagnostics = self._diagnostic_state()
        if not callable(getattr(environment, "download_file", None)):
            diagnostics["daemon_log"] = {"status": "download_unsupported"}
            return
        target = self._private_artifact("daemon", "log")
        remote = f"/tmp/sfx-daemon-evidence-{uuid.uuid4().hex}.log"
        source = shlex.quote(DAEMON_LOG)
        try:
            # Shell tools still work after `killall python3`; no daemon/socket
            # request is involved. The separate file avoids downloading a FIFO
            # or an unbounded log, and umask keeps raw evidence private.
            result = await asyncio.wait_for(original_exec(
                command=f"umask 077; set -C; test ! -L {source} && test -f {source} "
                        f"&& tail -c {DAEMON_LOG_CAP} -- {source} >{shlex.quote(remote)} "
                        f"&& wc -c <{source}",
                user=environment.default_user, timeout_sec=10), timeout=12)
            if result.return_code != 0:
                diagnostics["daemon_log"] = {"status": "capture_failed",
                                              "result": _diagnostic_result(result)}
                return
            _write_private(target, b"")
            await asyncio.wait_for(environment.download_file(remote, str(target)), timeout=10)
            target.chmod(0o600)
            data = target.read_bytes()
            if len(data) > DAEMON_LOG_CAP:
                raise ValueError("daemon evidence exceeded its capture limit")
            count = (result.stdout or "").strip()
            diagnostics["daemon_log"] = {
                "status": "archived", "artifact": target.name,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "tail_limit_bytes": DAEMON_LOG_CAP,
                "observed_source_bytes": int(count) if count.isdecimal() else None}
        except BaseException as exc:
            diagnostics["daemon_log"] = {"status": "archive_failed",
                                          "error": _diagnostic_error(exc)}
            if not isinstance(exc, Exception):
                raise

    def _persist(self, context):
        run_index = getattr(self, "_run_index", 1)
        suffix = f"-run-{run_index:04d}" if run_index > 1 else ""
        artifact = self.logs_dir / f"sfx-live-{self.arm}{suffix}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        record = {"arm": self.arm, "depth": self.depth,
                  "scaffold": getattr(self, "live_config", {}).get("scaffold", "task"),
                  "speculate_writes": self.speculate_writes, "counts": self._counts,
                  "receipts": self._receipts, "raw": self._raw,
                  "wall_s": getattr(self, "_wall_s", None),
                  "lifecycle_wall_s": getattr(self, "_lifecycle_wall_s", None),
                  "run_index": run_index,
                  "instruction_sha256": getattr(self, "_instruction_sha256", None),
                  "failure": getattr(self, "_failure", None),
                  "cleanup_errors": getattr(self, "_cleanup_errors", []),
                  "completed": getattr(self, "_completed", False),
                  "env_id": getattr(self, "_env_id", None),
                  "session_id": self._session,
                  "initial_fs_hash": getattr(self, "_initial_fs_hash", None),
                  **getattr(self, "_snapshot", {})}
        record["early_edit_events"] = getattr(self, "_early_edit_events", [])
        trajectory = (context.metadata or {}).get("sfx_live_trajectory")
        if trajectory is not None:
            record["sfx_live_trajectory"] = trajectory
        diagnostics = self._diagnostic_state()
        private_path = self._private_artifact("diagnostics", "json")
        try:
            data = json.dumps(diagnostics, allow_nan=False).encode()
            _write_private(private_path, data)
            record["diagnostics"] = {
                "artifact": private_path.name, "sha256": hashlib.sha256(data).hexdigest(),
                "tool_attempts": len(diagnostics["tools"]),
                "control_failures": len(diagnostics["control_failures"]),
                "daemon_log": diagnostics.get("daemon_log")}
        except Exception as exc:
            record["diagnostics"] = {"status": "persist_failed", "error": _diagnostic_error(exc)}
        artifact.write_text(json.dumps(record))
        context.metadata = {**(context.metadata or {}), "sfx_live": record}
