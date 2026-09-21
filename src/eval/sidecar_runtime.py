"""Host-owned, checkpoint-scoped controller lifecycle for Docker Harbor tasks.

Only the host broker has Docker authority. The controller has a private PID
namespace and never executes task commands. This protects controller liveness;
brokered speculation still shares the task's process/global state, not a new
security boundary for arbitrary speculative side effects.
"""
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import time
import uuid

from eval.control_transport import InSandboxControlTransport
from sfx.script_contracts import ScriptContracts, ScriptInvocationRejected


SOURCE = "/root/src"
SOCKET = "/sfx-control/daemon.sock"
TRACE = "/logs/agent/sfx-trace.jsonl"
LABEL = "sfx.controller.token"
CONTROLLER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _bootstrap(module):
    # Module names are constants chosen by this module, never task input.
    return (f"import runpy,sys;sys.path.insert(0,{SOURCE!r});"
            f"sys.argv=[{module!r},*sys.argv[1:]];runpy.run_module({module!r},run_name='__main__')")


class ControllerError(RuntimeError):
    pass


@dataclass
class Result:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _overlap(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _path(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or path in map(Path, ("/", "/root", "/home", "/tmp", "/var", "/etc", "/proc", "/sys", "/dev"))):
        raise ControllerError("unbounded or aliased host path")
    return path


class Controller:
    def __init__(self, task_container_id, exec_fn, *, repo, scratch, source_root,
                 control_root, depth, table, script_contracts, fork_path_view,
                 expected_mounts):
        if not isinstance(task_container_id, str) or not re.fullmatch(r"[0-9a-f]{64}", task_container_id):
            raise ControllerError("an exact Docker task container ID is required")
        if type(depth) is not int or depth < 1 or not re.fullmatch(r"[\w-]+(?:\.json)?", table):
            raise ControllerError("invalid controller depth or table")
        if fork_path_view not in ("cwd", "proot"):
            raise ControllerError("invalid fork path view")
        targets = [PurePosixPath(value) for value in (repo, scratch)]
        reserved = [PurePosixPath(value) for value in ("/root/src", "/root/data", "/sfx-control", "/logs/agent")]
        if any(not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/")
               or any(_overlap(path, other) for other in reserved) for path in targets):
            raise ControllerError("unsafe task mount target")
        if _overlap(*targets) or set(expected_mounts) != {repo, scratch}:
            raise ControllerError("exact disjoint repo and scratch bindings are required")
        self.task_id, self.exec_fn = task_container_id, exec_fn
        self.repo, self.scratch = repo, scratch
        self.source_root, self.control_root = _path(source_root), _path(control_root)
        self.expected_mounts = {target: _path(source) for target, source in expected_mounts.items()}
        paths = [*self.expected_mounts.values(), self.source_root, self.control_root]
        if any(not path.is_dir() for path in paths) or any(
                _overlap(left, right) for i, left in enumerate(paths) for right in paths[i + 1:]):
            raise ControllerError("host workspace, code and control paths must be disjoint directories")
        if any(self.control_root.iterdir()) or len(os.fsencode(self.control_root / "broker.sock")) >= 104:
            raise ControllerError("control root must be empty and support a short Unix socket path")
        if stat.S_IMODE(self.control_root.stat().st_mode) & 0o077:
            raise ControllerError("control root must be private to its owner")
        control_info = self.control_root.stat()
        self._control_root_identity = (control_info.st_dev, control_info.st_ino)
        self._control_socket_identity = None
        self._control_channel = None
        self._control_lock = asyncio.Lock()
        for name in ("src", "data"):
            if _path(self.source_root / name) != self.source_root / name:
                raise ControllerError("aliased controller source")
        self.depth, self.table = depth, table
        self.contracts = ScriptContracts(script_contracts).definitions
        self.fork_path_view = fork_path_view
        self.token = uuid.uuid4().hex
        self.name = "sfx-controller-" + self.token
        self.sidecar_id = None
        self.broker = None
        self._identity = None
        self._closed = False
        self._started = False
        self._poisoned = False
        self._uncertain = False
        self._task_remove_lock = asyncio.Lock()
        self.report = {"runtime": "docker_controller_sidecar_v1", "task_container_id": self.task_id,
                       "control_transport": "persistent_unix_v1",
                       "started": False, "closed": False, "valid": False, "poisoned": False,
                       "task_removed": False, "errors": []}

    @property
    def poisoned(self):
        return self._poisoned or self.broker is not None and self.broker.poisoned

    async def _docker(self, *argv, timeout=15):
        process = await asyncio.create_subprocess_exec(
            "docker", *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return Result(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    @staticmethod
    def _checked(result, operation):
        if result.return_code:
            raise ControllerError(f"Docker {operation} failed (exit code {result.return_code})")
        return result

    async def _inspect(self, target, *, missing_ok=False):
        result = await self._docker("inspect", target)
        if result.return_code:
            # Only an explicit not-found response proves that an owned object
            # is gone. An unavailable Docker daemon is not successful cleanup.
            if missing_ok and re.search(r"no such (?:object|container):", result.stderr, re.IGNORECASE):
                return None
            self._checked(result, "inspect")
        values = json.loads(result.stdout)
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise ControllerError("invalid Docker inspection")
        return values[0]

    def _task_identity(self, info, *, running):
        config, host = info.get("Config", {}), info.get("HostConfig", {})
        if (info.get("Id") != self.task_id or not re.fullmatch(r"sha256:[0-9a-f]{64}", info.get("Image", ""))
                or config.get("User", "") not in ("", "root", "0", "0:0", "root:root")
                or host.get("Privileged", False) or host.get("PidMode", "") not in ("", "private")
                or host.get("CapAdd") or host.get("UsernsMode", "") not in ("", "private")):
            raise ControllerError("task requires an unprivileged private-PID root container")
        from eval.sfx_daemon_run import _validate_tool_environment

        environment = config.get("Env")
        if environment is not None:
            if not isinstance(environment, list):
                raise ControllerError("invalid task environment")
            for entry in environment:
                if (not isinstance(entry, str) or "=" not in entry
                        or not entry.split("=", 1)[0] or "\x00" in entry):
                    raise ControllerError("invalid task environment")
                name, value = entry.split("=", 1)
                try:
                    _validate_tool_environment({name: value})
                except ScriptInvocationRejected:
                    raise ControllerError("task shell environment contains unsupported startup or function hooks") from None
        state = info.get("State", {})
        if running and (state.get("Running") is not True or state.get("Paused") or not state.get("Pid")):
            raise ControllerError("verified task container is not running")
        mounts = info.get("Mounts", [])
        matched = {}
        for mount in mounts:
            source = _path(mount["Source"])
            if (any(_overlap(source, private) for private in (self.source_root, self.control_root))
                    or stat.S_ISSOCK(source.stat().st_mode)
                    or mount.get("Propagation", "rprivate") not in ("", "private", "rprivate")):
                raise ControllerError("task mount exposes controller/code or unsafe host state")
            target = mount["Destination"]
            if target in self.expected_mounts:
                if (target in matched or mount.get("Type") != "bind" or mount.get("RW") is not True
                        or source != self.expected_mounts[target]):
                    raise ControllerError("task binding differs from the explicit Harbor mount")
                matched[target] = str(source)
            elif any(_overlap(source, expected) for expected in self.expected_mounts.values()):
                raise ControllerError("additional task mount aliases a workspace binding")
        if set(matched) != set(self.expected_mounts):
            raise ControllerError("task is missing an explicitly bound workspace")
        # Docker constructs this list from a mount map: identical containers
        # can expose different list orders on successive inspect calls.
        canonical_mounts = sorted(mounts, key=lambda mount: json.dumps(mount, sort_keys=True))
        identity = {"Id": info["Id"], "Image": info["Image"], "Config": config,
                    "HostConfig": host, "Mounts": canonical_mounts,
                    "StartedAt": state.get("StartedAt")}
        # Only a digest is retained in ordinary lifecycle evidence, not Env.
        return _digest(identity)

    async def _verify_task(self, *, running=True):
        info = await self._inspect(self.task_id)
        identity = self._task_identity(info, running=running)
        if self._identity is not None and identity != self._identity:
            raise ControllerError("task container identity changed")
        return info, identity

    def _error(self, phase, exc, *, uncertain=False):
        self._poisoned = True
        self._uncertain |= uncertain
        self.report.update(valid=False, poisoned=True)
        self.report["errors"].append({"phase": phase, "type": type(exc).__name__})
        # Exception text can contain private task/transport data. Keep it out
        # of the public lifecycle report, but retain bounded owner-only
        # evidence so a failed lifecycle is diagnosable without another run.
        try:
            message = str(exc).encode("utf-8", errors="replace")
            row = {"phase": phase, "type": type(exc).__name__,
                   "message": message[:8192].decode("utf-8", errors="replace"),
                   "message_bytes": len(message),
                   "message_sha256": hashlib.sha256(message).hexdigest(),
                   "message_truncated": len(message) > 8192}
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
            with os.fdopen(os.open(self.control_root / "controller-errors.jsonl", flags, 0o600), "w") as log:
                os.fchmod(log.fileno(), 0o600)
                log.write(json.dumps(row) + "\n")
        except Exception:
            # Secondary evidence-write failure must not replace the original
            # failure or reverse the sticky invalidation above.
            pass

    async def _task_exec(self, **kwargs):
        try:
            await self._verify_task()
            result = await self.exec_fn(**kwargs)
            await self._verify_task()
            return result
        except BaseException as exc:
            self._error("task_worker", exc, uncertain=True)
            raise

    async def start(self):
        if self._started or self._closed:
            raise ControllerError("controller lifecycle cannot be reused")
        try:
            info, self._identity = await self._verify_task()
            self.report["task_identity_sha256"] = self._identity
            self.report["image"] = info["Image"]
            (self.control_root / "logs").mkdir(mode=0o700)
            config = {"token": self.token, "repo": self.repo, "scratch": self.scratch,
                      "depth": self.depth, "table": self.table,
                      "script_contracts": self.contracts, "fork_path_view": self.fork_path_view}
            config_path = self.control_root / "config.json"
            config_path.write_text(json.dumps(config))
            config_path.chmod(0o600)
            from eval.sidecar_rpc import HostBroker

            self.broker = HostBroker(self.control_root / "broker.sock", self._task_exec,
                                    repo=self.repo, scratch=self.scratch, source_dir=SOURCE,
                                    script_contracts=self.contracts, fork_path_view=self.fork_path_view,
                                    on_poison=self._on_broker_poison)
            await self.broker.start()
            argv = ["create", "--name", self.name, "--label", f"{LABEL}={self.token}",
                    "--label", f"sfx.controller.task={self.task_id}", "--network", "none",
                    "--runtime", "runc", "--no-healthcheck",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--cpus", "1", "--memory", "512m", "--pids-limit", "128", "--user", "0",
                    "--workdir", "/", "--env", "PATH=" + CONTROLLER_PATH,
                    "--entrypoint", "python3"]
            bindings = [(self.expected_mounts[self.repo], self.repo, True),
                        (self.expected_mounts[self.scratch], self.scratch, False),
                        (self.source_root / "src", SOURCE, True),
                        (self.source_root / "data", "/root/data", True),
                        (self.control_root, "/sfx-control", False),
                        (self.control_root / "logs", "/logs/agent", False)]
            for source, target, readonly in bindings:
                if "," in str(source) + target:
                    raise ControllerError("unsupported Docker bind path")
                argv += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
            created = self._checked(await self._docker(*argv, info["Image"], "-I", "-S", "-B", "-u", "-c",
                                                      _bootstrap("eval.sidecar_daemon"),
                                                      "/sfx-control/config.json", timeout=30), "create")
            self.sidecar_id = created.stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", self.sidecar_id):
                raise ControllerError("Docker did not return an exact owned controller ID")
            self.report["sidecar_id"] = self.sidecar_id
            await self._owned_sidecar()
            self._checked(await self._docker("start", self.sidecar_id), "start")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = await self._owned_sidecar()
                if not state["State"]["Running"]:
                    raise ControllerError("controller exited during startup")
                ready = self.control_root / "ready.json"
                if ready.is_file() and json.loads(ready.read_text()).get("token") == self.token:
                    break
                await asyncio.sleep(0.1)
            else:
                raise ControllerError("controller startup deadline exceeded")
            await self._verify_task()
            self._verify_control_socket()
            self.raise_if_poisoned()
            self._started = True
            self.report.update(started=True, valid=True)
            return self
        except BaseException as exc:
            self._error("start", exc)
            try:
                await self.aclose()
            except BaseException:
                pass
            raise

    async def _owned_sidecar(self, *, missing_ok=False):
        info = await self._inspect(self.sidecar_id or self.name, missing_ok=missing_ok)
        if info is None:
            return None
        labels = info.get("Config", {}).get("Labels") or {}
        if (labels.get(LABEL) != self.token or labels.get("sfx.controller.task") != self.task_id
                or info.get("Image") != self.report.get("image")
                or not re.fullmatch(r"[0-9a-f]{64}", info.get("Id", ""))
                or self.sidecar_id is not None and info["Id"] != self.sidecar_id):
            raise ControllerError("controller ownership identity mismatch")
        self.sidecar_id = info["Id"]
        return info

    def raise_if_poisoned(self):
        marker = self.control_root / "uncertain.json"
        if marker.is_file() and json.loads(marker.read_text()).get("token") == self.token:
            self._error("retained_fork", ControllerError("task worker completion is uncertain"), uncertain=True)
        if self.broker is not None:
            try:
                self.broker.raise_if_poisoned()
            except BaseException as exc:
                self._error("broker", exc, uncertain=True)
        if self._poisoned:
            raise ControllerError("controller lifecycle is invalid; no fallback is permitted")

    def _control_argv(self, command):
        words = shlex.split(command)
        prefix = ["PYTHONPATH=" + SOURCE, "SFX_SOCKET=" + SOCKET,
                  "python3", "-m", "eval.sfx_client_cli"]
        if words[:5] != prefix or len(words) < 7:
            raise ControllerError("controller accepts only its fixed SFX control CLI")
        argv = words[5:]
        op, session = argv[:2]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session):
            raise ControllerError("invalid controller session identity")
        if op == "snapshot":
            valid = argv[2:] == [self.repo, TRACE]
        elif op == "begin":
            valid = len(argv) in (5, 6) and argv[2:4] == [self.repo, self.scratch] and argv[4] in ("0", "1")
        elif op == "end":
            valid = len(argv) == 2
        else:
            valid = op in {"resolve", "report", "feed", "mutation_begin", "mutation_end"} and len(argv) == 3
            if valid:
                valid = len(argv[2]) <= 2 * 1024 * 1024 and isinstance(
                    json.loads(base64.b64decode(argv[2], validate=True)), dict)
        if not valid:
            raise ControllerError("invalid scoped SFX control arguments")
        return argv

    def _verify_control_socket(self):
        root = self.control_root.lstat()
        if (not stat.S_ISDIR(root.st_mode) or stat.S_IMODE(root.st_mode) & 0o077
                or (root.st_dev, root.st_ino) != self._control_root_identity
                or self.control_root.resolve(strict=True) != self.control_root):
            raise ControllerError("private controller directory identity changed")
        ready = self.control_root / "ready.json"
        if (not stat.S_ISREG(ready.lstat().st_mode)
                or json.loads(ready.read_text()).get("token") != self.token):
            raise ControllerError("controller ready identity changed")
        socket = self.control_root / "daemon.sock"
        info = socket.lstat()
        identity = (info.st_dev, info.st_ino)
        if (not stat.S_ISSOCK(info.st_mode)
                or self._control_socket_identity is not None and identity != self._control_socket_identity):
            raise ControllerError("controller socket identity changed")
        self._control_socket_identity = identity

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self._started or self._closed or cwd is not None or env or user not in (None, "root", "0", 0):
            raise ControllerError("unsupported controller execution context")
        argv = self._control_argv(command)
        async with self._control_lock:
            if self._closed:
                raise ControllerError("controller control transport is closed")
            self.raise_if_poisoned()
            try:
                timeout = 30 if timeout_sec is None else min(30, timeout_sec)
                self._verify_control_socket()
                if argv[0] == "snapshot":
                    result = await self._docker("exec", "--user", "0", "--workdir", "/",
                                                "--env", "PATH=" + CONTROLLER_PATH,
                                                "--env", "SFX_SOCKET=" + SOCKET,
                                                self.sidecar_id, "python3", "-I", "-S", "-B", "-c",
                                                _bootstrap("eval.sfx_client_cli"), *argv, timeout=timeout)
                else:
                    if self._control_channel is None:
                        self._control_channel = InSandboxControlTransport(self.control_root / "daemon.sock")
                    reply = await asyncio.wait_for(self._control_channel.request(*argv), timeout=timeout)
                    result = Result(json.dumps(reply) + "\n")
                self._verify_control_socket()
                self.raise_if_poisoned()
                if result.return_code:
                    self._error("control_exit", ControllerError("control CLI failed"))
                return result
            except BaseException as exc:
                self._error("control_transport", exc, uncertain=True)
                raise

    async def _close_control(self):
        async def drain():
            async with self._control_lock:
                if self._control_channel is not None:
                    await self._control_channel.aclose()

        task = asyncio.create_task(drain())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
                except BaseException:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _remove_uncertain_task(self):
        async with self._task_remove_lock:
            if self.report["task_removed"]:
                return
            info = await self._inspect(self.task_id, missing_ok=True)
            if info is not None:
                if self._task_identity(info, running=False) != self._identity:
                    raise ControllerError("refusing to remove a changed task identity")
                self._checked(await self._docker("rm", "--force", self.task_id, timeout=20), "task removal")
            if await self._inspect(self.task_id, missing_ok=True) is not None:
                raise ControllerError("owned task removal was not verified")
            from eval.sidecar_daemon import _atomic_json

            _atomic_json(self.control_root / "task-terminated.json", {"token": self.token})
            self.report["task_removed"] = True

    async def _on_broker_poison(self):
        self._error("broker_poison", ControllerError("worker completion is uncertain"), uncertain=True)
        await self._remove_uncertain_task()
        return True

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                await self._close_control()
            except BaseException as exc:
                self._error("control_close", exc, uncertain=True)
            try:
                sidecar = await self._owned_sidecar(missing_ok=True)
                if self._started and (sidecar is None or not sidecar["State"]["Running"]):
                    self._error("controller_exited", ControllerError("controller exited unexpectedly"),
                                uncertain=bool(self.broker and self.broker.inflight_count))
                if sidecar is not None and sidecar["State"]["Running"]:
                    # TERM stops admission and cancels queued work. Do not use
                    # `docker stop`: its forced kill could delete forks before
                    # their task-side workers have stopped accessing them.
                    self._checked(await self._docker("kill", "--signal", "TERM", self.sidecar_id), "quiesce")
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        sidecar = await self._owned_sidecar()
                        if not sidecar["State"]["Running"]:
                            break
                        await asyncio.sleep(0.1)
                    else:
                        raise ControllerError("controller workers did not quiesce before the deadline")
                    if sidecar["State"].get("ExitCode", 0) != 0:
                        raise ControllerError("controller did not exit cleanly")
                self.report["controller_quiesced"] = True
            except BaseException as exc:
                self._error("controller_quiesce", exc, uncertain=True)
            if self._uncertain and self._identity is not None:
                try:
                    await self._remove_uncertain_task()
                except BaseException as exc:
                    self._error("task_removal", exc, uncertain=True)
            if self.broker is not None:
                try:
                    await asyncio.wait_for(self.broker.aclose(), timeout=10)
                    self.broker.raise_if_poisoned()
                    if self.broker.inflight_count:
                        raise ControllerError("broker still has in-flight task execution")
                    self.report["broker_drained"] = True
                except BaseException as exc:
                    self._error("broker_drain", exc, uncertain=True)
            if self._uncertain and self._identity is not None and not self.report["task_removed"]:
                try:
                    await self._remove_uncertain_task()
                except BaseException as exc:
                    self._error("task_removal", exc, uncertain=True)
            try:
                sidecar = await self._owned_sidecar(missing_ok=True)
                if sidecar is not None:
                    if self._uncertain and not self.report["task_removed"]:
                        raise ControllerError("retaining controller forks until task termination is verified")
                    self._checked(await self._docker("rm", "--force", self.sidecar_id, timeout=20), "controller removal")
                    if await self._owned_sidecar(missing_ok=True) is not None:
                        raise ControllerError("owned controller removal was not verified")
                self.report["controller_removed"] = True
            except BaseException as exc:
                self._error("controller_removal", exc)
        finally:
            self.report.update(closed=True, valid=self._started and not self._poisoned,
                               poisoned=self._poisoned)
        self.raise_if_poisoned()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self.aclose()
        except BaseException:
            if exc_type is None:
                raise
        return False
