"""Docker-only live adapter with a controller outside the task PID namespace."""
import asyncio
import hashlib
import json
from pathlib import Path
import re
import shlex
import tempfile
import time

from eval.sfx_live_agent import (
    DAEMON_LOG_CAP, SFXLiveAgent, _diagnostic_error, _repo_root, _write_private,
)


async def task_identity(environment, repo, scratch):
    from harbor.environments.docker.docker import DockerEnvironment

    if not isinstance(environment, DockerEnvironment):
        raise ValueError("The sidecar controller requires the local Harbor Docker backend")
    if environment.default_user not in (None, "", "root", "0", 0):
        raise ValueError("The first sidecar adapter requires a root task user")
    expected = {}
    for mount in environment._mounts:
        if mount.get("target") not in (repo, scratch):
            continue
        if mount.get("type") != "bind" or mount.get("read_only") or mount["target"] in expected:
            raise ValueError("Repository and scratch need distinct writable bind mounts")
        expected[mount["target"]] = str(Path(mount["source"]).resolve(strict=True))
    if set(expected) != {repo, scratch}:
        raise ValueError("Declare repository and scratch bind mounts in the Harbor job")
    result = await environment._run_docker_compose_command(
        ["ps", "-q", "main"], timeout_sec=10)
    container_id = (result.stdout or "").strip()
    if result.return_code or not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise RuntimeError("Could not resolve exactly one task container")
    return container_id, expected


class SidecarSFXAgent(SFXLiveAgent):
    def __init__(self, *args, **kwargs):
        fixed = {"source_dir": "/root/src", "socket_path": "/sfx-control/daemon.sock",
                 "trace_path": "/logs/agent/sfx-trace.jsonl", "control_transport": "exec",
                 "python_executable": "python3"}
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"Sidecar controller requires {key}={value}")
            kwargs[key] = value
        if kwargs.get("control_exec") is not None:
            raise ValueError("Sidecar controller owns the control transport")
        self._controller = None
        self._sidecar_poisoned = False
        super().__init__(*args, **kwargs)

    async def _launch_daemon(self, environment, table_file):
        await task_identity(environment, self.repo, self.scratch_dir)

    async def _route_one(self, *args, **kwargs):
        self._controller.raise_if_poisoned()
        result = await super()._route_one(*args, **kwargs)
        self._controller.raise_if_poisoned()
        return result

    async def _stream_event(self, environment, event):
        if event["event"] != "model_abort":
            self._controller.raise_if_poisoned()
        await super()._stream_event(environment, event)

    async def _archive_daemon_log(self, environment, original_exec):
        diagnostics = self._diagnostic_state()
        try:
            source = self._sidecar_root / "daemon.log"
            target = self._private_artifact("daemon", "log")
            with source.open("rb") as log:
                log.seek(0, 2)
                count = log.tell()
                log.seek(max(0, count - DAEMON_LOG_CAP))
                data = log.read(DAEMON_LOG_CAP)
            _write_private(target, data)
            diagnostics["daemon_log"] = {
                "status": "archived", "artifact": target.name, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "tail_limit_bytes": DAEMON_LOG_CAP, "observed_source_bytes": count}
        except OSError as exc:
            diagnostics["daemon_log"] = {"status": "archive_failed",
                                          "error": _diagnostic_error(exc)}

    async def _after_cleanup_hash(self, original_exec, environment):
        result = await asyncio.wait_for(original_exec(
            command=(f"PYTHONPATH={shlex.quote(self.source_dir)} python3 -m eval.sfx_client_cli "
                     f"snapshot {shlex.quote(self._session)} {shlex.quote(self.repo)} "
                     "/tmp/sfx-sidecar-no-trace"),
            user=environment.default_user, timeout_sec=15), timeout=17)
        if result.return_code:
            raise RuntimeError("Post-controller filesystem snapshot failed")
        value = json.loads((result.stdout or "").strip())
        digest = value.get("final_fs_hash")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise RuntimeError("Invalid post-controller filesystem snapshot")
        if digest != self._snapshot.get("final_fs_hash"):
            raise RuntimeError("Filesystem changed during controller cleanup")
        trace = self._sidecar_root / "logs/sfx-trace.jsonl"
        complete_trace = ([json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
                          if trace.exists() else [])
        previous = self._snapshot.get("trace", [])
        if complete_trace[:len(previous)] != previous:
            raise RuntimeError("Controller trace changed during cleanup")
        self._snapshot["trace"] = complete_trace
        return digest

    async def run(self, instruction, environment, context):
        from eval.sidecar_runtime import Controller

        context.metadata = {key: value for key, value in (context.metadata or {}).items()
                            if key not in ("sfx_live", "sfx_live_trajectory")}
        if self._sidecar_poisoned:
            raise RuntimeError("Prior controller failure invalidated this trial; no later model requests")
        if self._controller is not None:
            raise RuntimeError("Concurrent sidecar runs on one agent are unsupported")
        started = time.monotonic()
        container_id, expected = await task_identity(environment, self.repo, self.scratch_dir)
        self._sidecar_root = Path(tempfile.mkdtemp(prefix="sfx-ctl-", dir="/tmp"))
        original_exec = environment.exec
        controller = Controller(
            container_id, original_exec, repo=self.repo, scratch=self.scratch_dir,
            source_root=_repo_root(), control_root=self._sidecar_root, depth=self.depth,
            table=self.table, script_contracts=self.script_contracts.definitions,
            fork_path_view=self.fork_path_view, expected_mounts=expected)
        self._controller = controller
        self._control_exec = controller.exec
        primary = None
        cleanup = None
        entered = False
        after_hash = None
        try:
            await controller.start()
            entered = True
            await super().run(instruction, environment, context)
            controller.raise_if_poisoned()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                await controller.aclose()
                controller.raise_if_poisoned()
                if entered and self._snapshot.get("final_fs_hash"):
                    after_hash = await self._after_cleanup_hash(original_exec, environment)
            except BaseException as exc:
                cleanup = exc
            self._sidecar_poisoned = bool(cleanup) or bool(getattr(controller, "poisoned", False))
            self._controller = None
            self._control_exec = None
            if entered:
                if cleanup is not None:
                    self._completed = False
                    self._cleanup_errors.append({"phase": "sidecar_cleanup",
                                                 "type": type(cleanup).__name__})
                    if primary is None:
                        self._failure = {"type": type(cleanup).__name__, "message": str(cleanup)}
                self._lifecycle_wall_s = time.monotonic() - started
                self._snapshot.update(controller="isolated_docker_sidecar",
                                      controller_lifecycle=controller.report,
                                      after_cleanup_fs_hash=after_hash)
                self._persist(context)
            if primary is None and cleanup is not None:
                raise cleanup
