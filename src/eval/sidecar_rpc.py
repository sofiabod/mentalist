"""Private controller-to-host RPC; every tool process runs through task exec."""
import asyncio
import base64
import json
import math
import os
from pathlib import Path
import shlex
import socket
import stat
import time

from eval.spec_worker import (
    ADMISSION_TYPES, MAX_MESSAGE_BYTES, NonReusableToolOutput, ProtocolViolation, absolute_path,
    strict_loads, validate_request, validate_response, validate_settings,
)


WORKER_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from eval.spec_worker import main; main(sys.argv[2:])"
)


class BrokerPoisoned(RuntimeError):
    pass


class SpeculationDeclined(ValueError):
    pass


def _positive(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return float(value)


def _encode(value):
    data = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolViolation("message too large")
    return data


class HostBroker:
    def __init__(self, socket_path, exec_fn, *, repo, scratch, source_dir,
                 python_executable="python3", script_contracts=None, fork_path_view="proot",
                 worker_timeout_s=120, drain_timeout_s=5, on_poison=None, poison_timeout_s=15):
        self.socket_path = Path(absolute_path(os.fspath(socket_path)))
        if not callable(exec_fn):
            raise ValueError("exec_fn must be the captured task executor")
        self.exec_fn = exec_fn
        self.settings = validate_settings({
            "repo": os.fspath(repo), "scratch": os.fspath(scratch),
            "script_contracts": [] if script_contracts is None else script_contracts,
            "fork_path_view": fork_path_view,
        })
        self.settings = strict_loads(_encode(self.settings))
        self.source_dir = absolute_path(os.fspath(source_dir))
        if (not isinstance(python_executable, str) or not python_executable
                or "\x00" in python_executable):
            raise ValueError("invalid task Python executable")
        self.python_executable = python_executable
        self.worker_timeout_s = _positive(worker_timeout_s)
        self.drain_timeout_s = _positive(drain_timeout_s)
        if on_poison is not None and not callable(on_poison):
            raise ValueError("on_poison must be the controller's task termination callback")
        self.on_poison = on_poison
        self.poison_timeout_s = _positive(poison_timeout_s)
        self.workers_quiesced = False
        self._termination_task = None
        self._termination_deadline = None
        self._server = None
        self._identity = None
        self._inflight = set()
        self._handlers = set()
        self._closed = False
        self.poison_reason = None

    @property
    def poisoned(self):
        return self.poison_reason is not None

    @property
    def inflight_count(self):
        return len(self._inflight)

    def _poison(self, reason):
        if self.poison_reason is None:
            self.poison_reason = reason

    def raise_if_poisoned(self):
        if self.poisoned:
            raise BrokerPoisoned(f"speculative worker state is uncertain: {self.poison_reason}")

    async def _quiesce_after_poison(self):
        if not self.poisoned or self.on_poison is None:
            return
        if self._termination_task is None:
            async def terminate():
                try:
                    self.workers_quiesced = (await self.on_poison()) is True
                except Exception:
                    self.workers_quiesced = False
            self._termination_deadline = time.monotonic() + self.poison_timeout_s
            self._termination_task = asyncio.create_task(terminate())
        try:
            await asyncio.wait_for(asyncio.shield(self._termination_task),
                                   max(0, self._termination_deadline - time.monotonic()))
        except Exception:
            pass

    async def start(self):
        if self._closed or self._server is not None:
            raise RuntimeError("broker cannot be started twice")
        if not self.socket_path.parent.is_dir() or os.path.lexists(self.socket_path):
            raise ValueError("broker requires an unused socket in an existing owned directory")
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path), limit=MAX_MESSAGE_BYTES + 1)
        os.chmod(self.socket_path, 0o600)
        info = self.socket_path.stat()
        self._identity = info.st_dev, info.st_ino
        return self

    def _worker_done(self, task):
        self._inflight.discard(task)
        if not task.cancelled():
            task.exception()

    async def _dispatch(self, request):
        if self._closed:
            self._poison("request_after_close")
        self.raise_if_poisoned()
        validate_request(request, self.settings["scratch"])
        payload = base64.b64encode(_encode({"request": request, "settings": self.settings})).decode("ascii")
        command = shlex.join([self.python_executable, "-c", WORKER_BOOTSTRAP, self.source_dir, payload])
        task = asyncio.create_task(self.exec_fn(
            command=command, cwd=self.settings["repo"],
            env=None, timeout_sec=self.worker_timeout_s))
        self._inflight.add(task)
        task.add_done_callback(self._worker_done)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), self.worker_timeout_s + 1)
        except asyncio.CancelledError:
            self._poison("dispatch_cancelled")
            if task.cancelled():
                raise BrokerPoisoned("speculative task worker was cancelled") from None
            raise
        except Exception:
            self._poison("worker_transport_failure")
            raise BrokerPoisoned("speculative task worker did not complete reliably") from None
        if (type(result.return_code) is not int or result.return_code != 0
                or not isinstance(result.stdout, str) or result.stderr not in (None, "")):
            self._poison("worker_exit_or_channel_failure")
            raise BrokerPoisoned("speculative task worker returned an invalid envelope")
        try:
            response = validate_response(strict_loads(result.stdout), request["op"])
        except Exception:
            self._poison("malformed_worker_response")
            raise BrokerPoisoned("speculative task worker returned malformed data") from None
        if not response["ok"] and response["error"]["category"] not in {"admission", "non_reusable"}:
            self._poison("worker_reported_failure")
        return response

    async def _handle(self, reader, writer):
        handler = asyncio.current_task()
        self._handlers.add(handler)
        try:
            try:
                line = await asyncio.wait_for(reader.readline(), self.worker_timeout_s + 1)
                if not line.endswith(b"\n"):
                    raise ProtocolViolation("incomplete request")
                request = strict_loads(line)
                response = await self._dispatch(request)
            except asyncio.CancelledError:
                if self.inflight_count:
                    self._poison("request_cancelled_with_worker")
                raise
            except BrokerPoisoned:
                response = {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}
            except Exception:
                self._poison("malformed_or_failed_request")
                response = {"ok": False, "error": {"category": "protocol", "type": "ProtocolViolation"}}
            else:
                if self.poisoned and response["ok"]:
                    response = {"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}}
            await self._quiesce_after_poison()
            writer.write(_encode(response) + b"\n")
            await writer.drain()
        except asyncio.CancelledError:
            self._poison("controller_handler_cancelled")
            raise
        except (OSError, ProtocolViolation):
            self._poison("controller_reply_lost")
            await self._quiesce_after_poison()
        finally:
            writer.close()
            if handler.cancelling():
                writer.transport.abort()
            try:
                await writer.wait_closed()
            except asyncio.CancelledError:
                writer.transport.abort()
                raise
            except OSError:
                pass
            finally:
                self._handlers.discard(handler)

    async def aclose(self):
        self._closed = True
        deadline = time.monotonic() + self.drain_timeout_s
        if self._server is not None:
            self._server.close()
        if self._inflight:
            _, pending = await asyncio.wait(tuple(self._inflight), timeout=max(0, deadline - time.monotonic()))
            if pending:
                self._poison("worker_drain_timeout")
        handlers = tuple(self._handlers)
        pending_handlers = set()
        if handlers:
            _, pending_handlers = await asyncio.wait(handlers, timeout=max(0, deadline - time.monotonic()))
        for handler in pending_handlers:
            handler.cancel()
        if pending_handlers:
            await asyncio.gather(*pending_handlers, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
        await self._quiesce_after_poison()
        if self._identity is not None:
            try:
                info = self.socket_path.lstat()
                if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self._identity:
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass
        self.raise_if_poisoned()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


class SidecarClient:
    def __init__(self, socket_path, *, timeout_s=140):
        self.socket_path = absolute_path(os.fspath(socket_path))
        self.timeout_s = _positive(timeout_s)

    def _request(self, request):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout_s)
            try:
                channel.connect(self.socket_path)
                channel.sendall(_encode(request) + b"\n")
                with channel.makefile("rb") as stream:
                    data = stream.readline(MAX_MESSAGE_BYTES + 2)
                if not data.endswith(b"\n"):
                    raise ProtocolViolation("incomplete broker response")
                response = validate_response(strict_loads(data), request["op"])
            except Exception:
                raise BrokerPoisoned("controller could not obtain a complete worker result") from None
        if not response["ok"]:
            error = response["error"]
            if error["category"] == "admission":
                exception = ADMISSION_TYPES.get(error["type"], SpeculationDeclined)
                raise exception("speculation declined by the task execution guard")
            if error["category"] == "non_reusable":
                raise NonReusableToolOutput("completed speculation produced non-reusable mixed output streams")
            raise BrokerPoisoned("host broker rejected uncertain speculative execution")
        return response["result"]

    def run(self, kind, args):
        started = time.monotonic()
        result = self._request({"op": "run", "kind": kind, "args": args})
        return tuple(result["output"]), (time.monotonic() - started) * 1000

    def run_in_fork(self, path, hop):
        result = self._request({"op": "run_in_fork", "path": os.fspath(path), "hop": hop})
        return tuple(result["output"]), result["status"]
