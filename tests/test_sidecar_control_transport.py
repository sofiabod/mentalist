"""Real Unix-socket control with fake Docker lifecycle; no model or container."""
import asyncio
import base64
import json
import socket
import threading

import pytest

from adapters import protocol
from eval import control_transport, sidecar_runtime as runtime
from test_sidecar_runtime import SIDECAR, TASK, control, setup


def payload(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def test_socket_is_reused_and_only_snapshots_execute_inside_controller(setup, monkeypatch):
    controller, docker = setup
    clients = []
    client_type = protocol.Client

    def connect(path):
        client = client_type(path)
        clients.append(client)
        return client

    monkeypatch.setattr(control_transport, "Client", connect)

    async def run():
        await controller.start()
        begin = await controller.exec(control("begin", "/app", "/tmp/sfx-scratch", "0"))
        assert json.loads(begin.stdout) == {"ok": True}
        await controller.exec(control("mutation_begin", payload({})))
        await controller.exec(control("feed", payload({"call_id": "1", "tool": "edit", "body": "literal"})))
        await controller.exec(control("mutation_end", payload({"mutation_id": 1, "success": True})))
        await controller.exec(control("resolve", payload({"tool": "read", "args": {"cmd": "literal read"}})))
        await controller.exec(control("report", payload({"tool": "read", "verb": "free",
            "outcome": "miss", "args": {"cmd": "literal read"}, "latency": 0.1,
            "observation": "PRIVATE OBSERVATION"})))
        assert len(clients) == 1
        await controller.exec(control("end"))
        docker.control_result = runtime.Result('{"final_fs_hash": "container-digest", "trace": []}')
        snapshot = await controller.exec(control("snapshot", "/app", runtime.TRACE))
        assert json.loads(snapshot.stdout)["final_fs_hash"] == "container-digest"
        assert len(clients) == 1
        await controller.aclose()
        assert clients[0].sock.fileno() == -1

    asyncio.run(run())
    assert [row["type"] for row in docker.control_messages] == [
        "turn_begin", "mutation_begin", "feed", "mutation_end", "resolve", "call_executed",
        "turn_end", "session_end"]
    assert all(row["session"] == "session" for row in docker.control_messages)
    assert docker.control_messages[0]["repo"] == "/app"
    assert docker.control_messages[0]["scratch"] == "/tmp/sfx-scratch"
    assert docker.control_messages[5]["observation"] == "PRIVATE OBSERVATION"
    invocations = [row for row in docker.events if row[0] == "exec"]
    assert len(invocations) == 1
    assert invocations[0][-4:] == ("snapshot", "session", "/app", runtime.TRACE)
    assert invocations[0][invocations[0].index(SIDECAR) + 1:][:5] == ("python3", "-I", "-S", "-B", "-c")
    assert "PRIVATE OBSERVATION" not in json.dumps(controller.report)
    assert controller.report["valid"] and docker.task is not None


@pytest.mark.parametrize("disabled", ["0", "1"])
def test_off_and_enabled_sessions_use_identical_socket_transport(setup, disabled):
    controller, docker = setup

    async def run():
        await controller.start()
        await controller.exec(control("begin", "/app", "/tmp/sfx-scratch", disabled))
        await controller.exec(control("end"))
        await controller.aclose()

    asyncio.run(run())
    assert docker.control_messages[0]["spec_disabled"] is (disabled == "1")
    assert not any(row[0] == "exec" for row in docker.events)


@pytest.mark.parametrize("change", ["token", "directory_mode", "regular_file", "symlink", "socket"])
def test_socket_or_owner_identity_change_never_reconnects_or_falls_back(setup, change):
    controller, docker = setup
    replacement = None

    async def run():
        nonlocal replacement
        await controller.start()
        await controller.exec(control("begin", "/app", "/tmp/sfx-scratch", "0"))
        path = controller.control_root / "daemon.sock"
        if change == "token":
            (controller.control_root / "ready.json").write_text('{"token":"different-owner"}')
        elif change == "directory_mode":
            controller.control_root.chmod(0o755)
        else:
            path.unlink()
            if change == "regular_file":
                path.write_text("not a socket")
            elif change == "symlink":
                path.symlink_to(controller.control_root / "ready.json")
            else:
                replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                replacement.bind(str(path))
        with pytest.raises(runtime.ControllerError, match="identity changed"):
            await controller.exec(control("end"))
        assert controller.poisoned and controller._uncertain
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    try:
        asyncio.run(run())
    finally:
        if replacement is not None:
            replacement.close()
    assert len(docker.control_messages) == 1
    assert not any(row[0] == "exec" for row in docker.events)
    assert [row[-1] for row in docker.events if row[0] == "rm"] == [TASK, SIDECAR]


@pytest.mark.parametrize("reply", [None, [], {"error": "PRIVATE daemon rejection"}])
def test_rejection_poison_is_sticky_without_retry_or_exec_fallback(setup, reply):
    controller, docker = setup

    async def run():
        await controller.start()
        docker.control_result = runtime.Result(json.dumps(reply))
        with pytest.raises(RuntimeError, match="daemon rejected"):
            await controller.exec(control("begin", "/app", "/tmp/sfx-scratch", "0"))
        with pytest.raises(runtime.ControllerError, match="no fallback"):
            await controller.exec(control("end"))
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    asyncio.run(run())
    assert len(docker.control_messages) == 1
    assert not any(row[0] == "exec" for row in docker.events)
    assert "PRIVATE" not in json.dumps(controller.report)
    assert controller.report["task_removed"] and not controller.report["valid"]


def blocking_client(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    clients, calls = [], []

    class Client:
        def __init__(self, path):
            self.active = False
            self.closed = False
            clients.append(self)

        def turn_begin(self, session, **kwargs):
            assert not self.closed and not self.active
            self.active = True
            calls.append("begin")
            entered.set()
            assert release.wait(3)
            assert not self.closed
            self.active = False
            return {"result": "ok"}

        def turn_end(self, session):
            calls.append("end")
            return {"result": "ok"}

        def session_end(self, session):
            calls.append("session_end")
            return {"result": "ok"}

        def close(self):
            assert not self.active
            self.closed = True

    monkeypatch.setattr(control_transport, "Client", Client)
    return entered, release, clients, calls


def test_cancelled_request_drains_before_poison_and_queued_request_is_not_sent(setup, monkeypatch):
    controller, docker = setup
    entered, release, clients, calls = blocking_client(monkeypatch)

    async def run():
        await controller.start()
        first = asyncio.create_task(controller.exec(control("begin", "/app", "/tmp/sfx-scratch", "0")))
        assert await asyncio.to_thread(entered.wait, 2)
        queued = asyncio.create_task(controller.exec(control("end")))
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not queued.done() and not clients[0].closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(runtime.ControllerError, match="no fallback"):
            await queued
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert calls == ["begin"] and len(clients) == 1 and clients[0].closed
    assert not any(row[0] == "exec" for row in docker.events)
    assert [row[-1] for row in docker.events if row[0] == "rm"] == [TASK, SIDECAR]


def test_close_cancellation_waits_for_active_request_before_quiescing(setup, monkeypatch):
    controller, docker = setup
    entered, release, clients, calls = blocking_client(monkeypatch)

    async def run():
        await controller.start()
        request = asyncio.create_task(controller.exec(control("begin", "/app", "/tmp/sfx-scratch", "0")))
        assert await asyncio.to_thread(entered.wait, 2)
        closing = asyncio.create_task(controller.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done() and not clients[0].closed
        assert not any(row[0] in ("kill", "rm") for row in docker.events)
        release.set()
        assert (await request).return_code == 0
        with pytest.raises(runtime.ControllerError):
            await closing

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert calls == ["begin"] and clients[0].closed
    assert any(row["phase"] == "control_close" and row["type"] == "CancelledError"
               for row in controller.report["errors"])
    assert controller.report["task_removed"] and not controller.report["valid"]


def test_request_deadline_poison_is_preserved_after_worker_drains(setup, monkeypatch):
    controller, _ = setup
    entered, release, clients, calls = blocking_client(monkeypatch)

    async def run():
        await controller.start()
        request = asyncio.create_task(controller.exec(
            control("begin", "/app", "/tmp/sfx-scratch", "0"), timeout_sec=0.01))
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0.03)
        assert not request.done() and not clients[0].closed
        release.set()
        with pytest.raises(TimeoutError):
            await request
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert calls == ["begin"] and clients[0].closed
    assert controller.report["task_removed"] and not controller.report["valid"]


def test_client_bounds_connect_and_closes_socket_when_connect_fails(monkeypatch):
    calls = []

    class Socket:
        def settimeout(self, value):
            calls.append(("timeout", value))

        def connect(self, path):
            calls.append(("connect", path))
            raise TimeoutError("connect deadline")

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(protocol.socket, "socket", lambda *args: Socket())
    with pytest.raises(TimeoutError, match="connect deadline"):
        protocol.Client("/private/control.sock")
    assert calls == [("timeout", 2.0), ("connect", "/private/control.sock"), ("close",)]
