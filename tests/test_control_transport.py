import asyncio
import base64
import json
import tempfile
import threading
from pathlib import Path

import pytest

from adapters.protocol import Client
from eval import control_transport, sfx_live_agent
from eval.capture import fs_hash
from eval.control_transport import InSandboxControlTransport
from test_live_fenced_integration import LocalSandbox, live_stack, make_agent, review_greet


def payload(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def test_reuses_real_socket_and_ends_real_session(live_stack, tmp_path, monkeypatch):
    repo, scratch, daemon = live_stack
    clients = []

    def connect(path):
        client = Client(path)
        clients.append(client)
        return client

    monkeypatch.setattr(control_transport, "Client", connect)

    async def drive():
        async with InSandboxControlTransport(sfx_live_agent.SOCKET) as transport:
            snapshot = await transport.request("snapshot", "persistent", repo, tmp_path / "trace")
            assert snapshot == {"final_fs_hash": fs_hash(repo), "trace": []}
            assert not clients
            assert await transport.request("begin", "persistent", repo, scratch, "0") == {"ok": True}
            for _ in range(3):
                reply = await transport.request("resolve", "persistent", payload({
                    "tool": "run", "args": {"cmd": "python3 greet.py --exact"}}))
                assert reply == {"served": False, "outcome": "miss", "output": None}
            assert len(clients) == 1
            assert await transport.request("end", "persistent") == {"ok": True}
            assert "persistent" not in daemon.sessions
        assert clients[0].sock.fileno() == -1
        with pytest.raises(RuntimeError, match="closed"):
            await transport.request("begin", "again", repo, scratch, "0")
        await transport.aclose()

    asyncio.run(drive())


def test_persistent_transport_preserves_real_fork_commit_and_exact_serve(live_stack, tmp_path, monkeypatch):
    repo, scratch, daemon = live_stack
    agent = make_agent(tmp_path, repo)
    environment = LocalSandbox(repo, "persistent-fork")
    body = "import sys\nprint('PERSISTENT')\nsys.stderr.write('SEPARATE\\n')\n"
    review_greet(monkeypatch, body)

    async def drive():
        async with InSandboxControlTransport(sfx_live_agent.SOCKET) as transport:
            async def cli(environment, *args):
                return await transport.request(*args)

            agent._cli = cli
            await agent._cli(environment, "begin", agent._session, repo, scratch, "0")
            try:
                command = f"cat > greet.py <<'EOF'\n{body}EOF"
                await agent._route(environment, environment.exec, command, None, None, None, None)
                chain = daemon.sessions[agent._session].chain
                assert chain is not None and chain.future is not None
                await asyncio.to_thread(chain.future.result, 5)
                result = await agent._route(environment, environment.exec, "python3 greet.py",
                                            None, None, None, None)
                assert (result.stdout, result.stderr, result.return_code) == ("PERSISTENT\n", "SEPARATE\n", 0)
                assert agent._counts["hits"] == 1
                assert environment.commands == [command]
                assert (repo / "greet.py").read_text() == body
                mismatch = await transport.request("resolve", agent._session, payload({
                    "tool": "run", "args": {"cmd": "python3 greet.py --different"}}))
                assert mismatch["served"] is False
            finally:
                await agent._cli(environment, "end", agent._session)
        assert agent._session not in daemon.sessions
        assert not daemon._mutations
        assert not list(scratch.iterdir())

    asyncio.run(drive())


def test_request_cancellation_drains_worker_then_closes_without_retry(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    clients, requests = [], []

    class BlockingClient:
        def __init__(self, path):
            self.closed = False
            clients.append(self)

        def turn_begin(self, session, **kwargs):
            requests.append(session)
            if session == "cancelled":
                entered.set()
                assert release.wait(2)
                assert not self.closed
            return {"result": "ok"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(control_transport, "Client", BlockingClient)

    async def drive():
        async with InSandboxControlTransport("/sandbox/daemon.sock") as transport:
            request = asyncio.create_task(transport.request("begin", "cancelled", "/app", "/scratch", "0"))
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()
            await asyncio.sleep(0)
            assert not request.done()
            assert not clients[0].closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert clients[0].closed
            assert await transport.request("begin", "next", "/app", "/scratch", "0") == {"ok": True}
            assert len(clients) == 2
            assert requests == ["cancelled", "next"]
        assert all(client.closed for client in clients)

    asyncio.run(drive())


def test_close_and_concurrent_requests_do_not_race_worker(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    requests, clients = [], []

    class BlockingClient:
        def __init__(self, path):
            self.active = False
            self.closed = False
            clients.append(self)

        def turn_begin(self, session, **kwargs):
            assert not self.active and not self.closed
            self.active = True
            requests.append(session)
            if session == "first":
                entered.set()
                assert release.wait(2)
            self.active = False
            return {"result": "ok"}

        def close(self):
            assert not self.active
            self.closed = True

    monkeypatch.setattr(control_transport, "Client", BlockingClient)

    async def drive():
        transport = InSandboxControlTransport("/sandbox/daemon.sock")
        first = asyncio.create_task(transport.request("begin", "first", "/app", "/scratch", "0"))
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(transport.request("begin", "second", "/app", "/scratch", "0"))
        closing = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        assert not first.done() and not second.done() and not closing.done()
        release.set()
        assert await first == await second == {"ok": True}
        await closing
        assert requests == ["first", "second"]
        assert len(clients) == 1 and clients[0].closed

    asyncio.run(drive())


@pytest.mark.parametrize("reply", [None, [], {"error": "expired"}])
def test_daemon_rejections_fail_closed_without_retry(monkeypatch, reply):
    clients = []

    class RejectingClient:
        def __init__(self, path):
            self.closed = False
            self.calls = 0
            clients.append(self)

        def resolve(self, session, tool, args):
            self.calls += 1
            return reply

        def close(self):
            self.closed = True

    monkeypatch.setattr(control_transport, "Client", RejectingClient)

    async def drive():
        async with InSandboxControlTransport("/sandbox/daemon.sock") as transport:
            with pytest.raises(RuntimeError, match="daemon rejected"):
                await transport.request("resolve", "s", payload({"tool": "run", "args": {"cmd": "exact"}}))
            assert len(clients) == 1 and clients[0].calls == 1 and clients[0].closed

    asyncio.run(drive())


def test_requires_explicit_absolute_socket():
    with pytest.raises(ValueError, match="absolute socket"):
        InSandboxControlTransport("daemon.sock")


def test_missing_socket_has_no_fallback():
    async def drive():
        with tempfile.TemporaryDirectory(prefix="sfx-missing-", dir="/tmp") as directory:
            async with InSandboxControlTransport(Path(directory) / "missing.sock") as transport:
                with pytest.raises(FileNotFoundError):
                    await transport.request("begin", "s", "/app", "/scratch", "0")
                assert transport._client is None

    asyncio.run(drive())
