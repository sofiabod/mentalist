import asyncio
from concurrent.futures import CancelledError as WorkerCancelledError
from types import SimpleNamespace

import pytest

from eval import control_transport, sfx_live_agent
from test_live_fenced_integration import LocalSandbox, live_stack, review_greet


class RecordingSandbox(LocalSandbox):
    def __init__(self, root, session_id):
        super().__init__(root, session_id)
        self.control_commands = []

    async def exec(self, command, **kwargs):
        if "eval.sfx_client_cli" in command:
            self.control_commands.append(command)
        return await super().exec(command, **kwargs)


@pytest.fixture
def channels(monkeypatch):
    created = []

    class TrackedTransport(control_transport.InSandboxControlTransport):
        def __init__(self, socket_path):
            super().__init__(socket_path)
            self.operations = []
            self.clients = []
            created.append(self)

        async def request(self, *args):
            self.operations.append(args[0])
            reply = await super().request(*args)
            if self._client is not None and self._client not in self.clients:
                self.clients.append(self._client)
            return reply

    monkeypatch.setattr(control_transport, "InSandboxControlTransport", TrackedTransport)
    return created


@pytest.mark.parametrize("transport", [None, "exec", "in_sandbox"])
def test_wrapper_transport_opt_in_and_success_cleanup(live_stack, tmp_path, channels, transport):
    repo, _, daemon = live_stack
    kwargs = {} if transport is None else {"control_transport": transport}
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / "logs", arm="OFF", **kwargs)
    environment = RecordingSandbox(repo, "transport-success")
    original_exec = environment.exec
    context = SimpleNamespace(metadata={})

    class Script:
        async def run(self, instruction, environment, context):
            result = await environment.exec(command="python3 greet.py")
            assert result.stdout == "OLD\n" and result.return_code == 0

    agent._load_wrapped = lambda environment: Script()
    asyncio.run(agent.run("check the program", environment, context))

    assert agent.control_transport == (transport or "exec")
    assert agent._control_channel is None and agent._raw_exec is None
    assert environment.exec == original_exec
    assert not daemon.sessions and not daemon._mutations
    assert context.metadata["sfx_live"]["completed"] is True
    if transport == "in_sandbox":
        assert not environment.control_commands
        assert len(channels) == 1 and channels[0]._closed
        assert channels[0].operations[0] == channels[0].operations[-1] == "snapshot"
        assert "begin" in channels[0].operations and "end" in channels[0].operations
        assert len(channels[0].clients) == 1
        assert channels[0].clients[0].sock.fileno() == -1
    else:
        assert environment.control_commands and not channels


def test_wrapper_rejects_unknown_transport(tmp_path):
    with pytest.raises(ValueError, match="invalid control transport"):
        sfx_live_agent.SFXLiveAgent(tmp_path, control_transport="remote_socket")


def test_wrapper_error_preserves_failure_and_cleans_session(live_stack, tmp_path, channels):
    repo, _, daemon = live_stack
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / "logs", arm="OFF", control_transport="in_sandbox")
    environment = RecordingSandbox(repo, "transport-error")
    original_exec = environment.exec
    context = SimpleNamespace(metadata={})

    class Script:
        async def run(self, instruction, environment, context):
            await environment.exec(command="python3 greet.py")
            raise RuntimeError("wrapped failure")

    agent._load_wrapped = lambda environment: Script()
    with pytest.raises(RuntimeError, match="wrapped failure"):
        asyncio.run(agent.run("fail after a real command", environment, context))

    assert environment.exec == original_exec
    assert not daemon.sessions and not daemon._mutations
    assert agent._control_channel is None and agent._raw_exec is None
    assert len(channels) == 1 and channels[0]._closed
    assert channels[0].clients[0].sock.fileno() == -1
    assert context.metadata["sfx_live"]["completed"] is False
    assert agent._failure == {"type": "RuntimeError", "message": "wrapped failure"}
    assert agent._cleanup_errors == []


@pytest.mark.parametrize("failed_operation", ["begin", "resolve"])
def test_control_failure_does_not_run_tool_and_still_ends_session(
        live_stack, tmp_path, channels, monkeypatch, failed_operation):
    repo, _, daemon = live_stack

    class FailingTransport(control_transport.InSandboxControlTransport):
        async def request(self, *args):
            reply = await super().request(*args)
            if args[0] == failed_operation:
                raise RuntimeError(f"uncertain {failed_operation} reply")
            return reply

    monkeypatch.setattr(control_transport, "InSandboxControlTransport", FailingTransport)
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / "logs", arm="ON", control_transport="in_sandbox")
    environment = RecordingSandbox(repo, "transport-rpc-error")
    original_exec = environment.exec
    context = SimpleNamespace(metadata={})

    class Script:
        async def run(self, instruction, environment, context):
            await environment.exec(command="python3 greet.py")

    agent._load_wrapped = lambda environment: Script()
    with pytest.raises(RuntimeError, match=f"uncertain {failed_operation} reply"):
        asyncio.run(agent.run("fail closed on control errors", environment, context))

    assert environment.exec == original_exec
    assert environment.commands == ["pwd"]
    assert not daemon.sessions and not daemon._mutations
    assert agent._control_channel is None
    assert channels[0]._closed and channels[0].clients[0].sock.fileno() == -1
    assert channels[0].operations.count(failed_operation) == 1
    assert "end" in channels[0].operations
    assert context.metadata["sfx_live"]["completed"] is False
    assert agent._cleanup_errors == []


def test_wrapper_cancellation_aborts_pending_edit_and_closes_transport(live_stack, tmp_path, channels, monkeypatch):
    repo, scratch, daemon = live_stack
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / "logs", arm="ON", control_transport="in_sandbox")
    environment = RecordingSandbox(repo, "transport-cancel")
    original_exec = environment.exec
    context = SimpleNamespace(metadata={})
    review_greet(monkeypatch, "print('UNCOMMITTED')\n")

    async def drive():
        ready = asyncio.Event()
        futures = []

        class Script:
            async def run(self, instruction, environment, context):
                command = "cat > greet.py <<'EOF'\nprint('UNCOMMITTED')\nEOF"
                await agent._stream_event(environment, {
                    "event": "model_delta", "model_step": 0, "elapsed_s": 0,
                    "text": f"```bash\n{command}\n", "_exec_cwd": str(repo)})
                assert agent._pending_edit is not None
                chain = daemon.sessions[agent._session].chain
                if chain is not None and chain.future is not None:
                    futures.append(chain.future)
                ready.set()
                await asyncio.Event().wait()

        agent._load_wrapped = lambda environment: Script()
        running = asyncio.create_task(agent.run("cancel a streamed edit", environment, context))
        await asyncio.wait_for(ready.wait(), 3)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        for future in futures:
            try:
                await asyncio.to_thread(future.result, 5)
            except WorkerCancelledError:
                pass

    asyncio.run(drive())
    assert environment.exec == original_exec
    assert not daemon.sessions and not daemon._mutations
    assert agent._pending_edit is None and agent._control_channel is None
    assert (repo / "greet.py").read_text() == "print('OLD')\n"
    assert not list(scratch.iterdir())
    assert len(channels) == 1 and channels[0]._closed
    assert channels[0].clients[0].sock.fileno() == -1
    assert channels[0].operations.index("mutation_end") < channels[0].operations.index("end")
    assert context.metadata["sfx_live"]["completed"] is False
    assert agent._failure["type"] == "CancelledError"
    assert agent._cleanup_errors == []


def test_checkpoints_keep_repository_and_daemon_but_not_session_or_channel(live_stack, tmp_path, channels):
    repo, _, daemon = live_stack
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / "logs", arm="OFF", control_transport="in_sandbox")
    environment = RecordingSandbox(repo, "transport-checkpoints")
    sessions, contexts = [], []

    class Script:
        async def run(self, instruction, environment, context):
            sessions.append(agent._session)
            assert agent._session in daemon.sessions
            if len(sessions) == 1:
                command = "cat > greet.py <<'EOF'\nprint('CHECKPOINT_TWO')\nEOF"
                assert (await environment.exec(command=command)).return_code == 0
            else:
                assert (await environment.exec(command="python3 greet.py")).stdout == "CHECKPOINT_TWO\n"

    agent._load_wrapped = lambda environment: Script()

    async def drive():
        for checkpoint in range(2):
            context = SimpleNamespace(metadata={})
            await agent.run(f"checkpoint {checkpoint}", environment, context)
            contexts.append(context)
            assert not daemon.sessions and agent._control_channel is None
            assert channels[-1]._closed and channels[-1].clients[0].sock.fileno() == -1

    asyncio.run(drive())
    assert len(set(sessions)) == len(channels) == 2
    assert channels[0].socket_path == channels[1].socket_path == agent.socket_path
    assert len(channels[0].clients) == len(channels[1].clients) == 1
    assert (repo / "greet.py").read_text() == "print('CHECKPOINT_TWO')\n"
    assert all(context.metadata["sfx_live"]["completed"] for context in contexts)
    assert contexts[0].metadata["sfx_live"] is not contexts[1].metadata["sfx_live"]
    assert not environment.control_commands
