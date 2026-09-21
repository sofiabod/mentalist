import asyncio
import base64
import json
from pathlib import Path
import shlex
import tempfile
import time
from types import SimpleNamespace

import pytest

from eval.sidecar_rpc import (
    BrokerPoisoned, HostBroker, NonReusableToolOutput, SidecarClient, SpeculationDeclined, WORKER_BOOTSTRAP,
)
from sfx.script_contracts import ScriptInvocationRejected


def result(response, return_code=0, stderr=""):
    return SimpleNamespace(stdout=json.dumps(response), stderr=stderr, return_code=return_code)


def success(output=None, operation="run"):
    fields = {"output": ["hello\n", "", 0] if output is None else output}
    fields.update({"duration_ms": 0.1} if operation == "run" else {"status": "OK"})
    return {"ok": True, "result": fields}


def broker(path, execute, **kwargs):
    return HostBroker(path, execute, repo="/app", scratch="/sfx-scratch", source_dir="/root/src", **kwargs)


def test_bridge_dispatches_only_through_task_executor_and_preserves_result():
    async def check(socket_path):
        calls = []

        async def execute(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(.03)
            return result(success(["λ\n\x00tail", "", -9]))

        async with broker(socket_path, execute) as server:
            client = SidecarClient(socket_path)
            output, elapsed = await asyncio.to_thread(client.run, "read", {"cmd": "cat  /app/data"})
            assert output == ("λ\n\x00tail", "", -9)
            assert elapsed >= 25
            assert not server.poisoned and server.inflight_count == 0
        assert len(calls) == 1
        call = calls[0]
        argv = shlex.split(call["command"])
        assert argv[:4] == ["python3", "-c", WORKER_BOOTSTRAP, "/root/src"] and len(argv) == 5
        payload = json.loads(base64.b64decode(argv[4]))
        assert payload["request"] == {"op": "run", "kind": "read", "args": {"cmd": "cat  /app/data"}}
        assert payload["settings"] == {
            "repo": "/app", "scratch": "/sfx-scratch", "script_contracts": [], "fork_path_view": "proot"}
        assert call["cwd"] == "/app" and call["env"] is None
        assert call["timeout_sec"] == 120
        assert not socket_path.exists()

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


def test_bridge_supports_fork_callback_and_does_not_rewrite_exact_command():
    async def check(socket_path):
        captured = []

        async def execute(**kwargs):
            payload = json.loads(base64.b64decode(shlex.split(kwargs["command"])[-1]))
            captured.append(payload["request"])
            return result(success(operation="run_in_fork"))

        async with broker(socket_path, execute):
            response = await asyncio.to_thread(SidecarClient(socket_path).run_in_fork,
                                              "/sfx-scratch/job", ("read", "free", {"cmd": "cat  /app/data"}))
        assert response == (("hello\n", "", 0), "OK")
        assert captured == [{"op": "run_in_fork", "path": "/sfx-scratch/job",
                             "hop": ["read", "free", {"cmd": "cat  /app/data"}]}]

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


@pytest.mark.parametrize("error_type,exception", [
    ("ScriptInvocationRejected", ScriptInvocationRejected),
    ("SpeculationDeclined", SpeculationDeclined),
])
def test_known_admission_decline_does_not_poison_later_work(error_type, exception):
    async def check(socket_path):
        responses = [
            {"ok": False, "error": {"category": "admission", "type": error_type}}, success()]

        async def execute(**kwargs):
            return result(responses.pop(0))

        async with broker(socket_path, execute) as server:
            client = SidecarClient(socket_path)
            with pytest.raises(exception):
                await asyncio.to_thread(client.run, "run", {"cmd": "python generated.py"})
            assert not server.poisoned
            assert (await asyncio.to_thread(client.run, "read", {"cmd": "cat data"}))[0][0] == "hello\n"

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


@pytest.mark.parametrize("chain", [False, True])
def test_mixed_output_decline_discards_cache_reservation_without_poisoning_broker(chain):
    from sfx.cache import Cache
    from sfx.executor import Executor
    from sfx.ledger import Ledger

    async def check(socket_path):
        responses = [
            {"ok": False, "error": {"category": "non_reusable", "type": "NonReusableToolOutput"}},
            success(["clean\n", "", 0]),
        ]

        async def execute(**kwargs):
            return result(responses.pop(0))

        async with broker(socket_path, execute) as server:
            client = SidecarClient(socket_path)
            clock = lambda: time.monotonic() * 1000
            ledger = Ledger()
            cache = Cache(clock, ledger)
            executor = Executor(clock, cache, ledger, slots=1)
            args = {"cmd": "cat existing missing existing"}
            try:
                if chain:
                    _, future = executor.submit_chain("read", args, lambda: client.run_in_fork(
                        "/sfx-scratch/job", ("read", "free", args)))
                    with pytest.raises(NonReusableToolOutput):
                        await asyncio.wrap_future(future)
                else:
                    key = executor.speculate("read", args, 1, lambda: client.run("read", args))
                    await asyncio.wrap_future(executor._specs[key].future)
                    assert executor._specs[key].failed
                assert cache.serve("read", args, ask_time=clock())[0].startswith("miss")
                assert not server.poisoned and server.inflight_count == 0
                assert (await asyncio.to_thread(client.run, "read", {"cmd": "cat existing"}))[0] == (
                    "clean\n", "", 0)
                assert not server.poisoned
            finally:
                executor._pool.shutdown(wait=True)

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


@pytest.mark.parametrize("failure", [
    "lost", "bad_json", "stderr", "worker", "wrong_output", "nonempty_stderr", "forged_decline",
    "exception", "cancelled",
])
def test_uncertain_worker_failure_poison_is_sticky_and_prevents_more_dispatch(failure):
    async def check(socket_path):
        calls = []

        async def execute(**kwargs):
            calls.append(kwargs)
            if failure == "exception":
                raise RuntimeError("SECRET transport details")
            if failure == "cancelled":
                raise asyncio.CancelledError()
            if failure == "lost":
                return result(success(), return_code=-9)
            if failure == "bad_json":
                return SimpleNamespace(stdout="not JSON SECRET", stderr="", return_code=0)
            if failure == "stderr":
                return result(success(), stderr="SECRET stderr")
            if failure == "worker":
                return result({"ok": False, "error": {"category": "worker", "type": "WorkerFailure"}})
            if failure == "nonempty_stderr":
                return result(success(["out", "err", 0]))
            if failure == "forged_decline":
                return result({"ok": False, "error": {"category": "non_reusable", "type": "WorkerFailure"}})
            return result(success(["out", "err", True]))

        server = await broker(socket_path, execute).start()
        client = SidecarClient(socket_path)
        with pytest.raises(BrokerPoisoned) as caught:
            await asyncio.to_thread(client.run, "read", {"cmd": "cat data"})
        assert "SECRET" not in str(caught.value)
        assert server.poisoned
        with pytest.raises(BrokerPoisoned):
            await asyncio.to_thread(client.run, "read", {"cmd": "cat other"})
        assert len(calls) == 1
        with pytest.raises(BrokerPoisoned):
            await server.aclose()
        assert not socket_path.exists()

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


@pytest.mark.parametrize("payload", [
    b'{"op":"run","op":"shell"}\n',
    b'{"op":"shell","command":"touch escaped"}\n',
    b'{"op":"run_in_fork","path":"/outside","hop":["read","free",{"cmd":"cat data"}]}\n',
])
def test_malformed_controller_request_never_reaches_task_executor(payload):
    async def check(socket_path):
        async def execute(**kwargs):
            pytest.fail("unexpected dispatch")

        server = await broker(socket_path, execute).start()
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(payload)
        await writer.drain()
        response = json.loads(await reader.readline())
        assert response["error"]["category"] == "protocol"
        writer.close()
        await writer.wait_closed()
        assert server.poisoned and server.inflight_count == 0
        with pytest.raises(BrokerPoisoned):
            await server.aclose()

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


def test_bounded_close_reports_inflight_uncertainty_without_claiming_cleanup():
    async def check(socket_path):
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(**kwargs):
            started.set()
            await release.wait()
            return result(success())

        server = await broker(socket_path, execute, drain_timeout_s=.03).start()
        request = asyncio.create_task(asyncio.to_thread(SidecarClient(socket_path).run, "read", {"cmd": "cat data"}))
        await started.wait()
        assert server.inflight_count == 1
        before = time.monotonic()
        with pytest.raises(BrokerPoisoned, match="worker_drain_timeout"):
            await server.aclose()
        assert time.monotonic() - before < 1
        assert server.inflight_count == 1 and server.poisoned
        with pytest.raises(BrokerPoisoned):
            await request
        release.set()
        await asyncio.gather(*tuple(server._inflight))
        await asyncio.sleep(0)
        assert server.inflight_count == 0

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


def test_concurrent_callbacks_are_tracked_until_all_task_results_return():
    async def check(socket_path):
        both_started, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def execute(**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                both_started.set()
            await release.wait()
            return result(success())

        async with broker(socket_path, execute) as server:
            client = SidecarClient(socket_path)
            requests = [asyncio.create_task(asyncio.to_thread(client.run, "read", {"cmd": f"cat data{i}"}))
                        for i in range(2)]
            await both_started.wait()
            assert server.inflight_count == 2
            release.set()
            values = await asyncio.gather(*requests)
            assert all(value[0] == ("hello\n", "", 0) for value in values)
            assert server.inflight_count == 0

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


@pytest.mark.parametrize("acknowledgement", [True, False, "error"])
def test_unknown_failure_waits_for_single_verified_task_termination(acknowledgement):
    async def check(socket_path):
        terminated, calls = asyncio.Event(), []

        async def execute(**kwargs):
            return result(success(), return_code=-9)

        async def terminate():
            calls.append("termination")
            await asyncio.sleep(.02)
            if acknowledgement == "error":
                raise RuntimeError("SECRET termination detail")
            terminated.set()
            return acknowledgement

        server = await broker(socket_path, execute, on_poison=terminate).start()
        client = SidecarClient(socket_path)
        with pytest.raises(BrokerPoisoned):
            await asyncio.to_thread(client.run, "read", {"cmd": "cat data"})
        assert server.workers_quiesced is (acknowledgement is True)
        assert terminated.is_set() is (acknowledgement != "error")
        assert server.poisoned
        with pytest.raises(BrokerPoisoned):
            await asyncio.to_thread(client.run, "read", {"cmd": "cat data"})
        with pytest.raises(BrokerPoisoned) as caught:
            await server.aclose()
        assert "SECRET" not in str(caught.value)
        assert calls == ["termination"]

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


def test_task_termination_timeout_does_not_claim_quiescence_or_clear_poison():
    async def check(socket_path):
        release = asyncio.Event()
        calls = []

        async def execute(**kwargs):
            return result(success(), return_code=-9)

        async def terminate():
            calls.append("termination")
            await release.wait()
            return True

        server = await broker(socket_path, execute, on_poison=terminate, poison_timeout_s=.02).start()
        before = time.monotonic()
        with pytest.raises(BrokerPoisoned):
            await asyncio.to_thread(SidecarClient(socket_path).run, "read", {"cmd": "cat data"})
        assert time.monotonic() - before < 1
        assert server.poisoned and not server.workers_quiesced
        with pytest.raises(BrokerPoisoned):
            await server.aclose()
        assert calls == ["termination"] and not server.workers_quiesced
        release.set()
        await server._termination_task
        assert server.workers_quiesced and server.poisoned

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))


def test_close_aborts_a_nonreading_controller_instead_of_hanging_on_buffered_reply():
    async def check(socket_path):
        executed = asyncio.Event()

        async def execute(**kwargs):
            executed.set()
            return result(success(["x" * (4 * 1024 * 1024), "", 0]))

        server = await broker(socket_path, execute, drain_timeout_s=.03).start()
        reader, writer = await asyncio.open_unix_connection(str(socket_path), limit=100)
        writer.write(b'{"op":"run","kind":"read","args":{"cmd":"cat data"}}\n')
        await writer.drain()
        await executed.wait()
        await asyncio.sleep(.01)
        before = time.monotonic()
        with pytest.raises(BrokerPoisoned):
            await server.aclose()
        assert time.monotonic() - before < 1
        assert server.inflight_count == 0 and server.poisoned
        writer.close()
        await writer.wait_closed()

    with tempfile.TemporaryDirectory(prefix="sfx-rpc-", dir="/tmp") as directory:
        asyncio.run(check(Path(directory) / "broker.sock"))
