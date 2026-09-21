"""Deterministic Docker lifecycle checks; no Docker process or model is started."""
import asyncio
import base64
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import stat
import tempfile
import threading
from types import SimpleNamespace

import pytest

from eval import sidecar_runtime as runtime
from eval.sidecar_daemon import RetainedForks, build_controller
from eval.sidecar_rpc import BrokerPoisoned


TASK = "a" * 64
SIDECAR = "b" * 64
IMAGE = "sha256:" + "c" * 64


class FakeBroker:
    def __init__(self, socket_path, exec_fn, **kwargs):
        self.socket_path, self.exec_fn, self.kwargs = socket_path, exec_fn, kwargs
        self.poisoned = False
        self.inflight_count = 0
        self.events = []
        self.close_error = None

    async def start(self):
        self.events.append(("broker_start",))

    async def aclose(self):
        self.events.append(("broker_close",))
        if self.close_error:
            raise self.close_error

    def raise_if_poisoned(self):
        if self.poisoned:
            raise BrokerPoisoned("test worker completion uncertain")


class Docker:
    def __init__(self, controller):
        self.controller = controller
        self.events = []
        self.task = {
            "Id": TASK, "Image": IMAGE, "Config": {"User": "root", "Env": ["PRIVATE_KEY=secret"]},
            "HostConfig": {"PidMode": "", "Privileged": False, "CapAdd": None},
            "State": {"Running": True, "Paused": False, "Pid": 123, "StartedAt": "original-start"},
            "Mounts": [{"Type": "bind", "Source": str(source), "Destination": target,
                        "RW": True, "Propagation": "rprivate"}
                       for target, source in controller.expected_mounts.items()]}
        self.sidecar = None
        self.control_result = runtime.Result('{"ok": true}')
        self.control_messages = []
        self.control_server = None
        self.control_thread = None
        self.quiesce_error = None
        self.change_after_create = False

    async def __call__(self, *argv, **kwargs):
        self.events.append(argv)
        if argv[0] == "inspect":
            value = self.task if argv[1] == TASK else self.sidecar
            if value is None:
                return runtime.Result(stderr="Error: No such object: " + argv[1], return_code=1)
            return runtime.Result(json.dumps([value]))
        if argv[0] == "create":
            labels = dict(argv[index + 1].split("=", 1) for index, value in enumerate(argv) if value == "--label")
            self.sidecar = {"Id": SIDECAR, "Image": IMAGE, "Config": {"Labels": labels},
                            "State": {"Running": False, "ExitCode": 0}}
            if self.change_after_create:
                self.task["Config"]["Env"] = ["CHANGED=yes"]
            return runtime.Result(SIDECAR + "\n")
        if argv[0] == "start":
            from adapters.protocol import serve

            def reply(message):
                self.control_messages.append(message)
                return json.loads(self.control_result.stdout)

            self.control_server = serve(None, str(self.controller.control_root / "daemon.sock"), reply)
            self.control_thread = threading.Thread(
                target=lambda: self.control_server.serve_forever(poll_interval=0.01), daemon=True)
            self.control_thread.start()
            self.sidecar["State"]["Running"] = True
            (self.controller.control_root / "ready.json").write_text(json.dumps({"token": self.controller.token}))
            self.controller.broker.events = self.events
            return runtime.Result(SIDECAR + "\n")
        if argv[0] == "kill":
            if self.quiesce_error:
                raise self.quiesce_error
            self.sidecar["State"]["Running"] = False
            return runtime.Result(SIDECAR + "\n")
        if argv[0] == "rm":
            if argv[-1] == TASK:
                self.task = None
            elif argv[-1] == SIDECAR:
                self.sidecar = None
            else:
                pytest.fail("unscoped removal")
            return runtime.Result(argv[-1] + "\n")
        if argv[0] == "exec":
            return self.control_result
        pytest.fail(f"unexpected Docker action {argv[0]}")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from eval import sidecar_rpc

    root = tmp_path.resolve()
    for name in ("repo", "scratch", "runtime", "control"):
        (root / name).mkdir(mode=0o700)
    (root / "runtime/src").mkdir()
    (root / "runtime/data").mkdir()
    monkeypatch.setattr(sidecar_rpc, "HostBroker", FakeBroker)
    private = tempfile.TemporaryDirectory(prefix="sfx-ctl-test-", dir="/tmp")

    async def task_exec(**kwargs):
        return runtime.Result("TASK", "", 0)

    controller = runtime.Controller(
        TASK, task_exec, repo="/app", scratch="/tmp/sfx-scratch",
        source_root=root / "runtime", control_root=Path(private.name).resolve(), depth=1,
        table="benchmark", script_contracts=[], fork_path_view="proot",
        expected_mounts={"/app": root / "repo", "/tmp/sfx-scratch": root / "scratch"})
    docker = Docker(controller)
    monkeypatch.setattr(controller, "_docker", docker)
    try:
        yield controller, docker
    finally:
        if docker.control_server is not None:
            docker.control_server.shutdown()
            docker.control_server.server_close()
            docker.control_thread.join(timeout=2)
        private.cleanup()


def control(op="end", *args):
    return "PYTHONPATH=/root/src SFX_SOCKET=/sfx-control/daemon.sock python3 -m eval.sfx_client_cli " + " ".join((op, "session", *args))


def test_normal_lifecycle_is_scoped_cpu_only_and_keeps_task(setup):
    controller, docker = setup

    async def run():
        assert await controller.start() is controller
        assert (await controller.exec(control(), user="root")).return_code == 0
        await controller.aclose()
        before = list(docker.events)
        await controller.aclose()
        assert docker.events == before

    asyncio.run(run())
    create = next(row for row in docker.events if row[0] == "create")
    for flag, value in (("--runtime", "runc"), ("--network", "none"), ("--cap-drop", "ALL"),
                        ("--security-opt", "no-new-privileges"), ("--cpus", "1"),
                        ("--memory", "512m"), ("--pids-limit", "128")):
        assert create[create.index(flag) + 1] == value
    assert not any(flag in create for flag in ("--privileged", "--gpus", "--pid"))
    assert "--no-healthcheck" in create
    assert create[create.index(IMAGE) + 1:][:5] == ("-I", "-S", "-B", "-u", "-c")
    assert "PATH=" + runtime.CONTROLLER_PATH in create
    mounts = [create[i + 1] for i, value in enumerate(create) if value == "--mount"]
    assert len(mounts) == 6
    assert any("dst=/app,readonly" in mount for mount in mounts)
    assert any("dst=/root/src,readonly" in mount for mount in mounts)
    assert not any("docker.sock" in mount for mount in mounts)
    assert docker.task is not None and docker.sidecar is None
    assert controller.report["valid"] and controller.report["broker_drained"]
    assert controller.report["controller_quiesced"] and controller.report["controller_removed"]
    assert not controller.report["task_removed"] and not controller.poisoned
    assert controller.report["control_transport"] == "persistent_unix_v1"
    assert not any(row[0] == "exec" for row in docker.events)
    assert [row["type"] for row in docker.control_messages] == ["turn_end", "session_end"]
    assert "PRIVATE_KEY" not in json.dumps(controller.report)
    phases = [event[0] for event in docker.events]
    assert phases.index("kill") < phases.index("broker_close") < phases.index("rm")


@pytest.mark.parametrize("command", [
    "killall python3", control("snapshot", "/outside", "/logs/agent/sfx-trace.jsonl"),
    control("begin", "/app", "/wrong-scratch", "0"), control("end", "extra"),
    control("resolve", "not-base64"), control("end") + "; touch /app/changed",
])
def test_controller_rejects_arbitrary_or_unscoped_commands(setup, command):
    controller, docker = setup

    async def run():
        await controller.start()
        before = len([event for event in docker.events if event[0] == "exec"])
        with pytest.raises((runtime.ControllerError, ValueError)):
            await controller.exec(command)
        assert len([event for event in docker.events if event[0] == "exec"]) == before
        await controller.aclose()

    asyncio.run(run())


def test_control_payload_is_data_not_a_shell_command(setup):
    controller, docker = setup
    payload = base64.b64encode(json.dumps({"tool": "run", "args": {"cmd": "killall python3"}}).encode()).decode()

    async def run():
        await controller.start()
        await controller.exec(control("resolve", payload))
        await controller.aclose()

    asyncio.run(run())
    assert docker.control_messages == [{"type": "resolve", "session": "session",
                                       "tool": "run", "args": {"cmd": "killall python3"}}]
    assert not any(row[0] == "exec" for row in docker.events)


@pytest.mark.parametrize("change", [
    lambda task, ctl: task["HostConfig"].update(PidMode="host"),
    lambda task, ctl: task["HostConfig"].update(PidMode="container:other"),
    lambda task, ctl: task["HostConfig"].update(Privileged=True),
    lambda task, ctl: task["HostConfig"].update(CapAdd=["SYS_ADMIN"]),
    lambda task, ctl: task["Config"].update(User="1000"),
    lambda task, ctl: task["Mounts"][0].update(RW=False),
    lambda task, ctl: task["Mounts"][0].update(Source=str(ctl.control_root)),
    lambda task, ctl: task["Mounts"][0].update(Source="/"),
    lambda task, ctl: task["Mounts"].pop(),
])
def test_unsafe_task_fails_before_container_creation(setup, change):
    controller, docker = setup
    change(docker.task, controller)
    with pytest.raises(runtime.ControllerError):
        asyncio.run(controller.start())
    assert not any(event[0] in ("create", "rm") for event in docker.events)


@pytest.mark.parametrize("entries", [
    ["BASH_ENV=/app/startup.sh"],
    ["ENV=/app/startup.sh"],
    ["BASH_FUNC_python3%%=() { printf hijacked; }"],
    ["BASH_FUNC_cat%%="],
    ["BASH_ENV=/app/startup.sh", "BASH_ENV="],
])
def test_task_shell_hooks_fail_before_controller_or_task_bootstrap(setup, entries):
    controller, docker = setup
    docker.task["Config"]["Env"] = entries
    calls = []

    async def task_exec(**kwargs):
        calls.append(kwargs)
        pytest.fail("task bootstrap started with shell hooks")

    controller.exec_fn = task_exec
    with pytest.raises(runtime.ControllerError, match="startup or function hooks"):
        asyncio.run(controller.start())
    assert not calls and controller.broker is None
    assert not any(event[0] in ("create", "start", "exec", "rm") for event in docker.events)
    assert docker.task["Config"]["Env"] == entries


def test_empty_shell_startup_variables_remain_unchanged(setup):
    controller, docker = setup
    entries = ["BASH_ENV=", "ENV=", "PATH=/usr/bin:/bin"]
    docker.task["Config"]["Env"] = entries

    async def run():
        await controller.start()
        assert (await controller._task_exec(command="trusted worker", env=None)).stdout == "TASK"
        await controller.aclose()

    asyncio.run(run())
    assert docker.task["Config"]["Env"] == entries
    assert controller.report["valid"]


@pytest.mark.parametrize("entries", ["BASH_ENV=x", [None], ["BASH_ENV"], ["=value"], ["BASH_ENV=\x00"]])
def test_malformed_task_environment_fails_before_bootstrap(setup, entries):
    controller, docker = setup
    docker.task["Config"]["Env"] = entries
    with pytest.raises(runtime.ControllerError, match="invalid task environment"):
        asyncio.run(controller.start())
    assert not any(event[0] in ("create", "exec") for event in docker.events)


def test_legitimate_extra_harbor_log_bind_is_not_rejected(setup, tmp_path):
    controller, docker = setup
    logs = tmp_path.resolve() / "harbor-logs"
    logs.mkdir()
    docker.task["Mounts"].append({"Type": "bind", "Source": str(logs), "Destination": "/logs",
                                 "RW": True, "Propagation": "rprivate"})

    async def run():
        await controller.start()
        await controller.aclose()

    asyncio.run(run())
    assert controller.report["valid"]


def test_changed_task_identity_invalidates_start_without_running_tools(setup):
    controller, docker = setup
    docker.change_after_create = True
    with pytest.raises(runtime.ControllerError, match="identity changed"):
        asyncio.run(controller.start())
    assert not any(row[0] == "exec" for row in docker.events)
    assert docker.task is not None and docker.sidecar is None
    assert not controller.report["valid"]


def test_mount_order_changes_do_not_change_task_identity(setup):
    controller, docker = setup

    async def run():
        await controller.start()
        identity = controller._identity
        docker.task["Mounts"].reverse()
        assert (await controller._verify_task())[1] == identity
        assert (await controller._task_exec(command="trusted worker")).stdout == "TASK"
        await controller.aclose()

    asyncio.run(run())
    assert controller.report["valid"] and not controller.report["task_removed"]


def test_substantive_mount_change_still_invalidates_identity(setup):
    controller, docker = setup

    async def run():
        await controller.start()
        docker.task["Mounts"][0]["RW"] = False
        with pytest.raises(runtime.ControllerError, match="binding differs"):
            await controller._verify_task()
        docker.task["Mounts"][0]["RW"] = True
        await controller.aclose()

    asyncio.run(run())


def test_error_details_are_bounded_private_and_do_not_enter_public_report(setup):
    controller, _ = setup
    message = "PRIVATE_CREDENTIAL " + "x" * 9000
    controller._error("task_worker", runtime.ControllerError(message), uncertain=True)
    path = controller.control_root / "controller-errors.jsonl"
    row = json.loads(path.read_text())
    assert row["phase"] == "task_worker" and row["type"] == "ControllerError"
    assert row["message"] == message[:8192]
    assert row["message_bytes"] == len(message) and row["message_truncated"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "PRIVATE_CREDENTIAL" not in json.dumps(controller.report)
    assert controller.poisoned and not controller.report["valid"]


def test_private_evidence_failure_does_not_replace_original_invalidation(setup):
    controller, _ = setup
    target = controller.control_root / "not-an-error-log"
    target.write_text("unchanged")
    (controller.control_root / "controller-errors.jsonl").symlink_to(target)
    controller._error("task_worker", runtime.ControllerError("original"), uncertain=True)
    assert target.read_text() == "unchanged"
    assert controller.poisoned and controller._uncertain


def test_broker_poison_terminates_exact_task_before_controller_and_acknowledges(setup):
    controller, docker = setup

    async def run():
        await controller.start()
        callback = controller.broker.kwargs["on_poison"]
        assert await callback() is True
        assert docker.task is None and docker.sidecar is not None
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    asyncio.run(run())
    removals = [event[-1] for event in docker.events if event[0] == "rm"]
    assert removals == [TASK, SIDECAR]
    marker = json.loads((controller.control_root / "task-terminated.json").read_text())
    assert marker == {"token": controller.token}
    assert controller.report["task_removed"] and not controller.report["valid"]


@pytest.mark.parametrize("failure", ["quiesce", "broker"])
def test_uncertain_shutdown_removes_task_before_controller_forks(setup, failure):
    controller, docker = setup

    async def run():
        await controller.start()
        if failure == "quiesce":
            docker.quiesce_error = TimeoutError("controller drain unknown")
        else:
            controller.broker.close_error = TimeoutError("worker drain unknown")
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    asyncio.run(run())
    removals = [event[-1] for event in docker.events if event[0] == "rm"]
    assert removals == [TASK, SIDECAR]
    assert controller.poisoned and controller.report["task_removed"]


def test_changed_task_is_never_deleted_and_uncertain_forks_are_retained(setup):
    controller, docker = setup

    async def run():
        await controller.start()
        docker.quiesce_error = TimeoutError("controller drain unknown")
        docker.task["Config"]["Env"] = ["OTHER_OWNER=yes"]
        with pytest.raises(runtime.ControllerError):
            await controller.aclose()

    asyncio.run(run())
    assert not any(event[0] == "rm" for event in docker.events)
    assert docker.task is not None and docker.sidecar is not None
    assert not controller.report["valid"] and not controller.report["task_removed"]


def test_docker_unavailability_is_not_verified_absence(setup, monkeypatch):
    controller, _ = setup

    async def unavailable(*args, **kwargs):
        return runtime.Result(stderr="Cannot connect to Docker: PRIVATE_SOCKET", return_code=1)

    monkeypatch.setattr(controller, "_docker", unavailable)
    with pytest.raises(runtime.ControllerError) as caught:
        asyncio.run(controller._inspect(TASK, missing_ok=True))
    assert "PRIVATE_SOCKET" not in str(caught.value)


@pytest.mark.parametrize("message", [
    "Error: No such object: ",
    "error: no such object: ",
    "Error response from daemon: No such container: ",
    "error response from daemon: no such container: ",
])
def test_explicit_docker_not_found_is_case_insensitive(setup, monkeypatch, message):
    controller, _ = setup

    async def missing(*args, **kwargs):
        return runtime.Result(stdout="[]\n", stderr=message + TASK, return_code=1)

    monkeypatch.setattr(controller, "_docker", missing)
    assert asyncio.run(controller._inspect(TASK, missing_ok=True)) is None
    with pytest.raises(runtime.ControllerError):
        asyncio.run(controller._inspect(TASK))


def test_worker_exec_uses_captured_task_executor_and_rechecks_identity(setup):
    controller, docker = setup
    calls = []

    async def task_exec(**kwargs):
        calls.append(kwargs)
        return runtime.Result("task result", "", 0)

    controller.exec_fn = task_exec

    async def run():
        await controller.start()
        response = await controller.broker.exec_fn(command="fixed worker bootstrap", cwd="/app", env=None)
        assert response.stdout == "task result"
        await controller.aclose()

    asyncio.run(run())
    assert calls == [{"command": "fixed worker bootstrap", "cwd": "/app", "env": None}]
    assert not any(row[0] == "exec" for row in docker.events)


def test_controller_callbacks_are_remote_and_stock_handle_validator_survives(monkeypatch):
    from eval import sfx_daemon_run, sidecar_rpc

    class Client:
        def __init__(self, socket):
            self.socket = socket

        def run(self, *args):
            return "remote GET"

        def run_in_fork(self, *args):
            return "remote fork"

    monkeypatch.setattr(sidecar_rpc, "SidecarClient", Client)
    monkeypatch.setattr(sfx_daemon_run, "_run", lambda *args: pytest.fail("local task execution"))
    monkeypatch.setattr(sfx_daemon_run, "_run_in_fork", lambda *args: pytest.fail("local task execution"))
    daemon = build_controller(1, "/private/broker.sock", table={"k": 1, "table": {}})
    assert daemon.run("read", {}) == "remote GET"
    assert daemon.fs_substrate._run_in_fork("/fork", ()) == "remote fork"
    assert daemon.fs_substrate.validate_handle is sfx_daemon_run._validate_fork_handle
    daemon.shutdown()


def test_uncertain_remote_failure_retains_fork_until_matching_task_termination(tmp_path):
    events = []

    @contextmanager
    def fork(*args):
        events.append("created")
        try:
            yield "private fork"
        finally:
            events.append("deleted")

    guard = RetainedForks(fork, tmp_path, "owned-token")
    with pytest.raises(BrokerPoisoned):
        with guard.fork():
            raise BrokerPoisoned("worker completion unknown")
    assert events == ["created"] and len(guard.retained) == 1
    assert guard.release_if_terminated() is False
    (tmp_path / "task-terminated.json").write_text('{"token":"wrong-owner"}')
    assert guard.release_if_terminated() is False
    (tmp_path / "task-terminated.json").write_text('{"token":"owned-token"}')
    assert guard.release_if_terminated() is True
    assert events == ["created", "deleted"] and guard.retained == []


def test_completed_or_locally_declined_forks_are_cleaned_normally(tmp_path):
    events = []

    @contextmanager
    def fork(*args):
        try:
            yield "private fork"
        finally:
            events.append("deleted")

    guard = RetainedForks(fork, tmp_path, "owned-token")
    with guard.fork():
        pass
    with pytest.raises(ValueError):
        with guard.fork():
            raise ValueError("admission declined")
    assert events == ["deleted", "deleted"] and not guard.retained


@pytest.mark.parametrize("decision", [True, False, ValueError("malformed termination marker")])
def test_finalization_joins_ended_session_pools_and_bypasses_unsafe_finalizers(monkeypatch, decision):
    from eval import sidecar_daemon

    events = []

    def joined(**kwargs):
        assert kwargs == {"wait": True, "cancel_futures": True}
        events.append("pools joined")

    def release():
        assert events == ["pools joined"]
        if isinstance(decision, Exception):
            raise decision
        return decision

    class ImmediateExit(BaseException):
        pass

    def exit_without_finalizers(code):
        assert code == 74
        events.append("os._exit")
        raise ImmediateExit

    monkeypatch.setattr(sidecar_daemon.os, "_exit", exit_without_finalizers)
    daemon = SimpleNamespace(owned_pools=[SimpleNamespace(shutdown=joined)],
                             retained_forks=SimpleNamespace(release_if_terminated=release))
    if decision is True:
        sidecar_daemon.finish_controller(daemon)
        assert events == ["pools joined"]
    else:
        with pytest.raises(ImmediateExit):
            sidecar_daemon.finish_controller(daemon)
        assert events == ["pools joined", "os._exit"]
