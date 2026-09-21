import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from eval import sidecar_agent
from eval.sfx_live_agent import SFXLiveAgent
from eval.sidecar_agent import SidecarSFXAgent, task_identity


@pytest.mark.parametrize("key,value", [
    ("source_dir", "/task/source"), ("socket_path", "/tmp/shared.sock"),
    ("trace_path", "/app/trace"), ("control_transport", "in_sandbox"),
    ("python_executable", "/other/python"), ("control_exec", lambda: None),
])
def test_sidecar_rejects_unverified_control_overrides(tmp_path, key, value):
    with pytest.raises(ValueError):
        SidecarSFXAgent(tmp_path, **{key: value})


def test_identity_uses_exact_harbor_main_container_and_declared_bind_mounts(tmp_path, monkeypatch):
    from harbor.environments.docker import docker

    class Environment:
        default_user = "root"
        _mounts = [{"type": "bind", "source": str(tmp_path), "target": "/app"},
                   {"type": "bind", "source": str(tmp_path), "target": "/tmp/sfx-scratch"}]

        async def _run_docker_compose_command(self, command, **kwargs):
            assert command == ["ps", "-q", "main"]
            assert kwargs == {"timeout_sec": 10}
            return SimpleNamespace(return_code=0, stdout="a" * 64 + "\n")

    monkeypatch.setattr(docker, "DockerEnvironment", Environment)
    env = Environment()
    identity, mounts = asyncio.run(task_identity(env, "/app", "/tmp/sfx-scratch"))
    assert identity == "a" * 64
    assert mounts == {"/app": str(tmp_path), "/tmp/sfx-scratch": str(tmp_path)}
    env._mounts = [{"type": "bind", "source": str(tmp_path), "target": "/app", "read_only": True}]
    with pytest.raises(ValueError, match="writable"):
        asyncio.run(task_identity(env, "/app", "/tmp/sfx-scratch"))
    with pytest.raises(ValueError, match="Docker"):
        asyncio.run(task_identity(object(), "/app", "/tmp/sfx-scratch"))


def lifecycle(tmp_path, monkeypatch, *, primary=None, close_error=None, start_error=None,
              hash_error=None):
    from eval import sidecar_runtime

    events = []
    digest = "sha256:" + "b" * 64

    class Controller:
        poisoned = False
        report = {"workers_drained": True}

        def __init__(self, container_id, exec_fn, **kwargs):
            assert container_id == "a" * 64
            assert kwargs["expected_mounts"] == {"/app": "/owned/repo"}
            self.exec_fn = exec_fn

        async def start(self):
            events.append("start")
            if start_error:
                raise start_error

        async def exec(self, **kwargs):
            raise AssertionError("Unexpected control request")

        async def aclose(self):
            events.append("close")
            if close_error:
                raise close_error

        def raise_if_poisoned(self):
            if self.poisoned:
                raise RuntimeError("poisoned")

    async def identity(*args):
        return "a" * 64, {"/app": "/owned/repo"}

    async def run(self, instruction, environment, context):
        assert self._control_exec == self._controller.exec
        events.append("run")
        self._run_index += 1
        self._snapshot = {"final_fs_hash": digest}
        self._completed = primary is None
        self._cleanup_errors = []
        self._failure = {"type": type(primary).__name__} if primary else None
        self._wall_s = .1
        if primary:
            raise primary

    async def after(self, exec_fn, environment):
        events.append("after")
        if hash_error:
            raise hash_error
        return digest

    monkeypatch.setattr(sidecar_runtime, "Controller", Controller)
    monkeypatch.setattr(sidecar_agent, "task_identity", identity)
    monkeypatch.setattr(SFXLiveAgent, "run", run)
    monkeypatch.setattr(SidecarSFXAgent, "_after_cleanup_hash", after)
    monkeypatch.setattr(sidecar_agent.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    agent = SidecarSFXAgent(tmp_path)
    environment = SimpleNamespace(exec=lambda: None, default_user="root")
    context = SimpleNamespace(metadata={"sfx_live": {"stale": True}, "unrelated": 1})
    return agent, environment, context, events


def test_controller_brackets_entire_checkpoint_and_cleanup_is_measured(tmp_path, monkeypatch):
    agent, env, context, events = lifecycle(tmp_path, monkeypatch)
    asyncio.run(agent.run("task", env, context))
    assert events == ["start", "run", "close", "after"]
    assert agent._controller is agent._control_exec is None
    record = context.metadata["sfx_live"]
    assert record["controller"] == "isolated_docker_sidecar"
    assert record["completed"]
    assert record["after_cleanup_fs_hash"] == record["final_fs_hash"]
    assert record["lifecycle_wall_s"] > 0
    assert context.metadata["unrelated"] == 1
    assert not agent._sidecar_poisoned


def test_cleanup_error_invalidates_record_and_prevents_later_checkpoint(tmp_path, monkeypatch):
    error = RuntimeError("uncertain worker completion")
    agent, env, context, events = lifecycle(tmp_path, monkeypatch, close_error=error)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(agent.run("task", env, context))
    assert caught.value is error
    assert not context.metadata["sfx_live"]["completed"]
    assert context.metadata["sfx_live"]["cleanup_errors"][0]["phase"] == "sidecar_cleanup"
    assert agent._sidecar_poisoned
    with pytest.raises(RuntimeError, match="Prior controller failure"):
        asyncio.run(agent.run("next checkpoint", env, context))
    assert events == ["start", "run", "close"]
    assert "sfx_live" not in context.metadata


def test_primary_error_survives_cleanup_failure(tmp_path, monkeypatch):
    primary = TimeoutError("tool failed")
    agent, env, context, events = lifecycle(
        tmp_path, monkeypatch, primary=primary, close_error=RuntimeError("cleanup"))
    with pytest.raises(TimeoutError) as caught:
        asyncio.run(agent.run("task", env, context))
    assert caught.value is primary
    assert context.metadata["sfx_live"]["failure"]["type"] == "TimeoutError"


def test_startup_failure_still_closes_and_does_not_reuse_previous_record(tmp_path, monkeypatch):
    error = RuntimeError("startup failure")
    agent, env, context, events = lifecycle(tmp_path, monkeypatch, start_error=error)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(agent.run("task", env, context))
    assert caught.value is error
    assert events == ["start", "close"]
    assert "sfx_live" not in context.metadata
    assert agent._controller is agent._control_exec is None


def test_post_cleanup_filesystem_change_invalidates_trial(tmp_path, monkeypatch):
    agent, env, context, events = lifecycle(
        tmp_path, monkeypatch, hash_error=RuntimeError("Filesystem changed"))
    with pytest.raises(RuntimeError, match="Filesystem changed"):
        asyncio.run(agent.run("task", env, context))
    assert not context.metadata["sfx_live"]["completed"]
    assert context.metadata["sfx_live"]["after_cleanup_fs_hash"] is None
    assert agent._sidecar_poisoned


@pytest.mark.parametrize("digest,expected_error", [
    ("sha256:" + "c" * 64, None),
    ("c" * 64, "Invalid post-controller"),
    ("sha256:" + "d" * 64, "Filesystem changed"),
    (None, "Invalid post-controller"),
])
def test_post_cleanup_snapshot_checks_real_hash_format(tmp_path, digest, expected_error):
    agent = SidecarSFXAgent(tmp_path)
    agent._sidecar_root = tmp_path
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/sfx-trace.jsonl").write_text('{"ev":"fork_execution","phase":"completed"}\n')
    agent._snapshot = {"final_fs_hash": "sha256:" + "c" * 64, "trace": []}

    async def execute(**kwargs):
        assert "sfx_client_cli snapshot" in kwargs["command"]
        return SimpleNamespace(return_code=0, stdout=json.dumps({"final_fs_hash": digest}))

    call = agent._after_cleanup_hash(execute, SimpleNamespace(default_user="root"))
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == digest
        assert agent._snapshot["trace"] == [{"ev": "fork_execution", "phase": "completed"}]
