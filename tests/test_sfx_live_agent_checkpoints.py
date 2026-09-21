"""Harbor checkpoint lifecycle checks with no model endpoint or container."""
import asyncio
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from eval import live_ab, sfx_live_agent
from harbor.models.agent.context import AgentContext


class CheckpointEnvironment:
    default_user = "agent"
    session_id = "persistent-workspace"

    def __init__(self):
        self.workspace = "initial"
        self.commands = []

    async def exec(self, command, **kwargs):
        if command == "pwd":
            return SimpleNamespace(stdout="/app\n", stderr="", return_code=0)
        self.commands.append((command, self.workspace))
        self.workspace = command
        return SimpleNamespace(stdout=self.workspace, stderr="", return_code=0)


def checkpoint_agent(tmp_path, monkeypatch, model_fn):
    agent = sfx_live_agent.SFXLiveAgent(
        tmp_path, arm="OFF", wrapped="eval.sfx_live_agent:build_live_agent",
        base_url="https://cpu.invalid/v1", model="cpu-stub", api_key="unused",
        model_fn="checkpoint:stub", streaming=False, scaffold="general")
    monkeypatch.setattr(sfx_live_agent, "_resolve_model_fn", lambda spec: model_fn)
    environment = CheckpointEnvironment()
    trace = [{"checkpoint": "preexisting"}]
    operations = []

    async def control(env, op, *args):
        operations.append((op, args[0]))
        if op == "snapshot":
            return {"final_fs_hash": env.workspace, "trace": list(trace)}
        if op == "begin":
            trace.append({"checkpoint": agent._run_index, "event": "begin"})
        if op == "mutation_begin":
            return {"mutation_id": "fence"}
        if op == "end":
            trace.append({"checkpoint": agent._run_index, "event": "end"})
        return {"ok": True}

    monkeypatch.setattr(agent, "_cli", control)
    return agent, environment, operations


def test_checkpoints_preserve_workspace_but_reset_conversation_and_records(tmp_path, monkeypatch):
    prompts = []

    def model_fn(base_url, model, api_key, messages):
        if len(messages) == 2:
            prompts.append(copy.deepcopy(messages))
            return f"```bash\nprintf checkpoint-{len(prompts)}\n```"
        return "DONE"

    agent, environment, operations = checkpoint_agent(tmp_path, monkeypatch, model_fn)
    original_exec = environment.exec
    first = AgentContext(metadata={"private_diagnostic": "HIDDEN_GRADER_SENTINEL"})
    second = AgentContext()

    async def run():
        await agent.run("first public checkpoint", environment, first)
        first_copy = copy.deepcopy(first.metadata)
        await agent.run("second public checkpoint", environment, second)
        assert first.metadata == first_copy

    asyncio.run(run())
    records = [first.metadata["sfx_live"], second.metadata["sfx_live"]]
    assert environment.exec == original_exec
    assert environment.commands == [
        ("printf checkpoint-1", "initial"),
        ("printf checkpoint-2", "printf checkpoint-1"),
    ]
    assert [row["run_index"] for row in records] == [1, 2]
    assert records[0]["session_id"] != records[1]["session_id"]
    assert records[1]["initial_fs_hash"] == records[0]["final_fs_hash"]
    for index, record in enumerate(records, start=1):
        assert record["counts"]["authoritative"] == 1
        assert len(record["raw"]) == len(record["receipts"]) == 1
        assert record["raw"][0]["index"] == 0
        assert {event["checkpoint"] for event in record["trace"]} == {index}
        assert record["completed"] and record["failure"] is None
        assert record["lifecycle_wall_s"] >= record["wall_s"] >= 0
        assert record["scaffold"] == record["sfx_live_trajectory"]["scaffold"] == "general"
    assert prompts[1][1]["content"] == "Task:\nsecond public checkpoint\n\nBegin."
    assert "first public checkpoint" not in json.dumps(prompts[1])
    assert "HIDDEN_GRADER_SENTINEL" not in json.dumps(prompts)
    assert [op for op, _ in operations].count("begin") == 2
    paths = [tmp_path / "sfx-live-OFF.json", tmp_path / "sfx-live-OFF-run-0002.json"]
    assert [json.loads(path.read_text()) for path in paths] == records


def test_preflight_failure_does_not_reuse_previous_checkpoint_metadata(tmp_path, monkeypatch):
    agent, environment, operations = checkpoint_agent(tmp_path, monkeypatch, lambda *args: "DONE")
    context = AgentContext(metadata={"keep": "unrelated"})
    asyncio.run(agent.run("first", environment, context))
    original_control = agent._cli

    async def failed_begin(env, op, *args):
        if op == "begin":
            raise RuntimeError("checkpoint begin failed")
        return await original_control(env, op, *args)

    monkeypatch.setattr(agent, "_cli", failed_begin)
    with pytest.raises(RuntimeError, match="checkpoint begin failed"):
        asyncio.run(agent.run("second", environment, context))
    record = context.metadata["sfx_live"]
    assert context.metadata["keep"] == "unrelated"
    assert "sfx_live_trajectory" not in context.metadata
    assert "sfx_live_trajectory" not in record
    assert not record["completed"]
    assert record["wall_s"] is None and record["lifecycle_wall_s"] >= 0
    assert not any(record["counts"].values())
    assert record["raw"] == record["receipts"] == []
    assert record["failure"] == {"type": "RuntimeError", "message": "checkpoint begin failed"}
    assert operations[-2][0] == "end"
    assert agent._raw_exec is None
    assert json.loads((tmp_path / "sfx-live-OFF-run-0002.json").read_text()) == record


def test_initial_snapshot_failure_is_persisted_without_stale_trace(tmp_path, monkeypatch):
    agent, environment, _ = checkpoint_agent(tmp_path, monkeypatch, lambda *args: "DONE")
    original_control = agent._cli
    failed = False

    async def first_snapshot_fails(env, op, *args):
        nonlocal failed
        if op == "snapshot" and not failed:
            failed = True
            raise RuntimeError("snapshot unavailable")
        return await original_control(env, op, *args)

    monkeypatch.setattr(agent, "_cli", first_snapshot_fails)
    context = AgentContext()
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        asyncio.run(agent.run("public checkpoint", environment, context))
    record = context.metadata["sfx_live"]
    assert record["trace"] == []
    assert record["initial_fs_hash"] is None
    assert not record["completed"]
    assert (tmp_path / "sfx-live-OFF.json").is_file()


@pytest.mark.parametrize("model_fails", [False, True])
def test_cleanup_failure_keeps_primary_failure_and_final_snapshot(tmp_path, monkeypatch, model_fails):
    def model_fn(*args):
        if model_fails:
            raise ValueError("model interrupted")
        return "DONE"

    agent, environment, _ = checkpoint_agent(tmp_path, monkeypatch, model_fn)
    original_exec = environment.exec
    original_control = agent._cli

    async def failed_end(env, op, *args):
        if op == "end":
            raise RuntimeError("daemon cleanup failed")
        return await original_control(env, op, *args)

    monkeypatch.setattr(agent, "_cli", failed_end)
    context = AgentContext()
    expected = ValueError if model_fails else RuntimeError
    with pytest.raises(expected, match="model interrupted" if model_fails else "daemon cleanup failed"):
        asyncio.run(agent.run("public checkpoint", environment, context))
    record = context.metadata["sfx_live"]
    assert environment.exec == original_exec
    assert not record["completed"]
    assert record["failure"]["type"] == expected.__name__
    assert record["cleanup_errors"] == [{"phase": "end", "type": "RuntimeError",
                                           "message": "daemon cleanup failed"}]
    assert record["final_fs_hash"] == "initial"
    assert record["sfx_live_trajectory"]["stop_reason"] == ("model_error" if model_fails else "done")


@pytest.mark.parametrize("scaffold", ["task", "general"])
def test_scaffold_is_explicit_and_does_not_force_checks_for_general(tmp_path, monkeypatch, scaffold):
    captured = []

    def model_fn(base_url, model, api_key, messages):
        captured.append(messages[0]["content"])
        return "DONE"

    agent, environment, _ = checkpoint_agent(tmp_path, monkeypatch, model_fn)
    agent.live_config["scaffold"] = scaffold
    context = AgentContext()
    asyncio.run(agent.run("build a CLI", environment, context))
    if scaffold == "task":
        assert captured == [live_ab._TASK_SUBMISSION_SYSTEM]
    else:
        assert captured[0].startswith(live_ab.SYSTEM)
        assert "use checks you find appropriate" in captured[0]
        assert "submission requirements" in captured[0]
        assert "There are no native tools or function calls" in captured[0]
        assert "assistant FINAL response" in captured[0]
        assert "Do not issue native tool calls or tool handoffs" in captured[0]
        assert "reproduce.py" not in captured[0]
        assert "after EACH edit" not in captured[0]
    assert context.metadata["sfx_live_trajectory"]["scaffold"] == scaffold


def test_unknown_scaffold_fails_before_run(tmp_path):
    with pytest.raises(ValueError, match="scaffold"):
        sfx_live_agent.SFXLiveAgent(tmp_path, scaffold="typo")


def test_cancelled_model_thread_cannot_issue_a_late_command(tmp_path, monkeypatch):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    run_agent = live_ab.run_agent

    def model_fn(*args):
        entered.set()
        assert release.wait(5)
        return "```bash\nprintf late-command\n```"

    def tracked_run(*args, **kwargs):
        try:
            return run_agent(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(live_ab, "run_agent", tracked_run)
    agent, environment, _ = checkpoint_agent(tmp_path, monkeypatch, model_fn)
    original_exec = environment.exec
    context = AgentContext()

    async def run():
        task = asyncio.create_task(agent.run("interrupted checkpoint", environment, context))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert await asyncio.to_thread(exited.wait, 5)

    asyncio.run(run())
    assert environment.exec == original_exec
    assert environment.commands == []
    assert context.metadata["sfx_live"]["failure"]["type"] == "CancelledError"
    assert not context.metadata["sfx_live"]["completed"]
