"""Private failure evidence, without a daemon, shell execution or model endpoint."""
import asyncio
import hashlib
import json
import shlex
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.sfx_live_agent import DAEMON_LOG_CAP, PRIVATE_OUTPUT_CAP, SFXLiveAgent


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=returncode)


class Environment:
    default_user = "agent"
    session_id = "diagnostic-workspace"

    def __init__(self, *, fail_control=(), tool_error=None, tool_result=None,
                 archive_error=None):
        self.fail_control = set(fail_control)
        self.tool_error = tool_error
        self.tool_result = tool_result or result("out\n", "err\n", 3)
        self.archive_error = archive_error
        self.daemon_log = b"DAEMON_PRIVATE_SENTINEL\n" * 4000
        self.tool_commands, self.control_ops, self.archive_commands = [], [], []
        self.downloads = []

    async def exec(self, command, **kwargs):
        if command == "pwd":
            return result("/app\n")
        if "eval.sfx_client_cli" in command:
            words = shlex.split(command)
            op = words[words.index("eval.sfx_client_cli") + 1]
            self.control_ops.append(op)
            if op in self.fail_control:
                return result("CONTROL_STDOUT_SECRET", "[Errno 111] CONTROL_STDERR_SECRET", 1)
            reply = {"snapshot": {"final_fs_hash": "unchanged", "trace": []},
                     "mutation_begin": {"mutation_id": "token"},
                     "mutation_end": {"chain_preserved": False},
                     "resolve": {"served": False}}.get(op, {"ok": True})
            return result(json.dumps(reply))
        if command.startswith("umask 077; set -C;"):
            self.archive_commands.append(command)
            if self.archive_error:
                raise self.archive_error
            return result(str(len(self.daemon_log)) + "\n")
        self.tool_commands.append(command)
        if self.tool_error:
            raise self.tool_error
        return self.tool_result

    async def download_file(self, source_path, target_path):
        self.downloads.append((source_path, target_path))
        Path(target_path).write_bytes(self.daemon_log[-DAEMON_LOG_CAP:])


def run_tool(tmp_path, monkeypatch, environment, command="printf task", *, agent=None):
    agent = agent or SFXLiveAgent(tmp_path, arm="OFF")

    async def wrapped_run(instruction, environment, context):
        await environment.exec(command=command, user=environment.default_user)

    monkeypatch.setattr(agent, "_load_wrapped", lambda env: SimpleNamespace(run=wrapped_run))
    context = SimpleNamespace(metadata={})
    return agent, context, agent.run("public task", environment, context)


def private_record(context, logs):
    return json.loads((logs / context.metadata["sfx_live"]["diagnostics"]["artifact"]).read_text())


def test_command_is_retained_before_control_begin_failure(tmp_path, monkeypatch, capsys):
    environment = Environment(fail_control={"mutation_begin"})
    command = "printf COMMAND_PRIVATE_SENTINEL"
    agent, context, run = run_tool(tmp_path, monkeypatch, environment, command)
    with pytest.raises(RuntimeError, match="mutation_begin failed") as caught:
        asyncio.run(run)

    assert "CONTROL_STDERR_SECRET" not in str(caught.value)
    assert environment.tool_commands == []
    diagnostic = private_record(context, tmp_path)
    tool, = diagnostic["tools"]
    assert tool["command"] == command
    assert tool["tool_start_s"] <= tool["tool_abort_s"]
    assert tool["phase"] == "control:mutation_begin"
    assert tool["result"] is None
    assert "tool_end_s" not in tool
    failure, = diagnostic["control_failures"]
    assert failure["operation"] == "mutation_begin" and failure["tool_index"] == 0
    assert failure["result"]["returncode"] == 1
    assert failure["result"]["stderr"]["classification"] == "connection_refused"
    assert failure["result"]["stderr"]["errno"] == 111
    assert "CONTROL_STDERR_SECRET" not in json.dumps(diagnostic)
    assert "COMMAND_PRIVATE_SENTINEL" not in json.dumps(context.metadata["sfx_live"])
    assert context.metadata["sfx_live"]["raw"] == context.metadata["sfx_live"]["receipts"] == []
    assert not any(agent._counts[key] for key in ("authoritative", "hits", "writes_fed"))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failed_operation", ["mutation_end", "report"])
def test_authoritative_result_survives_later_control_failure(
        tmp_path, monkeypatch, failed_operation):
    stdout = "AUTHORITATIVE_PRIVATE_SENTINEL" + "o" * PRIVATE_OUTPUT_CAP
    stderr = "stderr-only\n"
    environment = Environment(fail_control={failed_operation},
                              tool_result=result(stdout, stderr, 137))
    _, context, run = run_tool(tmp_path, monkeypatch, environment)
    with pytest.raises(RuntimeError, match=failed_operation + " failed"):
        asyncio.run(run)

    tool, = private_record(context, tmp_path)["tools"]
    assert environment.tool_commands == ["printf task"]  # no retry or fallback
    assert tool["tool_start_s"] <= tool["authoritative_start_s"]
    assert tool["authoritative_start_s"] <= tool["authoritative_end_s"] <= tool["tool_abort_s"]
    assert "tool_end_s" not in tool
    assert tool["phase"] == "control:" + failed_operation
    observed = tool["result"]
    assert observed["returncode"] == 137
    assert observed["stdout"]["sha256"] == hashlib.sha256(stdout.encode()).hexdigest()
    assert observed["stderr"]["sha256"] == hashlib.sha256(stderr.encode()).hexdigest()
    assert observed["stdout"]["bytes"] == len(stdout.encode())
    assert observed["preview_truncated"]
    assert "stderr-only\n" in observed["stdout_then_stderr_preview"]
    assert len(observed["stdout_then_stderr_preview"].encode()) <= PRIVATE_OUTPUT_CAP
    main = context.metadata["sfx_live"]
    assert main["raw"] == main["receipts"] == []
    assert "AUTHORITATIVE_PRIVATE_SENTINEL" not in json.dumps(main)


def test_original_exec_error_wins_over_mutation_end_and_archive_errors(tmp_path, monkeypatch):
    primary = TimeoutError("original tool timed out")
    environment = Environment(fail_control={"mutation_end"}, tool_error=primary,
                              archive_error=OSError(2, "ARCHIVE_PRIVATE_SENTINEL"))
    _, context, run = run_tool(tmp_path, monkeypatch, environment)
    with pytest.raises(TimeoutError) as caught:
        asyncio.run(run)

    assert caught.value is primary
    diagnostic = private_record(context, tmp_path)
    tool, = diagnostic["tools"]
    assert tool["phase"] == "authoritative_execution"
    assert tool["execution_error"]["type"] == "TimeoutError"
    assert tool["mutation_end_error"]["type"] == "RuntimeError"
    assert tool["result"] is None
    assert tool["authoritative_start_s"] <= tool["authoritative_abort_s"] <= tool["tool_abort_s"]
    assert "authoritative_end_s" not in tool
    assert diagnostic["daemon_log"]["status"] == "archive_failed"
    assert diagnostic["daemon_log"]["error"]["classification"] == "not_found"
    assert "ARCHIVE_PRIVATE_SENTINEL" not in json.dumps(context.metadata)
    assert context.metadata["sfx_live"]["failure"]["type"] == "TimeoutError"
    assert environment.tool_commands == ["printf task"]


def test_feed_failure_does_not_invent_authoritative_execution(tmp_path, monkeypatch):
    environment = Environment(fail_control={"feed", "mutation_end"})
    agent = SFXLiveAgent(tmp_path, arm="ON")
    _, context, run = run_tool(tmp_path, monkeypatch, environment,
                              "cat > data.txt <<'EOF'\ncontents\nEOF", agent=agent)
    with pytest.raises(RuntimeError, match="feed failed"):
        asyncio.run(run)
    tool, = private_record(context, tmp_path)["tools"]
    assert tool["phase"] == "control:feed"
    assert tool["mutation_end_error"]["type"] == "RuntimeError"
    assert not any(key.startswith("authoritative_") for key in tool)
    assert tool["result"] is None and environment.tool_commands == []


@pytest.mark.parametrize("failed", [False, True])
def test_daemon_log_archived_privately_even_when_daemon_unavailable(tmp_path, monkeypatch, failed):
    environment = Environment(fail_control={"mutation_begin", "end"} if failed else ())
    _, context, run = run_tool(tmp_path, monkeypatch, environment)
    if failed:
        with pytest.raises(RuntimeError, match="mutation_begin failed"):
            asyncio.run(run)
    else:
        asyncio.run(run)

    summary = context.metadata["sfx_live"]["diagnostics"]
    archived = summary["daemon_log"]
    assert archived["status"] == "archived"
    assert archived["bytes"] == archived["tail_limit_bytes"] == DAEMON_LOG_CAP
    assert archived["observed_source_bytes"] == len(environment.daemon_log)
    path = tmp_path / archived["artifact"]
    assert path.read_bytes() == environment.daemon_log[-DAEMON_LOG_CAP:]
    assert archived["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    private_path = tmp_path / summary["artifact"]
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert summary["sha256"] == hashlib.sha256(private_path.read_bytes()).hexdigest()
    assert len(environment.archive_commands) == len(environment.downloads) == 1
    command, = environment.archive_commands
    assert "tail -c 65536 -- /tmp/sfx-daemon.log" in command
    assert "python" not in command and "sfx_client_cli" not in command
    assert "DAEMON_PRIVATE_SENTINEL" not in json.dumps(context.metadata)


def test_checkpoint_diagnostics_do_not_reuse_prior_tools_or_artifacts(tmp_path, monkeypatch):
    environment = Environment()
    agent, first, run = run_tool(tmp_path, monkeypatch, environment, "printf first")
    asyncio.run(run)
    first_record = private_record(first, tmp_path)
    _, second, run = run_tool(tmp_path, monkeypatch, environment, "printf second", agent=agent)
    asyncio.run(run)
    second_record = private_record(second, tmp_path)
    assert [tool["command"] for tool in first_record["tools"]] == ["printf first"]
    assert [tool["command"] for tool in second_record["tools"]] == ["printf second"]
    assert private_record(first, tmp_path) == first_record
    assert (first.metadata["sfx_live"]["diagnostics"]["artifact"]
            != second.metadata["sfx_live"]["diagnostics"]["artifact"])
    assert first_record["daemon_log"]["artifact"] != second_record["daemon_log"]["artifact"]


def test_in_sandbox_transport_failure_is_recorded_without_payload(tmp_path):
    agent = SFXLiveAgent(tmp_path, control_transport="in_sandbox")
    primary = ConnectionRefusedError(111, "CONTROL_PRIVATE_SENTINEL")

    async def request(*args):
        raise primary

    agent._control_channel = SimpleNamespace(request=request)
    with pytest.raises(ConnectionRefusedError) as caught:
        asyncio.run(agent._cli(Environment(), "mutation_begin", "SECRET_SESSION", "SECRET_PAYLOAD"))
    assert caught.value is primary
    error, = agent._diagnostic_state()["control_failures"]
    assert error["operation"] == "mutation_begin" and error["result"] is None
    assert error["error"]["classification"] == "connection_refused"
    assert "SECRET" not in json.dumps(error) and "PRIVATE_SENTINEL" not in json.dumps(error)


def test_private_artifact_error_cannot_mask_execution_failure(tmp_path, monkeypatch):
    from eval import sfx_live_agent

    primary = TimeoutError("original tool timed out")
    environment = Environment(tool_error=primary)
    _, context, run = run_tool(tmp_path, monkeypatch, environment)

    def fail_private(*args):
        raise PermissionError(13, "PRIVATE_FILESYSTEM_SENTINEL")

    monkeypatch.setattr(sfx_live_agent, "_write_private", fail_private)
    with pytest.raises(TimeoutError) as caught:
        asyncio.run(run)
    assert caught.value is primary
    summary = context.metadata["sfx_live"]["diagnostics"]
    assert summary["status"] == "persist_failed"
    assert summary["error"]["classification"] == "permission_denied"
    assert "PRIVATE_FILESYSTEM_SENTINEL" not in json.dumps(summary)


def test_private_result_preview_is_bounded_at_multibyte_boundary():
    from eval.sfx_live_agent import _diagnostic_result

    stdout = "A" + "€" * PRIVATE_OUTPUT_CAP
    diagnostic = _diagnostic_result(result(stdout), private_preview=True)
    assert len(diagnostic["stdout_then_stderr_preview"].encode()) <= PRIVATE_OUTPUT_CAP
    assert diagnostic["stdout"]["bytes"] == len(stdout.encode())
    assert diagnostic["stdout"]["sha256"] == hashlib.sha256(stdout.encode()).hexdigest()
    assert diagnostic["preview_truncated"]


@pytest.mark.parametrize("primary", [None, TimeoutError("original execution timeout")])
def test_archive_cancellation_is_not_swallowed_or_allowed_to_mask_primary(
        tmp_path, monkeypatch, primary):
    cancellation = asyncio.CancelledError("cancelled during daemon log capture")
    environment = Environment(tool_error=primary, archive_error=cancellation)
    _, context, run = run_tool(tmp_path, monkeypatch, environment)
    expected = primary if primary is not None else cancellation
    with pytest.raises(type(expected)) as caught:
        asyncio.run(run)
    assert caught.value is expected
    record = context.metadata["sfx_live"]
    assert record["completed"] is False
    assert record["failure"]["type"] == type(expected).__name__
    assert record["cleanup_errors"][-1]["phase"] == "daemon_log"
    assert record["cleanup_errors"][-1]["type"] == "CancelledError"
    assert record["diagnostics"]["daemon_log"]["status"] == "archive_failed"
