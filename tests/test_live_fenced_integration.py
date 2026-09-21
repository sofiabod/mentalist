import asyncio
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.protocol import serve
from eval import sfx_live_agent
from eval.capture import fs_hash
from eval.sfx_daemon_run import build_daemon, _run
from harbor.environments.base import ExecResult


def _multistep_model_fn(base_url, model, api_key, messages):
    responses = [
        "```bash\ncat > sfx_probe.py <<'SFX_EOF'\nprint('sfx multistep stub')\nSFX_EOF\n```",
        "```bash\npython sfx_probe.py\n```",
        "```bash\nls -la\n```",
    ]
    step = sum(message["role"] == "assistant" for message in messages)
    return responses[step] if step < len(responses) else "DONE"


class LocalSandbox:
    default_user = "agent"

    def __init__(self, root, session_id):
        self.root = root
        self.session_id = session_id
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "sfx_client_cli" not in command:
            self.commands.append(command)
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd or self.root, env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode)


def review_greet(monkeypatch, source):
    """Authorize this test's fixed, inspected program, not arbitrary model code."""
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([{
        "script": "greet.py", "positionals": 0, "path_options": [],
        "value_options": [], "flags": [], "required": [],
        "source_sha256": {"greet.py": hashlib.sha256(source.encode()).hexdigest()},
    }]))


@pytest.fixture
def live_stack(tmp_path, monkeypatch):
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    repo.mkdir()
    scratch.mkdir()
    (repo / "greet.py").write_text("print('OLD')\n")
    monkeypatch.setenv("SFX_REPO", str(repo))
    monkeypatch.setenv("SFX_SEPARATE_STDERR", "1")
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    review_greet(monkeypatch, "print('OLD')\n")
    monkeypatch.setattr(sfx_live_agent, "APP", str(repo))
    monkeypatch.setattr(sfx_live_agent, "SCRATCH", str(scratch))
    monkeypatch.setattr(sfx_live_agent, "TRACE", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(sfx_live_agent, "SFX_SRC", str(Path(__file__).resolve().parents[1] / "src"))
    table = {"k": 1, "min_support": 1, "tau": .35,
             "table": {"main|edit|edit:OK": {"support": 20, "p": {"run": .99}}}}
    daemon = build_daemon(1, table=table)
    daemon.resolve_args = lambda kind, ctx: {"cmd": "python3 greet.py"}
    with tempfile.TemporaryDirectory(prefix="sfx-socket-", dir="/tmp") as sockdir:
        socket = str(Path(sockdir) / "daemon.sock")
        monkeypatch.setattr(sfx_live_agent, "SOCKET", socket)
        server = serve(daemon, socket)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield repo, scratch, daemon
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            daemon.shutdown()


def make_agent(tmp_path, repo, arm="ON"):
    # Use the constructor so per-instance transport paths honor the fixture's
    # patched defaults, just as the hosted adapter's explicit paths do.
    agent = sfx_live_agent.SFXLiveAgent(tmp_path / f"logs-{arm}", arm=arm, repo=repo)
    agent.arm, agent.depth = arm, 1
    agent.logs_dir = tmp_path / f"logs-{arm}"
    agent._session = f"live-{arm}"
    agent._default_cwd = str(repo)
    agent._route_lock = asyncio.Lock()
    agent.trajectory = None
    agent._receipts = []
    agent._raw = []
    agent._last_served = None
    agent._counts = dict(resolves=0, hits=0, misses=0, never_routed=0,
                         authoritative=0, writes_fed=0)
    return agent


def test_real_wrapper_cli_and_fork_preserve_stderr_and_advance_after_edit(live_stack, tmp_path, monkeypatch):
    repo, scratch, daemon = live_stack
    agent = make_agent(tmp_path, repo)
    env = LocalSandbox(repo, "real-local-sandbox")
    body = "import sys\nprint('NEW')\nsys.stderr.write('ERROR_CHANNEL\\n')\n"
    review_greet(monkeypatch, body)

    async def drive():
        await agent._cli(env, "begin", agent._session, repo, scratch, "0")
        try:
            await agent._route(env, env.exec, f"cat > greet.py <<'EOF'\n{body}EOF",
                               None, None, None, None)
            chain = daemon.sessions[agent._session].chain
            assert chain is not None
            if chain.future is not None:
                await asyncio.to_thread(chain.future.result, 5)
            result = await agent._route(env, env.exec, "python3 greet.py", None, None, None, None)
            assert (result.stdout, result.stderr, result.return_code) == ("NEW\n", "ERROR_CHANNEL\n", 0)
            assert "python3 greet.py" not in env.commands
            assert agent._counts["hits"] == 1
            assert (repo / "greet.py").read_text() == body
            assert agent._receipts[-1]["stderr_sha"] == sfx_live_agent._sha("ERROR_CHANNEL\n")
        finally:
            await agent._cli(env, "end", agent._session)

    asyncio.run(drive())


def test_real_unextractable_write_invalidates_cached_result(live_stack, tmp_path):
    repo, scratch, daemon = live_stack
    agent = make_agent(tmp_path, repo)
    env = LocalSandbox(repo, "edit-sandbox")

    async def drive():
        await agent._cli(env, "begin", agent._session, repo, scratch, "0")
        try:
            daemon.sessions[agent._session].cache.put(
                "run", {"cmd": "python3 greet.py"}, 0, result=("STALE\n", "", 0))
            await agent._route(env, env.exec, "cp replacement.py greet.py", None, None, None, None)
            result = await agent._route(env, env.exec, "python3 greet.py", None, None, None, None)
            assert result.stdout == "REPLACED\n"
        finally:
            await agent._cli(env, "end", agent._session)

    (repo / "replacement.py").write_text("print('REPLACED')\n")
    asyncio.run(drive())


def test_wrapper_persists_actual_environment_identity_and_full_receipts(live_stack, tmp_path):
    repo, _, _ = live_stack
    agent = make_agent(tmp_path, repo, "OFF")
    env = LocalSandbox(repo, "harbor-instance-123")
    context = SimpleNamespace(metadata={})

    class Script:
        async def run(self, instruction, environment, context):
            await environment.exec(command="python3 greet.py")

    agent._load_wrapped = lambda environment: Script()
    asyncio.run(agent.run("probe", env, context))
    record = json.loads((agent.logs_dir / "sfx-live-OFF.json").read_text())
    assert record["env_id"] == env.session_id
    assert record["completed"] is True and record["wall_s"] > 0
    assert record["initial_fs_hash"] == record["final_fs_hash"] == fs_hash(repo)
    assert len(record["receipts"]) == 1
    assert not (repo / ".sfx").exists()


@pytest.mark.parametrize("arm", ["OFF", "ON"])
def test_existing_live_model_factory_runs_cpu_stub_through_fenced_wrapper(live_stack, tmp_path, arm):
    repo, _, _ = live_stack
    agent = make_agent(tmp_path, repo, arm)
    env = LocalSandbox(repo, f"model-loop-{arm}")
    agent.wrapped = "eval.sfx_live_agent:build_live_agent"
    agent.artifacts = []
    agent.live_config = {"base_url": "unused", "model": "cpu-stub", "api_key": "unused",
                         "max_steps": 6,
                         "model_fn": f"{__name__}:_multistep_model_fn"}
    context = SimpleNamespace(metadata={})
    asyncio.run(agent.run("CPU model-loop wiring probe", env, context))
    assert context.metadata["sfx_live_trajectory"]["steps"] == 3
    assert context.metadata["sfx_live"]["completed"] is True
    assert len(agent._receipts) == 3
    assert all(r["returncode"] == 0 for r in agent._receipts)
    assert (repo / "sfx_probe.py").read_text() == "print('sfx multistep stub')\n"


def test_production_runner_rechecks_resolver_command_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("SFX_REPO", str(tmp_path))
    with pytest.raises(ValueError, match="source-pinned read-only contract"):
        _run("test", {"cmd": "pytest --version && touch unexpected.txt"})
    assert not (tmp_path / "unexpected.txt").exists()
