"""SFXLiveAgent interception seam (no sandbox, no daemon): a FakeEnvironment records
exec calls and answers the in-sandbox client CLI with canned daemon replies, so we can
assert the host-side routing contract:
  - cache HIT  -> cached result returned, NO authoritative exec of the tool
  - MISS       -> authoritative exec + a report back to the daemon
  - never-kind -> straight to authoritative (no resolve round-trip)
  - OFF arm    -> begins the session spec_disabled
"""
import asyncio
import base64
import json

import pytest

from eval.sfx_live_agent import SFXLiveAgent


class FakeExecResult:
    def __init__(self, stdout="", stderr="", return_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnv:
    default_user = "agent"
    session_id = "sandbox-test"

    def __init__(self, resolve_reply):
        self.resolve_reply = resolve_reply
        self.tool_execs = []       # authoritative tool commands actually run
        self.cli_ops = []          # sfx_client_cli ops observed
        self.cli_payloads = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "sfx_client_cli" in command:
            parts = command.split("sfx_client_cli", 1)[1].split()
            op = parts[0]
            self.cli_ops.append(op)
            if op in ("mutation_begin", "mutation_end", "report", "feed", "resolve"):
                self.cli_payloads.append((op, json.loads(base64.b64decode(parts[-1]))))
            if op == "resolve":
                return FakeExecResult(stdout=json.dumps(self.resolve_reply))
            if op == "mutation_begin":
                return FakeExecResult(stdout=json.dumps({"mutation_id": "mutation-test"}))
            if op == "mutation_end":
                return FakeExecResult(stdout=json.dumps({"chain_preserved": False}))
            return FakeExecResult(stdout=json.dumps({"ok": True}))
        self.tool_execs.append(command)
        return FakeExecResult(stdout="AUTHORITATIVE:" + command, return_code=0)


def _agent(arm="ON"):
    import tempfile
    from pathlib import Path
    a = SFXLiveAgent.__new__(SFXLiveAgent)
    a.arm = arm
    a._session = "live-test"
    a._receipts = []
    a._raw = []
    a._last_served = None
    a._route_lock = asyncio.Lock()
    a._default_cwd = "/app"
    a.logs_dir = Path(tempfile.mkdtemp())
    a._counts = {"resolves": 0, "hits": 0, "misses": 0, "never_routed": 0,
                 "authoritative": 0, "writes_fed": 0}
    return a


def _route(agent, env, command):
    async def go():
        return await agent._route(env, env.exec, command, None, None, None, None)
    return asyncio.run(go())


def test_cache_hit_serves_cached_and_skips_authoritative():
    env = FakeEnv({"served": True, "outcome": "hit_completed",
                   "output": ["CACHED OUT", "CACHED ERR", 0]})
    agent = _agent("ON")
    result = _route(agent, env, "python3 greet.py")
    assert result.stdout == "CACHED OUT" and result.return_code == 0
    assert result.stderr == "CACHED ERR"
    assert env.tool_execs == []                # served from fork, never ran authoritatively
    assert agent._counts["hits"] == 1
    assert agent._receipts[0]["stderr_sha"]


def test_miss_runs_authoritative_and_reports():
    env = FakeEnv({"served": False, "outcome": "miss", "output": None})
    agent = _agent("ON")
    result = _route(agent, env, "python3 greet.py")
    assert result.stdout == "AUTHORITATIVE:python3 greet.py"
    assert env.tool_execs == ["python3 greet.py"]
    assert "report" in env.cli_ops
    assert agent._counts["misses"] == 1


def test_never_kind_goes_straight_to_authoritative_without_resolve():
    env = FakeEnv({"served": False})
    agent = _agent("ON")
    result = _route(agent, env, "pip install requests")
    assert result.stdout == "AUTHORITATIVE:pip install requests"
    assert "resolve" not in env.cli_ops        # never-kind bypasses the fork/cache entirely
    assert agent._counts["never_routed"] == 1


def test_network_and_append_are_never_routed():
    for cmd in ("curl https://x/y", "echo x >> log.txt"):
        env = FakeEnv({"served": False})
        agent = _agent("ON")
        _route(agent, env, cmd)
        assert "resolve" not in env.cli_ops
        assert agent._counts["never_routed"] == 1


@pytest.mark.parametrize("kwargs", [
    {"cwd": "/app/elsewhere"}, {"env": {"MODE": "different"}},
    {"user": "root"}, {"timeout_sec": 0.1},
])
def test_nondefault_context_cannot_consume_default_context_cache(kwargs):
    env = FakeEnv({"served": True, "output": ["WRONG CONTEXT", "", 0]})
    agent = _agent()
    options = dict(cwd=None, env=None, timeout_sec=None, user=None)
    options.update(kwargs)
    result = asyncio.run(agent._route(env, env.exec, "python3 greet.py", **options))
    assert result.stdout.startswith("AUTHORITATIVE:")
    assert "resolve" not in env.cli_ops
    report = next(p for op, p in env.cli_payloads if op == "report")
    assert report["speculate"] is False and report["args"] == {}


def test_ordinary_edit_fences_and_reports_even_without_extractable_contents():
    env = FakeEnv({"served": False})
    _route(_agent(), env, "sed -i 's/OLD/NEW/' data.txt")
    assert env.cli_ops == ["mutation_begin", "mutation_end", "report"]
    assert env.cli_payloads[-1][1]["verb"] == "fork"


def test_legacy_merged_output_cannot_be_served_as_separate_stderr():
    env = FakeEnv({"served": True, "output": ["MERGED", 0]})
    result = _route(_agent(), env, "python3 greet.py")
    assert result.stdout.startswith("AUTHORITATIVE:")
    assert "mutation_begin" in env.cli_ops


def test_off_never_uses_cached_results():
    env = FakeEnv({"served": True, "output": ["CACHE", "", 0]})
    result = _route(_agent("OFF"), env, "python3 greet.py")
    assert result.stdout.startswith("AUTHORITATIVE:")
    assert "resolve" not in env.cli_ops


def test_unquoted_heredoc_is_not_speculated_as_literal_contents():
    env = FakeEnv({"served": False})
    _route(_agent(), env, "cat > data.txt <<EOF\n$VALUE\nEOF")
    assert "feed" not in env.cli_ops


@pytest.mark.parametrize("path", ["/outside/data.txt", "/application/data.txt", "../data.txt", "."])
def test_unsupported_stream_path_still_runs_authoritatively(path):
    env = FakeEnv({"served": False})
    result = _route(_agent(), env, f"cat > {path} <<'EOF'\nliteral\nEOF")
    assert result.stdout.startswith("AUTHORITATIVE:")
    assert "feed" not in env.cli_ops
    begin = next(p for op, p in env.cli_payloads if op == "mutation_begin")
    assert begin["write_args"] is None


def test_authoritative_exception_closes_mutation_fence():
    env = FakeEnv({"served": False})
    agent = _agent()

    async def failing_exec(**kwargs):
        raise TimeoutError("tool timed out")

    with pytest.raises(TimeoutError):
        asyncio.run(agent._route(env, failing_exec, "touch data.txt", None, None, None, None))
    end = next(p for op, p in env.cli_payloads if op == "mutation_end")
    assert end["success"] is False


def test_concurrent_wrapper_calls_are_serialized():
    env = FakeEnv({"served": False})
    agent = _agent()
    active = maximum = 0

    async def slow_exec(**kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0.01)
        active -= 1
        return FakeExecResult(stdout=kwargs["command"])

    async def run():
        return await asyncio.gather(*(
            agent._route(env, slow_exec, f"touch file{i}", None, None, None, None)
            for i in range(2)))

    results = asyncio.run(run())
    assert maximum == 1
    assert [r.stdout for r in results] == ["touch file0", "touch file1"]
    assert len(agent._receipts) == 2


def test_build_live_agent_honors_task_required_git_submission(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from eval import sfx_live_agent
    from eval.live_ab import _TASK_SUBMISSION_SYSTEM

    task = (
        "Add streaming JSON iteration to HTTPX responses.\n"
        "IMPORTANT: Please work on this in a new branch from main and commit "
        "everything when you are done."
    )
    commands = [
        "git switch -c sfx-httpx main",
        "python reproduce.py",
        "git add reproduce.py",
        "git commit -m 'Implement streaming JSON iteration'",
    ]
    responses = [f"```bash\n{command}\n```" for command in commands] + ["DONE"]
    model_calls = []

    def model_fn(base_url, model, api_key, messages):
        system = messages[0]["content"]
        assert system == _TASK_SUBMISSION_SYSTEM
        assert "Do not use git" not in system
        assert "Use git only when the task requires it" in system
        assert "Honor any requested branch" in system
        assert "commit your work before DONE when required" in system
        assert "DONE only after the tests pass" in system
        assert "all task submission requirements are satisfied" in system
        assert "ONE bash command per step" in system
        assert "create reproduce.py" in system
        assert "run `python reproduce.py` again after EACH edit" in system
        assert messages[1] == {"role": "user", "content": f"Task:\n{task}\n\nBegin."}
        model_calls.append(model)
        return responses[len(model_calls) - 1]

    def resolve_model_fn(spec):
        assert spec == "cpu-submission-stub"
        return model_fn

    def no_network(*args, **kwargs):
        pytest.fail("injected CPU model must not make network requests")

    monkeypatch.setattr(sfx_live_agent, "_resolve_model_fn", resolve_model_fn)
    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    environment = FakeEnv({"served": False})
    context = SimpleNamespace(metadata={})
    wrapped = sfx_live_agent.build_live_agent(tmp_path, environment, config={
        "base_url": "https://cpu.invalid/v1", "model": "cpu-stub", "api_key": "stub",
        "model_fn": "cpu-submission-stub", "streaming": False,
        "max_steps": len(responses),
    })

    asyncio.run(wrapped.run(task, environment, context))

    assert environment.tool_execs == commands
    trajectory = context.metadata["sfx_live_trajectory"]
    assert trajectory["commands"] == commands
    assert trajectory["stop_reason"] == "done"
    assert trajectory["timing_source"] == "injected_response"
