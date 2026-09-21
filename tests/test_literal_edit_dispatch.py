import asyncio

import pytest

from eval import sfx_live_agent
from test_live_absolute_edits import emit
from test_live_fenced_integration import LocalSandbox, live_stack, make_agent, review_greet
from test_live_write_ablation import route, stack


BODY = "url = 'https://example.invalid'; text = 'pip install example; eval exec'\nprint('NEW')\n"
COMMAND = f"cat > target.py <<'EOF'\n{BODY}EOF"


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("followup,served", [("python3 greet.py", True), ("python3 greet.py ", False)])
def test_literal_body_keeps_one_confirmed_fence_and_exact_followup_guard(
        live_stack, tmp_path, monkeypatch, absolute, followup, served):
    repo, scratch, daemon = live_stack
    marker = repo / "must-not-exist"
    body = BODY + f"shell_text = '$(touch {marker})'\n"
    review_greet(monkeypatch, body)
    target = str(repo / "greet.py") if absolute else "greet.py"
    command = f"cat > {target} <<'EOF'\n{body}EOF"
    prefix = f"```bash\n{command}\n"
    agent = make_agent(tmp_path, repo)
    agent._pending_edit = None
    environment = LocalSandbox(repo, "literal-edit-dispatch")
    operations = []
    cli = agent._cli

    async def recorded_cli(environment, *args):
        operations.append(args[0])
        return await cli(environment, *args)

    agent._cli = recorded_cli

    async def drive():
        await agent._cli(environment, "begin", agent._session, repo, scratch, "0")
        try:
            await emit(agent, environment, "model_delta", text=prefix)
            chain = daemon.sessions[agent._session].chain
            assert chain is not None and chain.future is not None
            await asyncio.to_thread(chain.future.result, 5)
            assert (repo / "greet.py").read_text() == "print('OLD')\n"
            assert not marker.exists()
            assert environment.commands == []
            await emit(agent, environment, "model_end", text=prefix + "```", command=command)
            result = await agent._route(environment, environment.exec, command,
                                        str(repo), None, None, None)
            assert result.return_code == 0
            assert operations.count("mutation_begin") == operations.count("mutation_end") == 1
            assert operations.count("feed") == 1
            assert agent._pending_edit is None
            assert not daemon._mutations
            assert agent._early_edit_events[-1]["chain_preserved"] is True
            assert environment.commands == [command]
            before = list(operations)
            await emit(agent, environment, "model_start")
            assert operations == before
            result = await agent._route(environment, environment.exec, followup,
                                        str(repo), None, None, None)
            assert (result.stdout, result.stderr, result.return_code) == ("NEW\n", "", 0)
            assert agent._raw[-1]["served"] is served
            assert agent._counts["hits"] == int(served)
            assert agent._counts["writes_fed"] == 1
            assert environment.commands == ([command] if served else [command, followup])
            assert agent._receipts[0]["command"] == command
            assert agent._receipts[-1]["command"] == followup
            assert (repo / "greet.py").read_text() == body
            assert not marker.exists()
        finally:
            await agent._cancel_pending_edit(environment, "test_end")
            await agent._cli(environment, "end", agent._session)

    asyncio.run(drive())
    assert agent._pending_edit is None
    assert not daemon._mutations
    assert {path.name for path in scratch.iterdir()} <= {"state"}


@pytest.mark.parametrize("body", [
    "print('NEW'); number = 1\n",
    "url = 'https://example.invalid'\nprint('NEW')\n",
    "label = 'pip install example'\nprint('NEW')\n",
])
def test_unstreamed_literal_body_dispatches_as_edit_without_rewriting_command(tmp_path, body):
    agent, environment, events = stack(tmp_path)
    command = f"cat > target.py <<'EOF'\n{body}EOF"
    route(agent, environment, command)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "feed", "exec", "mutation_end", "resolve", "report"]
    assert events[-1][1]["tool"] == "edit"
    assert events[-1][1]["verb"] == "fork"
    assert events[-1][1]["args"] == {"cmd": command}
    assert environment.commands == [command]
    assert agent._counts["writes_fed"] == agent._counts["authoritative"] == 1


@pytest.mark.parametrize("arm,writes,options", [
    ("OFF", True, {}), ("OFF", False, {}), ("ON", False, {}),
    ("ON", True, {"cwd": "/elsewhere"}),
    ("ON", True, {"env": {"MODE": "different"}}),
    ("ON", True, {"timeout_sec": 0.1}),
    ("ON", True, {"user": "other-user"}),
])
def test_literal_body_dispatch_does_not_expand_arm_or_context_eligibility(
        tmp_path, arm, writes, options):
    agent, environment, events = stack(tmp_path, arm=arm, speculate_writes=writes)
    route(agent, environment, COMMAND, **options)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "exec", "mutation_end", "report"]
    assert events[0][1] == {"write_args": None}
    assert events[-1][1]["tool"] == "edit"
    assert events[-1][1]["verb"] == "fork"
    if options:
        assert events[-1][1]["speculate"] is False
    assert environment.commands == [COMMAND]
    assert agent._counts["writes_fed"] == agent._counts["hits"] == 0


@pytest.mark.parametrize("command", [
    "cat > target.py <<EOF\n$(touch should-not-run)\nEOF",
    COMMAND + "\necho additional; echo command",
    COMMAND.replace("target.py", "../target.py"),
    COMMAND.replace("target.py", "/outside/target.py"),
    "python3 greet.py; echo additional",
    "curl https://example.invalid",
])
def test_unsupported_or_genuine_unsafe_commands_keep_never_policy(tmp_path, command):
    assert sfx_live_agent._literal_write(command, "/app") is None
    agent, environment, events = stack(tmp_path)
    route(agent, environment, command)
    assert [operation for operation, _ in events] == [
        "mutation_begin", "exec", "mutation_end", "report"]
    assert events[0][1] == {"write_args": None}
    assert events[-1][1]["verb"] == "never"
    assert agent._counts["never_routed"] == 1
    assert agent._counts["writes_fed"] == agent._counts["hits"] == 0
    assert environment.commands == [command]
