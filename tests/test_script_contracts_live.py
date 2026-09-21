import asyncio
import hashlib
import json

import pytest

from eval import sfx_daemon_run
from eval.capture import fs_hash
from sfx.script_contracts import ScriptContracts, ScriptSourceMismatch
from test_live_fenced_integration import live_stack
from test_live_stream_integration import _live_agent


SOURCE = (
    "import argparse, json, sys\n"
    "from pathlib import Path\n"
    "parser = argparse.ArgumentParser()\n"
    "parser.add_argument('input')\n"
    "parser.add_argument('--rules', required=True)\n"
    "args = parser.parse_args()\n"
    "rules = json.loads(Path(args.rules).read_text())\n"
    "print(rules['prefix'] + ':' + Path(args.input).read_text().strip())\n"
    "sys.stderr.write('checked\\n')\n"
)
COMMAND = "python3 code_search.py input.txt --rules=rules.json"
CONTRACT = {
    "script": "code_search.py", "positionals": 1, "path_options": ["--rules"],
    "value_options": [], "flags": [], "required": ["--rules"],
    "source_sha256": {"code_search.py": hashlib.sha256(SOURCE.encode()).hexdigest()},
}


@pytest.mark.parametrize("case", ["data", "rules", "different_args", "intervening_state", "source"])
def test_reviewed_argumentful_cli_uses_real_fork_and_exact_prior_history(
        live_stack, tmp_path, monkeypatch, case):
    repo, _, daemon = live_stack
    (repo / "code_search.py").write_text(SOURCE)
    (repo / "input.txt").write_text("OLD\n")
    (repo / "other.txt").write_text("OTHER\n")
    (repo / "rules.json").write_text('{"prefix":"OLD_RULE"}\n')
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([CONTRACT]))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    daemon.resolve_args = sfx_daemon_run._resolve_args
    events = []
    daemon.log = events.append
    path, content = "input.txt", "NEW\n"
    if case == "rules":
        path, content = "rules.json", '{"prefix":"NEW_RULE"}\n'
    elif case == "source":
        path, content = "code_search.py", "print('UNREVIEWED_SOURCE')\n"
    previous = (repo / path).read_text()
    edit = f"cat > {path} <<'SFX_REVIEWED_EOF'\n{content}SFX_REVIEWED_EOF"
    actual = COMMAND.replace("input.txt", "other.txt") if case == "different_args" else COMMAND
    intervening = "python3 -c \"open('input.txt', 'w').write('LATEST\\n')\""
    observed = []

    def stream(*args, **kwargs):
        messages = args[3]
        step = sum(message["role"] == "assistant" for message in messages)
        if step == 0:
            assert not daemon.sessions[agent._session].arg_by_kind
            yield {"type": "delta", "content": f"```bash\n{COMMAND}\n```"}
        elif step == 1:
            session = daemon.sessions[agent._session]
            assert session.arg_by_kind["run"] == COMMAND
            assert session.trajectory is None
            yield {"type": "delta", "content": f"```bash\n{edit}\n"}
            chain = session.chain
            assert chain is not None and chain.future is not None
            assert chain.hops[0].args == {"cmd": COMMAND}
            if case == "source":
                with pytest.raises(ScriptSourceMismatch):
                    chain.future.result(timeout=5)
            else:
                chain.future.result(timeout=5)
                result = next(job.result for job in session.cache.jobs.values()
                              if job.spec_id == session.chain_spec_ids[0])
                observed.append(result)
                expected = "NEW_RULE:OLD\n" if case == "rules" else "OLD_RULE:NEW\n"
                assert result == (expected, "checked\n", 0)
            assert (repo / path).read_text() == previous
            assert not agent._pending_edit["confirmed"]
            yield {"type": "delta", "content": "```"}
        elif step == 2 and case == "intervening_state":
            yield {"type": "delta", "content": f"```bash\n{intervening}\n```"}
        elif step == (3 if case == "intervening_state" else 2):
            yield {"type": "delta", "content": f"```bash\n{actual}\n```"}
        else:
            yield {"type": "delta", "content": "DONE"}
        yield {"type": "done", "finish_reason": "stop"}

    agent, environment, context = _live_agent(tmp_path, repo, monkeypatch, stream)
    agent.script_contracts = ScriptContracts([CONTRACT])
    agent.live_config["max_steps"] = 6
    asyncio.run(agent.run("Controlled reviewed-CLI mechanism test", environment, context))

    should_hit = case in ("data", "rules")
    assert agent._counts["hits"] == int(should_hit)
    assert agent._counts["writes_fed"] == 1
    assert agent._raw[0]["served"] is False
    assert agent._raw[-1]["command"] == actual
    assert agent._raw[-1]["served"] is should_hit
    assert agent._raw[0]["stdout"] == "OLD_RULE:OLD\n"
    expected = {
        "data": "OLD_RULE:NEW\n", "rules": "NEW_RULE:OLD\n",
        "different_args": "OLD_RULE:OTHER\n", "intervening_state": "OLD_RULE:LATEST\n",
        "source": "UNREVIEWED_SOURCE\n",
    }[case]
    assert agent._raw[-1]["stdout"] == expected
    assert agent._raw[-1]["returncode"] == 0
    if should_hit:
        assert environment.commands.count(COMMAND) == 1
        assert agent._counts["misses"] == 1
    else:
        assert agent._counts["misses"] == 2
    before = fs_hash(repo)
    native = asyncio.run(environment.exec(actual))
    assert (native.stdout, native.stderr, native.return_code) == (
        agent._raw[-1]["stdout"], agent._raw[-1]["stderr"], agent._raw[-1]["returncode"])
    assert fs_hash(repo) == before
    if case == "source":
        assert any(event.get("ev") == "fork_execution" and event.get("phase") == "error"
                   and event.get("error_type") == "ScriptSourceMismatch" for event in events)
    else:
        assert observed
    assert context.metadata["sfx_live"]["completed"] is True
    assert context.metadata["sfx_live_trajectory"]["stream_provenance"]["kind"] == "injected_test"
