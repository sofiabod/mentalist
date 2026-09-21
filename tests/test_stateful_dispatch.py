"""Controlled edit dispatch is shared transport delay, not live token preview."""
import asyncio
import json

import pytest

from eval import stateful_probe
from eval.live_ab import _parse_action
from eval.sfx_live_agent import _literal_write


@pytest.mark.parametrize("command", [
    "python probe.py",
    "python probe.py --different",
    "python -m eval.sfx_client_cli --socket /unused/socket resolve live run '{}'",
    "printf '%s' 'replacement' > probe.py",
])
def test_transport_does_not_delay_tools_cli_or_nonliteral_edits(tmp_path, monkeypatch, command):
    dispatched = []

    async def unexpected_sleep(seconds):
        pytest.fail(f"nonliteral command received edit dispatch delay {seconds}")

    class Process:
        returncode = 0

        async def communicate(self):
            return b"result", b"diagnostic"

    async def subprocess(command, **kwargs):
        dispatched.append((command, kwargs["cwd"]))
        return Process()

    monkeypatch.setattr(stateful_probe.asyncio, "sleep", unexpected_sleep)
    monkeypatch.setattr(stateful_probe.asyncio, "create_subprocess_shell", subprocess)
    sandbox = stateful_probe.LocalSandbox(tmp_path, edit_dispatch_s=0.25)
    result = asyncio.run(sandbox.exec(command))
    assert dispatched == [(command, tmp_path)]
    assert (result.stdout, result.stderr, result.return_code) == ("result", "diagnostic", 0)


def test_default_transport_does_not_delay_literal_edit(tmp_path, monkeypatch):
    fixture = tmp_path / "probe.py"
    fixture.write_text("old\n")

    async def unexpected_sleep(seconds):
        pytest.fail(f"default transport added dispatch delay {seconds}")

    monkeypatch.setattr(stateful_probe.asyncio, "sleep", unexpected_sleep)
    result = asyncio.run(stateful_probe.LocalSandbox(tmp_path).exec(
        "cat > probe.py <<'EOF'\nnew\nEOF"))
    assert result.return_code == 0
    assert fixture.read_text() == "new\n"


@pytest.mark.parametrize("mode", stateful_probe.MODES)
def test_all_arms_delay_before_real_edit_and_persist_the_same_setting(tmp_path, monkeypatch, mode):
    sleeps = []
    yield_once = asyncio.sleep

    async def observe_delay(seconds):
        # At this transport boundary FULL may have prepared its private fork,
        # but no arm may have applied the real edit to the live workspace yet.
        assert (tmp_path / "workspace/probe.py").read_text() == stateful_probe.script(1)
        assert (tmp_path / "workspace/revision.txt").read_text() == "0\n"
        sleeps.append(seconds)
        await yield_once(0)

    monkeypatch.setattr(stateful_probe.asyncio, "sleep", observe_delay)
    record = stateful_probe.run_one(
        tmp_path, mode=mode, work_units=1, gap_s=0, rounds=1,
        edit_dispatch_s=0.25,
    )
    assert sleeps == [0.25]
    assert (tmp_path / "workspace/probe.py").read_text() == stateful_probe.script(1)
    assert (tmp_path / "workspace/revision.txt").read_text() == "1\n"
    assert record["completed"] is True
    assert record["edit_dispatch_s"] == 0.25
    assert record["sfx_live_trajectory"]["stop_reason"] == "done"
    assert record["workers_drained"] is True
    assert record["after_cleanup_fs_hash"] == record["final_fs_hash"]
    persisted = json.loads((tmp_path / f"probe-rep0-{mode}/record.json").read_text())
    assert persisted["edit_dispatch_s"] == 0.25


@pytest.mark.parametrize("delay", [-0.1, float("nan")])
def test_invalid_dispatch_delay_fails_before_touching_fixture(tmp_path, monkeypatch, delay):
    fixture = tmp_path / "untouched.py"
    fixture.write_text("preserve me\n")

    def unexpected_workspace(*args, **kwargs):
        pytest.fail("invalid dispatch delay reached workspace mutation")

    monkeypatch.setattr(stateful_probe, "_workspace", unexpected_workspace)
    with pytest.raises(ValueError, match="edit_dispatch_s"):
        stateful_probe.run_one(
            tmp_path, mode="ON", work_units=1, gap_s=0, rounds=1,
            edit_dispatch_s=delay,
        )
    assert fixture.read_text() == "preserve me\n"
    assert list(tmp_path.iterdir()) == [fixture]


def test_bootstrap_model_emits_eight_calls_then_done_without_extra_gap(monkeypatch):
    sleeps = []
    monkeypatch.setattr(stateful_probe.time, "sleep", sleeps.append)
    model = stateful_probe.model_fn(json.dumps(dict(
        work_units=1, gap_s=0.25, rounds=3, bootstrap=True,
    )))
    messages = [{"role": "user", "content": "controlled fixture"}]
    actual = []
    for _ in range(8):
        answer = model("unused", "unused", "unused", messages)
        actual.append(_parse_action(answer))
        messages.extend([
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "tool result"},
        ])
    assert _literal_write(actual[0]) == {
        "path": stateful_probe.REVISION_FILE, "contents": stateful_probe.revision_data(0),
    }
    assert actual[1:] == stateful_probe.commands(1, 3)
    assert model("unused", "unused", "unused", messages) == "DONE"
    assert sleeps == [0.25] * 7
