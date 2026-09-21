import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eval import stateful_probe
from eval.live_ab import _parse_action
from eval.sfx_live_agent import _literal_write


def _execute_probe(root, revision, *args):
    (root / "probe.py").write_text(stateful_probe.script(50))
    (root / stateful_probe.REVISION_FILE).write_text(stateful_probe.revision_data(revision))
    return subprocess.run(
        [sys.executable, "probe.py", *args], cwd=root,
        capture_output=True, text=True, check=True, timeout=10,
    )


def test_tool_performs_deterministic_work_and_exposes_revision_and_arguments(tmp_path):
    original = _execute_probe(tmp_path, 1)
    repeated = _execute_probe(tmp_path, 1)
    edited = _execute_probe(tmp_path, 2)
    different_args = _execute_probe(tmp_path, 1, "--different")
    assert (original.stdout, original.stderr) == (repeated.stdout, repeated.stderr)
    assert original.stdout != edited.stdout
    assert original.stderr != edited.stderr
    assert original.stdout != different_args.stdout
    assert json.loads(original.stdout) == {
        "revision": 1,
        "digest": hashlib.pbkdf2_hmac("sha256", b"1", b"sfx-probe", 50).hex(),
        "args": [],
    }
    assert json.loads(different_args.stdout)["args"] == ["--different"]
    assert "sleep" not in stateful_probe.script(50)


@pytest.mark.parametrize("control", stateful_probe.CONTROLS)
def test_model_emits_exact_commands_and_only_synthetic_between_call_gaps(control, monkeypatch):
    config = dict(work_units=50, gap_s=0.25, rounds=2, control=control)
    expected = stateful_probe.commands(50, 2, control)
    sleeps = []
    monkeypatch.setattr(stateful_probe.time, "sleep", sleeps.append)
    model = stateful_probe.model_fn(json.dumps(config))
    messages = [{"role": "user", "content": "irrelevant instruction"}]
    for index, command in enumerate(expected):
        answer = model("unused", "unused", "unused", messages)
        assert _parse_action(answer) == command
        assert sleeps == [0.25] * index
        messages.extend([
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "authoritative result"},
        ])
    assert model("unused", "unused", "unused", messages) == "DONE"
    assert sleeps == [0.25] * (len(expected) - 1)


def test_controls_change_exact_arguments_or_invalidate_the_existing_write_chain():
    normal = stateful_probe.commands(50, 2)
    wrong = stateful_probe.commands(50, 2, "wrong_args")
    intervening = stateful_probe.commands(50, 2, "intervening_edit")
    assert normal[0] == wrong[0] == intervening[0] == "python probe.py"
    for revision in (1, 2):
        write = normal[2 * revision - 1]
        assert _literal_write(write) == {
            "path": stateful_probe.REVISION_FILE,
            "contents": stateful_probe.revision_data(revision),
        }
        assert wrong[2 * revision - 1] == write
        assert wrong[2 * revision] == "python probe.py --different"
        alternate = intervening[3 * revision - 1]
        assert alternate.startswith("printf ")
        assert _literal_write(alternate) is None
        assert str(revision + 1000) in alternate
        assert alternate.endswith(f"> {stateful_probe.REVISION_FILE}")


@pytest.mark.parametrize("gap", [-1, 11, float("nan"), float("inf"), True, "1"])
def test_model_rejects_invalid_gap(gap):
    with pytest.raises(ValueError, match="gap_s"):
        stateful_probe.model_fn(json.dumps(dict(work_units=50, gap_s=gap, rounds=1)))


@pytest.mark.parametrize("work_units", [0, -1, 10_000_001, True, 1.5])
def test_script_rejects_unbounded_or_invalid_work(work_units):
    with pytest.raises(ValueError, match="work_units"):
        stateful_probe.script(work_units)


@pytest.mark.parametrize("revision", [-1, True, 1.5, "1"])
def test_revision_data_rejects_invalid_revision(revision):
    with pytest.raises(ValueError, match="revision"):
        stateful_probe.revision_data(revision)


@pytest.mark.parametrize("control", stateful_probe.CONTROLS)
def test_commands_mutate_only_revision_data_and_never_the_pinned_checker(tmp_path, control):
    workspace = stateful_probe._workspace(tmp_path, 50)
    source = (workspace / "probe.py").read_bytes()
    expected_pin = stateful_probe.fixture_identity(50)["fixture_source_sha256"]
    assert hashlib.sha256(source).hexdigest() == expected_pin
    for command in stateful_probe.commands(50, 2, control):
        subprocess.run(command, shell=True, cwd=workspace, check=True,
                       capture_output=True, text=True, timeout=10)
        assert (workspace / "probe.py").read_bytes() == source
    expected_revision = 1002 if control == "intervening_edit" else 2
    assert (workspace / stateful_probe.REVISION_FILE).read_text() == f"{expected_revision}\n"


def test_all_arms_share_owned_scratch_and_get_on_serve_current_revision_without_oracle(tmp_path, monkeypatch):
    from eval.live_driver import metrics
    from eval.sfx_daemon_run import _contracts, _resolve_args
    from eval.table import load_table
    from sfx.daemon import Daemon

    starts = []
    session_start = Daemon.session_start
    expected_table = load_table(
        Path(stateful_probe.__file__).resolve().parents[2] / "data/tables/benchmark.json"
    )

    def inspect_session(daemon, *args, **kwargs):
        session = session_start(daemon, *args, **kwargs)
        starts.append((session.trajectory, session.spec_disabled))
        expected_scratch = Path(os.environ["SFX_SCRATCH"])
        assert expected_scratch.is_dir()
        assert expected_scratch.parent.parent == tmp_path
        assert expected_scratch.name == "forks"
        assert Path(session.scratch) == expected_scratch
        assert daemon.resolve_args is _resolve_args
        assert daemon.global_table == expected_table
        assert _contracts().definitions == [{
            "script": "probe.py", "positionals": 0, "path_options": [],
            "value_options": [], "flags": [], "required": [],
            "source_sha256": {"probe.py": stateful_probe.fixture_identity(50)["fixture_source_sha256"]},
        }]
        return session

    monkeypatch.setattr(Daemon, "session_start", inspect_session)
    records = {mode: stateful_probe.run_one(
        tmp_path, mode=mode, work_units=50, gap_s=0.25, rounds=2,
    ) for mode in stateful_probe.MODES}
    off, get, on = (records[mode] for mode in ("OFF", "GET", "ON"))
    assert starts == [(None, True), (None, True), (None, False), (None, False)]
    assert off["env_id"] != on["env_id"]
    assert off["counts"]["hits"] == off["counts"]["writes_fed"] == 0
    assert on["counts"]["writes_fed"] == 2
    pair = metrics(off, on)
    assert pair["pair_valid"] is True
    assert pair["served_compared"] > 0
    assert pair["served_unverified"] == pair["served_mismatches"] == 0
    assert pair["spec_writes_committed"] > 0
    get_pair = metrics(off, get)
    assert get_pair["pair_valid"] is True
    assert get_pair["served_compared"] > 0
    assert get_pair["served_unverified"] == get_pair["served_mismatches"] == 0
    assert get["counts"]["hits"] > 0
    assert get["counts"]["writes_fed"] == 0
    for record in records.values():
        outputs = [json.loads(raw["stdout"]) for raw in record["raw"]
                   if raw["command"] == "python probe.py"]
        assert [output["revision"] for output in outputs] == [0, 1, 2]
        assert all(output["args"] == [] for output in outputs)
        assert record["sfx_live_trajectory"]["stop_reason"] == "done"
        assert record["fixture_version"] == stateful_probe.FIXTURE_VERSION
        assert record["fixture_source_sha256"] == hashlib.sha256(
            (tmp_path / "workspace/probe.py").read_bytes()).hexdigest()
        assert record["final_fs_hash"] == record["after_cleanup_fs_hash"]


@pytest.mark.parametrize("tail", [-1, 11, float("nan"), True, "1"])
def test_stream_fixture_rejects_invalid_tail(tail):
    with pytest.raises(ValueError, match="stream_tail_s"):
        stateful_probe.stream_model_fn(json.dumps(
            dict(work_units=50, gap_s=0, rounds=1, stream_tail_s=tail)))


def test_stream_fixture_exposes_complete_edit_before_closing_fence():
    from eval.sfx_live_agent import _streamed_literal_write

    model = stateful_probe.stream_model_fn(json.dumps(
        dict(work_units=50, gap_s=0, rounds=1, stream_tail_s=0)))
    events = list(model("unused", "unused", "unused", [{"role": "assistant"}]))
    partial = events[0]["content"] + events[1]["content"]
    expected = stateful_probe.commands(50, 1)[1]
    assert _parse_action(partial) is None
    assert _streamed_literal_write(partial) == (expected, _literal_write(expected))
    assert _parse_action(partial + events[2]["content"]) == expected
    assert events[-1] == {"type": "done", "finish_reason": "stop"}
    assert not any(event["type"] == "usage" for event in events)


def test_legacy_workspace_marker_is_not_silently_reused(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "probe.py"
    source.write_text("legacy source\n")
    (tmp_path / "experiment-owned.json").write_text(json.dumps({
        "experiment": "stateful_probe", "workspace": str(workspace),
    }))
    with pytest.raises(ValueError, match="ownership marker"):
        stateful_probe._workspace(tmp_path, 50)
    assert source.read_text() == "legacy source\n"
    assert not (workspace / stateful_probe.REVISION_FILE).exists()


def test_cli_runs_only_the_selected_mode_and_reports_scope(tmp_path, monkeypatch, capsys):
    calls = []

    def run_one(root, **config):
        calls.append((root, config))
        return {"completed": True, "workers_drained": True,
                "after_cleanup_fs_hash": "same", "final_fs_hash": "same",
                "raw": [{"served": False}]}

    monkeypatch.setattr(stateful_probe, "run_one", run_one)
    result = stateful_probe.main([
        "--output", str(tmp_path), "--mode", "GET", "--control", "intervening_edit",
        "--work-units", "50", "--rounds", "1", "--gap-s", "0", "--stream-tail-s", "0",
    ])
    assert result == 0
    assert calls == [(tmp_path, {"mode": "GET", "control": "intervening_edit",
                                "work_units": 50, "rounds": 1, "gap_s": 0,
                                "stream_tail_s": 0})]
    summary = json.loads(capsys.readouterr().out)
    assert summary["scope"] == "controlled CPU mechanism demo, not live-model performance"
    assert summary["record"] == str(tmp_path / "probe-rep0-GET" / "record.json")
    assert summary["completed"] and summary["workers_drained"]
    assert summary["workspace_unchanged_after_cleanup"]
    assert summary["served_result"] is False
    assert not any("speed" in key or "wall" in key for key in summary)


@pytest.mark.parametrize("failure", ["incomplete", "not_drained", "changed_after_cleanup"])
def test_cli_fails_when_lifecycle_evidence_is_invalid(tmp_path, monkeypatch, failure):
    record = {"completed": True, "workers_drained": True,
              "after_cleanup_fs_hash": "same", "final_fs_hash": "same", "raw": []}
    if failure == "incomplete":
        record["completed"] = False
    elif failure == "not_drained":
        record["workers_drained"] = False
    else:
        record["after_cleanup_fs_hash"] = "changed"
    monkeypatch.setattr(stateful_probe, "run_one", lambda *args, **kwargs: record)
    assert stateful_probe.main(["--output", str(tmp_path)]) == 1


@pytest.mark.parametrize("option,value", [
    ("--work-units", "0"), ("--rounds", "0"), ("--gap-s", "nan"),
    ("--stream-tail-s", "-1"),
])
def test_cli_rejects_invalid_fixture_before_creating_output(tmp_path, monkeypatch, option, value):
    output = tmp_path / "not-created"
    monkeypatch.setattr(stateful_probe, "run_one",
                        lambda *args, **kwargs: pytest.fail("invalid fixture was executed"))
    with pytest.raises(SystemExit) as error:
        stateful_probe.main(["--output", str(output), option, value])
    assert error.value.code == 2
    assert not output.exists()
