"""An eligible command spelling mismatch must resolve-miss, not policy-bypass."""
import json

from eval import stateful_probe
from eval.live_driver import metrics
from mining.normalize import classify


def test_post_edit_eligible_command_mismatch_discards_chain_and_runs_authoritatively(
        tmp_path, monkeypatch):
    original_commands = stateful_probe.commands

    def changed_post_edit_commands(work_units, rounds, control="normal"):
        tape = original_commands(work_units, rounds, control)
        return [command if index == 0 or command != "python probe.py"
                else f"python {'./' * (index // 2)}probe.py"
                for index, command in enumerate(tape)]

    assert classify("Bash", "python ./probe.py") == ("run", "free")
    monkeypatch.setattr(stateful_probe, "commands", changed_post_edit_commands)
    config = dict(work_units=1000, gap_s=0.1, rounds=2, case_id="exact-command")
    off = stateful_probe.run_one(tmp_path, mode="OFF", **config)
    on = stateful_probe.run_one(tmp_path, mode="ON", **config)

    assert on["counts"]["resolves"] == 3  # Initial run plus both changed requests.
    assert on["counts"]["misses"] == 3
    assert on["counts"]["hits"] == on["counts"]["never_routed"] == 0
    assert on["counts"]["writes_fed"] == 2
    assert on["counts"]["authoritative"] == 5
    changed_resolves = [event for event in on["trace"]
                        if event.get("ev") == "resolve"
                        and event.get("args") in ({"cmd": "python ./probe.py"},
                                                  {"cmd": "python ././probe.py"})]
    assert len(changed_resolves) == 2
    assert all(event["kind"] == "run" and event["outcome"] == "miss"
               and event["has_result"] is False for event in changed_resolves)
    prefix_breaks = [event for event in on["trace"]
                     if event.get("ev") == "chain"
                     and event.get("discard_reason") == "prefix_break"]
    assert len(prefix_breaks) == 2
    assert all(event["commit"] is False for event in prefix_breaks)

    pair = metrics(off, on)
    assert pair["pair_valid"] is True
    assert pair["receipt_parity"] is True
    assert pair["losslessness_violations"] == 0
    assert off["raw"] == on["raw"]
    assert off["initial_fs_hash"] == on["initial_fs_hash"]
    assert off["final_fs_hash"] == on["final_fs_hash"]
    assert off["env_id"] != on["env_id"]
    for record in (off, on):
        assert record["completed"] is True
        assert record["sfx_live_trajectory"]["stop_reason"] == "done"
        assert record["workers_drained"] is True
        assert record["after_cleanup_fs_hash"] == record["final_fs_hash"]
        assert all(row["served"] is False and row["returncode"] == 0
                   for row in record["raw"])
        checks = [row for row in record["raw"] if row["command"].startswith("python ")]
        assert [json.loads(row["stdout"])["revision"] for row in checks] == [0, 1, 2]
        assert all(json.loads(row["stdout"])["args"] == [] for row in checks)
        assert [row["stderr"] for row in checks] == [
            "checked revision 0\n", "checked revision 1\n", "checked revision 2\n",
        ]


def test_observed_exact_spelling_is_learned_only_after_authoritative_miss(tmp_path, monkeypatch):
    original_commands = stateful_probe.commands

    def changed_commands(work_units, rounds, control="normal"):
        return [command if index == 0 or command != "python probe.py"
                else "python ./probe.py"
                for index, command in enumerate(original_commands(work_units, rounds, control))]

    monkeypatch.setattr(stateful_probe, "commands", changed_commands)
    config = dict(work_units=1000, gap_s=.1, rounds=2, case_id="learned-spelling")
    off = stateful_probe.run_one(tmp_path, mode="OFF", **config)
    on = stateful_probe.run_one(tmp_path, mode="ON", **config)

    changed = [event for event in on["trace"] if event.get("ev") == "resolve"
               and event.get("args") == {"cmd": "python ./probe.py"}]
    assert [event["outcome"] for event in changed] == ["miss", "hit_completed"]
    assert on["counts"]["hits"] == 1
    assert on["counts"]["misses"] == 2
    pair = metrics(off, on)
    assert pair["pair_valid"] and pair["receipt_parity"]
    assert pair["losslessness_violations"] == 0
    assert off["final_fs_hash"] == on["final_fs_hash"]
    checks = [row for row in on["raw"] if row["command"].startswith("python ")]
    assert [json.loads(row["stdout"])["revision"] for row in checks] == [0, 1, 2]
    assert [row["served"] for row in checks] == [False, False, True]
