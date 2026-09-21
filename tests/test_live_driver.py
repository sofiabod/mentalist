"""Exact receipt/state parity and explicitly bounded served-result evidence."""
import pytest

from eval.live_driver import losslessness_violations, metrics


def _arm(wall_s, receipts, fs_hash, trace=None, env_id="e"):
    return {"wall_s": wall_s, "receipts": receipts, "final_fs_hash": fs_hash,
            "trace": trace or [], "env_id": env_id}


def _r(sha, rc=0):
    return {"stdout_sha": sha, "stderr_sha": "empty-stderr", "returncode": rc,
            "signal": -rc if rc < 0 else 0}


def _arm_served(wall_s, receipts, fs_hash, served_flags, env_id="e"):
    a = _arm(wall_s, receipts, fs_hash, env_id=env_id)
    a["raw"] = [{"index": i, "served": s, "command": receipts[i].get("command")}
                for i, s in enumerate(served_flags)]
    return a


def _identified(sha, command="cat result"):
    return {**_r(sha), "command": command, "context": {"cwd": "/app"}}


def _comparable_pair(off_receipts, on_receipts, served_flags):
    off = _arm(2.0, off_receipts, "H")
    on = _arm_served(1.0, on_receipts, "H", served_flags)
    off["initial_fs_hash"] = on["initial_fs_hash"] = "INITIAL"
    return off, on


def test_identical_arms_have_zero_losslessness_violations():
    off = _arm(2.0, [_r("a"), _r("b")], "H")
    on = _arm(1.2, [_r("a"), _r("b")], "H")
    assert losslessness_violations(off, on) == 0
    m = metrics(off, on)
    assert m["success_parity"] is True
    assert m["losslessness_violations"] == 0


def test_wrong_served_result_is_a_losslessness_violation():
    off = _arm(2.0, [_r("a")], "H")
    on = _arm(1.0, [_r("WRONG")], "H")
    assert losslessness_violations(off, on) == 1
    assert metrics(off, on)["success_parity"] is False


def test_final_fs_divergence_is_a_losslessness_violation():
    off = _arm(2.0, [_r("a")], "H1")
    on = _arm(1.0, [_r("a")], "H2")
    assert losslessness_violations(off, on) == 1


def test_served_divergence_is_flagged_as_diagnostic_without_weakening_gate():
    off, on = _comparable_pair([_identified("a"), _identified("b")],
                               [_identified("WRONG"), _identified("b")], [True, True])
    m = metrics(off, on)
    assert m["served_divergences"] == 1
    assert m["miss_divergences"] == 0
    assert m["losslessness_violations"] == 1
    assert m["success_parity"] is False


def test_miss_divergence_still_counts_toward_losslessness():
    off, on = _comparable_pair([_identified("a"), _identified("b")],
                               [_identified("a"), _identified("DIFF")], [True, False])
    m = metrics(off, on)
    assert m["miss_divergences"] == 1
    assert m["served_divergences"] == 0
    assert m["losslessness_violations"] == 1


def test_unequal_trajectories_leave_all_twelve_hits_unverified():
    off, on = _comparable_pair([_identified("a")] * 7,
                               [_identified("a")] * 30, [True] * 12 + [False] * 18)
    m = metrics(off, on)
    assert m["served_total"] == 12
    assert m["served_compared"] == m["served_matches"] == m["served_mismatches"] == 0
    assert m["served_unverified"] == 12
    assert m["served_comparison_status"] == "unverified"
    assert m["served_comparison_reason"] == "receipt_count"
    assert m["served_divergences"] == 0  # No longer the sole evidence field.
    assert m["pair_valid"] is False
    assert m["wall_speedup"] is m["wall_delta_ms"] is None


def test_identical_served_calls_have_explicit_comparison_counts():
    receipts = [_identified("a"), _identified("b")]
    off, on = _comparable_pair(receipts, receipts, [False, True])
    m = metrics(off, on)
    assert m["served_total"] == m["served_compared"] == m["served_matches"] == 1
    assert m["served_unverified"] == m["served_mismatches"] == 0
    assert m["served_comparison_status"] == "compared"
    assert m["served_comparison_reason"] is None
    assert m["pair_valid"] is True


@pytest.mark.parametrize("served_first", [False, True])
def test_no_comparisons_after_first_output_divergence(served_first):
    off, on = _comparable_pair([_identified("a"), _identified("b")],
                               [_identified("WRONG"), _identified("b")],
                               [served_first, True])
    m = metrics(off, on)
    assert m["served_compared"] == m["served_mismatches"] == int(served_first)
    assert m["served_matches"] == 0
    assert m["served_unverified"] == 1
    assert m["served_comparison_reason"] == "receipt[0]:result_divergence"
    assert m["pair_valid"] is False


@pytest.mark.parametrize("axis,value", [("command", "different command"),
                                        ("context", {"cwd": "/elsewhere"})])
def test_identity_divergence_does_not_align_later_repeated_commands(axis, value):
    receipts = [_identified("same")] * 3
    off, on = _comparable_pair(receipts,
                               [receipts[0], {**receipts[1], axis: value}, receipts[2]],
                               [True, True, True])
    m = metrics(off, on)
    assert m["served_compared"] == m["served_matches"] == 1
    assert m["served_mismatches"] == 0
    assert m["served_unverified"] == 2
    assert m["served_comparison_status"] == "partial"
    assert m["served_comparison_reason"] == "receipt[1]:call_identity"
    assert m["wall_speedup"] is None


@pytest.mark.parametrize("axis", ["initial_fs_hash", "final_fs_hash"])
def test_state_divergence_prevents_served_equivalence_claim(axis):
    off, on = _comparable_pair([_identified("a")], [_identified("a")], [True])
    on[axis] = "DIFFERENT"
    m = metrics(off, on)
    assert m["served_compared"] == 0
    assert m["served_unverified"] == 1
    assert m["served_comparison_reason"] == axis
    assert m["pair_valid"] is False


@pytest.mark.parametrize("axis", ["pre_fs_hash", "post_fs_hash", "fs_hash"])
def test_per_call_state_divergence_prevents_comparison(axis):
    off, on = _comparable_pair([{**_identified("a"), axis: "STATE_A"}],
                               [{**_identified("a"), axis: "STATE_B"}], [True])
    m = metrics(off, on)
    assert m["served_compared"] == 0
    assert m["served_unverified"] == 1
    assert m["served_comparison_reason"] == "receipt[0]:filesystem_state"
    assert m["pair_valid"] is False
    assert m["wall_speedup"] is m["wall_delta_ms"] is None
    assert f"receipt[0]:{axis}" in m["invalid_reasons"]


@pytest.mark.parametrize("axis", ["pre_fs_hash", "post_fs_hash", "fs_hash"])
@pytest.mark.parametrize("value", [None, "", 123])
def test_incomplete_per_call_state_also_invalidates_pair(axis, value):
    off, on = _comparable_pair([{**_identified("a"), axis: "STATE"}],
                               [{**_identified("a"), axis: value}], [True])
    m = metrics(off, on)
    assert m["pair_valid"] is False
    assert m["wall_speedup"] is m["wall_delta_ms"] is None
    assert m["served_compared"] == 0
    assert m["served_unverified"] == 1


@pytest.mark.parametrize("raw", [
    [{"index": 0, "served": True, "command": "cat result"}] * 2,
    [{"served": True, "command": "cat result"}],
    [{"index": 1, "served": True, "command": "cat result"}],
    [{"index": -1, "served": True, "command": "cat result"}],
    [{"index": True, "served": True, "command": "cat result"}],
    [{"index": 0, "served": True, "command": "wrong command"}],
    [{"index": 0, "served": True}],
    [{"index": 0, "served": "true", "command": "cat result"}],
    [None],
])
def test_malformed_raw_mapping_never_claims_verified_matches(raw):
    off, on = _comparable_pair([_identified("a")], [_identified("a")], [True])
    on["raw"] = raw
    on["counts"] = {"hits": 1}
    m = metrics(off, on)
    known_hits = max(1, sum(isinstance(r, dict) and r.get("served") is True for r in raw))
    assert m["served_total"] == m["served_unverified"] == known_hits
    assert m["served_compared"] == m["served_matches"] == m["served_mismatches"] == 0
    assert m["served_comparison_status"] == "unverified"
    assert m["served_comparison_reason"] == "invalid_served_mapping"


def test_missing_raw_index_invalidates_the_entire_mapping():
    receipts = [_identified("a"), _identified("b")]
    off, on = _comparable_pair(receipts, receipts, [True, True])
    on["raw"].pop()
    on["counts"] = {"hits": 2}
    m = metrics(off, on)
    assert m["served_total"] == m["served_unverified"] == 2
    assert m["served_matches"] == m["served_compared"] == 0
    assert m["served_comparison_reason"] == "missing_served_mapping"


@pytest.mark.parametrize("reported_hits", [0, 2])
def test_inconsistent_hit_counter_is_not_a_complete_comparison(reported_hits):
    off, on = _comparable_pair([_identified("a")], [_identified("a")], [True])
    on["counts"] = {"hits": reported_hits}
    m = metrics(off, on)
    assert m["served_total"] == m["served_unverified"] == max(1, reported_hits)
    assert m["served_matches"] == m["served_compared"] == 0
    assert m["served_comparison_reason"] == "inconsistent_served_count"


def test_explicit_unique_indices_allow_raw_records_in_different_order():
    receipts = [_identified("a", "cat a"), _identified("b", "cat b")]
    off, on = _comparable_pair(receipts, receipts, [True, True])
    on["raw"].reverse()
    m = metrics(off, on)
    assert m["served_matches"] == m["served_compared"] == 2
    assert m["served_unverified"] == 0


@pytest.mark.parametrize("missing", ["command", "context", "stderr_sha"])
def test_missing_comparison_evidence_is_not_a_verified_match(missing):
    receipt = _identified("a")
    del receipt[missing]
    off, on = _comparable_pair([receipt], [receipt], [True])
    m = metrics(off, on)
    assert m["served_compared"] == m["served_matches"] == 0
    assert m["served_unverified"] == 1
    assert m["served_comparison_status"] == "unverified"


def test_hit_counter_without_raw_mapping_does_not_imply_comparison():
    off, on = _comparable_pair([_identified("a")], [_identified("a")], [])
    on["counts"] = {"hits": 1}
    m = metrics(off, on)
    assert m["served_total"] == m["served_unverified"] == 1
    assert m["served_compared"] == 0
    assert m["served_comparison_reason"] == "missing_served_mapping"


def test_signed_wall_delta_does_not_claim_attributed_tool_overlap():
    off = _arm(3.0, [_r("a")], "H")
    on = _arm(1.5, [_r("a")], "H")
    assert metrics(off, on)["wall_delta_ms"] == 1500.0
    assert metrics(off, on)["hidden_tool_ms"] is None
    assert metrics(on, off)["wall_delta_ms"] == -1500.0


def test_committed_writes_and_rollbacks_from_trace():
    trace = [
        {"ev": "chain", "commit": True},
        {"ev": "chain", "commit": False},
        {"ev": "spec_end", "terminal": "discarded"},
        {"ev": "spec_end", "terminal": "hit_completed"},
    ]
    on = _arm(1.0, [_r("a")], "H", trace=trace)
    m = metrics(_arm(1.0, [_r("a")], "H"), on)
    assert m["spec_writes_committed"] == 1
    assert m["rollbacks"] == 1


@pytest.mark.parametrize("axis,value", [
    ("stderr_sha", "different-error"), ("signal", 15),
    ("command", "cat wrong-file"), ("context", {"cwd": "/different"}),
    ("call_id", "different-call"),
])
def test_full_receipt_divergence_invalidates_speed(axis, value):
    receipt = {**_r("same-output"), "command": "cat expected-file",
               "context": {"cwd": "/app"}, "call_id": "call-1"}
    off = _arm(2.0, [receipt], "H")
    on = _arm(1.0, [{**receipt, axis: value}], "H")
    result = metrics(off, on)
    assert result["pair_valid"] is False
    assert result["wall_speedup"] is None
    assert result["wall_delta_ms"] is None
    assert result["losslessness_violations"] == 1


@pytest.mark.parametrize("axis", ["stdout_sha", "stderr_sha", "returncode", "signal"])
def test_missing_receipt_axis_is_unverified_even_when_both_missing(axis):
    receipt = _r("a")
    del receipt[axis]
    result = metrics(_arm(2, [receipt], "H"), _arm(1, [receipt], "H"))
    assert not result["pair_valid"]
    assert result["wall_speedup"] is None
    assert any(f"missing_{axis}" in reason for reason in result["invalid_reasons"])


def test_one_sided_call_identity_and_missing_fs_are_invalid():
    off = _arm(2, [{**_r("a"), "command": "cat a"}], None)
    on = _arm(1, [_r("a")], None)
    result = metrics(off, on)
    assert result["losslessness_violations"] == 2
    assert not result["pair_valid"]


@pytest.mark.parametrize("axis,value", [("stdout_sha", ""), ("stderr_sha", None),
                                        ("returncode", False), ("signal", "unknown")])
def test_unsupported_receipt_values_cannot_be_equal_evidence(axis, value):
    receipt = {**_r("a"), axis: value}
    result = metrics(_arm(2, [receipt], "H"), _arm(1, [receipt], "H"))
    assert not result["pair_valid"]
    assert result["wall_speedup"] is None


def test_no_call_or_invalid_timing_cannot_claim_speed():
    for off, on in [(_arm(2, [], "H"), _arm(1, [], "H")),
                    (_arm(2, [_r("a")], "H"), _arm(float("nan"), [_r("a")], "H"))]:
        assert metrics(off, on)["wall_speedup"] is None


@pytest.mark.parametrize("on_hash", ["different", None])
def test_initial_workspace_must_match_when_supplied(on_hash):
    off = {**_arm(2, [_r("a")], "FINAL"), "initial_fs_hash": "INITIAL"}
    on = {**_arm(1, [_r("a")], "FINAL"), "initial_fs_hash": on_hash}
    result = metrics(off, on)
    assert not result["pair_valid"]
    assert "initial_fs_hash" in result["invalid_reasons"]
    assert result["wall_speedup"] is None


def test_one_sided_initial_hash_is_unverified():
    off = {**_arm(2, [_r("a")], "H"), "initial_fs_hash": "INITIAL"}
    on = _arm(1, [_r("a")], "H")
    assert not metrics(off, on)["pair_valid"]


@pytest.mark.parametrize("arm", ["OFF", "ON"])
def test_incomplete_arm_cannot_claim_parity_or_speed(arm):
    arms = {"OFF": {**_arm(2, [_r("a")], "H"), "completed": True},
            "ON": {**_arm(1, [_r("a")], "H"), "completed": True}}
    arms[arm]["completed"] = False
    result = metrics(arms["OFF"], arms["ON"])
    assert not result["pair_valid"]
    assert not result["success_parity"]
    assert result["wall_speedup"] is None
    assert f"{arm}:not_completed" in result["invalid_reasons"]
