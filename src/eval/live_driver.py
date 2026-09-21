"""Read-only parity checks for recorded command execution.

Compare call identities, output bytes/hashes, return codes and repository state.
Served results without aligned evidence remain unverified. Timing differences are
observations only, not attributed speedups. This module runs no experiments.
"""
import math

from sfx.schema import HIT_TERMINALS


def committed_writes(trace):
    """Count admitted chains; the historical name does not mean filesystem merges."""
    return sum(1 for r in trace if r.get("ev") == "chain" and r.get("commit"))


def rollbacks(trace):
    return sum(1 for r in trace
               if r.get("ev") == "spec_end" and r.get("terminal") not in HIT_TERMINALS)


_RECEIPT_AXES = ("stdout_sha", "stderr_sha", "returncode", "signal")
_CALL_AXES = ("command", "context", "args", "kind", "verb", "tool", "call_id",
              "id", "seq", "cwd", "user", "env", "stdout", "stderr")
_STATE_AXES = ("pre_fs_hash", "post_fs_hash", "fs_hash")


def _receipt_issues(off, on):
    left, right = off.get("receipts"), on.get("receipts")
    if not isinstance(left, list) or not isinstance(right, list):
        return ["missing_receipts"]
    if len(left) != len(right):
        return ["receipt_count"]
    issues = []
    for i, (a, b) in enumerate(zip(left, right)):
        if not isinstance(a, dict) or not isinstance(b, dict):
            issues.append(f"receipt[{i}]:invalid_receipt")
            continue
        for axis in _RECEIPT_AXES:
            if axis not in a or axis not in b or a[axis] is None or b[axis] is None:
                issues.append(f"receipt[{i}]:missing_{axis}")
            elif axis.endswith("_sha") and any(
                    not isinstance(r[axis], str) or not r[axis] for r in (a, b)):
                issues.append(f"receipt[{i}]:invalid_{axis}")
            elif axis in ("returncode", "signal") and any(
                    type(r[axis]) is not int for r in (a, b)):
                issues.append(f"receipt[{i}]:invalid_{axis}")
            elif a[axis] != b[axis]:
                issues.append(f"receipt[{i}]:{axis}")
        for axis in _CALL_AXES:
            if axis in a or axis in b:
                if axis not in a or axis not in b or a[axis] != b[axis]:
                    issues.append(f"receipt[{i}]:{axis}")
        for axis in _STATE_AXES:
            if axis in a or axis in b:
                if any(not isinstance(r.get(axis), str) or not r[axis] for r in (a, b)):
                    issues.append(f"receipt[{i}]:invalid_{axis}")
                elif a[axis] != b[axis]:
                    issues.append(f"receipt[{i}]:{axis}")
    return issues


def _receipts_identical(off, on):
    return not _receipt_issues(off, on)


def _filesystem_identical(off, on, key="final_fs_hash"):
    a, b = off.get(key), on.get(key)
    return isinstance(a, str) and bool(a) and a == b


def losslessness_violations(off, on):
    """Legacy count of failed OFF/ON parity checks; zero is required by the gate.

    Divergent live trajectories make a pair invalid, but do not establish that
    SFX corrupted a served result. See the explicit served comparison counts for
    how much paired result evidence is actually available.
    """
    v = 0
    if not _receipts_identical(off, on):
        v += 1
    if not _filesystem_identical(off, on):
        v += 1
    if (("initial_fs_hash" in off or "initial_fs_hash" in on)
            and not _filesystem_identical(off, on, "initial_fs_hash")):
        v += 1
    return v


def _served_index_map(on):
    """Require an explicit one-to-one mapping, never infer or deduplicate indices."""
    raw, receipts = on.get("raw"), on.get("receipts")
    if not isinstance(raw, list) or not isinstance(receipts, list):
        return {}, "missing_served_mapping"
    mapped = {}
    for row in raw:
        if not isinstance(row, dict):
            return {}, "invalid_served_mapping"
        index = row.get("index")
        if (type(index) is not int or not 0 <= index < len(receipts)
                or index in mapped or type(row.get("served")) is not bool):
            return {}, "invalid_served_mapping"
        receipt = receipts[index]
        if (not isinstance(receipt, dict) or not isinstance(row.get("command"), str)
                or not row["command"] or row["command"] != receipt.get("command")):
            return {}, "invalid_served_mapping"
        mapped[index] = row["served"]
    if len(mapped) != len(receipts):
        return {}, "missing_served_mapping"
    return mapped, None


def divergence_diagnostics(off, on):
    """Count only comparable served receipts, not zeroes caused by a failed gate.

    A paired comparison is not an independent authoritative rerun. Require the
    recorded boundary states and call identities to match, and stop at the first
    divergence: a later identical command may run against a different state.
    Unequal-length trajectories are deliberately not aligned, even by prefix.
    """
    served, mapping_issue = _served_index_map(on)
    raw = on.get("raw")
    # Preserve claimed hits even if their indices/commands are malformed. Such
    # records remain unverified instead of disappearing when mapping fails.
    total = sum(isinstance(r, dict) and r.get("served") is True for r in raw) \
        if isinstance(raw, list) else 0
    counts = on.get("counts")
    reported_hits = counts.get("hits") if isinstance(counts, dict) else None
    if type(reported_hits) is int and reported_hits >= 0:
        # A counter without corresponding raw receipts is evidence of hits, not
        # evidence that those hits were compared successfully.
        total = max(total, reported_hits)
        if mapping_issue is None and reported_hits != sum(served.values()):
            mapping_issue = "inconsistent_served_count"
    matched, mismatched, miss_div = [], [], []
    reason = None
    left, right = off.get("receipts"), on.get("receipts")
    if not isinstance(left, list) or not isinstance(right, list):
        reason = "missing_receipts"
    elif len(left) != len(right):
        reason = "receipt_count"
    elif not _filesystem_identical(off, on, "initial_fs_hash"):
        reason = "initial_fs_hash"
    elif not _filesystem_identical(off, on):
        reason = "final_fs_hash"
    elif any("completed" in r and r["completed"] is not True for r in (off, on)):
        reason = "incomplete_execution"
    elif mapping_issue:
        reason = mapping_issue
    else:
        identity_axes = tuple(a for a in _CALL_AXES if a not in ("stdout", "stderr"))
        for i, (a, b) in enumerate(zip(left, right)):
            if (not isinstance(a, dict) or not isinstance(b, dict)
                    or any(not isinstance(r.get("command"), str) or not r["command"]
                           or not isinstance(r.get("context"), dict) or not r["context"]
                           for r in (a, b))):
                reason = f"receipt[{i}]:missing_call_identity"
                break
            if any((axis in a or axis in b)
                   and (axis not in a or axis not in b or a[axis] != b[axis])
                   for axis in identity_axes):
                reason = f"receipt[{i}]:call_identity"
                break
            # Capture formats may also include per-call state fingerprints. A
            # known mismatch invalidates correspondence before comparing output.
            if any((axis in a or axis in b)
                   and (not isinstance(a.get(axis), str) or not a[axis]
                        or a[axis] != b.get(axis))
                   for axis in _STATE_AXES):
                reason = f"receipt[{i}]:filesystem_state"
                break
            issues = _receipt_issues({"receipts": [a]}, {"receipts": [b]})
            if any(":missing_" in issue or ":invalid_" in issue for issue in issues):
                reason = f"receipt[{i}]:incomplete_result"
                break
            if issues:
                if served.get(i) is True:
                    mismatched.append(i)
                elif served.get(i) is False:
                    miss_div.append(i)
                reason = f"receipt[{i}]:result_divergence"
                break
            if served.get(i) is True:
                matched.append(i)

    compared = len(matched) + len(mismatched)
    unverified = total - compared
    if unverified and reason is None:
        reason = "missing_served_mapping"
    status = ("no_served_calls" if total == 0 and mapping_issue is None else
              "unverified" if compared == 0 else
              "partial" if unverified else "compared")
    return {"served_total": total,
            "served_compared": compared,
            "served_matches": len(matched),
            "served_mismatches": len(mismatched),
            "served_unverified": unverified,
            "served_comparison_status": status,
            "served_comparison_reason": reason,
            # Compatibility aliases describe only comparable mismatches now.
            "served_divergences": len(mismatched),
            "miss_divergences": len(miss_div),
            "divergent_served_calls": mismatched,
            "divergent_miss_calls": miss_div}


def metrics(off, on):
    issues = _receipt_issues(off, on)
    receipt_parity = not issues
    if not _filesystem_identical(off, on):
        issues.append("final_fs_hash")
    if (("initial_fs_hash" in off or "initial_fs_hash" in on)
            and not _filesystem_identical(off, on, "initial_fs_hash")):
        issues.append("initial_fs_hash")
    if not off.get("receipts") and not on.get("receipts"):
        issues.append("no_tool_calls")
    for arm, record in (("OFF", off), ("ON", on)):
        if "completed" in record and record["completed"] is not True:
            issues.append(f"{arm}:not_completed")
        wall = record.get("wall_s")
        if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
                or not math.isfinite(wall) or wall <= 0):
            issues.append(f"{arm}:invalid_wall_s")
    valid = not issues
    delta = round((off["wall_s"] - on["wall_s"]) * 1000.0, 1) if valid else None
    return {
        "pair_valid": valid,
        "invalid_reasons": issues,
        "wall_speedup": round(off["wall_s"] / on["wall_s"], 4) if valid else None,
        "e2e_wall_off_s": off.get("wall_s"),
        "e2e_wall_on_s": on.get("wall_s"),
        "receipt_parity": receipt_parity,
        "success_parity": valid,
        "wall_delta_ms": delta,
        # Deprecated: an OFF/ON clock difference does not identify hidden tool
        # work. Keep the old key explicitly unsupported instead of inventing it.
        "hidden_tool_ms": None,
        "spec_writes_committed": committed_writes(on["trace"]),
        "rollbacks": rollbacks(on["trace"]),
        "losslessness_violations": losslessness_violations(off, on),
        **divergence_diagnostics(off, on),
    }
