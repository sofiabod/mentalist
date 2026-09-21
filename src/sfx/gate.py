from dataclasses import dataclass

from sfx.schema import SPECULABLE, STREAM_ONLY

TAU_GET = 0.35
TAU_FORK = 0.50


@dataclass
class Candidate:
    kind: str
    verb: str
    p: float
    args: dict | None
    hidden_ms: float = 0.0
    cost_ms: float = 1.0


@dataclass
class Admission:
    candidate: Candidate
    action: str
    priority: float
    reason: str = ""


def _tau(verb):
    return TAU_FORK if verb == "fork" else TAU_GET


def admit(cand, free_slots):
    if free_slots <= 0:
        return Admission(cand, "reject", 0.0, "no_slots")
    if cand.verb not in SPECULABLE:
        return Admission(cand, "reject", 0.0, "not_speculable")
    if cand.verb in STREAM_ONLY:
        return Admission(cand, "reject", 0.0, "stream_only")
    if cand.args is None:
        return Admission(cand, "reject", 0.0, "unresolved_args")
    if cand.p < _tau(cand.verb):
        return Admission(cand, "reject", 0.0, "below_tau")
    priority = cand.p * cand.hidden_ms / cand.cost_ms if cand.cost_ms > 0 else 0.0
    return Admission(cand, "execute", priority, "admitted")


def rank(candidates, free_slots):
    admitted = [admit(c, free_slots) for c in candidates]
    admitted = [a for a in admitted if a.action != "reject"]
    admitted.sort(key=lambda a: a.priority, reverse=True)
    return [a.candidate for a in admitted]
