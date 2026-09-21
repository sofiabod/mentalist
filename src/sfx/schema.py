from dataclasses import dataclass, field, asdict

POLICIES = {"free", "fork", "never"}
SPECULABLE = {"free", "fork"}
STREAM_ONLY = {"fork"}
HIT_TERMINALS = ("hit_completed", "hit_promoted")
TERMINAL_STATES = {
    "hit_completed",
    "hit_promoted",
    "miss",
    "discarded",
    "preempted",
    "epoch_expired",
}


@dataclass(frozen=True)
class Outcome:
    kind: str
    status: str

    def klass(self):
        return f"{self.kind}:{self.status}"


@dataclass
class ToolEvent:
    t: float
    kind: str
    verb: str
    role: str
    epoch: int
    args: dict = field(default_factory=dict)
    outcome: Outcome | None = None

    def __post_init__(self):
        if self.verb not in POLICIES:
            raise ValueError(f"unknown policy: {self.verb}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if d.get("outcome") is not None:
            d["outcome"] = Outcome(**d["outcome"])
        return cls(**d)


def context_key(events, k, outcome_visible=True):
    role = events[-1].role
    kinds = tuple(e.kind for e in list(events)[-k:])
    last_outcome = events[-1].outcome.klass() if outcome_visible else "?"
    return (role, kinds, last_outcome)


INTERCEPTION_GRADES = {"full", "envelope", "additive", "capture"}
STATE_MODELS = {"fs-only", "capture"}


@dataclass(frozen=True)
class Profile:
    """Declared per-adapter capabilities; a missing one degrades a feature rather than crashing."""
    emission_format: str
    state_model: str
    interception_grade: str
    stream_visibility: str
    fork_substrate: str

    def __post_init__(self):
        if self.state_model not in STATE_MODELS:
            raise ValueError(f"unsupported state model: {self.state_model}")
        if self.fork_substrate not in {"clonefile", "none"}:
            raise ValueError(f"unsupported fork substrate: {self.fork_substrate}")
        if self.interception_grade not in INTERCEPTION_GRADES:
            raise ValueError(f"unknown interception grade: {self.interception_grade}")

    @property
    def streams(self):
        return self.stream_visibility not in ("none", None)

    @property
    def intercepts(self):
        return self.interception_grade in ("full", "envelope")

    @property
    def outcome_visible(self):
        return self.state_model != "capture"

    def modes(self):
        m = []
        m.append("patch-chains" if self.streams else "get-only")
        m.append("intercept" if self.intercepts else "additive-tool")
        m.append("outcome-conditioned" if self.outcome_visible
                 else "outcome-unconditioned")
        return m

    def degradation_mode(self):
        if self.interception_grade == "capture":
            return "replay"
        if not self.streams:
            return "get-only"
        return {"full": "intercept", "envelope": "intercept+verify",
                "additive": "fallback"}[self.interception_grade]


PROFILES = {
    "claude-code": Profile("json-stream", "fs-only", "envelope", "full", "clonefile"),
    "codex": Profile("json-stream", "fs-only", "additive", "full", "clonefile"),
    "capture": Profile("replay", "capture", "capture", "none", "none"),
}
