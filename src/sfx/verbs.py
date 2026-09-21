from dataclasses import dataclass
from enum import IntEnum

from sfx.schema import POLICIES


class Annotation(IntEnum):
    DECLARED = 1
    INFERRED = 2
    VALIDATED = 3


@dataclass(frozen=True)
class Policy:
    writes_from_stream_only: bool
    commit: str


POLICY = {
    "free": Policy(False, "serve"),
    "fork": Policy(True, "diff-apply-under-epoch"),
    "never": Policy(False, "none"),
}

assert set(POLICY) == POLICIES


def valid_prefix(chain):
    prefix = []
    for verb in chain:
        if verb == "never":
            break
        prefix.append(verb)
    return prefix
