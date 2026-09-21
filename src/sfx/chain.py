from dataclasses import dataclass

from sfx.schema import STREAM_ONLY
from sfx.fork import WritePathError, validate_write_path as _validate_write_path


@dataclass
class Hop:
    kind: str
    verb: str
    args: dict
    outcome: str
    result: object = None
    duration: float = 0.0
    launch: float = 0.0


@dataclass
class Chain:
    write: object
    hops: list
    future: object = None


def run_chain(write, repo, scratch, next_hop, depth_cap, clock,
              substrate=None, apply_write=None, run_get=None):
    if substrate is None:
        from sfx.substrate import FilesystemSubstrate
        substrate = FilesystemSubstrate(apply_write, run_get)
    if "path" not in write.args:
        raise WritePathError("streamed write missing 'path'")
    _validate_write_path(write.args["path"])
    hops = []
    with substrate.fork(repo, scratch) as h:
        substrate.apply(h, write)
        outcomes = []
        while len(hops) < depth_cap:
            pred = next_hop(outcomes)
            if pred is None:
                break
            kind, verb, args = pred
            if verb == "never" or verb in STREAM_ONLY:
                break
            t0 = clock()
            result, status = substrate.run_get(h, (kind, verb, args))
            hops.append(Hop(kind, verb, args, status, result,
                            duration=clock() - t0, launch=t0))
            outcomes.append(status)
    return Chain(write=write, hops=hops)


def run_hop_in_fork(write, hop, repo, scratch, substrate, clock):
    """Fork, apply the streamed write, run one precomputed hop; return (result, status,
    duration). Runs on the background executor thread; the fork lives only here."""
    _validate_write_path(write.args["path"])
    kind, verb, args = hop
    with substrate.fork(repo, scratch) as h:
        substrate.apply(h, write)
        t0 = clock()
        result, status = substrate.run_get(h, (kind, verb, args))
    return result, status, clock() - t0


def longest_prefix(speculated, real):
    n = 0
    for spec, actual in zip(speculated, real):
        if spec != actual:
            break
        n += 1
    return speculated[:n], speculated[n:]
