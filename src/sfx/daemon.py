import json
import os
import stat
import threading
import uuid
from collections import Counter
from functools import wraps
from pathlib import Path

from adapters.stream import StreamParser
from sfx import resolver
from sfx.cache import Cache, _canon
from sfx.chain import (Chain, Hop, WritePathError, _validate_write_path,
                       longest_prefix, run_hop_in_fork)
from sfx.fork import ForkError, validate_write_path
from sfx.executor import Executor
from sfx.gate import Candidate, admit
from sfx.ledger import Ledger
from sfx.predictor import Predictor, _key_str
from sfx.schema import (HIT_TERMINALS, PROFILES, STATE_MODELS,
                        Outcome, Profile, ToolEvent, context_key)
from sfx.substrate import FilesystemSubstrate

GET_BREADTH = 2
RESOLVE_JOIN_TIMEOUT_S = 1.0


def _synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


def _write_is_committed(repo, args):
    """Compare one private regular file without following links or blocking on FIFOs."""
    parent_fd = None
    try:
        relative = validate_write_path(args["path"])
        expected = args["contents"].encode("utf-8")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_fd = os.open(repo, directory_flags)
        for part in relative.parts[:-1]:
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != len(expected):
                return False
            return source.read(len(expected) + 1) == expected
    except (OSError, UnicodeError, WritePathError):
        return False
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

KIND_POLICY = {
    "read": "free", "grep": "free", "search": "free", "test": "free",
    "lint": "free", "typecheck": "free", "build": "free",
    "run": "free", "edit": "fork", "sub-LLM": "free",
    # non-speculable: state a fork cannot faithfully reproduce
    "install": "never", "network": "never", "append": "never", "git": "never",
    "unknown": "never", "unsafe": "never",
}


def _resolve_profile(profile, state_model):
    if state_model not in STATE_MODELS:
        raise ValueError(f"unsupported state model: {state_model}")
    if profile is not None:
        if isinstance(profile, Profile):
            return profile
        if isinstance(profile, str) and profile in PROFILES:
            return PROFILES[profile]
        raise ValueError(f"unknown profile: {profile}")
    return Profile(emission_format="json-stream", state_model=state_model,
                   interception_grade="full", stream_visibility="full",
                   fork_substrate="clonefile")


def _step_tau(gate, verb):
    from sfx.gate import TAU_FORK, TAU_GET
    if verb == "fork":
        return TAU_FORK
    return TAU_GET


def _load_repo_table(repo, state_dir=None):
    p = (Path(state_dir) if state_dir else repo / ".sfx") / "table.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _persist_repo_table(repo, session_counts, state_dir=None):
    if not session_counts:
        return
    d = Path(state_dir) if state_dir else repo / ".sfx"
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    existing = _load_repo_table(repo, state_dir)
    if existing:
        for key, entry in existing["table"].items():
            counts[key] = Counter({kind: round(entry["support"] * prob)
                                   for kind, prob in entry["p"].items()})
    for key, ctr in session_counts.items():
        counts.setdefault(key, Counter()).update(ctr)
    table = {}
    for key, ctr in counts.items():
        support = sum(ctr.values())
        table[key] = {"support": support,
                      "p": {kind: c / support for kind, c in ctr.items()}}
    p = d / "table.json"
    tmp = d / "table.json.tmp"
    tmp.write_text(json.dumps({"table": table}, sort_keys=True))
    os.replace(tmp, p)


class Session:
    def __init__(self, sid, repo, role, clock, global_table, k, scratch=None,
                 repo_table=None, profile=None, state_dir=None):
        self.sid = sid
        self.repo = repo
        self.repo_key = repo.resolve()
        self.scratch = scratch
        self.state_dir = state_dir
        self.role = role
        self.profile = profile
        self.ledger = Ledger()
        self.cache = Cache(clock, self.ledger)
        self.executor = Executor(clock, self.cache, self.ledger, slots=4)
        self.predictor = Predictor(global_table, k, repo_table=repo_table,
                                   outcome_visible=profile.outcome_visible)
        self.stream = StreamParser()
        self.last_event = clock()
        self.chain = None
        self.chain_claim = None
        self.chain_cursor = 0
        self.chain_spec_ids = []
        self.real_hops = []
        self._seq = 0
        self._flushed = 0
        self.arg_by_kind = {}
        self.last_edit_path = ""
        self.last_observation = ""
        self.registered_kinds = None
        self.spec_disabled = False
        self.trajectory = None
        self.traj_cursor = 0
        self.streamed_write = False
        self.streamed_gets = set()
        self.sptc_enabled = False
        self.chain_mutation_id = None
        self.completed_mutations = {}

    def next_id(self):
        self._seq += 1
        return self._seq, f"{self.sid}:{self._seq}"


class Daemon:
    def __init__(self, clock, global_table, k, run=None, resolve_args=None,
                 on_idle=None, slots=4, apply_write=None, run_in_fork=None,
                 depth_cap=1, log=None):
        self.clock = clock
        self.global_table = global_table
        self.k = global_table["k"]
        self.run = run
        self.resolve_args = resolve_args or (lambda kind, ctx: resolver.resolve(kind, ctx))
        self.on_idle = on_idle or (lambda sid: None)
        self.fs_substrate = FilesystemSubstrate(apply_write, run_in_fork)
        self.depth_cap = depth_cap
        self.log = log or (lambda rec: None)
        self.sessions = {}
        self._lock = threading.RLock()
        self._mutations = {}

    def _flush_spec_end(self, s):
        for rec in s.ledger.records[s._flushed:]:
            if rec["ev"] == "spec_end" and rec["terminal"] not in HIT_TERMINALS:
                self.log({"ev": "spec_end", "id": rec["id"], "terminal": rec["terminal"],
                          "wasted_cpu_ms": rec["wasted_ms"]})
        s._flushed = len(s.ledger.records)

    @_synchronized
    def turn_end(self, sid):
        self._flush_spec_end(self.sessions[sid])

    @_synchronized
    def session_end(self, sid):
        s = self.sessions[sid]
        if any(m["sid"] == sid for m in self._mutations.get(s.repo_key, {}).values()):
            raise ValueError("cannot end a session with an active mutation")
        _persist_repo_table(s.repo, s.predictor.session_counts, s.state_dir)
        s.cache.bump_epoch()
        s.executor.shutdown()
        self._flush_spec_end(s)
        del self.sessions[sid]

    @_synchronized
    def shutdown(self):
        for s in self.sessions.values():
            s.cache.bump_epoch()
            s.executor.shutdown()
            self._flush_spec_end(s)
        self.sessions.clear()
        self._mutations.clear()

    @_synchronized
    def session_start(self, sid, repo, role, scratch=None, state_model="fs-only",
                      profile=None, registered_kinds=None,
                      state_dir=None, spec_disabled=False, trajectory=None,
                      sptc=True):
        repo = Path(repo)
        profile = _resolve_profile(profile, state_model)
        s = Session(sid, repo, role, self.clock, self.global_table, self.k,
                    scratch=scratch, repo_table=_load_repo_table(repo, state_dir),
                    profile=profile, state_dir=state_dir)
        s.registered_kinds = set(registered_kinds) if registered_kinds else None
        s.spec_disabled = spec_disabled
        s.sptc_enabled = sptc
        s.trajectory = trajectory
        self.sessions[sid] = s
        return s

    def _invalidate_repo(self, repo_key, preserve=None):
        for s in self.sessions.values():
            if s.repo_key != repo_key or s is preserve:
                continue
            if s.chain is not None:
                self._discard_chain(s, from_hop=0, reason="epoch_bump")
            s.streamed_write = False
            s.cache.bump_epoch()
            seq, _ = s.next_id()
            self.log({"ev": "epoch", "seq": seq, "epoch": s.cache.epoch})

    @_synchronized
    def on_fs_change(self, sid):
        repo_key = self.sessions[sid].repo_key
        for mutation in self._mutations.get(repo_key, {}).values():
            mutation["conflicted"] = True
        self._invalidate_repo(repo_key)

    @_synchronized
    def mutation_begin(self, sid, write_args=None):
        """Fence every reader of the repo before an authoritative mutation starts."""
        s = self.sessions[sid]
        if write_args is not None:
            if (not isinstance(write_args, dict)
                    or set(write_args) != {"path", "contents"}
                    or not isinstance(write_args["path"], str)
                    or not isinstance(write_args["contents"], str)):
                raise ValueError("write_args must contain only string path and contents")
            _validate_write_path(write_args["path"])
            write_args = dict(write_args)
        active = self._mutations.setdefault(s.repo_key, {})
        for mutation in active.values():
            mutation["conflicted"] = True
        mutation_id = uuid.uuid4().hex
        active[mutation_id] = {"sid": sid, "write_args": write_args,
                               "conflicted": bool(active)}
        self._invalidate_repo(s.repo_key)
        return mutation_id

    @_synchronized
    def mutation_end(self, sid, mutation_id, success=False):
        s = self.sessions[sid]
        active = self._mutations.get(s.repo_key, {})
        mutation = active.get(mutation_id)
        if mutation is None or mutation["sid"] != sid:
            raise ValueError("unknown mutation for this session")
        preserve = (success is True and not mutation["conflicted"] and len(active) == 1
                    and s.chain is not None and s.chain_mutation_id == mutation_id
                    and mutation["write_args"] == s.chain.write.args)
        if preserve and s.chain.future is not None and s.chain.future.done():
            preserve = (not s.chain.future.cancelled()
                        and s.chain.future.exception() is None)
        if preserve:
            preserve = _write_is_committed(s.repo_key, mutation["write_args"])
        del active[mutation_id]
        if not active:
            self._mutations.pop(s.repo_key, None)
        self._invalidate_repo(s.repo_key, preserve=s if preserve else None)
        s.completed_mutations[mutation_id] = bool(preserve)
        s.chain_mutation_id = None
        return bool(preserve)

    @_synchronized
    def tick(self, idle_after):
        now = self.clock()
        for sid, s in self.sessions.items():
            if now - s.last_event >= idle_after:
                self.on_idle(sid)

    @_synchronized
    def call_executed(self, sid, kind, verb, outcome, args, latency, observation="",
                      mutation_id=None, speculate=True):
        s = self.sessions[sid]
        t0 = self.clock()
        s.last_event = t0
        s.last_observation = observation
        if speculate and args and args.get("cmd"):
            s.arg_by_kind[kind] = args["cmd"]
        preserved = False
        if mutation_id is not None:
            if mutation_id not in s.completed_mutations:
                raise ValueError("report requires a completed mutation for this session")
            preserved = s.completed_mutations.pop(mutation_id)
            s.streamed_write = False
        elif verb != "free":
            streamed_match = (s.streamed_write and s.chain is not None
                              and args == s.chain.write.args)
            if not streamed_match:
                self.on_fs_change(sid)
            s.streamed_write = False
        if not preserved:
            s.predictor.observe(ToolEvent(t=t0, kind=kind, verb=verb,
                                          role=s.role, epoch=s.cache.epoch,
                                          args=args, outcome=Outcome(kind, outcome)))
        oracle_next = self._oracle_advance(s, kind)
        if (not speculate or preserved or s.spec_disabled
                or self._mutations.get(s.repo_key)):
            seq, cid = s.next_id()
            self._emit_step(s, seq, cid, None, None, [], None, "none",
                            "reject", "mutation_active" if self._mutations.get(s.repo_key)
                            else "speculation_suppressed", 0.0, None, t0)
            return
        ranked = [(oracle_next["kind"], 1.0)] if oracle_next else s.predictor.propose()
        seq, cid = s.next_id()
        if not ranked:
            self._emit_step(s, seq, cid, None, None, ranked, None, "none",
                            "reject", "no_proposal", 0.0, None, t0)
            return
        rctx = resolver.Ctx(repo=s.repo, session=s.arg_by_kind,
                            last_edit_path=s.last_edit_path,
                            last_observation=s.last_observation)
        launched = 0
        for i, (pred_kind, p) in enumerate(ranked):
            pred_verb = KIND_POLICY[pred_kind]
            pred_args = oracle_next["args"] if oracle_next else self.resolve_args(pred_kind, rctx)
            tier = resolver.resolve_tier(pred_kind, rctx)
            cand = Candidate(kind=pred_kind, verb=pred_verb, p=p, args=pred_args,
                             hidden_ms=latency)
            slots = 0 if launched >= GET_BREADTH else s.executor.free_slots()
            a = admit(cand, free_slots=slots)
            spec = None
            if a.action == "execute" and not s.spec_disabled:
                key = s.executor.speculate(pred_kind, pred_args, priority=a.priority,
                                           job=lambda k=pred_kind, ar=pred_args: self.run(k, ar))
                if key is not None:
                    spec = {"fork_id": s.executor._specs[key].spec_id, "kind": pred_kind}
                    launched += 1
            if i == 0:
                self._emit_step(s, seq, cid, pred_kind, pred_verb, ranked, pred_args,
                                tier, a.action, a.reason, a.priority, spec, t0)
            elif spec is not None:
                eseq, ecid = s.next_id()
                self._emit_step(s, eseq, ecid, pred_kind, pred_verb, ranked, pred_args,
                                tier, a.action, a.reason, a.priority, spec, t0)
            if a.reason == "no_slots":
                break

    def _oracle_advance(self, s, kind):
        if s.trajectory is None:
            return None
        s.traj_cursor += 1
        if s.traj_cursor >= len(s.trajectory):
            return None
        return s.trajectory[s.traj_cursor]

    def _emit_step(self, s, seq, cid, kind, verb, ranked, args, tier, gate,
                   reason, priority, spec, t0):
        self.log({"ev": "step", "seq": seq, "id": cid, "kind": kind, "verb": verb,
                  "ctx": _key_str(*context_key(s.predictor.events, s.predictor.k)),
                  "session_support": sum(sum(c.values())
                                         for c in s.predictor.session_counts.values()),
                  "proposal": [[k, round(p, 4)] for k, p in ranked[:3]],
                  "resolved_args": args, "args_tier": tier,
                  "gate": gate, "gate_reason": reason,
                  "tau": _step_tau(gate, verb), "free_slots": s.executor.free_slots(),
                  "spec": spec, "wall_ms": self.clock() - t0, "epoch": s.cache.epoch})
        self._flush_spec_end(s)

    @_synchronized
    def call_stream_delta(self, sid, call_id, tool, delta, mutation_id=None):
        s = self.sessions[sid]
        s.last_event = self.clock()
        if not s.profile.streams or s.spec_disabled:
            return None
        write = s.stream.feed(call_id, tool, delta)
        if write is None:
            return None
        active = self._mutations.get(s.repo_key, {})
        mutation = active.get(mutation_id)
        if active or mutation_id is not None:
            if (mutation is None or mutation["sid"] != sid or len(active) != 1
                    or mutation["conflicted"] or mutation["write_args"] != write.args):
                return None
        carried = []
        if s.chain is not None:
            carried = [h.kind for h in s.chain.hops[max(0, s.chain_cursor - 1):]]
        if mutation is None:
            self.on_fs_change(sid)
        elif s.chain is not None:
            self._discard_chain(s, from_hop=0, reason="replaced_stream")
        s.chain_mutation_id = mutation_id
        s.streamed_write = True
        s.last_edit_path = write.args["path"]
        s.predictor.observe(ToolEvent(t=self.clock(), kind="edit", verb=write.verb,
                                      role=s.role, epoch=s.cache.epoch,
                                      args=write.args, outcome=Outcome("edit", "OK")))
        try:
            _validate_write_path(write.args["path"])
        except WritePathError:
            self._emit_chain(s, write.verb, [], commit=False,
                             reason="write_path_error")
            return None
        pred = self._next_hop(s, carried, write.args)([]) if self.depth_cap >= 1 else None
        if pred is None:
            s.chain = Chain(write=write, hops=[])
            s.chain_cursor = 0
            s.chain_spec_ids = []
            s.real_hops = []
            self._emit_chain(s, write.verb, [], commit=True, reason=None)
            return s.chain
        return self._submit_chain(s, write, pred)

    @_synchronized
    def stream_get_delta(self, sid, call_id, kind, delta):
        """sPTC for GET-class calls: parse the emerging read/run from its own tokens and
        speculate it on the async lane as soon as the args are unambiguous from a PARTIAL
        prefix, before generation completes. Stateless: args come from tokens, not history.
        """
        s = self.sessions[sid]
        s.last_event = self.clock()
        if (not s.profile.streams or not s.sptc_enabled or s.spec_disabled
                or self._mutations.get(s.repo_key)):
            return None
        partial, done = s.stream.feed_get(call_id, delta)
        if partial is None or call_id in s.streamed_gets:
            return None
        verb = KIND_POLICY[kind]
        if verb not in ("free",):
            return None
        s.streamed_gets.add(call_id)
        s.executor.speculate(kind, partial, priority=1.0,
                             job=lambda k=kind, a=partial: self.run(k, a))
        return partial

    def _submit_chain(self, s, write, pred):
        kind, verb, args = pred
        substrate = self.fs_substrate
        try:
            substrate.probe(s.repo, s.scratch)
        except ForkError:
            s.chain = None
            self._emit_chain(s, write.verb, [], commit=False, reason="no_substrate")
            return None

        execution_id = uuid.uuid4().hex

        def job():
            started = self.clock()
            self.log({"ev": "fork_execution", "execution_id": execution_id,
                      "kind": kind, "phase": "started", "monotonic_ms": started,
                      "elapsed_ms": 0.0})
            try:
                result, _status, dur = run_hop_in_fork(write, pred, s.repo, s.scratch,
                                                       substrate, self.clock)
            except Exception as exc:
                ended = self.clock()
                self.log({"ev": "fork_execution", "execution_id": execution_id,
                          "kind": kind, "phase": "error",
                          "monotonic_ms": ended, "elapsed_ms": ended - started,
                          "error_type": type(exc).__name__})
                raise
            ended = self.clock()
            event = {"ev": "fork_execution", "execution_id": execution_id,
                     "kind": kind, "phase": "completed",
                     "monotonic_ms": ended, "elapsed_ms": ended - started}
            if isinstance(result, (tuple, list)) and len(result) in (2, 3) and type(result[-1]) is int:
                event["return_code"] = result[-1]
            self.log(event)
            return result, dur

        spec_id, future = s.executor.submit_chain(kind, args, job)
        hop = Hop(kind, verb, args, None)
        s.chain = Chain(write=write, hops=[hop], future=future)
        s.chain_cursor = 0
        s.chain_spec_ids = [spec_id]
        s.real_hops = []
        self._emit_chain(s, write.verb, [kind], commit=True, reason=None,
                         execution_id=execution_id)
        return s.chain

    def _emit_chain(self, s, verb, hops, commit, reason, execution_id=None):
        seq, cid = s.next_id()
        event = {"ev": "chain", "seq": seq, "id": cid, "verb": verb, "hops": hops,
                 "commit": commit, "discard_reason": reason, "epoch": s.cache.epoch}
        if execution_id is not None:
            event["execution_id"] = execution_id
        self.log(event)
        self._flush_spec_end(s)

    def _next_hop(self, s, carried=None, write_args=None):
        prior = {"kind": None}
        carried = list(carried or [])

        def next_hop(outcomes):
            if outcomes:
                last_kind = prior["kind"]
                s.predictor.observe(ToolEvent(
                    t=self.clock(), kind=last_kind, verb=KIND_POLICY[last_kind],
                    role=s.role, epoch=s.cache.epoch, args={},
                    outcome=Outcome(last_kind, outcomes[-1])))
            ranked = s.predictor.propose()
            candidates = [(kind, probability, "predictor") for kind, probability in ranked]
            if carried:
                kind = carried.pop(0)
                candidates = [(kind, 1.0, "carried")] + [
                    entry for entry in candidates if entry[0] != kind]
            if not candidates:
                self.log({"ev": "chain_prediction", "session": s.sid,
                          "gate": "reject", "gate_reason": "no_proposal",
                          "epoch": s.cache.epoch})
                return None
            ctx = resolver.Ctx(repo=s.repo, session=s.arg_by_kind,
                               last_edit_path=s.last_edit_path,
                               last_edit_contents=(write_args or {}).get("contents"),
                               last_observation=s.last_observation)
            for kind, probability, source in candidates:
                verb = KIND_POLICY.get(kind, "never")
                args = self.resolve_args(kind, ctx) if verb == "free" else None
                _resolved, tier, resolution_reason = resolver.resolve_details(kind, ctx)
                admission = admit(Candidate(kind=kind, verb=verb, p=probability, args=args),
                                  free_slots=s.executor.free_slots())
                reason = admission.reason
                if reason == "unresolved_args":
                    reason = resolution_reason if resolution_reason != "resolved" else reason
                self.log({"ev": "chain_prediction", "session": s.sid,
                          "kind": kind, "verb": verb, "source": source,
                          "probability": probability if source == "predictor" else None,
                          "resolved_args": args, "args_tier": tier,
                          "gate": admission.action, "gate_reason": reason,
                          "epoch": s.cache.epoch})
                if admission.action == "execute":
                    prior["kind"] = kind
                    return kind, verb, args
                if reason == "no_slots":
                    break
            return None
        return next_hop

    def resolve(self, sid, kind, args):
        chain_claim = None
        with self._lock:
            s = self.sessions[sid]
            t0 = self.clock()
            s.last_event = t0
            if (s.spec_disabled or self._mutations.get(s.repo_key)
                    or (not s.profile.intercepts and s.registered_kinds is not None
                        and kind not in s.registered_kinds)
                    or KIND_POLICY.get(kind, "never") == "never"):
                self._emit_resolve(s, kind, args, "miss", None, 0.0, t0)
                return "miss", None
            if s.chain is not None:
                before = s.ledger.total_saved_ms
                outcome, result = self._resolve_chain(s, kind, args)
                if outcome != "join":
                    self._emit_resolve(s, kind, args, outcome, result,
                                       s.ledger.total_saved_ms - before, t0)
                    return outcome, result
                job = result
                chain_claim = s.chain_claim
            else:
                job = s.cache.claim(kind, args)

        # A producer or another client may need the daemon lock to finish or
        # invalidate this job. Never wait for speculative execution under it.
        completed = False
        if job is not None:
            # Bound the optimization join below the control socket's timeout.
            # A miss runs authoritatively with the caller's own command timeout;
            # it never publishes an incomplete speculative result.
            completed = job.done.wait(timeout=RESOLVE_JOIN_TIMEOUT_S)
        with self._lock:
            before = s.ledger.total_saved_ms
            if chain_claim is not None:
                outcome, result = self._finish_chain_claim(s, chain_claim, t0, completed)
                self._emit_resolve(s, kind, args, outcome, result,
                                   s.ledger.total_saved_ms - before, t0)
                return outcome, result
            if self.sessions.get(sid) is not s or self._mutations.get(s.repo_key):
                outcome, key, result = "miss", None, None
            else:
                outcome, key, result = s.cache.finish(job, ask_time=t0)
            if outcome == "hit_promoted" and key in s.executor._specs:
                s.executor.promote(key)
            self._emit_resolve(s, kind, args, outcome, result,
                               s.ledger.total_saved_ms - before, t0)
            return outcome, result

    def _emit_resolve(self, s, kind, args, outcome, result, saved_ms, t0):
        seq, cid = s.next_id()
        self.log({"ev": "resolve", "seq": seq, "id": cid, "kind": kind, "args": args,
                  "outcome": outcome, "has_result": result is not None,
                  "saved_ms": saved_ms, "wall_ms": self.clock() - t0,
                  "epoch": s.cache.epoch})
        self._flush_spec_end(s)

    def _resolve_chain(self, s, kind, args):
        if s.chain_claim is not None:
            return "miss", None
        write = s.chain.write
        if s.chain_cursor == 0:
            stable = kind == "edit" and args == write.args
            if not stable:
                self._discard_chain(s, from_hop=0, reason="prefix_break")
                return "miss", None
            contents = s.chain.write.args.get("contents")
            s.chain_cursor = 1
            if not s.chain.hops:
                s.chain = None
            return "hit_completed", contents

        hop_i = s.chain_cursor - 1
        speculated = [(h.kind, _canon(h.args)) for h in s.chain.hops]
        real = [(h.kind, _canon(h.args)) for h in s.real_hops]
        real.append((kind, _canon(args)))
        served, _ = longest_prefix(speculated, real)
        if len(served) < len(real):
            self._discard_chain(s, from_hop=hop_i, reason="prefix_break")
            return "miss", None
        job = s.cache.claim(kind, args)
        if job is None or job.spec_id != s.chain_spec_ids[hop_i]:
            self._discard_chain(s, from_hop=hop_i, reason="chain_unavailable")
            return "miss", None
        s.real_hops.append(Hop(kind, KIND_POLICY[kind], args, None))
        s.chain_cursor += 1
        s.chain_claim = (s.chain, job, hop_i)
        return "join", job

    def _finish_chain_claim(self, s, claim, ask_time, completed):
        chain, job, hop_i = claim
        owned = s.chain is chain and s.chain_claim is claim
        valid = (owned and self.sessions.get(s.sid) is s
                 and not self._mutations.get(s.repo_key) and job.epoch == s.cache.epoch)
        outcome, result = "miss", None
        if valid and completed:
            outcome, _, result = s.cache.finish(job, ask_time=ask_time)
        if outcome == "miss":
            if owned:
                self._discard_chain(s, from_hop=hop_i,
                                    reason="chain_join_timeout" if not completed
                                    else "chain_unavailable")
            else:
                if job.spec_id not in s.ledger._terminated:
                    s.ledger.terminal(job.spec_id, "discarded")
                s.cache.discard_spec(job.spec_id)
                if s.chain_claim is claim:
                    s.chain_claim = None
            return outcome, result
        s.chain_claim = None
        if hop_i + 1 >= len(chain.hops):
            s.chain = None
        return outcome, result

    def _discard_chain(self, s, from_hop, reason):
        for spec_id in s.chain_spec_ids[from_hop:]:
            if spec_id not in s.ledger._terminated:
                s.ledger.terminal(spec_id, "discarded")
                s.cache.discard_spec(spec_id)
        s.chain = None
        s.chain_claim = None
        s.chain_mutation_id = None
        self._emit_chain(s, "fork", [], commit=False, reason=reason)
