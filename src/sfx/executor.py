from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Spec:
    key: tuple
    spec_id: str
    priority: float
    launch: float
    future: Any = None
    duration: float = None
    promoted: bool = False
    failed: bool = False


class Executor:
    def __init__(self, clock, cache, ledger, slots):
        self.clock = clock
        self.cache = cache
        self.ledger = ledger
        self.slots = slots
        self.running = set()
        self._specs = {}
        self._pool = ThreadPoolExecutor(max_workers=slots,
                                        thread_name_prefix="sfx-spec")

    def free_slots(self):
        now = self.clock()
        for key in list(self.running):
            spec = self._specs[key]
            if spec.failed:
                self.running.discard(key)
                continue
            done = spec.future is None or spec.future.done()
            if done and spec.duration is not None and spec.launch + spec.duration <= now:
                self.running.discard(key)
        return self.slots - len(self.running)

    def authoritative(self, job, need_slot=False):
        if need_slot and self.free_slots() <= 0:
            self._preempt_one()
        result, _ = job()
        return result

    def speculate(self, kind, args, priority, job):
        if self.free_slots() <= 0:
            return None
        launch = self.clock()
        key, spec_id = self.cache.reserve(kind, args, launch)
        spec = Spec(key=key, spec_id=spec_id, priority=priority, launch=launch)
        self._specs[key] = spec
        self.running.add(key)
        spec.future = self._pool.submit(self._run, spec, job)
        return key

    def submit_chain(self, kind, args, job):
        """Reserve an async cache slot and run `job` on the pool, off the GET-class
        slots. `job` returns (result, duration); on failure the slot is discarded.
        Returns (spec_id, future)."""
        launch = self.clock()
        _, spec_id = self.cache.reserve(kind, args, launch, synchronous=False)
        future = self._pool.submit(self._run_chain, spec_id, job)
        return spec_id, future

    def _run_chain(self, spec_id, job):
        try:
            result, duration = job()
        except Exception:
            self.cache.discard_spec(spec_id)
            raise
        self.cache.complete(spec_id, result, duration)

    def _run(self, spec, job):
        try:
            result, duration = job()
        except Exception:
            self.cache.discard_spec(spec.spec_id)
            spec.failed = True
            return
        spec.duration = duration
        self.cache.complete(spec.spec_id, result, duration)

    def drain(self):
        for spec in list(self._specs.values()):
            if spec.future is not None:
                spec.future.result()
        self.free_slots()

    join = drain

    def promote(self, key):
        self._specs[key].promoted = True

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _preempt_one(self):
        candidates = [s for k, s in self._specs.items()
                      if k in self.running and not s.promoted
                      and s.spec_id not in self.ledger._terminated]
        if not candidates:
            return
        victim = min(candidates, key=lambda s: s.priority)
        dur = victim.duration if victim.duration is not None else self.clock() - victim.launch
        wasted = min(dur, self.clock() - victim.launch)
        self.ledger.terminal(victim.spec_id, "preempted", wasted_ms=wasted)
        self.running.discard(victim.key)
