import threading


class Job:
    def __init__(self, spec_id, kind, args, occurrence, epoch, launch, duration,
                 result=None, synchronous=False):
        self.spec_id = spec_id
        self.kind = kind
        self.args = args
        self.occurrence = occurrence
        self.epoch = epoch
        self.launch = launch
        self.duration = duration
        self.result = result
        self.synchronous = synchronous
        self.done = threading.Event()

    @property
    def key(self):
        return (self.kind, _canon(self.args), self.occurrence, self.epoch)

    def done_by(self, t):
        return self.done.is_set() and self.launch + self.duration <= t


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _canon(args):
    return _freeze(args)


class Cache:
    def __init__(self, clock, ledger):
        self.clock = clock
        self.ledger = ledger
        self.epoch = 0
        self.jobs = {}
        self._put_counts = {}
        self._serve_counts = {}
        self._n = 0
        self._lock = threading.RLock()

    def reserve(self, kind, args, launch, synchronous=False):
        with self._lock:
            slot = (kind, _canon(args), self.epoch)
            occ = self._put_counts.get(slot, 0)
            self._put_counts[slot] = occ + 1
            spec_id = f"spec{self._n}"
            self._n += 1
            job = Job(spec_id, kind, args, occ, self.epoch, launch, None, None,
                      synchronous=synchronous)
            self.jobs[job.key] = job
            self.ledger.record(spec_id, "spec_launch", lane="spec")
            return job.key, spec_id

    def complete(self, spec_id, result, duration):
        with self._lock:
            job = next((j for j in self.jobs.values() if j.spec_id == spec_id), None)
            if job is None:
                return
            job.result = result
            job.duration = duration
            job.done.set()

    def put(self, kind, args, duration, result=None, launch=None, synchronous=False):
        with self._lock:
            launch = self.clock() if launch is None else launch
            key, spec_id = self.reserve(kind, args, launch, synchronous=synchronous)
        self.complete(spec_id, result, duration)
        return key, spec_id

    def claim(self, kind, args):
        """Claim one occurrence before waiting, so concurrent asks cannot share it."""
        with self._lock:
            slot = (kind, _canon(args), self.epoch)
            occ = self._serve_counts.get(slot, 0)
            # Failed/discarded reservations leave holes. Never reuse their numbers:
            # a later in-flight reservation may already own the next occurrence.
            while occ < self._put_counts.get(slot, 0):
                job = self.jobs.get((kind, _canon(args), occ, self.epoch))
                if job is not None:
                    break
                occ += 1
            else:
                job = None
            self._serve_counts[slot] = occ
            if job is None:
                return None
            self._serve_counts[slot] = occ + 1
            return job

    def finish(self, job, ask_time):
        """Validate a claimed, completed job without blocking the caller's lock."""
        with self._lock:
            if (job is None or not job.done.is_set() or job.duration is None
                    or job.epoch != self.epoch or self.jobs.get(job.key) is not job
                    or job.spec_id in self.ledger._terminated):
                return "miss", None, None
            # synchronous hops ran on the request thread: cost was paid inline, not
            # hidden behind model work, so no latency was actually saved
            saved = 0.0 if job.synchronous else max(0.0, min(job.duration, ask_time - job.launch))
            if job.done_by(ask_time):
                self.ledger.terminal(job.spec_id, "hit_completed", saved_ms=saved)
                return "hit_completed", job.key, job.result
            self.ledger.terminal(job.spec_id, "hit_promoted", saved_ms=saved)
            return "hit_promoted", job.key, job.result

    def serve(self, kind, args, ask_time):
        job = self.claim(kind, args)
        if job is None:
            return "miss", None, None
        job.done.wait()
        return self.finish(job, ask_time)

    def discard_spec(self, spec_id):
        with self._lock:
            key = next((k for k, j in self.jobs.items() if j.spec_id == spec_id), None)
            if key is None:
                return
            job = self.jobs.pop(key)
            job.done.set()

    def bump_epoch(self):
        with self._lock:
            old = self.epoch
            self.epoch += 1
            for job in self.jobs.values():
                if job.epoch <= old:
                    if job.spec_id not in self.ledger._terminated:
                        dur = job.duration if job.duration is not None else self.clock() - job.launch
                        wasted = min(dur, self.clock() - job.launch)
                        self.ledger.terminal(job.spec_id, "epoch_expired", wasted_ms=wasted)
                    job.done.set()
            self.jobs = {k: j for k, j in self.jobs.items() if j.epoch > old}
