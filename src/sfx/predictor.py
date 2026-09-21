from collections import Counter, defaultdict, deque

from sfx.schema import context_key

SUPPORT_CAP = 20


def _key_str(role, kinds, last_outcome):
    return f"{role}|{','.join(kinds)}|{last_outcome}"


class Predictor:
    def __init__(self, global_table, k, repo_table=None, outcome_visible=True):
        self.k = k
        self.outcome_visible = outcome_visible
        self.global_table = global_table["table"]
        self.repo_table = (repo_table or {}).get("table", {})
        self.session_counts = defaultdict(Counter)
        self.events = deque(maxlen=self.k + 1)

    def observe(self, event):
        if self.events:
            key = _key_str(*context_key(self.events, self.k, self.outcome_visible))
            self.session_counts[key][event.kind] += 1
        self.events.append(event)

    def propose(self):
        if not self.events:
            return []
        key = _key_str(*context_key(self.events, self.k, self.outcome_visible))
        sources = []
        session = self.session_counts.get(key)
        n_session = sum(session.values()) if session else 0
        decay = SUPPORT_CAP / (SUPPORT_CAP + n_session)
        for table in (self.global_table, self.repo_table):
            entry = table.get(key)
            if entry:
                sources.append((min(entry["support"], SUPPORT_CAP) * decay, entry["p"]))
        if session:
            sources.append((min(n_session, SUPPORT_CAP),
                            {kind: c / n_session for kind, c in session.items()}))

        if not sources:
            return []

        total = sum(w for w, _ in sources)
        blended = defaultdict(float)
        for w, dist in sources:
            for kind, prob in dist.items():
                blended[kind] += (w / total) * prob
        return sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
