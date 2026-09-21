from collections import Counter

from sfx.schema import TERMINAL_STATES


class Ledger:
    def __init__(self):
        self.records = []
        self._terminated = set()
        self.total_saved_ms = 0.0
        self.total_wasted_ms = 0.0

    def record(self, spec_id, ev, **fields):
        self.records.append({"id": spec_id, "ev": ev, **fields})

    def terminal(self, spec_id, state, saved_ms=0.0, wasted_ms=0.0):
        if state not in TERMINAL_STATES:
            raise ValueError(f"unknown terminal state: {state}")
        if spec_id in self._terminated:
            raise ValueError(f"speculation already terminal: {spec_id}")
        self._terminated.add(spec_id)
        self.total_saved_ms += saved_ms
        self.total_wasted_ms += wasted_ms
        self.records.append(
            {"id": spec_id, "ev": "spec_end", "terminal": state,
             "saved_ms": saved_ms, "wasted_ms": wasted_ms}
        )

    def terminal_counts(self):
        return Counter(r["terminal"] for r in self.records if r["ev"] == "spec_end")
