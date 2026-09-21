from sfx.gate import admit, rank, Candidate, TAU_GET, TAU_FORK


def _get(p=0.9, args={"cmd": "pytest"}):
    return Candidate(kind="test", verb="free", p=p, args=args)


def test_admits_resolvable_speculable_confident_get_with_budget():
    a = admit(_get(), free_slots=2)
    assert a is not None
    assert a.action == "execute"


def test_rejects_unresolvable_get_regardless_of_confidence():
    a = admit(_get(p=0.99, args=None), free_slots=2)
    assert a.action == "reject"
    assert a.reason == "unresolved_args"


def test_rejects_non_speculable_never():
    c = Candidate(kind="push", verb="never", p=0.99, args={"cmd": "git push"})
    assert admit(c, free_slots=2).action == "reject"


def test_rejects_confidence_below_tau_rung():
    assert admit(_get(p=TAU_GET - 0.01), free_slots=2).action == "reject"
    assert admit(_get(p=TAU_GET), free_slots=2) is not None


def test_rejects_when_no_budget_slot():
    assert admit(_get(), free_slots=0).action == "reject"


def test_fork_rung_needs_higher_tau_than_get():
    assert TAU_FORK > TAU_GET
    patch = Candidate(kind="edit", verb="fork", p=TAU_GET + 0.001, args={"cmd": "x"})
    assert admit(patch, free_slots=1).action == "reject"
    patch_ok = Candidate(kind="edit", verb="fork", p=TAU_FORK, args={"cmd": "x"})
    assert admit(patch_ok, free_slots=1) is not None


def test_unresolved_args_rejects_even_when_kind_confident():
    c = Candidate(kind="test", verb="free", p=0.9, args=None)
    a = admit(c, free_slots=1)
    assert a.action == "reject"
    assert a.reason == "unresolved_args"


def test_rejects_stream_only_fork_even_when_resolved_and_confident():
    c = Candidate(kind="edit", verb="fork", p=TAU_FORK + 0.1, args={"path": "a.py"})
    a = admit(c, free_slots=2)
    assert a.action == "reject"
    assert a.reason == "stream_only"


def test_stream_only_reject_does_not_affect_free():
    a = admit(_get(p=TAU_GET + 0.1), free_slots=2)
    assert a.action == "execute"


def test_ev_ranking_orders_two_admitted_candidates():
    cheap_slow = Candidate(kind="test", verb="free", p=0.5, args={"cmd": "pytest"},
                           hidden_ms=8000, cost_ms=10)
    fast_certain = Candidate(kind="lint", verb="free", p=0.9, args={"cmd": "ruff"},
                             hidden_ms=500, cost_ms=5)
    ranked = rank([fast_certain, cheap_slow], free_slots=2)
    assert [c.kind for c in ranked] == ["test", "lint"]
