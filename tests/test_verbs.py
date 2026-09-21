from sfx.schema import POLICIES
from sfx.verbs import POLICY, valid_prefix, Annotation


def test_policy_table_matches_site():
    expected = {
        "free": (False, "serve"),
        "fork": (True, "diff-apply-under-epoch"),
        "never": (False, "none"),
    }
    assert set(POLICY) == POLICIES
    for policy, (stream_only, commit) in expected.items():
        assert POLICY[policy].writes_from_stream_only == stream_only
        assert POLICY[policy].commit == commit


def test_fork_free_free_chain_is_fully_valid():
    assert valid_prefix(["fork", "free", "free"]) == ["fork", "free", "free"]


def test_never_truncates_chain_before_it():
    assert valid_prefix(["fork", "free", "never", "free"]) == ["fork", "free"]


def test_leading_never_yields_empty_prefix():
    assert valid_prefix(["never", "free"]) == []


def test_annotation_tiers_are_declared_inferred_validated():
    assert (Annotation.DECLARED, Annotation.INFERRED, Annotation.VALIDATED) == (1, 2, 3)
