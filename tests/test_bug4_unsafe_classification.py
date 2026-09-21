"""BUG 4 (SAFETY): unknown commands must default to the safest non-speculable
policy (never), and install must classify consistently across normalize and
daemon.

verb contract (sfx.schema): free/fork = speculable, never = not. An unknown
command whose side effects we cannot reason about must not be speculated, so
it classifies never and gate.admit rejects it.
"""

from mining.normalize import classify
from sfx.gate import Candidate, admit
from sfx.daemon import KIND_POLICY
from sfx.schema import SPECULABLE


def v(cmd):
    return classify("Bash", cmd)[1]


def test_unknown_command_is_not_speculable():
    verb = v("terraform apply -auto-approve")
    assert verb == "never"
    assert verb not in SPECULABLE


def test_unknown_command_rejected_by_gate():
    kind, verb = classify("Bash", "kubectl delete pod x")
    a = admit(Candidate(kind=kind, verb=verb, p=0.99, args={"cmd": "x"}), free_slots=4)
    assert a.action == "reject"
    assert a.reason == "not_speculable"


def test_install_consistent_between_normalize_and_daemon():
    assert classify("Bash", "pip install requests") == ("install", "never")
    assert KIND_POLICY["install"] == "never"
