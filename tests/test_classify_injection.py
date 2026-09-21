"""SECURITY: classify() must not let a safe leading token (pytest/cat/grep/find)
mask shell compounding, command substitution, or a destructive token/flag. Such
a command mutates real state, so it must classify NON-speculable (never), never
free. verb contract (sfx.schema): free/fork = speculable, never = not. Values
hand-derived: each command below performs or hides a WRITE/DELETE, so free is a
lossless-serve of a mutation and is wrong; the safe policy is never.
"""

from mining.normalize import classify
from sfx.schema import SPECULABLE


def v(cmd):
    return classify("Bash", cmd)[1]


# the four bypass examples from the report
def test_pytest_then_touch_is_never():
    assert v("pytest tests/ ; touch X") == "never"


def test_cat_substitution_touch_is_never():
    assert v("cat $(touch X)") == "never"


def test_find_delete_is_never():
    assert v("find . -delete") == "never"


def test_grep_and_rm_is_never():
    assert v("grep foo x && rm -rf y") == "never"


def test_never_cases_not_speculable():
    for c in ("pytest tests/ ; touch X", "cat $(touch X)", "find . -delete",
              "grep foo x && rm -rf y"):
        assert v(c) not in SPECULABLE


# more masked forms
def test_backtick_substitution_is_never():
    assert v("cat `touch X`") == "never"


def test_find_long_delete_is_never():
    assert v("find . -name '*.log' --delete") == "never"


def test_pytest_then_shred_is_never():
    assert v("pytest ; shred secret") == "never"


def test_grep_pipe_xargs_rm_is_never():
    assert v("grep -l foo . | xargs rm -f") == "never"


def test_dd_after_chain_is_never():
    assert v("cat x && dd if=/dev/zero of=disk") == "never"


# safe commands stay free (no false positives)
def test_plain_pytest_free():
    assert classify("Bash", "pytest") == ("test", "free")


def test_plain_cat_free():
    assert classify("Bash", "cat src/foo.py") == ("read", "free")


def test_plain_grep_free():
    assert classify("Bash", "grep x src") == ("grep", "free")
