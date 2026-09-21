import json
import shlex

import pytest

from sfx.resolver import (
    Ctx,
    canonical_args,
    canonical_command,
    resolve,
    resolve_tier,
)


def _pkg(d, scripts):
    (d / "package.json").write_text(json.dumps({"scripts": scripts}))


# --- edit-derived tier: path shapes ---------------------------------------


def test_empty_edit_path_does_not_resolve_run(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="")
    assert resolve("run", ctx) is None
    assert resolve_tier("run", ctx) == "none"


def test_non_python_edit_path_does_not_resolve_run(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="notes.txt")
    assert resolve("run", ctx) is None
    assert resolve_tier("run", ctx) == "none"


def test_edit_path_with_spaces_is_quoted_as_one_argument(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="my file.py")
    got = resolve("run", ctx)
    assert got == {"cmd": "python 'my file.py'"}
    assert shlex.split(got["cmd"]) == ["python", "my file.py"]
    requested = 'python "my file.py"'
    assert canonical_args("run", requested, ctx) == {"cmd": requested}
    assert canonical_args("run", requested, ctx) != got


def test_directory_ending_in_py_resolves_as_if_a_script(tmp_path):
    """last_edit_path pointing at a directory named '*.py' still resolves 'run' as a python invocation."""
    pydir = tmp_path / "pkg.py"
    pydir.mkdir()
    ctx = Ctx(repo=tmp_path, last_edit_path=str(pydir))
    got = resolve("run", ctx)
    assert got == {"cmd": f"python {pydir}"}
    assert resolve_tier("run", ctx) == "edit"


# --- precedence: config vs edit vs session --------------------------------


def test_config_run_wins_over_edit_derived(tmp_path):
    _pkg(tmp_path, {"run": "node server.js"})
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py")
    assert resolve("run", ctx) == {"cmd": "node server.js"}
    assert resolve_tier("run", ctx) == "config"


def test_edit_derived_wins_over_session_for_run(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py", session={"run": "python old.py"})
    assert resolve("run", ctx) == {"cmd": "python repro.py"}
    assert resolve_tier("run", ctx) == "edit"


# --- resolve_tier label must match what resolve actually served -----------


def test_tier_label_agrees_with_resolve_across_tiers(tmp_path):
    """Every tier label resolve_tier reports must correspond to resolve returning a command."""
    _pkg(tmp_path, {"test": "jest"})
    cases = [
        Ctx(repo=tmp_path),  # config
        Ctx(repo=tmp_path, last_edit_path="repro.py"),  # edit (for run)
        Ctx(repo=tmp_path, session={"lint": "ruff ."}),  # session
    ]
    for ctx in cases:
        for kind in ("test", "run", "lint", "read"):
            tier = resolve_tier(kind, ctx)
            served = resolve(kind, ctx)
            if tier == "none":
                assert served is None
            else:
                assert served is not None


# --- canonical_args collision: LOSSLESSNESS / NO-CROSS-CONTAMINATION ------


def test_canonical_args_keeps_distinct_run_commands_separate(tmp_path):
    """Config predictions cannot replace the identity of a real requested program."""
    _pkg(tmp_path, {"run": "node server.js"})
    ctx = Ctx(repo=tmp_path)
    a = canonical_args("run", "python attack.py", ctx)
    b = canonical_args("run", "python victim.py", ctx)
    assert a != b, (
        f"distinct commands must not collapse to one cache key, both became {a}"
    )


def test_canonical_args_keeps_distinct_test_commands_separate(tmp_path):
    """Different test selections remain distinct even when config predicts one command."""
    _pkg(tmp_path, {"test": "jest"})
    ctx = Ctx(repo=tmp_path)
    a = canonical_args("test", "pytest tests/a.py", ctx)
    b = canonical_args("test", "pytest tests/b.py", ctx)
    assert a != b, (
        f"distinct test commands must not collapse to one cache key, both became {a}"
    )


def test_config_prediction_does_not_serve_output_for_another_actual_command(tmp_path):
    from sfx.cache import Cache
    from sfx.ledger import Ledger

    _pkg(tmp_path, {"test": "pytest tests/a.py"})
    ctx = Ctx(repo=tmp_path, session={"test": "pytest tests/old.py"})
    predicted = resolve("test", ctx)
    cache = Cache(lambda: 10, Ledger())
    cache.put("test", predicted, 0, result="tests/a.py passed")
    wrong_request = canonical_args("test", "pytest tests/b.py", ctx)
    assert cache.serve("test", wrong_request, 10) == ("miss", None, None)
    exact_request = canonical_args("test", "pytest tests/a.py", ctx)
    assert cache.serve("test", exact_request, 10)[2] == "tests/a.py passed"


def test_session_command_does_not_replace_real_request(tmp_path):
    ctx = Ctx(repo=tmp_path, session={"read": "cat old.txt"})
    assert canonical_args("read", "cat actual.txt", ctx) == {"cmd": "cat actual.txt"}


def test_canonical_args_edit_tier_ignored_diverges_from_resolve(tmp_path):
    """resolve() serves the edit-derived command but canonical_args ignores it, so serve key != put key."""
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py")
    put_side = resolve("run", ctx)
    serve_side = canonical_args("run", "python repro.py", ctx)
    assert put_side == serve_side


# --- run kind vs non-run kind ---------------------------------------------


def test_edit_tier_only_fires_for_run_kind(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py")
    for kind in ("read", "test", "build", "lint"):
        assert resolve(kind, ctx) is None
        assert resolve_tier(kind, ctx) == "none"


def test_canonical_command_has_no_edit_tier(tmp_path):
    """The legacy command selector has no edit tier; request keys no longer use it."""
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py")
    assert resolve("run", ctx) == {"cmd": "python repro.py"}
    assert canonical_command("run", ctx) is None
