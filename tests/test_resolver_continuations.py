import os

import pytest

from sfx.resolver import Ctx, canonical_args, resolve, resolve_details, resolve_tier


REQUIRED = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--rules', required=True)
parser.parse_args()
"""


def test_observed_argv_wins_for_same_edited_script_without_rewriting(tmp_path):
    command = "python3 -u ./code_search.py 'sample root' --rules=rules.json"
    context = Ctx(repo=tmp_path, last_edit_path="code_search.py",
                  last_edit_contents=REQUIRED, session={"run": command})

    assert resolve("run", context) == {"cmd": command}
    assert resolve_tier("run", context) == "session"
    assert canonical_args("run", "python3 code_search.py", context) == {
        "cmd": "python3 code_search.py"}


def test_observed_argv_reused_after_python_file_inside_scan_input_changes(tmp_path):
    (tmp_path / "repo").mkdir()
    command = "python code_search.py repo --rules rules.json"
    context = Ctx(repo=tmp_path, last_edit_path="repo/A.py",
                  last_edit_contents="print('new fixture')\n", session={"run": command})

    assert resolve("run", context) == {"cmd": command}
    assert resolve_tier("run", context) == "session"


@pytest.mark.parametrize("argument", ["--input=fixture.py", "fixture.py"])
def test_explicit_python_input_file_grounds_prior_run(tmp_path, argument):
    command = f"python analyze.py {argument}"
    context = Ctx(repo=tmp_path, last_edit_path="fixture.py", session={"run": command})

    assert resolve("run", context) == {"cmd": command}


def test_absolute_observed_script_under_repo_preserves_exact_command(tmp_path):
    command = f"python {tmp_path}/code_search.py repo --rules rules.json"
    context = Ctx(repo=tmp_path, last_edit_path="code_search.py", session={"run": command})

    assert resolve("run", context) == {"cmd": command}


def test_unrelated_script_does_not_override_new_script(tmp_path):
    (tmp_path / "repo").mkdir()
    context = Ctx(repo=tmp_path, last_edit_path="new_script.py",
                  session={"run": "python code_search.py repo --rules rules.json"})

    assert resolve("run", context) == {"cmd": "python new_script.py"}
    assert resolve_tier("run", context) == "edit"


def test_unrelated_script_not_used_when_new_script_requires_args(tmp_path):
    context = Ctx(repo=tmp_path, last_edit_path="new_script.py",
                  last_edit_contents=REQUIRED, session={"run": "python old.py --flag"})

    assert resolve_details("run", context) == (None, "none", "required_python_arguments")


def test_data_edit_retains_prior_full_command(tmp_path):
    command = "python code_search.py repo --rules rules.json"
    context = Ctx(repo=tmp_path, last_edit_path="rules.json", session={"run": command})

    assert resolve("run", context) == {"cmd": command}


@pytest.mark.parametrize("declaration", [
    "parser.add_argument('root_dir')",
    "parser.add_argument('files', nargs='+')",
    "parser.add_argument('--rules', required=True)",
    "parser.add_subparsers(required=True)",
])
def test_cold_argless_guess_suppressed_for_required_cli_input(tmp_path, declaration):
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", last_edit_contents=declaration)

    assert resolve_details("run", context) == (None, "none", "required_python_arguments")


@pytest.mark.parametrize("declaration", [
    "print('no arguments')",
    "parser.add_argument('--verbose', action='store_true')",
    "parser.add_argument('files', nargs='*')",
    "parser.add_argument('file', nargs='?')",
])
def test_optional_inputs_do_not_disable_cold_guess(tmp_path, declaration):
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", last_edit_contents=declaration)

    assert resolve("run", context) == {"cmd": "python cli.py"}


def test_pending_source_overrides_old_authoritative_source(tmp_path):
    (tmp_path / "cli.py").write_text("print('OLD')\n")
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", last_edit_contents=REQUIRED)

    assert resolve("run", context) is None
    context.last_edit_contents = None
    assert resolve("run", context) == {"cmd": "python cli.py"}


def test_current_source_suppresses_required_arg_guess_after_commit(tmp_path):
    (tmp_path / "cli.py").write_text(REQUIRED)

    assert resolve("run", Ctx(repo=tmp_path, last_edit_path="cli.py")) is None


def test_previously_argless_command_is_not_reused_when_edit_adds_required_args(tmp_path):
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", last_edit_contents=REQUIRED,
                  session={"run": "python cli.py"})

    assert resolve_details("run", context) == (None, "none", "required_python_arguments")


def test_source_inspection_does_not_read_symlink_target(tmp_path):
    (tmp_path / "other.py").write_text(REQUIRED)
    (tmp_path / "linked.py").symlink_to("other.py")

    assert resolve("run", Ctx(repo=tmp_path, last_edit_path="linked.py")) == {
        "cmd": "python linked.py"}


def test_source_inspection_does_not_block_on_fifo(tmp_path):
    os.mkfifo(tmp_path / "pipe.py")

    assert resolve("run", Ctx(repo=tmp_path, last_edit_path="pipe.py")) == {
        "cmd": "python pipe.py"}


@pytest.mark.parametrize("condition", [
    "len(sys.argv) != 3", "len(sys.argv) < 3", "len(sys.argv) <= 1",
    "len(sys.argv) == 1",
])
def test_cold_argless_guess_rejected_by_required_sys_argv_guard(tmp_path, condition):
    source = f"import sys\nif {condition}:\n    sys.exit(1)\n"
    context = Ctx(repo=tmp_path, last_edit_path="code_search.py", last_edit_contents=source)

    assert resolve_details("run", context) == (None, "none", "required_python_arguments")


def test_sys_argv_guard_preserves_observed_exact_arguments(tmp_path):
    source = "import sys\nif len(sys.argv) < 3:\n    sys.exit(1)\n"
    command = "python3 ./code_search.py 'sample root' --rules=rules.json"
    context = Ctx(repo=tmp_path, last_edit_path="code_search.py", last_edit_contents=source,
                  session={"run": command})

    assert resolve_details("run", context) == ({"cmd": command}, "session", "resolved")
    assert canonical_args("run", "python3 code_search.py other --rules=rules.json", context) == {
        "cmd": "python3 code_search.py other --rules=rules.json"}


def test_sys_argv_guard_invalidates_previously_argless_guess(tmp_path):
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", session={"run": "python cli.py"},
                  last_edit_contents="import sys\nif len(sys.argv) < 2:\n    raise ValueError('input')\n")

    assert resolve_details("run", context) == (None, "none", "required_python_arguments")


@pytest.mark.parametrize("source", [
    "import sys\nif len(sys.argv) > 1:\n    sys.exit(1)\n",
    "import sys\nif len(sys.argv) < 1:\n    sys.exit(1)\n",
    "import sys\nif len(sys.argv) == 1:\n    print('default')\n",
    "import sys\nif len(sys.argv) == 1:\n    sys.exit(0)\n",
    "example = 'if len(sys.argv) < 3: sys.exit(1)'\n",
])
def test_optional_or_non_executed_sys_argv_does_not_suppress_guess(tmp_path, source):
    context = Ctx(repo=tmp_path, last_edit_path="cli.py", last_edit_contents=source)

    assert resolve("run", context) == {"cmd": "python cli.py"}
