from sfx.resolver import resolve, resolve_tier, Ctx


TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "foo/bar.py", line 12, in <module>\n'
    "    main()\n"
    "ZeroDivisionError: division by zero\n"
)

PYTEST_FAIL = (
    "=========================== short test summary info ===========================\n"
    "FAILED tests/test_x.py::test_y - AssertionError: 1 != 2\n"
    "1 failed in 0.03s\n"
)

LS_OUTPUT = "setup.py\nfoo/bar.py\nREADME.md\n"


def test_read_derives_cat_of_file_named_in_traceback(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=TRACEBACK)

    assert resolve("read", ctx) == {"cmd": "cat foo/bar.py"}
    assert resolve_tier("read", ctx) == "observation"


def test_test_derives_failing_node_id_from_pytest_summary(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=PYTEST_FAIL)

    assert resolve("test", ctx) == {"cmd": "pytest tests/test_x.py::test_y"}
    assert resolve_tier("test", ctx) == "observation"


def test_read_derives_first_path_from_ls_listing(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=LS_OUTPUT)

    assert resolve("read", ctx) == {"cmd": "cat setup.py"}


def test_observation_ignored_when_no_path_or_node(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation="all good, nothing to see\n")

    assert resolve("read", ctx) is None
    assert resolve("test", ctx) is None


def test_edit_run_still_wins_over_observation(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py", last_observation=TRACEBACK)

    assert resolve("run", ctx) == {"cmd": "python repro.py"}
    assert resolve_tier("run", ctx) == "edit"


def test_config_still_wins_over_observation(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    ctx = Ctx(repo=tmp_path, last_observation=PYTEST_FAIL)

    assert resolve("test", ctx) == {"cmd": "pytest"}
    assert resolve_tier("test", ctx) == "config"


def test_no_observation_leaves_read_unresolved(tmp_path):
    ctx = Ctx(repo=tmp_path)

    assert resolve("read", ctx) is None
