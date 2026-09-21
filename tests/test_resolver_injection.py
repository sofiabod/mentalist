from sfx.resolver import _from_observation, resolve, resolve_tier, Ctx


def _tb(path):
    return f'Traceback (most recent call last):\n  File "{path}", line 1, in <module>\n'


def _pathline(path):
    return f"{path}\n"


def _fail(node):
    return f"FAILED {node} - AssertionError\n"


def test_semicolon_in_traceback_path_is_safe_miss(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_tb("foo.py;touch_X.py"))
    assert _from_observation("read", ctx) is None
    assert resolve("read", ctx) is None
    assert resolve_tier("read", ctx) == "none"


def test_dollar_paren_in_path_line_is_safe_miss(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_pathline("foo$(touch X).py"))
    assert _from_observation("read", ctx) is None


def test_backtick_in_path_is_safe_miss(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_pathline("foo`id`.py"))
    assert _from_observation("read", ctx) is None


def test_redirect_in_path_is_safe_miss(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_pathline("foo>evil.py"))
    assert _from_observation("read", ctx) is None


def test_pipe_amp_glob_quote_space_flag_in_path_are_safe_miss(tmp_path):
    for bad in ["foo|bar.py", "foo&bar.py", "foo*.py", "foo'.py", 'foo".py']:
        ctx = Ctx(repo=tmp_path, last_observation=_pathline(bad))
        assert _from_observation("read", ctx) is None, bad


def test_injection_in_pytest_node_id_is_safe_miss(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_fail("a.py::b;touch_X"))
    assert _from_observation("test", ctx) is None
    assert resolve("test", ctx) is None


def test_normal_path_still_derives_cleanly(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_tb("src/foo/bar.py"))
    assert _from_observation("read", ctx) == "cat src/foo/bar.py"
    assert resolve("read", ctx) == {"cmd": "cat src/foo/bar.py"}


def test_normal_node_id_still_derives_cleanly(tmp_path):
    ctx = Ctx(repo=tmp_path, last_observation=_fail("tests/test_x.py::test_y"))
    assert _from_observation("test", ctx) == "pytest tests/test_x.py::test_y"
