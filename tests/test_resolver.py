import json

from sfx.resolver import resolve, resolve_tier, Ctx


def _write(d, name, content):
    p = d / name
    p.write_text(content)
    return p


def test_config_detection_resolves_test_command_from_package_json(tmp_path):
    _write(tmp_path, "package.json", json.dumps({"scripts": {"test": "jest --ci"}}))
    ctx = Ctx(repo=tmp_path)

    assert resolve("test", ctx) == {"cmd": "jest --ci"}


def test_config_detection_resolves_cargo_test(tmp_path):
    _write(tmp_path, "Cargo.toml", "[package]\nname='x'\n")
    ctx = Ctx(repo=tmp_path)

    assert resolve("test", ctx) == {"cmd": "cargo test"}


def test_config_detection_resolves_pyproject_pytest(tmp_path):
    _write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    ctx = Ctx(repo=tmp_path)

    assert resolve("test", ctx) == {"cmd": "pytest"}


def test_session_command_used_when_config_absent(tmp_path):
    ctx = Ctx(repo=tmp_path, session={"lint": "ruff check ."})

    assert resolve("lint", ctx) == {"cmd": "ruff check ."}


def test_config_wins_over_session_for_same_kind(tmp_path):
    _write(tmp_path, "package.json", json.dumps({"scripts": {"lint": "eslint ."}}))
    ctx = Ctx(repo=tmp_path, session={"lint": "ruff check ."})

    assert resolve("lint", ctx) == {"cmd": "eslint ."}


def test_no_resolution_returns_none(tmp_path):
    ctx = Ctx(repo=tmp_path)

    assert resolve("read", ctx) is None
    assert resolve("test", ctx) is None


def test_run_resolves_from_the_edit_just_streamed(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="reproduce_error.py")

    assert resolve("run", ctx) == {"cmd": "python reproduce_error.py"}
    assert resolve_tier("run", ctx) == "edit"


def test_edit_tier_only_fires_for_run_kind(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="reproduce_error.py")

    assert resolve("read", ctx) is None
    assert resolve("test", ctx) is None


def test_edit_tier_ignores_non_python_edits(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="README.md")

    assert resolve("run", ctx) is None


def test_edit_tier_wins_over_session_for_run(tmp_path):
    ctx = Ctx(repo=tmp_path, last_edit_path="repro.py", session={"run": "python old.py"})

    assert resolve("run", ctx) == {"cmd": "python repro.py"}
    assert resolve_tier("run", ctx) == "edit"
