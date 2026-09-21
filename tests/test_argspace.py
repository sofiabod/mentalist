from adapters import mini_swe
from sfx import resolver
from sfx.cache import Cache, _canon
from sfx.ledger import Ledger


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _pyrepo(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = ''\n")
    return tmp_path


def test_fact_kind_resolves_to_canonical_cmd(tmp_path):
    ctx = resolver.Ctx(repo=_pyrepo(tmp_path))
    assert resolver.resolve("test", ctx) == {"cmd": "pytest"}


def test_config_speculation_matches_only_the_exact_command(tmp_path, monkeypatch):
    repo = _pyrepo(tmp_path)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    ctx = resolver.Ctx(repo=repo)
    assert _canon(mini_swe._get_args("test", "pytest")) == \
        _canon(resolver.resolve("test", ctx))
    assert _canon(mini_swe._get_args("test", "pytest -q tests/x.py")) != \
        _canon(resolver.resolve("test", ctx))


def test_content_kind_abstains(tmp_path):
    ctx = resolver.Ctx(repo=_pyrepo(tmp_path))
    assert resolver.resolve("read", ctx) is None


def test_content_kind_get_args_falls_back_to_raw(tmp_path, monkeypatch):
    monkeypatch.setenv(mini_swe.REPO_ENV, str(_pyrepo(tmp_path)))
    assert mini_swe._get_args("read", "cat foo.py") == {"cmd": "cat foo.py"}


def test_served_bytes_are_the_speculations_bytes(tmp_path, monkeypatch):
    repo = _pyrepo(tmp_path)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    clock = FakeClock(0)
    cache = Cache(clock, Ledger())
    ctx = resolver.Ctx(repo=repo)
    cache.put("test", resolver.resolve("test", ctx), duration=100,
              result="SPECULATED-BYTES")
    clock.t = 500
    outcome, _, result = cache.serve(
        "test", mini_swe._get_args("test", "pytest"), ask_time=500)
    assert outcome == "hit_completed"
    assert result == "SPECULATED-BYTES"
    miss, _, _ = cache.serve(
        "test", mini_swe._get_args("test", "pytest -q tests/other.py"),
        ask_time=500)
    assert miss == "miss"


def test_resolve_tier(tmp_path):
    ctx = resolver.Ctx(repo=_pyrepo(tmp_path))
    assert resolver.resolve_tier("test", ctx) == "config"
    assert resolver.resolve_tier("read", ctx) == "none"
    ctx2 = resolver.Ctx(repo=tmp_path / "empty", session={"lint": "ruff"})
    assert resolver.resolve_tier("lint", ctx2) == "session"
