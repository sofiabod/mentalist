from adapters import mini_swe


class FakeClient:
    def __init__(self, sock):
        self.sock = sock
        self.began = None
        self.resolves = []
        self.reply = {"result": "miss"}

    def turn_begin(self, session, repo, role, scratch=None, state_dir=None):
        self.began = (session, repo, role, scratch, state_dir)

    def resolve(self, session, tool, args):
        self.resolves.append((session, tool, args))
        return self.reply


def test_connect_returns_none_without_socket(monkeypatch):
    monkeypatch.delenv(mini_swe.SOCKET_ENV, raising=False)
    assert mini_swe.connect() is None


def test_connect_begins_turn_with_socket(monkeypatch):
    monkeypatch.setenv(mini_swe.SOCKET_ENV, "/tmp/x.sock")
    monkeypatch.setenv(mini_swe.REPO_ENV, "/workspace")
    monkeypatch.setattr(mini_swe, "Client", FakeClient)
    monkeypatch.setenv(mini_swe.STATE_DIR_ENV, "/shared")
    c = mini_swe.connect()
    assert c.began == ("mini", "/workspace", "main", None, "/shared")


def test_claim_returns_none_on_miss():
    c = FakeClient("s")
    c.reply = {"result": "miss"}
    assert mini_swe.claim_or_none(c, "test", "pytest") is None


def test_claim_returns_cached_dict_on_hit():
    c = FakeClient("s")
    c.reply = {"result": "hit_completed", "output": ["real grep bytes", 0]}
    out = mini_swe.claim_or_none(c, "read", "cat foo")
    assert out["returncode"] == 0
    assert out["output"] == "real grep bytes"
    assert out["exception_info"] == ""


def test_claim_none_client_is_noop():
    assert mini_swe.claim_or_none(None, "read", "x") is None


def test_config_yaml_selects_sfx_environment():
    y = mini_swe.mini_config_yaml()
    assert "environment_class: adapters.mini_swe.SfxEnvironment" in y
