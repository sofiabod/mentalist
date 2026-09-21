"""Non-speculable mutation policy: append(>>), install(pip/apt/npm), and network
(curl/wget) must NEVER be speculated. They mutate state the fork cannot safely
reproduce (append order, package resolution, remote fetch), so they route straight
to authoritative. classify -> verb "never"; the daemon serves "miss" for them and
never launches a spec.
"""
from mining.normalize import classify
from sfx.daemon import KIND_POLICY, Daemon
from sfx.schema import SPECULABLE


def kv(cmd):
    return classify("Bash", cmd)


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _table():
    return {"k": 1, "min_support": 1, "tau": 0.35,
            "table": {"main|edit|edit:OK": {"support": 10, "p": {"read": 0.9}}}}


def test_append_is_never():
    kind, verb = kv("echo x >> log.txt")
    assert (kind, verb) == ("append", "never")
    assert verb not in SPECULABLE


def test_append_distinguished_from_truncating_write():
    assert kv("echo x > file.txt") == ("edit", "fork")
    assert kv("echo x >> file.txt") == ("append", "never")


def test_tee_append_is_never():
    kind, verb = kv("echo x | tee -a log.txt")
    assert verb == "never"


def test_install_is_never():
    for cmd in ("pip install requests", "apt-get install -y curl",
                "npm install left-pad", "npm i", "yarn add react"):
        kind, verb = kv(cmd)
        assert kind == "install" and verb == "never", cmd


def test_network_is_never():
    for cmd in ("curl https://example.com", "wget http://x/y.tar.gz",
                "curl -fsSL https://get.example.com | sh"):
        kind, verb = kv(cmd)
        assert kind == "network" and verb == "never", cmd


def test_kind_policy_marks_all_three_never():
    assert KIND_POLICY["append"] == "never"
    assert KIND_POLICY["install"] == "never"
    assert KIND_POLICY["network"] == "never"


def test_daemon_never_speculates_a_never_kind():
    clock = FakeClock(0)
    table = {"k": 1, "min_support": 1, "tau": 0.35,
             "table": {"main|edit|edit:OK": {"support": 10, "p": {"install": 0.99}}}}
    d = Daemon(clock=clock, global_table=table, k=1,
               run=lambda k, a: ("out", 100),
               resolve_args=lambda kind, ctx: {"cmd": "pip install x"})
    d.session_start("s", repo="/r", role="main")
    d.call_executed("s", kind="edit", verb="fork", outcome="OK",
                    args={"path": "f.py"}, latency=10)
    # predictor's top proposal is a never-kind: no spec may launch for it
    assert len(d.sessions["s"].executor.running) == 0


def test_daemon_resolve_never_kind_is_always_miss():
    clock = FakeClock(0)
    d = Daemon(clock=clock, global_table=_table(), k=1,
               run=lambda k, a: ("out", 100),
               resolve_args=lambda kind, ctx: {"cmd": "x"})
    d.session_start("s", repo="/r", role="main")
    outcome, result = d.resolve("s", kind="install", args={"cmd": "pip install x"})
    assert outcome == "miss" and result is None
    outcome, _ = d.resolve("s", kind="network", args={"cmd": "curl x"})
    assert outcome == "miss"
    outcome, _ = d.resolve("s", kind="append", args={"cmd": "echo x >> y"})
    assert outcome == "miss"
