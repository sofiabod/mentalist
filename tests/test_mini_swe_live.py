import hashlib
import json
import tempfile
import threading
import uuid
from pathlib import Path

from adapters import mini_swe
from adapters.protocol import serve
from eval.sfx_daemon_run import _file_log
from sfx import resolver
from sfx.daemon import Daemon


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 100.0
        return self.t


def _global_table():
    return {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {"main|edit|edit:OK": {"support": 10, "p": {"test": 0.99}}},
    }


def _sock_dir():
    d = Path(tempfile.gettempdir()) / uuid.uuid4().hex[:8]
    d.mkdir()
    return d


def test_live_seam_produces_serve_event_over_socket(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts=''\n")
    trace = tmp_path / "trace.jsonl"
    d = Daemon(clock=FakeClock(), global_table=_global_table(), k=1,
               run=lambda kind, args: (("out", 0), 3000),
               resolve_args=lambda kind, ctx: resolver.resolve(
                   kind, resolver.Ctx(repo=repo)),
               log=_file_log(str(trace)))
    sp = str(_sock_dir() / "s.sock")
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setenv(mini_swe.SOCKET_ENV, sp)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    client = mini_swe.connect()

    served_any = False
    for _ in range(3):
        if mini_swe.claim_or_none(client, "edit", "sed -i s/a/b/ foo.py") is None:
            mini_swe.report_executed(client, "edit", "fork", "sed -i s/a/b/ foo.py",
                                     returncode=0, latency=3000)
        assert mini_swe.claim_or_none(client, "test", "pytest -q") is None
        if mini_swe.claim_or_none(client, "test", "pytest") is not None:
            served_any = True
        else:
            mini_swe.report_executed(client, "test", "free", "pytest",
                                     returncode=0, latency=3000)
    client.turn_end(mini_swe._session())
    srv.shutdown()

    lines = [json.loads(l) for l in trace.read_text().splitlines()]
    assert served_any
    assert any(l["ev"] == "resolve" and l["outcome"] != "miss" for l in lines)
    assert any(l["ev"] == "step" for l in lines)


def _run_loop_table():
    return {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {"main|run|run:OK": {"support": 10, "p": {"edit": 0.99}},
                  "main|edit|edit:OK": {"support": 10, "p": {"run": 0.99}}},
    }


def _review_read_only_repro_fixture(monkeypatch, source, scratch):
    # The two callers below contain fixed, reviewed programs that only emit
    # output and exit. This is not blanket authorization for arbitrary scripts.
    monkeypatch.setenv("SFX_SCRIPT_CONTRACTS", json.dumps([{
        "script": "repro.py", "positionals": 0, "path_options": [],
        "value_options": [], "flags": [], "required": [],
        "source_sha256": {"repro.py": hashlib.sha256(source.encode()).hexdigest()},
    }]))
    scratch.mkdir()
    monkeypatch.setenv("SFX_SCRATCH", str(scratch))
    monkeypatch.setenv("SFX_FORK_PATH_VIEW", "cwd")


def test_failed_run_serves_with_real_nonzero_returncode(monkeypatch, tmp_path):
    from eval.sfx_daemon_run import _run
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "print('boom', end='')\nraise SystemExit(3)\n"
    (repo / "repro.py").write_text(source)
    monkeypatch.setenv("SFX_REPO", str(repo))
    _review_read_only_repro_fixture(monkeypatch, source, tmp_path / "scratch")
    trace = tmp_path / "trace.jsonl"
    d = Daemon(clock=FakeClock(), global_table=_run_loop_table(), k=1,
               run=_run,
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               log=_file_log(str(trace)))
    sp = str(_sock_dir() / "s.sock")
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setenv(mini_swe.SOCKET_ENV, sp)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    client = mini_swe.connect()

    cmd = "python3 repro.py"
    mini_swe.report_executed(client, "run", "free", cmd, returncode=3, latency=3000)
    mini_swe.report_executed(client, "edit", "fork", "sed -i s/1/2/ repro.py",
                             returncode=0, latency=3000)
    for _ in range(20):
        served = mini_swe.claim_or_none(client, "run", cmd)
        if served is not None:
            break
        mini_swe.report_executed(client, "run", "free", cmd, returncode=3, latency=3000)
    client.turn_end(mini_swe._session())
    srv.shutdown()

    assert served is not None
    assert served["returncode"] == 3
    assert served["output"] == "boom"


def test_served_run_preserves_stderr_over_socket(monkeypatch, tmp_path):
    from eval.sfx_daemon_run import _run
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "import sys\nsys.stderr.write('err-only\\n')\nsys.exit(3)\n"
    (repo / "repro.py").write_text(source)
    monkeypatch.setenv("SFX_REPO", str(repo))
    _review_read_only_repro_fixture(monkeypatch, source, tmp_path / "scratch")
    trace = tmp_path / "trace.jsonl"
    d = Daemon(clock=FakeClock(), global_table=_run_loop_table(), k=1,
               run=_run,
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               log=_file_log(str(trace)))
    sp = str(_sock_dir() / "s.sock")
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setenv(mini_swe.SOCKET_ENV, sp)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    client = mini_swe.connect()

    cmd = "python3 repro.py"
    mini_swe.report_executed(client, "run", "free", cmd, returncode=3, latency=3000)
    mini_swe.report_executed(client, "edit", "fork", "sed -i s/1/2/ repro.py",
                             returncode=0, latency=3000)
    for _ in range(20):
        served = mini_swe.claim_or_none(client, "run", cmd)
        if served is not None:
            break
        mini_swe.report_executed(client, "run", "free", cmd, returncode=3, latency=3000)
    client.turn_end(mini_swe._session())
    srv.shutdown()

    assert served is not None
    assert served["output"] == "err-only\n"
    assert served["returncode"] == 3


def _read_loop_table():
    return {
        "k": 1, "min_support": 1, "tau": 0.35,
        "table": {"main|read|read:OK": {"support": 10, "p": {"read": 0.99}}},
    }


def test_edit_fences_stale_cached_read_over_socket(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "foo.py").write_text("V=1\n")
    trace = tmp_path / "trace.jsonl"
    d = Daemon(clock=FakeClock(), global_table=_read_loop_table(), k=1,
               run=lambda kind, args: (("OLD", 0), 3000),
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               log=_file_log(str(trace)))
    sp = str(_sock_dir() / "s.sock")
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setenv(mini_swe.SOCKET_ENV, sp)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    client = mini_swe.connect()

    cmd = "cat foo.py"
    mini_swe.report_executed(client, "read", "free", cmd, returncode=0, latency=3000)
    served = mini_swe.claim_or_none(client, "read", cmd)
    assert served is not None and served["output"] == "OLD"

    mini_swe.report_executed(client, "read", "free", cmd, returncode=0, latency=3000)
    mini_swe.report_executed(client, "edit", "fork", "sed -i s/1/2/ foo.py",
                             returncode=0, latency=3000)
    stale = mini_swe.claim_or_none(client, "read", cmd)
    client.turn_end(mini_swe._session())
    srv.shutdown()

    assert stale is None


def test_live_seam_session_tier_rerun_serves_over_socket(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "repro.py").write_text("print(1)\n")
    trace = tmp_path / "trace.jsonl"
    d = Daemon(clock=FakeClock(), global_table=_run_loop_table(), k=1,
               run=lambda kind, args: ((f"{args['cmd']}-out", 0), 3000),
               resolve_args=lambda kind, ctx: resolver.resolve(kind, ctx),
               log=_file_log(str(trace)))
    sp = str(_sock_dir() / "s.sock")
    srv = serve(d, sp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setenv(mini_swe.SOCKET_ENV, sp)
    monkeypatch.setenv(mini_swe.REPO_ENV, str(repo))
    client = mini_swe.connect()

    cmd = "python repro.py"
    mini_swe.report_executed(client, "run", "free", cmd, returncode=0, latency=3000)
    mini_swe.report_executed(client, "edit", "fork", "sed -i s/1/2/ repro.py",
                             returncode=0, latency=3000)
    served = mini_swe.claim_or_none(client, "run", cmd)
    client.turn_end(mini_swe._session())
    srv.shutdown()

    lines = [json.loads(l) for l in trace.read_text().splitlines()]
    assert served is not None
    assert any(l["ev"] == "step" and l["kind"] == "run"
               and l["args_tier"] == "session" for l in lines)
    assert any(l["ev"] == "resolve" and l["kind"] == "run"
               and l["outcome"] != "miss" for l in lines)
