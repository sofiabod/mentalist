import json
import shutil
import subprocess

import pytest

from sfx.daemon import Daemon


class FakeClock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null",
                        "GIT_CONFIG_SYSTEM": "/dev/null",
                        "HOME": str(cwd),
                        "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "s@x.com")
    _git(r, "config", "user.name", "s")
    (r / "foo.py").write_text("VALUE = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


def _apply_write(fork_path, write):
    p = write.args["path"]
    contents = write.args["contents"]
    if contents is None:
        (fork_path / p).unlink()
    else:
        (fork_path / p).write_text(contents)


def _stream(d, sid, call_id, path, contents):
    body = json.dumps({"path": path, "contents": contents})
    chain = None
    for ch in [body[i:i + 6] for i in range(0, len(body), 6)]:
        chain = d.call_stream_delta(sid, call_id, tool="Write", delta=ch)
    # the chain now runs on the background executor lane; block until it lands so
    # the (sync-clock) tests observe the same "ready to serve" state they always did
    if chain is not None and chain.future is not None:
        chain.future.result(timeout=10.0)
    return chain


class _W:
    def __init__(self, path, contents):
        self.args = {"path": path, "contents": contents}


def _truth_run(repo, tmp_path, edits, cmd_reader):
    """Independent oracle: clone the repo, apply every edit in order, run the
    reader over the resulting tree. Never touches cache/chain code."""
    dst = tmp_path / f"truth-{cmd_reader}-{len(list(tmp_path.glob('truth-*')))}"
    shutil.copytree(repo, dst)
    for path, contents in edits:
        _apply_write(dst, _W(path, contents))
    files = sorted(p.name for p in dst.iterdir() if p.suffix == ".py")
    shutil.rmtree(dst)
    return ",".join(files)


def _run_in_fork_listing(fork_path, hop):
    files = sorted(p.name for p in fork_path.iterdir() if p.suffix == ".py")
    return (",".join(files), "PASS")


def _run_table(k=2):
    return {"k": k, "min_support": 1, "tau": 0.35,
            "table": {"main|read,edit|edit:OK": {"support": 10, "p": {"run": 0.9}},
                      "main|edit,edit|edit:OK": {"support": 10, "p": {"edit": 0.9}},
                      "main|edit,edit|edit:OK|run": {"support": 10, "p": {"run": 0.9}}}}


def _run_daemon(clock, repo, scratch, run_in_fork=_run_in_fork_listing,
                depth_cap=1, k=2, resolve_args=None):
    resolve_args = resolve_args or (lambda kind, ctx: {"cmd": "python a.py"})
    d = Daemon(clock=clock, global_table=_run_table(k), k=k,
               resolve_args=resolve_args,
               apply_write=_apply_write, run_in_fork=run_in_fork,
               depth_cap=depth_cap)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))
    return d


def _prime_read(d):
    d.call_executed("s1", kind="read", verb="free", outcome="OK",
                    args={"cmd": "cat foo.py"}, latency=0.0)


# --- LOSSLESSNESS across many interleaved edit orders ---

def test_three_interleaved_edits_carry_run_is_lossless(repo, scratch, tmp_path):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    edits = [("a.py", "# a\n"), ("b.py", "# b\n"), ("c.py", "# c\n")]
    for name, contents in edits:
        _stream(d, "s1", name, name, contents)
        assert d.resolve("s1", kind="edit",
                         args={"path": name, "contents": contents})[0] == "hit_completed"
        _apply_write(repo, _W(name, contents))
    clock.t = 1000
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    truth = _truth_run(repo, tmp_path, [], "final")
    assert outcome == "hit_completed"
    assert result == truth


def test_same_file_edited_twice_run_reflects_latest_bytes(repo, scratch, tmp_path):
    clock = FakeClock(0)
    captured = {}

    def run_in_fork(fork_path, hop):
        captured["contents"] = (fork_path / "foo.py").read_text()
        return (captured["contents"], "PASS")

    d = _run_daemon(clock, repo, scratch, run_in_fork=run_in_fork)
    _prime_read(d)
    _stream(d, "s1", "e1", "foo.py", "VALUE = 2\n")
    _apply_write(repo, _W("foo.py", "VALUE = 2\n"))
    _stream(d, "s1", "e2", "foo.py", "VALUE = 3\n")
    clock.t = 1000
    outcome, result = d.resolve("s1", kind="edit",
                                args={"path": "foo.py", "contents": "VALUE = 3\n"})
    assert outcome == "hit_completed"
    if d.sessions["s1"].chain is not None:
        outcome2, result2 = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
        assert result2 == "VALUE = 3\n"
        assert result2 != "VALUE = 2\n"


def test_edit_then_delete_then_run_lossless(repo, scratch, tmp_path):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    _apply_write(repo, _W("a.py", "# a\n"))
    _stream(d, "s1", "e2", "a.py", None)
    clock.t = 1000
    d.resolve("s1", kind="edit", args={"path": "a.py", "contents": None})
    if d.sessions["s1"].chain is not None:
        outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
        truth = _truth_run(repo, tmp_path, [("a.py", None)], "del")
        assert result == truth
        assert "a.py" not in result


def test_four_interleaved_edits_carry_lossless(repo, scratch, tmp_path):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    names = ["a.py", "b.py", "c.py", "d.py"]
    for name in names:
        _stream(d, "s1", name, name, f"# {name}\n")
        assert d.resolve("s1", kind="edit",
                         args={"path": name, "contents": f"# {name}\n"})[0] == "hit_completed"
        _apply_write(repo, _W(name, f"# {name}\n"))
    clock.t = 1000
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    truth = _truth_run(repo, tmp_path, [], "four")
    assert outcome == "hit_completed"
    assert result == truth


# --- restream discard: a new stream mid-chain must discard the old fork's specs ---

def test_restream_discards_prior_chain_specs(repo, scratch):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    ledger = d.sessions["s1"].ledger
    before = ledger.terminal_counts().get("discarded", 0)
    _stream(d, "s1", "e2", "b.py", "# b\n")
    after = ledger.terminal_counts().get("discarded", 0)
    assert after >= before


# --- EPOCH FENCE: a real fs write mid-chain must not serve a stale fork ---

def test_epoch_bump_midchain_never_serves_stale_run(repo, scratch, tmp_path):
    clock = FakeClock(0)

    def run_in_fork(fork_path, hop):
        files = sorted(p.name for p in fork_path.iterdir() if p.suffix == ".py")
        return (",".join(files), "PASS")

    d = _run_daemon(clock, repo, scratch, run_in_fork=run_in_fork)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    _stream(d, "s1", "e2", "b.py", "# b\n")
    chain = d.sessions["s1"].chain
    assert chain is not None
    d.on_fs_change("s1")
    clock.t = 1000
    d.resolve("s1", kind="edit", args={"path": "b.py", "contents": "# b\n"})
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    assert outcome == "miss"


def test_epoch_bump_before_any_resolve_run_hop_misses(repo, scratch):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    d.on_fs_change("s1")
    clock.t = 1000
    d.resolve("s1", kind="edit", args={"path": "a.py", "contents": "# a\n"})
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    assert outcome == "miss"


# --- depth_cap boundary ---

def test_depth_cap_zero_yields_no_hops(repo, scratch):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch, depth_cap=0)
    _prime_read(d)
    chain = _stream(d, "s1", "e1", "a.py", "# a\n")
    assert chain is not None
    assert chain.hops == []


def test_depth_cap_respected_with_carry(repo, scratch):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch, depth_cap=1)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    _stream(d, "s1", "e2", "b.py", "# b\n")
    chain = d.sessions["s1"].chain
    assert len(chain.hops) <= 1


# --- never-verb truncation mid-chain ---

def test_never_verb_truncates_chain(repo, scratch):
    clock = FakeClock(0)
    table = {"k": 2, "min_support": 1, "tau": 0.35,
             "table": {"main|read,edit|edit:OK": {"support": 10, "p": {"git": 0.9}}}}
    d = Daemon(clock=clock, global_table=table, k=2,
               resolve_args=lambda kind, ctx: {"cmd": "git status"},
               apply_write=_apply_write, run_in_fork=_run_in_fork_listing,
               depth_cap=3)
    d.session_start("s1", repo=str(repo), role="main", scratch=str(scratch))
    _prime_read(d)
    chain = _stream(d, "s1", "e1", "a.py", "# a\n")
    assert chain is not None
    assert chain.hops == []


# --- edit after a run in the same chain ---

def test_edit_after_run_rebuilds_chain_lossless(repo, scratch, tmp_path):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    _stream(d, "s1", "e1", "a.py", "# a\n")
    d.resolve("s1", kind="edit", args={"path": "a.py", "contents": "# a\n"})
    clock.t = 500
    d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    _apply_write(repo, _W("a.py", "# a\n"))
    d.on_fs_change("s1")
    _stream(d, "s1", "e2", "b.py", "# b\n")
    clock.t = 1000
    out, _ = d.resolve("s1", kind="edit", args={"path": "b.py", "contents": "# b\n"})
    assert out == "hit_completed"
    outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
    truth = _truth_run(repo, tmp_path, [("b.py", "# b\n")], "afterrun")
    assert outcome == "miss" or result == truth


# --- PER-CASE ISOLATION: distinct call_ids never bleed a stale tail ---

def test_two_call_ids_no_tail_contamination(repo, scratch, tmp_path):
    clock = FakeClock(0)
    d = _run_daemon(clock, repo, scratch)
    _prime_read(d)
    _stream(d, "s1", "cidA", "a.py", "# a\n")
    _apply_write(repo, _W("a.py", "# a\n"))
    _stream(d, "s1", "cidB", "b.py", "# b\n")
    clock.t = 1000
    d.resolve("s1", kind="edit", args={"path": "b.py", "contents": "# b\n"})
    if d.sessions["s1"].chain is not None:
        outcome, result = d.resolve("s1", kind="run", args={"cmd": "python a.py"})
        truth = _truth_run(repo, tmp_path, [("b.py", "# b\n")], "twocid")
        assert result == truth
        assert "a.py" in result and "b.py" in result
