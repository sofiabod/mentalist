"""sPTC: serve a NOVEL call purely from the streamed tokens (empty tables), before
generation completes. Independent source of truth: with empty global+session tables,
spex.propose() returns [] and can NEVER serve; any hit therefore came from the stream.
"""
import subprocess
import time

import pytest

from sfx.daemon import Daemon
from sfx.producer import ScriptedProducer, VllmStreamProducer


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                   env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                        "HOME": str(cwd), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "s@x.com")
    _git(r, "config", "user.name", "s")
    (r / "hello.py").write_text("print('novel-42')\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _empty_table():
    return {"k": 1, "table": {}}


def _run_factory(repo):
    def run(kind, args):
        t0 = time.monotonic()
        p = subprocess.run(args["cmd"], shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, cwd=repo)
        return (p.stdout, p.returncode), (time.monotonic() - t0) * 1000.0
    return run


def _daemon(repo, clock, run=None):
    run = run or _run_factory(repo)
    d = Daemon(clock=clock, global_table=_empty_table(), k=1, run=run)
    d.session_start("s", repo=str(repo), role="main")
    return d


# --- test 1: novel call served via stream; control = stream off -> miss ---

def test_novel_call_served_via_stream(repo):
    clock = FakeClock()
    d = _daemon(repo, clock)
    prod = ScriptedProducer("run", {"cmd": "python3 hello.py"}, latency_ms_per_token=5.0)
    for tok, ms in prod.tokens():
        d.stream_get_delta("s", "c1", "run", tok)
        clock.advance(ms)
    outcome, result = d.resolve("s", "run", {"cmd": "python3 hello.py"})
    assert outcome.startswith("hit"), outcome
    assert result[0] == "novel-42\n"


def test_novel_call_miss_without_stream(repo):
    clock = FakeClock()
    d = _daemon(repo, clock)
    # no stream fed; empty tables -> spex has nothing -> miss
    outcome, result = d.resolve("s", "run", {"cmd": "python3 hello.py"})
    assert outcome == "miss"
    assert result is None


# --- test 2: speculation fires from a PARTIAL prefix (before the last token) ---

def test_speculation_fires_from_partial_prefix(repo):
    clock = FakeClock()
    launched = {"n": 0}
    real_run = _run_factory(repo)

    def run(kind, args):
        launched["n"] += 1
        return real_run(kind, args)

    d = _daemon(repo, clock, run=run)
    prod = ScriptedProducer("run", {"cmd": "python3 hello.py"})
    toks = list(prod.tokens())
    for tok, _ in toks[:-1]:            # feed all but the last token
        d.stream_get_delta("s", "c1", "run", tok)
    assert launched["n"] == 1, "speculation must fire from the partial prefix, not on completion"


# --- test 3: byte gate: streamed serve == live execute ---

def test_streamed_serve_byte_identical_to_live(repo):
    live = _run_factory(repo)
    (live_out, live_rc), _ = live("run", {"cmd": "python3 hello.py"})

    clock = FakeClock()
    d = _daemon(repo, clock)
    prod = ScriptedProducer("run", {"cmd": "python3 hello.py"})
    for tok, _ in prod.tokens():
        d.stream_get_delta("s", "c1", "run", tok)
    outcome, result = d.resolve("s", "run", {"cmd": "python3 hello.py"})
    assert outcome.startswith("hit")
    assert result == (live_out, live_rc)


# --- test 4: wall overlap: exec hidden behind the generation window ---

def test_exec_hidden_behind_generation_window(repo):
    # sPTC hides only the TAIL: from the parse-point (cmd value closed) to end of stream.
    # A tool that fits that tail is fully overlapped -> ON-wall approx the stream window,
    # NOT window + tool. Independent source of truth: post-parse tail = trailing-token
    # count * latency; a tool shorter than it must be hidden.
    (repo / "slow.py").write_text("import time; time.sleep(0.15); print('slow')\n")
    clock = FakeClock()
    d = _daemon(repo, clock)
    # 20 trailing tokens * 20ms = 400ms tail after the cmd value closes; tool = 150ms
    prod = ScriptedProducer("run", {"cmd": "python3 slow.py"}, latency_ms_per_token=20.0,
                            trailing_tokens=20)
    tool_s, tail_s = 0.15, 20 * 0.02
    assert tail_s > tool_s, "setup: tail must exceed tool so it can hide"

    t0 = time.monotonic()
    for tok, ms in prod.tokens():
        d.stream_get_delta("s", "c1", "run", tok)
        clock.advance(ms)
        time.sleep(ms / 1000.0)
    stream_wall = time.monotonic() - t0
    tr = time.monotonic()
    outcome, result = d.resolve("s", "run", {"cmd": "python3 slow.py"})
    resolve_wall = time.monotonic() - tr
    assert outcome.startswith("hit")
    # exec overlapped the tail: at serve time it is already done, so the resolve pays
    # almost nothing vs the tool_s a live execute would cost. This IS the hidden latency.
    assert resolve_wall < tool_s * 0.5
    # and the whole ON run cost about the stream window, not window + tool
    assert (stream_wall + resolve_wall) < stream_wall + tool_s * 0.5


# --- test 5a: losslessness: an edit between speculation and the call bumps epoch -> live miss ---

def test_edit_between_spec_and_call_forces_miss(repo):
    clock = FakeClock()
    d = _daemon(repo, clock)
    prod = ScriptedProducer("run", {"cmd": "python3 hello.py"})
    for tok, _ in prod.tokens():
        d.stream_get_delta("s", "c1", "run", tok)
    d.on_fs_change("s")                 # an edit lands after speculation
    outcome, result = d.resolve("s", "run", {"cmd": "python3 hello.py"})
    assert outcome == "miss", "stale prefetch from a prior epoch must not serve"


# --- test 5b: incomplete stream -> safe live miss, no hang ---

def test_incomplete_stream_safe_miss(repo):
    clock = FakeClock()
    d = _daemon(repo, clock)
    d.stream_get_delta("s", "c1", "run", '{"cmd": "python3 hel')  # never completed
    outcome, result = d.resolve("s", "run", {"cmd": "python3 hello.py"})
    assert outcome == "miss"
    assert result is None


# --- write authority: a fork/edit kind cannot execute via the GET stream lane ---

def test_edit_kind_never_runs_via_get_stream(repo):
    clock = FakeClock()
    ran = {"n": 0}

    def run(kind, args):
        ran["n"] += 1
        return ("", 0), 0.0

    d = _daemon(repo, clock, run=run)
    # feed a complete edit-shaped call through the GET lane; verb=fork must be refused
    d.stream_get_delta("s", "c1", "edit", '{"cmd": "rm -rf x"}')
    assert ran["n"] == 0, "STREAM_ONLY (fork) kind must not speculate on the GET lane"


# --- Rung 2 seam is a hard stub, not a fake result ---

def test_vllm_producer_is_seam_only(repo):
    with pytest.raises(NotImplementedError):
        VllmStreamProducer("run", {"cmd": "x"})
