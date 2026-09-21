import subprocess

import pytest

from adapters.stream import StreamedWrite
from sfx.chain import run_chain, longest_prefix


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
    (fork_path / write.args["path"]).write_text(write.args["contents"])


def _read_value(fork_path):
    ns = {}
    exec((fork_path / "foo.py").read_text(), ns)
    return ns["VALUE"]


def test_streamed_write_lands_in_fork_verbatim_repo_untouched(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})
    seen = {}

    def run_get(fork_path, hop):
        seen["fork_contents"] = (fork_path / "foo.py").read_text()
        return ("test-out", "PASS")

    run_chain(write, repo, scratch,
              apply_write=_apply_write,
              next_hop=lambda outs: ("test", "free", {"cmd": "pytest"}) if not outs else None,
              run_get=run_get, depth_cap=3, clock=lambda: 0.0)

    assert seen["fork_contents"] == "VALUE = 42\n"
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"


def test_predicted_test_runs_against_edited_fork_state(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})

    def run_get(fork_path, hop):
        return ("out", "PASS" if _read_value(fork_path) == 42 else "FAIL")

    chain = run_chain(write, repo, scratch,
                      apply_write=_apply_write,
                      next_hop=lambda outs: ("test", "free", {"cmd": "pytest"}) if not outs else None,
                      run_get=run_get, depth_cap=3, clock=lambda: 0.0)

    assert chain.hops[0].kind == "test"
    assert chain.hops[0].outcome == "PASS"


def test_observed_outcome_drives_next_hop(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})

    def next_hop(outs):
        if not outs:
            return ("test", "free", {"cmd": "pytest"})
        if len(outs) == 1 and outs[-1] == "PASS":
            return ("lint", "free", {"cmd": "ruff"})
        return None

    chain = run_chain(write, repo, scratch,
                      apply_write=_apply_write, next_hop=next_hop,
                      run_get=lambda fp, hop: ("out", "PASS"), depth_cap=3, clock=lambda: 0.0)

    assert [h.kind for h in chain.hops] == ["test", "lint"]


def test_chain_truncates_before_post(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})

    def next_hop(outs):
        if not outs:
            return ("test", "free", {"cmd": "pytest"})
        if len(outs) == 1:
            return ("push", "never", {"cmd": "git push"})
        return None

    chain = run_chain(write, repo, scratch,
                      apply_write=_apply_write, next_hop=next_hop,
                      run_get=lambda fp, hop: ("out", "PASS"), depth_cap=5, clock=lambda: 0.0)

    assert [h.verb for h in chain.hops] == ["free"]
    assert all(h.verb != "never" for h in chain.hops)


def test_depth_cap_bounds_chain(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})

    chain = run_chain(write, repo, scratch,
                      apply_write=_apply_write,
                      next_hop=lambda outs: ("test", "free", {"cmd": "pytest"}),
                      run_get=lambda fp, hop: ("out", "PASS"), depth_cap=2, clock=lambda: 0.0)

    assert len(chain.hops) == 2


@pytest.mark.parametrize("bad_path", ["/etc/passwd", "../escape.py", "a/../../x.py", "", "."])
def test_streamed_write_with_escaping_path_is_rejected_repo_untouched(repo, scratch, bad_path):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": bad_path, "contents": "PWNED\n"})
    touched = {"n": 0}

    def apply_write(fork_path, w):
        touched["n"] += 1
        (fork_path / w.args["path"]).write_text(w.args["contents"])

    with pytest.raises(Exception):
        run_chain(write, repo, scratch,
                  apply_write=apply_write,
                  next_hop=lambda outs: None,
                  run_get=lambda fp, hop: ("out", "PASS"), depth_cap=3, clock=lambda: 0.0)

    assert touched["n"] == 0
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"
    assert not (repo.parent / "escape.py").exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_copied_symlink_cannot_redirect_streamed_write_into_repo(repo, scratch, ancestor):
    from sfx.chain import WritePathError

    (repo / "link").symlink_to(repo if ancestor else repo / "foo.py")
    write = StreamedWrite(id="c1", tool="Write", verb="fork", args={
        "path": "link/foo.py" if ancestor else "link", "contents": "CORRUPTED\n"})
    called = []
    def apply(fp, w):
        called.append(True)
        _apply_write(fp, w)
    with pytest.raises(WritePathError):
        run_chain(write, repo, scratch, apply_write=apply,
                  next_hop=lambda outcomes: None, run_get=lambda fp, hop: None,
                  depth_cap=1, clock=lambda: 0.0)
    assert called == []
    assert (repo / "foo.py").read_text() == "VALUE = 1\n"
    assert list(scratch.iterdir()) == []


def test_predicted_write_hop_never_executes_in_fork(repo, scratch):
    write = StreamedWrite(id="c1", tool="Write", verb="fork",
                          args={"path": "foo.py", "contents": "VALUE = 42\n"})
    executed = []

    def next_hop(outs):
        if not outs:
            return ("edit", "fork", {"path": "bar.py", "contents": "PWNED\n"})
        return None

    def run_get(fork_path, hop):
        executed.append(hop)
        return ("out", "PASS")

    chain = run_chain(write, repo, scratch,
                      apply_write=_apply_write, next_hop=next_hop,
                      run_get=run_get, depth_cap=5, clock=lambda: 0.0)

    assert executed == []
    assert all(h.verb != "fork" for h in chain.hops)


def test_longest_prefix_serves_matching_head_discards_divergent_tail():
    speculated = [("edit", {"path": "foo.py"}),
                  ("test", {"cmd": "pytest"}),
                  ("lint", {"cmd": "ruff"})]
    real = [("edit", {"path": "foo.py"}),
            ("test", {"cmd": "pytest"}),
            ("typecheck", {"cmd": "mypy"})]

    served, discarded = longest_prefix(speculated, real)

    assert served == [("edit", {"path": "foo.py"}), ("test", {"cmd": "pytest"})]
    assert discarded == [("lint", {"cmd": "ruff"})]
