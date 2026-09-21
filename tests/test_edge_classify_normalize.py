"""Adversarial edge tests for mining.normalize.classify / _classify_command.

Expected verbs are grounded in real shell semantics and the verb contract in
sfx.schema / sfx.daemon: free = speculate + serve GET-style (no epoch fence),
fork = speculate under diff-apply-under-epoch, never = do not speculate. A
command that mutates real state must therefore never be classified free.
"""

import pytest

from mining.normalize import classify


def v(cmd):
    return classify("Bash", cmd)[1]


def k(cmd):
    return classify("Bash", cmd)[0]


# ---- DEFECTS: real mutations classified free (lossless-serve of a write) ----

def test_sed_long_inplace_flag_is_edit():
    """sed --in-place rewrites the file in place, same as sed -i."""
    assert v("sed --in-place s/a/b/ f.py") == "fork"


def test_gsed_inplace_is_edit():
    """gsed -i (gnu sed on macos) edits in place."""
    assert v("gsed -i s/a/b/ f.py") == "fork"


def test_chain_read_then_rm_is_not_free():
    """ls succeeds then rm deletes files; a free serve would drop the rm."""
    assert v("ls && rm -rf build") != "free"


def test_chain_run_then_rm_is_not_free():
    assert v("python x.py && rm -rf /tmp/x") != "free"


def test_chain_cat_then_mkdir_is_not_free():
    assert v("cat a && mkdir b") != "free"


def test_pipe_into_tee_is_not_free():
    """tee writes b.txt; the leading cat masks it."""
    assert v("cat a | tee b.txt") != "free"


def test_pipe_echo_into_tee_append_is_not_free():
    assert v("echo x | tee -a log") != "free"


def test_pipe_into_xargs_rm_is_not_free():
    assert v("grep foo x.py | xargs rm") != "free"


def test_pipe_cat_into_sed_inplace_is_not_free():
    assert v("cat f | sed -i s/a/b/ g") != "free"


# ---- HARDENING: cases the classifier already gets right, lock them in ----

def test_git_push_never():
    assert classify("Bash", "git push origin main") == ("git", "never")


def test_git_commit_never():
    assert classify("Bash", "git commit -am wip") == ("git", "never")


def test_git_pull_never():
    assert classify("Bash", "git pull") == ("git", "never")


def test_trailing_git_in_chain_never():
    assert v("echo hi && git push") == "never"


def test_subshell_git_never():
    assert v("(cd d && git push)") == "never"


def test_env_prefixed_git_never():
    assert v("env X=1 git commit") == "never"


def test_or_chain_git_never():
    assert v("false || git push") == "never"


def test_leading_whitespace_git_never():
    assert v("  git push") == "never"


def test_sed_n_print_is_read():
    assert classify("Bash", "sed -n 1,5p f.py") == ("read", "free")


def test_sed_i_short_flag_is_edit():
    assert v("sed -i s/a/b/ f.py") == "fork"


def test_rm_is_edit():
    assert v("rm -rf build") == "fork"


def test_mkdir_is_edit():
    assert v("mkdir -p a/b") == "fork"


def test_bare_tee_is_edit():
    assert v("tee out.txt") == "fork"


def test_pip_install_is_never():
    # install is non-speculable: real package resolution/mutation a fork cannot reproduce
    assert classify("Bash", "pip install requests") == ("install", "never")


def test_python_m_pytest_is_test():
    assert classify("Bash", "python -m pytest") == ("test", "free")


def test_bare_pytest_is_test():
    assert classify("Bash", "pytest") == ("test", "free")


def test_bare_script_run_is_run():
    assert classify("Bash", "python x.py") == ("run", "free")


def test_manage_py_not_run():
    assert k("python manage.py migrate") != "run"


def test_heredoc_write_is_edit():
    assert v("cat <<'EOF' > f.txt") == "fork"


def test_redirect_write_is_edit():
    assert v("printf hi > f.txt") == "fork"


def test_redirect_inside_quotes_not_edit():
    """A > inside a quoted string is not a real redirect."""
    assert v('echo ">"') != "fork"


# ---- Write vs Edit vs read tools; unknown tool must not crash ----

def test_write_tool_is_fork():
    assert classify("Write") == ("edit", "fork")


def test_edit_tool_is_fork():
    assert classify("Edit") == ("edit", "fork")


def test_read_tool_is_free():
    assert classify("Read") == ("read", "free")


def test_unknown_tool_valid_policy_no_crash():
    kind, verb = classify("TotallyMadeUpTool")
    assert verb in {"free", "fork", "never"}


def test_bash_no_skeleton_no_crash():
    kind, verb = classify("Bash", None)
    assert verb in {"free", "fork", "never"}


def test_empty_command_no_crash():
    kind, verb = classify("Bash", "")
    assert verb in {"free", "fork", "never"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
