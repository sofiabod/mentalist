import json

import pytest

from mining.ingest_openhands import (
    trajectory_repo,
    messages_to_events,
    ingest_trajectories,
)


def _asst(content):
    return {"role": "assistant", "content": content}


def _obs(content):
    return {"role": "user", "content": content}


def _view(path):
    return _asst(f"<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>{path}</parameter>\n</function>")


def _str_replace(path):
    return _asst(f"<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>{path}</parameter>\n</function>")


def _create(path):
    return _asst(f"<function=str_replace_editor>\n<parameter=command>create</parameter>\n<parameter=path>{path}</parameter>\n</function>")


def _bash(cmd):
    return _asst(f"<function=execute_bash>\n<parameter=command>{cmd}</parameter>\n</function>")


def _bash_result(exit_code):
    return _obs(f"EXECUTION RESULT of [execute_bash]:\nsome output\n[Command finished with exit code {exit_code}]")


FIXTURE = [
    {"role": "system", "content": "you are helpful"},
    _obs("<uploaded_files>\n/workspace/python__mypy__0.820\n</uploaded_files>\nfix the bug"),
    _view("/workspace/python__mypy__0.820/mypy"),
    _obs("EXECUTION RESULT of [str_replace_editor]:\nfiles..."),
    _bash("grep -rn Protocol mypy/"),
    _bash_result(0),
    _create("/workspace/python__mypy__0.820/reproduce.py"),
    _obs("EXECUTION RESULT of [str_replace_editor]:\nFile created successfully"),
    _bash("python3 reproduce.py"),
    _bash_result(1),
    _str_replace("/workspace/python__mypy__0.820/mypy/checker.py"),
    _obs("EXECUTION RESULT of [str_replace_editor]:\nThe file has been edited."),
    _bash("python -m pytest test_checker.py"),
    _bash_result(0),
    _asst("<function=finish>\n<parameter=message>done</parameter>\n</function>"),
]


def test_trajectory_repo_from_uploaded_files():
    assert trajectory_repo(FIXTURE) == "python/mypy"


def test_message_mapping_kinds_and_verbs():
    events = messages_to_events(FIXTURE)
    # expected from command semantics, hand-computed:
    # view -> read/free, grep -> grep/free, create -> edit/fork,
    # python3 script -> run/free, str_replace -> edit/fork, pytest -> test/free
    # finish -> dropped
    kv = [(e.kind, e.verb) for e in events]
    assert kv == [
        ("read", "free"),
        ("grep", "free"),
        ("edit", "fork"),
        ("run", "free"),
        ("edit", "fork"),
        ("test", "free"),
    ]


def test_bash_outcome_from_exit_code():
    events = messages_to_events(FIXTURE)
    # third bash is pytest with exit 0 -> PASS; second bash python3 exit 1 -> ERR
    py = events[3]
    assert py.kind == "run" and py.outcome.status == "ERR"
    pt = events[5]
    assert pt.kind == "test" and pt.outcome.status == "PASS"


def test_provenance_tag_present():
    events = messages_to_events(FIXTURE)
    assert events and all(e.args.get("provenance") == "benchmark" for e in events)


def test_blocklist_filter_drops_matching_repo():
    traj_kept = FIXTURE
    traj_dropped = [
        _obs("<uploaded_files>\n/workspace/abs-lang__abs__1.0\n</uploaded_files>\nfix"),
        _view("/workspace/abs-lang__abs__1.0/main.go"),
        _obs("EXECUTION RESULT of [str_replace_editor]:\nfiles..."),
    ]
    blocklist = {"abs-lang/abs"}
    sessions, kept, dropped = ingest_trajectories([traj_kept, traj_dropped], blocklist)
    assert kept == 1
    assert dropped == 1
    assert "abs-lang/abs" not in {trajectory_repo(t) for t in [traj_kept]}


def test_ingest_asserts_disjoint_from_blocklist():
    traj = FIXTURE
    with pytest.raises(AssertionError):
        ingest_trajectories([traj], {"python/mypy"}, drop_blocked=False)


def test_global_table_untouched(tmp_path):
    import hashlib, pathlib
    p = pathlib.Path("data/tables/global.json")
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    messages_to_events(FIXTURE)
    ingest_trajectories([FIXTURE], set())
    after = hashlib.sha256(p.read_bytes()).hexdigest()
    assert before == after
