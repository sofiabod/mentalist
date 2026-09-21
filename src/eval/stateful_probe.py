"""CPU mechanism demonstration using the live wrapper, CLI, daemon and resolver.

The tool performs real deterministic CPU work. The agent/schedule are controlled,
not a live model or a benchmark task. No future commands enter the predictor.
Local runs are sequential in a reset, experiment-owned workspace with a fresh
daemon/session each time; this is not container/security isolation.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid


MODES = ("native", "OFF", "GET", "ON")
CONTROLS = ("normal", "wrong_args", "intervening_edit")
FIXTURE_VERSION = 2
REVISION_FILE = "revision.txt"


def script(work_units):
    """Reviewed immutable checker; only its revision data changes during a run."""
    if type(work_units) is not int or not 1 <= work_units <= 10_000_000:
        raise ValueError("work_units must be an integer in [1, 10000000]")
    return (
        "import hashlib, json, sys\n"
        f"with open({REVISION_FILE!r}, encoding='utf-8') as source:\n"
        "    revision = int(source.read())\n"
        f"digest = hashlib.pbkdf2_hmac('sha256', str(revision).encode(), b'sfx-probe', {work_units})\n"
        "assert len(digest) == 32\n"
        "print(json.dumps({'revision': revision, 'digest': digest.hex(), "
        "'args': sys.argv[1:]}, sort_keys=True))\n"
        "sys.stderr.write('checked revision %d\\n' % revision)\n"
    )


def revision_data(revision):
    if type(revision) is not int or revision < 0:
        raise ValueError("revision must be a nonnegative integer")
    return f"{revision}\n"


def fixture_identity(work_units):
    return {"fixture": "stateful_probe", "fixture_version": FIXTURE_VERSION,
            "fixture_source_sha256": hashlib.sha256(script(work_units).encode()).hexdigest()}


def commands(work_units, rounds, control="normal"):
    script(work_units)
    if type(rounds) is not int or not 1 <= rounds <= 20:
        raise ValueError("rounds must be an integer in [1, 20]")
    if control not in CONTROLS:
        raise ValueError("unknown control")
    result = ["python probe.py"]
    for revision in range(1, rounds + 1):
        body = revision_data(revision)
        result.append(f"cat > {REVISION_FILE} <<'SFX_PROBE_EOF'\n{body}SFX_PROBE_EOF")
        if control == "intervening_edit":
            # Deliberately nonextractable: the old chain must be invalidated.
            alternate = revision_data(revision + 1000)
            result.append(f"printf '%s' {shlex.quote(alternate)} > {REVISION_FILE}")
        result.append("python probe.py" + (" --different" if control == "wrong_args" else ""))
    return result


def model_fn(spec):
    """CPU model replacement; its private command list is not a daemon oracle."""
    cfg = json.loads(spec)
    gap = cfg["gap_s"]
    if (isinstance(gap, bool) or not isinstance(gap, (int, float))
            or not math.isfinite(gap) or not 0 <= gap <= 10):
        raise ValueError("gap_s must be finite and in [0, 10]")
    tape = commands(cfg["work_units"], cfg["rounds"], cfg.get("control", "normal"))
    bootstrap = cfg.get("bootstrap", False)
    if type(bootstrap) is not bool:
        raise ValueError("bootstrap must be a boolean")
    if bootstrap:
        initial = revision_data(0)
        tape.insert(0, f"cat > {REVISION_FILE} <<'SFX_PROBE_EOF'\n{initial}SFX_PROBE_EOF")

    def predict(base_url, model, api_key, messages):
        index = sum(m.get("role") == "assistant" for m in messages)
        if index >= len(tape):
            return "DONE"
        if index and gap:
            time.sleep(gap)
        return f"```bash\n{tape[index]}\n```"

    return predict


def stream_model_fn(spec):
    """Synthetic chunk arrivals for CPU wiring tests, never live-model evidence.

    A complete edit arrives before its closing fence. The configured tail is an
    explicit experimental input, not measured decoding time or a token count.
    """
    cfg = json.loads(spec)
    model_fn(spec)  # Reuse the command/gap validation.
    tail = cfg["stream_tail_s"]
    if (type(tail) not in (int, float) or not math.isfinite(tail)
            or not 0 <= tail <= 10):
        raise ValueError("stream_tail_s must be finite and in [0, 10]")
    tape = commands(cfg["work_units"], cfg["rounds"], cfg.get("control", "normal"))
    if cfg.get("bootstrap", False):
        raise ValueError("stream fixture does not support bootstrap")

    def stream(base_url, model, api_key, messages, **sampling):
        index = sum(m.get("role") == "assistant" for m in messages)
        start = time.monotonic()
        gap = cfg["gap_s"] if index else 0
        time.sleep(max(0, start + gap - time.monotonic()))
        if index >= len(tape):
            yield {"type": "delta", "content": "DONE"}
        else:
            text = f"```bash\n{tape[index]}\n"
            split = len(text) // 2
            yield {"type": "delta", "content": text[:split]}
            yield {"type": "delta", "content": text[split:]}
            # Absolute deadlines: callback work consumes, not extends, this gap.
            time.sleep(max(0, start + gap + tail - time.monotonic()))
            yield {"type": "delta", "content": "```"}
        yield {"type": "done", "finish_reason": "stop"}

    return stream


class LocalSandbox:
    """The real-subprocess transport already used by the live integration tests."""
    default_user = None

    def __init__(self, root, edit_dispatch_s=0):
        self.root = root
        self.edit_dispatch_s = edit_dispatch_s
        self.session_id = f"local-reset-{uuid.uuid4().hex}"

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult
        from eval.sfx_live_agent import _literal_write

        # Explicit supplemental control: simulate pre-authoritative dispatch
        # latency equally in ALL arms, not faster/slower computation in one arm.
        # This is not a real model token stream or a measured network delay.
        if self.edit_dispatch_s and _literal_write(command) is not None:
            await asyncio.sleep(self.edit_dispatch_s)
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd or self.root, env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec or 60)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            raise
        return ExecResult(stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode)


def _workspace(output_root, work_units):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    marker = root / "experiment-owned.json"
    ownership = {"experiment": "stateful_probe", "fixture_version": FIXTURE_VERSION,
                 "workspace": str(workspace)}
    if workspace.exists() and not marker.exists():
        raise ValueError("refusing to reset an unowned workspace")
    if not marker.exists():
        with marker.open("x") as f:
            json.dump(ownership, f)
    if json.loads(marker.read_text()) != ownership:
        raise ValueError("workspace ownership marker does not match")
    workspace.mkdir(exist_ok=True)
    if workspace.is_symlink() or any(p.name not in {"probe.py", REVISION_FILE}
                                     or p.is_symlink() for p in workspace.iterdir()):
        raise ValueError("unexpected workspace entries; preserving them")
    # Only the two owned fixture files are reset; never delete a user repository.
    (workspace / "probe.py").write_text(script(work_units))
    (workspace / REVISION_FILE).write_text(revision_data(0))
    return workspace


def run_one(output_root, *, mode, work_units, gap_s, rounds, control="normal",
            case_id="probe", rep=0, edit_dispatch_s=0, stream_tail_s=None):
    from adapters.protocol import serve
    from eval import sfx_live_agent
    from eval.capture import fs_hash
    from eval.sfx_daemon_run import build_daemon, _file_log
    from eval.table import load_table

    if mode not in MODES or not case_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid mode or case ID")
    if (isinstance(edit_dispatch_s, bool) or not isinstance(edit_dispatch_s, (int, float))
            or not math.isfinite(edit_dispatch_s) or not 0 <= edit_dispatch_s <= 10):
        raise ValueError("edit_dispatch_s must be finite and in [0, 10]")
    cfg = dict(work_units=work_units, gap_s=gap_s, rounds=rounds, control=control)
    model_fn(json.dumps(cfg))  # Validate before touching the fixture.
    model_config = {"model": "controlled-cpu", "temperature": 0.0, "seed": 0,
                    "max_tokens": 1024, "model_fn": f"eval.stateful_probe:model_fn({json.dumps(cfg)})"}
    if stream_tail_s is not None:
        stream_cfg = {**cfg, "stream_tail_s": stream_tail_s}
        stream_model_fn(json.dumps(stream_cfg))
        model_config.update(model_fn=None,
                            stream_model_fn=f"eval.stateful_probe:stream_model_fn({json.dumps(stream_cfg)})")
    root = Path(output_root).resolve()
    workspace = _workspace(root, work_units)
    run_dir = root / f"{case_id}-rep{rep}-{mode}"
    run_dir.mkdir(exist_ok=False)
    scratch = run_dir / "forks"
    scratch.mkdir()
    trace = run_dir / "trace.jsonl"
    source = Path(__file__).resolve().parents[1]
    table_path = source.parent / "data/tables/benchmark.json"
    script_contracts = [{
        "script": "probe.py", "positionals": 0, "path_options": [],
        "value_options": [], "flags": [], "required": [],
        "source_sha256": {"probe.py": fixture_identity(work_units)["fixture_source_sha256"]},
    }]
    begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sfx-probe-socket-", dir="/tmp") as socket_root:
        socket = str(Path(socket_root) / "daemon.sock")
        envvars = {"SFX_REPO": str(workspace), "SFX_SCRATCH": str(scratch),
                   "SFX_SEPARATE_STDERR": "1",
                   "SFX_SCRIPT_CONTRACTS": json.dumps(script_contracts),
                   "PYTHONDONTWRITEBYTECODE": "1", "SFX_TABLE": "benchmark",
                   "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        with patch.dict(os.environ, envvars), patch.multiple(
                sfx_live_agent, APP=str(workspace), SCRATCH=str(scratch),
                TRACE=str(trace), SFX_SRC=str(source), SOCKET=socket):
            daemon = build_daemon(1, table=load_table(table_path), log=_file_log(trace))
            owned_pools = []
            start_session = daemon.session_start

            def remember_session(*args, **kwargs):
                session = start_session(*args, **kwargs)
                owned_pools.append(session.executor._pool)
                return session

            daemon.session_start = remember_session
            server = serve(daemon, socket)
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            agent = sfx_live_agent.SFXLiveAgent(
                run_dir / "agent", wrapped="eval.sfx_live_agent:build_live_agent",
                arm="ON" if mode in ("ON", "GET") else "OFF",
                speculate_writes=mode == "ON", table="benchmark", depth=1,
                script_contracts=script_contracts,
                base_url="http://unused", api_key="unused",
                max_steps=len(commands(work_units, rounds, control)) + 1,
                **model_config)
            environment = LocalSandbox(workspace, edit_dispatch_s=edit_dispatch_s)
            if mode == "native":
                # Retain identical receipt capture, bypass all per-call daemon
                # work. OFF separately measures the existing bookkeeping cost.
                async def plain_exec(environment, original_exec, command, cwd, env, timeout_sec, user):
                    agent._counts["authoritative"] += 1
                    return await original_exec(command=command, cwd=cwd, env=env,
                                               timeout_sec=timeout_sec, user=user)
                agent._route_one = plain_exec
            context = SimpleNamespace(metadata={})
            startup_wall = time.monotonic() - begin
            run_started = time.monotonic()
            try:
                asyncio.run(agent.run("Controlled stateful CPU probe", environment, context))
            finally:
                cleanup_started = time.monotonic()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                daemon.shutdown()
                # session_end removes sessions and Executor.shutdown is nonblocking.
                # Retain and join OUR pools outside the daemon lock so discarded
                # CPU work cannot leak into the next timed arm.
                for pool in owned_pools:
                    pool.shutdown(wait=True, cancel_futures=True)
                cleanup_wall = time.monotonic() - cleanup_started
            total_wall = time.monotonic() - run_started
    record = context.metadata["sfx_live"]
    record.update(experiment_mode=mode, case_id=case_id, rep=rep, **cfg,
                  **fixture_identity(work_units), stream_tail_s=stream_tail_s,
                  edit_dispatch_s=edit_dispatch_s,
                  startup_wall_s=startup_wall, cleanup_wall_s=cleanup_wall,
                  total_wall_s=total_wall,
                  workers_drained=all(not t.is_alive() for pool in owned_pools for t in pool._threads),
                  table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                  environment_isolation="sequential fixture reset, fresh daemon/session; not containers",
                  timing_kind="synthetic fixed gap, real CPU tool execution",
                  predictor="stock benchmark table and resolver; no oracle trajectory")
    if stream_tail_s is not None:
        record["timing_kind"] = "synthetic stream arrival schedule, real CPU tool execution"
    record["after_cleanup_fs_hash"] = fs_hash(workspace)
    if record["after_cleanup_fs_hash"] != record["final_fs_hash"]:
        record["completed"] = False
        record["error"] = "workspace changed after end-of-session snapshot"
    # End-of-session snapshot precedes worker draining; retain the final trace too.
    record["trace"] = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    with (run_dir / "record.json").open("x") as f:
        json.dump(record, f, indent=2)
    (run_dir / "final-probe.py").write_text((workspace / "probe.py").read_text())
    (run_dir / "final-revision.txt").write_text((workspace / REVISION_FILE).read_text())
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="owned output directory; defaults to a fresh temporary directory")
    parser.add_argument("--mode", choices=MODES, default="ON")
    parser.add_argument("--control", choices=CONTROLS, default="normal")
    parser.add_argument("--work-units", type=int, default=50_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gap-s", type=float, default=.05,
                        help="synthetic delay before the next response, not model latency")
    parser.add_argument("--stream-tail-s", type=float, default=.15,
                        help="synthetic delay between complete edit and closing fence")
    args = parser.parse_args(argv)
    config = dict(work_units=args.work_units, gap_s=args.gap_s, rounds=args.rounds,
                  control=args.control, stream_tail_s=args.stream_tail_s)
    try:
        stream_model_fn(json.dumps(config))  # Validate before creating a workspace.
    except ValueError as exc:
        parser.error(str(exc))
    root = args.output or Path(tempfile.mkdtemp(prefix="sfx-stateful-"))
    record = run_one(root, mode=args.mode, **config)
    unchanged = record["after_cleanup_fs_hash"] == record["final_fs_hash"]
    valid = record["completed"] and record["workers_drained"] and unchanged
    print(json.dumps({
        "scope": "controlled CPU mechanism demo, not live-model performance",
        "mode": args.mode, "control": args.control,
        "record": str((root / f"probe-rep0-{args.mode}" / "record.json").resolve()),
        "completed": record["completed"], "workers_drained": record["workers_drained"],
        "workspace_unchanged_after_cleanup": unchanged,
        "served_result": any(row["served"] for row in record["raw"]),
    }), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
