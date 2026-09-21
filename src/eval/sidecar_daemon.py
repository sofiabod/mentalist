"""Controller-only daemon entrypoint. Task execution is always brokered out."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import threading

from adapters.protocol import serve
from eval.sfx_daemon_run import build_daemon


class RetainedForks:
    def __init__(self, factory, control_root=None, token=None):
        self.factory, self.control_root, self.token = factory, control_root, token
        self.retained = []

    @contextmanager
    def fork(self, *args, **kwargs):
        from eval.sidecar_rpc import BrokerPoisoned

        context = self.factory(*args, **kwargs)
        handle = context.__enter__()
        try:
            yield handle
        except BrokerPoisoned:
            # Keep the generator itself alive: garbage collection would run its
            # finally block and delete the fork beneath an uncertain worker.
            self.retained.append((context, sys.exc_info()))
            if self.control_root is not None:
                try:
                    _atomic_json(self.control_root / "uncertain.json", {"token": self.token})
                except OSError:
                    pass  # Retention/finalization remains fail-closed without the marker.
            raise
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        else:
            context.__exit__(None, None, None)

    def release_if_terminated(self):
        if not self.retained:
            return True
        marker = self.control_root / "task-terminated.json" if self.control_root is not None else None
        if marker is None or not marker.is_file() or json.loads(marker.read_text()).get("token") != self.token:
            return False
        for context, error in self.retained:
            context.__exit__(*error)
        self.retained.clear()
        return True


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.chmod(0o600)
    temporary.replace(path)


def finish_controller(daemon):
    try:
        for pool in daemon.owned_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        safe = daemon.retained_forks.release_if_terminated()
    except BaseException:
        safe = False
    if not safe:
        # Never let interpreter finalizers delete an uncertain worker's fork.
        os._exit(74)


def build_controller(depth, broker_socket, *, table=None, log=None, control_root=None, token=None):
    from eval.sidecar_rpc import SidecarClient

    client = SidecarClient(broker_socket)
    daemon = build_daemon(depth, table=table, log=log)
    # Keep the stock fork-handle validator and write/snapshot implementation.
    # Neither execution callback may run task code in this PID namespace.
    daemon.run = client.run
    daemon.fs_substrate._run_in_fork = client.run_in_fork
    daemon.retained_forks = RetainedForks(daemon.fs_substrate.fork, control_root, token)
    daemon.fs_substrate.fork = daemon.retained_forks.fork
    daemon.owned_pools = []
    session_start = daemon.session_start

    def tracked_start(*args, **kwargs):
        session = session_start(*args, **kwargs)
        daemon.owned_pools.append(session.executor._pool)
        return session

    daemon.session_start = tracked_start
    return daemon


def main(argv):
    from eval.sfx_daemon_run import _file_log

    config = json.loads(Path(argv[0]).read_text())
    root = Path("/sfx-control")
    fd = os.open(root / "daemon.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    os.environ.update(SFX_REPO=config["repo"], SFX_SCRATCH=config["scratch"],
                      SFX_TRACE="/logs/agent/sfx-trace.jsonl", SFX_TABLE=config["table"],
                      SFX_SCRIPT_CONTRACTS=json.dumps(config["script_contracts"]),
                      SFX_FORK_PATH_VIEW=config["fork_path_view"], SFX_SEPARATE_STDERR="1")
    daemon = build_controller(config["depth"], "/sfx-control/broker.sock",
                              log=_file_log("/logs/agent/sfx-trace.jsonl"),
                              control_root=root, token=config["token"])
    server = serve(daemon, "/sfx-control/daemon.sock")
    stopping = threading.Event()

    def stop(_signum, _frame):
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _atomic_json(root / "ready.json", {"token": config["token"]})
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        try:
            daemon.shutdown()
            server.server_close()
        except BaseException:
            os._exit(74)
        # Include pools from sessions already removed by session_end. No new
        # protocol requests are admitted, while the host broker stays open.
        finish_controller(daemon)


if __name__ == "__main__":
    main(sys.argv[1:])
