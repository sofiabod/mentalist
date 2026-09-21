import json
import os
import socket
import socketserver
import threading


class SfxDaemonError(Exception):
    pass


def _dispatch(daemon, msg):
    t = msg["type"]
    sid = msg["session"]
    if t == "turn_begin":
        with daemon._lock:
            if sid not in daemon.sessions:
                daemon.session_start(sid, repo=msg["repo"], role=msg["role"],
                                     scratch=msg.get("scratch"),
                                     state_model=msg.get("state_model", "fs-only"),
                                     profile=msg.get("profile"),
                                     registered_kinds=msg.get("registered_kinds"),
                                     state_dir=msg.get("state_dir"),
                                     spec_disabled=msg.get("spec_disabled", False),
                                     trajectory=msg.get("trajectory"))
        return {"result": "ok"}
    if t == "mutation_begin":
        mutation_id = daemon.mutation_begin(sid, write_args=msg.get("write_args"))
        return {"result": "ok", "mutation_id": mutation_id}
    if t == "mutation_end":
        preserved = daemon.mutation_end(sid, msg["mutation_id"],
                                        success=msg.get("success", False))
        return {"result": "ok", "chain_preserved": preserved}
    if t == "feed":
        chain = daemon.call_stream_delta(sid, msg["call_id"], msg["tool"], msg["delta"],
                                         mutation_id=msg.get("mutation_id"))
        return {"chain_len": len(chain.hops) if chain else 0}
    if t == "resolve":
        if msg["args"] is None:
            return {"error": "args must not be null"}
        outcome, result = daemon.resolve(sid, kind=msg["tool"], args=msg["args"])
        return {"result": outcome, "output": result}
    if t == "call_executed":
        daemon.call_executed(sid, kind=msg["tool"], verb=msg["verb"],
                             outcome=msg["outcome"], args=msg["args"],
                             latency=msg["latency"],
                             observation=msg.get("observation", ""),
                             mutation_id=msg.get("mutation_id"),
                             speculate=msg.get("speculate", True))
        return {"result": "ok"}
    if t == "turn_end":
        daemon.turn_end(sid)
        return {"result": "ok"}
    if t == "session_end":
        daemon.session_end(sid)
        return {"result": "ok"}
    return {"error": f"unknown type: {t}"}


def _handle_line(dispatch, line):
    try:
        msg = json.loads(line)
        return dispatch(msg)
    except Exception as e:
        return {"error": str(e)}


def serve(daemon, sock_path, on_dispatch=None):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    dispatch = on_dispatch or (lambda msg: _dispatch(daemon, msg))

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                reply = _handle_line(dispatch, line)
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    class Server(socketserver.ThreadingUnixStreamServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server(sock_path, Handler)


class Client:
    def __init__(self, sock_path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        try:
            self.sock.connect(sock_path)
            self.f = self.sock.makefile("rwb")
        except BaseException:
            self.sock.close()
            raise
        self._lock = threading.Lock()

    def _send(self, msg):
        with self._lock:
            try:
                self.f.write((json.dumps(msg) + "\n").encode())
                self.f.flush()
                return json.loads(self.f.readline())
            except (OSError, json.JSONDecodeError) as e:
                raise SfxDaemonError(str(e))

    def turn_begin(self, session, repo, role, scratch=None, state_model="fs-only",
                   profile=None, registered_kinds=None, state_dir=None,
                   spec_disabled=False, trajectory=None):
        return self._send({"type": "turn_begin", "session": session,
                           "repo": repo, "role": role, "scratch": scratch,
                           "state_model": state_model, "profile": profile,
                           "registered_kinds": registered_kinds,
                           "state_dir": state_dir, "spec_disabled": spec_disabled,
                           "trajectory": trajectory})

    def mutation_begin(self, session, write_args=None):
        return self._send({"type": "mutation_begin", "session": session,
                           "write_args": write_args})

    def mutation_end(self, session, mutation_id, success=False):
        return self._send({"type": "mutation_end", "session": session,
                           "mutation_id": mutation_id, "success": success})

    def feed(self, session, call_id, tool, delta, mutation_id=None):
        return self._send({"type": "feed", "session": session,
                           "call_id": call_id, "tool": tool, "delta": delta,
                           "mutation_id": mutation_id})

    def resolve(self, session, tool, args):
        return self._send({"type": "resolve", "session": session,
                           "tool": tool, "args": args})

    def call_executed(self, session, tool, verb, outcome, args, latency, observation="",
                      mutation_id=None, speculate=True):
        return self._send({"type": "call_executed", "session": session,
                           "tool": tool, "verb": verb, "outcome": outcome,
                           "args": args, "latency": latency,
                           "observation": observation, "mutation_id": mutation_id,
                           "speculate": speculate})

    def turn_end(self, session):
        return self._send({"type": "turn_end", "session": session})

    def session_end(self, session):
        return self._send({"type": "session_end", "session": session})

    def close(self):
        self.f.close()
        self.sock.close()
