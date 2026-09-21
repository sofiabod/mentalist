"""In-sandbox client CLI for the LIVE sfx integration.

Runs INSIDE the same sandbox as the daemon, talks to it over the Unix socket, and
prints a one-line JSON reply. The host-side SFXLiveAgent shells this out through
environment.exec for every intercepted tool call, so the daemon, the fork, and the
socket client all live inside the sandbox on SFX_REPO=/app. No host-side daemon, no
replay: the wrapped agent's real calls drive it live.

Usage (argv):
  begin  <session> <repo> <scratch> <spec_disabled 0|1>
    resolve <session> <b64 json {tool,args}>            -> cached result envelope
  report  <session> <b64 json {tool,verb,outcome,args,latency,observation}>
  feed    <session> <b64 json {call_id,tool,body}>
  end     <session>
"""
import base64
import json
import os
import sys

from adapters.protocol import Client

SOCKET = os.environ.get("SFX_SOCKET", "/tmp/sfx.sock")


def _b64(s):
    return json.loads(base64.b64decode(s).decode())


def _checked(reply):
    if not isinstance(reply, dict) or reply.get("error"):
        raise RuntimeError(f"sfx daemon rejected request: {reply}")
    return reply


def dispatch(client, argv):
    op = argv[0]
    session = argv[1]
    if op == "snapshot":
        from pathlib import Path
        from eval.capture import fs_hash
        repo, trace = argv[2], Path(argv[3])
        records = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] \
            if trace.exists() else []
        return {"final_fs_hash": fs_hash(repo), "trace": records}
    if client is None:
        raise RuntimeError("sfx control request needs a socket client")
    if op == "begin":
        repo, scratch, spec_disabled = argv[2], argv[3], argv[4] == "1"
        trajectory = _b64(argv[5]) if len(argv) > 5 else None
        _checked(client.turn_begin(session, repo=repo, role="main", scratch=scratch,
                                   state_dir=f"{scratch}/state/{session}",
                                   spec_disabled=spec_disabled, trajectory=trajectory))
        return {"ok": True}
    if op == "mutation_begin":
        p = _b64(argv[2])
        return _checked(client.mutation_begin(session, write_args=p.get("write_args")))
    if op == "mutation_end":
        p = _b64(argv[2])
        return _checked(client.mutation_end(session, p["mutation_id"],
                                            success=p.get("success", False)))
    if op == "resolve":
        payload = _b64(argv[2])
        reply = _checked(client.resolve(session, tool=payload["tool"], args=payload["args"]))
        served = isinstance(reply.get("result"), str) and reply["result"].startswith("hit")
        return {"served": served, "outcome": reply.get("result"), "output": reply.get("output")}
    if op == "report":
        p = _b64(argv[2])
        _checked(client.call_executed(session, tool=p["tool"], verb=p["verb"], outcome=p["outcome"],
                                     args=p["args"], latency=p["latency"],
                                     observation=p.get("observation", ""),
                                     mutation_id=p.get("mutation_id"),
                                     speculate=p.get("speculate", True)))
        return {"ok": True}
    if op == "feed":
        p = _b64(argv[2])
        reply = _checked(client.feed(session, p["call_id"], p["tool"], p["body"],
                                    mutation_id=p.get("mutation_id")))
        return {"chain_len": reply.get("chain_len", 0)}
    if op == "end":
        _checked(client.turn_end(session))
        _checked(client.session_end(session))
        return {"ok": True}
    return {"error": f"unknown op {op}"}


def main(argv):
    if argv[0] == "snapshot":
        print(json.dumps(dispatch(None, argv)))
        return
    client = Client(SOCKET)
    try:
        print(json.dumps(dispatch(client, argv)))
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1:])
