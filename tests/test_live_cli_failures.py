import base64
import json

import pytest

from eval import sfx_client_cli


@pytest.mark.parametrize("operation,payload", [
    ("begin", None),
    ("resolve", {"tool": "run", "args": {"cmd": "python probe.py"}}),
    ("report", {"tool": "run", "verb": "free", "outcome": "OK",
                "args": {"cmd": "python probe.py"}, "latency": 1}),
    ("feed", {"call_id": "c1", "tool": "Edit", "body": "{}"}),
    ("end", None),
])
def test_daemon_rejection_cannot_be_printed_as_success(monkeypatch, capsys, operation, payload):
    class RejectingClient:
        def __init__(self, socket):
            pass

        def __getattr__(self, name):
            if name == "close":
                return lambda: None
            return lambda *args, **kwargs: {"error": "unknown or expired session"}

    monkeypatch.setattr(sfx_client_cli, "Client", RejectingClient)
    argv = [operation, "session"]
    if operation == "begin":
        argv += ["/app", "/tmp/sfx", "0"]
    elif payload is not None:
        argv.append(base64.b64encode(json.dumps(payload).encode()).decode())
    with pytest.raises(RuntimeError, match="daemon rejected"):
        sfx_client_cli.main(argv)
    assert capsys.readouterr().out == ""
