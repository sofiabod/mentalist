import json
import re
from dataclasses import dataclass

from mining.normalize import classify
from sfx.schema import STREAM_ONLY

_CMD_RE = re.compile(r'"cmd"\s*:\s*("(?:[^"\\]|\\.)*")')


@dataclass(frozen=True)
class StreamedWrite:
    id: str
    tool: str
    verb: str
    args: dict


class StreamParser:
    def __init__(self):
        self._buffers = {}
        self._get_buffers = {}

    def feed(self, call_id, tool, delta):
        verb = classify(tool)[1]
        if verb not in STREAM_ONLY:
            return None
        buf = self._buffers.get(call_id, "") + delta
        self._buffers[call_id] = buf
        try:
            args = json.loads(buf)
        except json.JSONDecodeError:
            return None
        del self._buffers[call_id]
        return StreamedWrite(id=call_id, tool=tool, verb=verb, args=args)

    def feed_get(self, call_id, delta):
        """Accumulate a GET-class call's arg stream. Returns (partial_args, done):
        partial_args once the "cmd" value is unambiguous (a complete JSON string,
        possibly before the closing brace), else None; done True once the full JSON
        object parses. Unambiguous-before-complete is what lets speculation fire early.
        """
        buf = self._get_buffers.get(call_id, "") + delta
        self._get_buffers[call_id] = buf
        try:
            args = json.loads(buf)
            del self._get_buffers[call_id]
            return args, True
        except json.JSONDecodeError:
            m = _CMD_RE.search(buf)
            if m:
                return {"cmd": json.loads(m.group(1))}, False
            return None, False
