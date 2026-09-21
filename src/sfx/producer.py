"""Token producers for sPTC: emit a tool call token-by-token with inter-token latency.

sPTC (stateless parse-then-call) reads the exact call from the model's token stream
mid-generation and speculates it before generation completes. A producer is the token
SOURCE. Offline we use ScriptedProducer (tokens + timing scripted, the parse/speculate/
serve mechanism is real); Rung 2 swaps in a live vLLM/OpenAI stream (VllmStreamProducer).
"""
import json


class TokenProducer:
    """Yields (token_text, inter_token_ms) for a target tool call's surface syntax."""

    def __init__(self, kind, args):
        self.kind = kind
        self.args = args

    def tokens(self):
        raise NotImplementedError


class ScriptedProducer(TokenProducer):
    """Renders `args` to the JSON surface stream.py parses, emits it char-by-char.

    Latency per emitted token models generation time; the sum is the generation window.
    This is a stand-in for the token SOURCE only. The mechanism (incremental parse,
    early speculation from a PARTIAL prefix, serve-on-completion) is genuine.
    """

    def __init__(self, kind, args, latency_ms_per_token=20.0, trailing_tokens=0):
        super().__init__(kind, args)
        self.latency_ms_per_token = latency_ms_per_token
        # tokens the model still streams AFTER the parseable args close (wrapper/close
        # syntax). They widen the post-parse TAIL sPTC can hide the exec behind.
        self.trailing_tokens = trailing_tokens

    def surface(self):
        return json.dumps(self.args)

    def tokens(self):
        for ch in self.surface():
            yield ch, self.latency_ms_per_token
        for _ in range(self.trailing_tokens):
            yield " ", self.latency_ms_per_token


class VllmStreamProducer(TokenProducer):
    """Rung 2 seam: wrap a live vLLM/OpenAI streaming completion.

    Not built (no GPU offline). To implement: iterate the streaming chat/completions
    response, accumulate the tool-call arguments delta, yield (delta_text, wall_gap_ms)
    where wall_gap_ms is measured between successive token timestamps. The daemon-side
    parse/speculate/serve path is identical to the scripted one; only the token SOURCE
    changes. That is the whole point of the abstraction.
    """

    def __init__(self, kind, args, client=None, model=None):
        super().__init__(kind, args)
        raise NotImplementedError(
            "VllmStreamProducer needs a live vLLM/OpenAI endpoint; offline uses "
            "ScriptedProducer. Rung 2 upgrade: stream the completion, yield "
            "(arg_delta, inter_token_ms) from real token timestamps.")
