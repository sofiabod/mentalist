"""Model-driven command loop used by the live streaming integration.

The executor and stream callbacks are supplied by the adapter; this module does
not start a daemon, reset a repository, replay a tape, or run a benchmark.
"""
import copy
import hashlib
import re
import time

from eval.model_stream import ContextBudgetExceeded, ModelStreamError, _usage
from eval.model_stream import stream_chat_completion as _model_stream
from eval.model_tools import parse_native_action, stream_native_bash

SYSTEM = (
    "You are a coding agent working in the current repository. Each step, respond with "
    "EXACTLY ONE bash command inside a ```bash\n...\n``` block, or the single token DONE "
    "when the task is complete. No prose, no explanation, one command per step."
)
_BASH = re.compile(r"```bash\s*\n(.*?)```", re.S)


# Optional task scaffold; general and native modes are configured by the adapter.
_REPRO_WORKFLOW = (
    "You are a coding agent in the current repo. Work the task in this loop, ONE bash "
    "command per step inside a ```bash\\n...\\n``` block:\n"
    "1. read the relevant source files.\n"
    "2. create reproduce.py and run `python reproduce.py` to confirm the issue.\n"
    "3. edit the source to fix it.\n"
    "4. run `python reproduce.py` again after EACH edit.\n"
    "5. when it passes, run the related tests with pytest.\n"
)
_TASK_SUBMISSION_SYSTEM = _REPRO_WORKFLOW + (
    "Use git only when the task requires it. Honor any requested branch and commit "
    "your work before DONE when required. Reply DONE only after the tests pass and "
    "all task submission requirements are satisfied. No prose, one command per step."
)


def _record_context_budget(event, requested):
    value = event.get("budget")
    integer_fields = {"prompt_tokens", "reserved_tokens", "context_length",
                      "safety_margin_tokens", "requested_max_tokens", "effective_max_tokens"}
    fingerprints = {"request_sha256": 64, "server_instance_id": 32, "renderer_sha256": 64}
    fields = integer_fields | set(fingerprints) | {"provider", "protocol"}
    if (set(event) != {"type", "budget"} or not isinstance(value, dict)
            or set(value) != fields
            or value.get("provider") != "sglang"
            or value.get("protocol") != "sfx-chat-token-budget-v1"):
        raise ModelStreamError("Invalid context-budget event")
    if any(type(value[key]) is not int or value[key] < 0 for key in integer_fields):
        raise ModelStreamError("Invalid context-budget token counts")
    if any(not isinstance(value[key], str)
           or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value[key])
           for key, length in fingerprints.items()):
        raise ModelStreamError("Invalid context-budget provenance")
    remaining = value["context_length"] - value["prompt_tokens"] - value["reserved_tokens"] - 1
    if (value["requested_max_tokens"] != requested or value["safety_margin_tokens"] != 1
            or value["effective_max_tokens"] != min(requested, max(0, remaining))):
        raise ModelStreamError("Inconsistent context-budget calculation")
    return dict(value)


def _model_action(base_url, model, api_key, messages, *,
                  temperature=0.0, seed=0, max_tokens=1024):
    import requests
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": temperature,
              "seed": seed, "max_tokens": max_tokens},
        timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_action(text):
    m = _BASH.search(text)
    if m:
        return m.group(1).strip()
    return None


def run_agent(exec_fn, task, *, base_url, model, api_key, max_steps,
              model_fn=_model_action, system=SYSTEM, stream_fn=None,
              on_stream_event=None, temperature=0.0, seed=0, max_tokens=1024,
              context_budget=None):
    """Drive the agent loop; exec_fn(cmd)->stdout runs each command. Returns wall + trajectory.

    Wall spans the whole loop, so it INCLUDES the model's think time (the real API latency)
    plus tool exec. sfx hides tool exec behind that think time on the ON arm.
    Timings are offsets from loop start, not savings estimates. With a stub model_fn,
    they measure the stub's delay, not live inference latency.
    An optional stream_fn yields delta/usage/done events. Stream callbacks expose
    cumulative content immediately, and model_end precedes authoritative tool
    execution. They never modify the model's generated text. First-chunk timing
    is not token-level timing; completion counts come only from supplied usage.
    Persisted delta_v1 rows contain deltas and incremental content fingerprints,
    not repeated cumulative prefixes; final/abort events retain complete text.
    """
    if context_budget not in (None, "sglang"):
        raise ValueError("context_budget must be None or 'sglang'")
    if context_budget is not None and stream_fn is not _model_stream:
        raise ValueError("context budgeting requires the live fenced streaming transport")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Task:\n{task}\n\nBegin."}]
    cmds = []
    timings = []
    stream_events = []
    stream_provenance = copy.deepcopy(getattr(stream_fn, "sfx_stream_provenance", None))
    if stream_provenance is not None and not isinstance(stream_provenance, dict):
        raise ValueError("stream provenance must be a dictionary")
    replay_stream = (stream_provenance is not None
                     and stream_provenance.get("kind") == "recorded_stream_replay")
    live_stream = stream_fn in (_model_stream, stream_native_bash)
    native_tools = (stream_fn is stream_native_bash or
                    (replay_stream and stream_provenance.get("source_transport") == "native_bash_tools"))
    stop_reason = "max_steps"
    t0 = time.monotonic()

    def result(reason, *, model_error=None, tool_error=None):
        """Build success or partial diagnostics without treating aborts as samples."""
        timing_source = (("live_stream" if live_stream else "injected_stream")
                         if stream_fn is not None else
                         ("live_response" if model_fn is _model_action else "injected_response"))
        if replay_stream:
            timing_source = "recorded_stream_replay"
        complete = model_error is None and tool_error is None
        generation_s = (sum(row["model_end_s"] - row["model_start_s"] for row in timings)
                        if complete else None)
        counts = [row.get("completion_tokens") for row in timings]
        completion_tokens = (sum(counts) if complete and counts
                             and all(type(n) is int for n in counts) else None)
        token_usage_source = "unavailable"
        if completion_tokens is not None:
            token_usage_source = ("recorded_usage" if replay_stream else
                                  "endpoint_usage" if live_stream else "injected_usage")
        record = {
            "wall_s": round(time.monotonic() - t0, 3), "commands": cmds,
            "steps": len(cmds), "stop_reason": reason,
            "timing_clock": "monotonic_elapsed_seconds", "timings": timings,
            "model": model, "temperature": temperature, "seed": seed, "max_tokens": max_tokens,
            "context_budget": context_budget,
            "timing_source": timing_source, "stream_events": stream_events,
            "stream_event_format": "delta_v1",
            "transport": "native_bash_tools" if native_tools else "fenced_bash_text",
            "stream_provenance": stream_provenance,
            "model_latency_semantics": (
                "Observed request latency, including stream-reader and callback overhead before "
                "completion; not isolated GPU decoding. stream_callback_s also includes the "
                "model_end callback, which runs after model_end_s."),
            "token_usage_source": token_usage_source,
            "completion_tokens": completion_tokens,
            "mean_generation_latency_s": (generation_s / len(timings)
                                           if complete and timings else None),
            "mean_request_s_per_completion_token": (generation_s / completion_tokens
                                                    if completion_tokens else None),
        }
        if model_error is not None:
            record["model_error"] = model_error
        if tool_error is not None:
            record["tool_error"] = tool_error
        return record

    def emit(event, model_step, **fields):
        row = {"event": event, "model_step": model_step,
               "elapsed_s": time.monotonic() - t0, **fields}
        if (stream_fn is not None or on_stream_event is not None
                or event in {"model_abort", "tool_start", "tool_end", "tool_abort"}):
            stream_events.append({key: value for key, value in row.items()
                                  if not (event == "model_delta" and key == "text")})
        if on_stream_event is not None:
            callback_start = time.monotonic()
            try:
                # Same process, different callback thread: share the exact clock
                # origin internally without persisting machine-local timestamps.
                on_stream_event({**row, "_monotonic_origin": t0})
            finally:
                if stream_fn is not None and event.startswith("model_"):
                    timing["stream_callback_s"] += time.monotonic() - callback_start

    for model_step in range(max_steps):
        model_start_s = time.monotonic() - t0
        timing = {"model_step": model_step, "command_index": None,
                  "model_start_s": model_start_s,
                  "model_end_s": None,
                  "tool_start_s": None, "tool_end_s": None}
        if stream_fn is not None:
            timing["stream_callback_s"] = 0.0
        text, stream = "", None
        text_digest = hashlib.sha256()
        try:
            emit("model_start", model_step)
            if stream_fn is None:
                if model_fn is _model_action:
                    text = model_fn(base_url, model, api_key, messages,
                                    temperature=temperature, seed=seed, max_tokens=max_tokens)
                else:
                    # Existing injected model functions keep their four-argument API.
                    text = model_fn(base_url, model, api_key, messages)
            else:
                timing.update({"first_chunk_s": None, "first_token_chunk_s": None,
                               "first_content_s": None, "first_reasoning_s": None,
                               "usage": None, "completion_tokens": None,
                               "finish_reason": None})
                timing["stream_request_start_s"] = time.monotonic() - t0
                stream_config = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}
                if context_budget is not None:
                    stream_config["context_budget"] = context_budget
                timing["stream_complete"] = False
                stream = iter(stream_fn(base_url, model, api_key, messages, **stream_config))
                for event in stream:
                    if not isinstance(event, dict) or timing["finish_reason"] is not None:
                        raise ModelStreamError("Invalid event or data after stream completion")
                    kind = event.get("type")
                    if kind == "context_budget":
                        if (context_budget is None or "context_budget" in timing
                                or timing["first_chunk_s"] is not None or timing["usage"] is not None):
                            raise ModelStreamError("Unexpected or late context-budget event")
                        timing["context_budget"] = _record_context_budget(event, max_tokens)
                        timing["context_budget_s"] = time.monotonic() - t0 - timing["stream_request_start_s"]
                        emit("model_context_budget", model_step,
                             budget=dict(timing["context_budget"]), raw_stream_event=copy.deepcopy(event))
                    elif kind == "delta":
                        if context_budget is not None and "context_budget" not in timing:
                            raise ModelStreamError("Missing context budget before model output")
                        if timing.get("context_budget", {}).get("effective_max_tokens") == 0:
                            raise ModelStreamError("Model output arrived after context exhaustion")
                        delta, reasoning = event.get("content", ""), event.get("reasoning")
                        if not isinstance(delta, str) or (reasoning is not None
                                                        and not isinstance(reasoning, str)):
                            raise ModelStreamError("Model deltas must be text")
                        now = time.monotonic() - t0
                        if timing["first_chunk_s"] is None:
                            timing["first_chunk_s"] = now
                        if (delta or reasoning) and timing["first_token_chunk_s"] is None:
                            timing["first_token_chunk_s"] = now
                        if delta and timing["first_content_s"] is None:
                            timing["first_content_s"] = now
                        if reasoning and timing["first_reasoning_s"] is None:
                            timing["first_reasoning_s"] = now
                        text += delta
                        text_digest.update(delta.encode("utf-8", errors="surrogatepass"))
                        emit("model_delta", model_step, text=text, delta=delta,
                             text_chars=len(text), text_sha256=text_digest.hexdigest(),
                             reasoning_delta=reasoning, raw_stream_event=dict(event))
                    elif kind == "usage":
                        timing["usage"] = dict(_usage(event.get("usage")))
                        budget = timing.get("context_budget")
                        if budget is not None:
                            if (timing["usage"].get("prompt_tokens") != budget["prompt_tokens"]
                                    or type(timing["usage"].get("completion_tokens")) is not int
                                    or timing["usage"]["completion_tokens"] > budget["effective_max_tokens"]):
                                raise ModelStreamError("Actual model usage differs from its context budget")
                        timing["completion_tokens"] = timing["usage"].get("completion_tokens")
                        emit("model_usage", model_step, usage=dict(timing["usage"]),
                             raw_stream_event=dict(event))
                    elif kind == "done":
                        reason = event.get("finish_reason")
                        if not isinstance(reason, str) or not reason:
                            raise ModelStreamError("Stream completion needs a finish reason")
                        timing["finish_reason"] = reason
                    else:
                        raise ModelStreamError(f"Unknown model stream event: {kind!r}")
                timing["stream_complete"] = timing["finish_reason"] is not None
                allowed_finishes = {"stop", "tool_calls"} if native_tools else {"stop"}
                if context_budget is not None and "context_budget" not in timing:
                    raise ModelStreamError("Model stream did not supply its context budget")
                if timing.get("context_budget", {}).get("effective_max_tokens") == 0:
                    raise ModelStreamError("Model stream completed after context exhaustion")
                if timing["finish_reason"] not in allowed_finishes:
                    raise ModelStreamError(
                        f"Model generation did not finish normally: {timing['finish_reason']!r}")
                if hasattr(stream, "close"):
                    stream.close()
                stream = None
            if not isinstance(text, str):
                raise ModelStreamError("Model content must be text")
            timing["model_end_s"] = time.monotonic() - t0
            if stream_fn is not None:
                timing["generation_latency_s"] = timing["model_end_s"] - model_start_s
            cmd = (parse_native_action(text) if native_tools else
                   None if text.strip().startswith("DONE") else _parse_action(text))
            emit("model_end", model_step, text=text, command=cmd,
                 finish_reason=timing.get("finish_reason"))
        except BaseException as exc:
            # The original failure remains authoritative, even if closing the
            # generator or cancelling its speculative reservation also fails.
            if stream is not None:
                closing, stream = stream, None
                try:
                    if hasattr(closing, "close"):
                        closing.close()
                except BaseException as cleanup_exc:
                    exc.add_note(f"model stream close also failed: {cleanup_exc!r}")
            timing["model_abort_s"] = time.monotonic() - t0
            timings.append(timing)
            error = {"model_step": model_step, "error_type": type(exc).__name__,
                     "error": str(exc)}
            if isinstance(exc, ContextBudgetExceeded):
                error["context_budget"] = _record_context_budget(
                    {"type": "context_budget", "budget": exc.context_budget}, max_tokens)
            try:
                emit("model_abort", model_step, text=text,
                     error_type=type(exc).__name__, error=str(exc))
            except BaseException as cleanup_exc:
                exc.add_note(f"model_abort callback also failed: {cleanup_exc!r}")
            try:
                exc.sfx_live_trajectory = result("model_error", model_error=error)
            except Exception as diagnostic_exc:
                exc.add_note(f"could not attach partial trajectory: {diagnostic_exc!r}")
            raise
        timings.append(timing)
        messages.append({"role": "assistant", "content": text})
        if text.strip().startswith("DONE"):
            stop_reason = "done"
            break
        if cmd is None:
            stop_reason = "unparseable"
            break
        timing["command_index"] = len(cmds)
        cmds.append(cmd)
        timing["tool_start_s"] = time.monotonic() - t0
        phase = "tool_start"
        try:
            emit("tool_start", model_step, command=cmd, command_index=timing["command_index"])
            phase = "execute"
            out = exec_fn(cmd)
            timing["tool_end_s"] = time.monotonic() - t0
            phase = "tool_end"
            emit("tool_end", model_step, command=cmd, command_index=timing["command_index"])
            phase = "observation"
            messages.append({"role": "user", "content": f"Output:\n{out[:4000]}"})
        except BaseException as exc:
            timing["tool_abort_s"] = time.monotonic() - t0
            error = {"model_step": model_step, "command_index": timing["command_index"],
                     "command": cmd, "phase": phase,
                     "error_type": type(exc).__name__, "error": str(exc)}
            try:
                emit("tool_abort", model_step, **{key: value for key, value in error.items()
                                                 if key != "model_step"})
            except BaseException as cleanup_exc:
                exc.add_note(f"tool_abort callback also failed: {cleanup_exc!r}")
            try:
                exc.sfx_live_trajectory = result("tool_error", tool_error=error)
            except Exception as diagnostic_exc:
                exc.add_note(f"could not attach partial trajectory: {diagnostic_exc!r}")
            raise
    return result(stop_reason)
