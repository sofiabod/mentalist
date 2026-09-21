"""Opt-in, process-local exact chat budgeting for the pinned SGLang 0.5.8 server.

Launch with ``python -m eval.sglang_budget <ordinary SGLang arguments>``.
Only authenticated loopback HTTP, one tokenizer worker, and text-only base
models are supported. No installed package files are changed. Counting reuses
the serving process's real chat renderer; it never enqueues generation. A
receipt is rechecked against the actual rendered generation request before it
can reach the tokenizer manager/queue.
"""
import copy
import functools
import hashlib
import hmac
import importlib.metadata
import inspect
import ipaddress
import json
import os
from pathlib import Path
import sys
import uuid


PROTOCOL = "sfx-chat-token-budget-v1"
ROUTE = "/sfx/chat-token-count"
HEADER = "X-SFX-Context-Budget"
VERSION = "0.5.8"
RENDERER_SHA256 = "1410f5c9ed763751b59c00b332208e1cca669dcc18e4e52b317f333ec957988d"
TOKENIZER_MANAGER_SHA256 = "e3df12f41a837883e42d1540b5d61b995dd7eea8afd859e6deb41f77d9b5e355"
_RECEIPT_FIELDS = {"protocol", "server_instance_id", "renderer_sha256", "model",
                   "prompt_tokens", "prompt_sha256", "reserved_tokens", "context_length",
                   "safety_margin_tokens"}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _authorize(raw_request, manager):
    client = getattr(raw_request, "client", None)
    if client is None or not _loopback(client.host):
        raise ValueError("SFX budget endpoint requires a loopback client")
    key = getattr(manager.server_args, "api_key", None)
    supplied = raw_request.headers.get("authorization", "")
    if not isinstance(key, str) or not key or not hmac.compare_digest(
            supplied.encode(), ("Bearer " + key).encode()):
        raise ValueError("SFX budget endpoint requires server API-key authentication")


def _count_adapted(serving, request, adapted, instance_id):
    manager = serving.tokenizer_manager
    ids = adapted.input_ids
    if (manager.model_config.is_multimodal or getattr(adapted, "lora_path", None)
            or request.model != manager.served_model_name
            or getattr(manager.server_args, "tokenizer_worker_num", None) != 1
            or getattr(manager.server_args, "allow_auto_truncate", None) is not False
            or manager.validate_total_tokens is not True
            or type(ids) is not list or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("SFX budget counting requires the supported text-only serving configuration")
    context, reserved = manager.context_len, manager.num_reserved_tokens
    if (type(context) is not int or context <= 1 or type(reserved) is not int
            or reserved < 0 or reserved >= context):
        raise ValueError("SFX budget server limits are invalid")
    return {"protocol": PROTOCOL, "server_instance_id": instance_id,
            "renderer_sha256": RENDERER_SHA256, "model": request.model,
            "prompt_tokens": len(ids), "prompt_sha256": canonical_sha256(ids),
            "reserved_tokens": reserved, "context_length": context,
            "safety_margin_tokens": 1}


def _guard_conversion(original, instance_id):
    @functools.wraps(original)
    def convert(serving, request, raw_request=None):
        receipt = raw_request.headers.get(HEADER) if raw_request is not None else None
        if receipt is not None:
            _authorize(raw_request, serving.tokenizer_manager)
            if len(receipt) > 2048:
                raise ValueError("SFX context budget receipt is invalid")
            try:
                receipt = json.loads(receipt)
            except (TypeError, ValueError):
                raise ValueError("SFX context budget receipt is invalid") from None
            if (type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS | {"max_tokens"}
                    or type(receipt["max_tokens"]) is not int or receipt["max_tokens"] < 1):
                raise ValueError("SFX context budget receipt is invalid")
        adapted, normalized = original(serving, request, raw_request)
        if receipt is not None:
            actual = _count_adapted(serving, normalized, adapted, instance_id)
            # Exact field types matter: bool must not compare equal to integer 1.
            if any(type(receipt[key]) is not type(value) or receipt[key] != value
                   for key, value in actual.items()):
                raise ValueError("SFX context budget receipt no longer matches the serving request")
            maximum = adapted.sampling_params.get("max_new_tokens")
            available = actual["context_length"] - actual["prompt_tokens"] - actual["reserved_tokens"] - 1
            if type(maximum) is not int or maximum != receipt["max_tokens"] or maximum > available:
                raise ValueError("SFX context budget receipt no longer matches the serving request")
        return adapted, normalized
    return convert


def _make_endpoint(request_type, instance_id):
    from fastapi.responses import JSONResponse

    async def count(raw_request):
        serving = raw_request.app.state.openai_serving_chat
        try:
            _authorize(raw_request, serving.tokenizer_manager)
        except ValueError:
            return JSONResponse({"error": {"type": "AuthenticationError",
                                            "message": "SFX budget authorization failed"}}, status_code=403)
        try:
            if raw_request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
                raise ValueError("json required")
            payload = await raw_request.json()
            if type(payload) is not dict or HEADER in raw_request.headers:
                raise ValueError("invalid count request")
            request = request_type.model_validate(copy.deepcopy(payload))
            if (serving.tokenizer_manager.model_config.is_multimodal
                    or request.model != serving.tokenizer_manager.served_model_name
                    or any(message.content is not None and not isinstance(message.content, str)
                           for message in request.messages)
                    or getattr(request, "lora_path", None)
                    or getattr(request, "n", 1) not in (None, 1)):
                raise ValueError("unsupported count request")
            if serving._validate_request(request):
                raise ValueError("invalid chat request")
            adapted, normalized = serving._convert_to_internal_request(request, raw_request)
            result = _count_adapted(serving, normalized, adapted, instance_id)
            result["request_sha256"] = canonical_sha256(payload)
            return JSONResponse(result)
        except Exception:
            # Validation/render errors can echo prompts or credentials. Never
            # return or log those exception strings from this private endpoint.
            return JSONResponse({"error": {"type": "BadRequestError",
                                            "message": "SFX exact chat budgeting is unavailable for this request"}},
                                status_code=400)
    return count


def install():
    """Install only into this process, failing before patching on source drift."""
    if importlib.metadata.version("sglang") != VERSION:
        raise RuntimeError("SFX budgeting requires pinned SGLang 0.5.8")
    from fastapi import Request
    from sglang.srt.entrypoints import http_server
    from sglang.srt.entrypoints.openai import serving_chat
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.managers import tokenizer_manager

    for module, expected in ((serving_chat, RENDERER_SHA256),
                             (tokenizer_manager, TOKENIZER_MANAGER_SHA256)):
        if hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest() != expected:
            raise RuntimeError("SFX budgeting installed serving source changed")
    if any(getattr(route, "path", None) == ROUTE for route in http_server.app.routes):
        raise RuntimeError("SFX budgeting is already installed")
    instance_id = uuid.uuid4().hex
    cls = serving_chat.OpenAIServingChat
    original = cls._convert_to_internal_request
    endpoint = _make_endpoint(ChatCompletionRequest, instance_id)
    endpoint.__annotations__["raw_request"] = Request
    http_server.app.add_api_route(ROUTE, endpoint, methods=["POST"], include_in_schema=False)
    cls._convert_to_internal_request = _guard_conversion(original, instance_id)
    return {"protocol": PROTOCOL, "server_instance_id": instance_id,
            "renderer_sha256": RENDERER_SHA256, "sglang_version": VERSION}


def main():
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    if (not _loopback(args.host) or not args.api_key or args.tokenizer_worker_num != 1
            or args.grpc_mode or args.encoder_only or args.allow_auto_truncate):
        raise RuntimeError("SFX budgeting requires authenticated loopback HTTP, one tokenizer worker, and no auto-truncation")
    print(json.dumps({"sfx_context_budget_extension": install()}, sort_keys=True), flush=True)
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
