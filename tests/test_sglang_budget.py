"""CPU-only tests: no SGLang installation, model, tokenizer download, or server."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

from eval import sglang_budget as budget


class ChatRequest:
    @classmethod
    def model_validate(cls, payload):
        value = copy.deepcopy(payload)
        value["messages"] = [SimpleNamespace(**message) for message in value["messages"]]
        return SimpleNamespace(**value)


def fixture():
    args = SimpleNamespace(api_key="local-secret", tokenizer_worker_num=1, allow_auto_truncate=False)
    manager = SimpleNamespace(server_args=args, served_model_name="test-model",
                              model_config=SimpleNamespace(is_multimodal=False),
                              context_len=100, num_reserved_tokens=3, validate_total_tokens=True)
    calls = []
    class Serving:
        tokenizer_manager = manager
        def _validate_request(self, request):
            return None
        def _convert_to_internal_request(self, request, raw_request=None):
            calls.append(copy.deepcopy(request))
            return SimpleNamespace(input_ids=[10, 20, 30], lora_path=None,
                                   sampling_params={"max_new_tokens": request.max_tokens}), request
    serving = Serving()
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "private task"}],
               "max_tokens": 90, "stream": True, "tools": [{"name": "example"}]}
    async def read_json():
        return copy.deepcopy(payload)
    raw = SimpleNamespace(headers=Headers({"authorization": "Bearer local-secret", "content-type": "application/json"}),
                          client=SimpleNamespace(host="127.0.0.1"), json=read_json,
                          app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=serving)))
    return serving, payload, raw, calls


def test_count_endpoint_reuses_exact_renderer_and_returns_only_counts_hashes():
    serving, payload, raw, calls = fixture()
    endpoint = budget._make_endpoint(ChatRequest, "a" * 32)
    response = asyncio.run(endpoint(raw))
    receipt = json.loads(response.body)
    assert response.status_code == 200
    assert len(calls) == 1 and calls[0].tools == payload["tools"]
    assert receipt["prompt_tokens"] == 3 and receipt["reserved_tokens"] == 3
    assert receipt["prompt_sha256"] == budget.canonical_sha256([10, 20, 30])
    assert receipt["request_sha256"] == budget.canonical_sha256(payload)
    assert "private task" not in response.body.decode() and "local-secret" not in response.body.decode()
    assert set(receipt) == budget._RECEIPT_FIELDS | {"request_sha256"}


@pytest.mark.parametrize("host,key", [("203.0.113.1", "local-secret"),
                                     ("127.0.0.1", "wrong"), ("127.0.0.1", "")])
def test_count_endpoint_requires_both_loopback_and_matching_authentication(host, key):
    _, _, raw, calls = fixture()
    raw.client.host = host
    raw.headers = Headers({"authorization": "Bearer " + key, "content-type": "application/json"})
    response = asyncio.run(budget._make_endpoint(ChatRequest, "a" * 32)(raw))
    assert response.status_code == 403 and calls == []


def test_renderer_error_is_redacted_and_never_exposes_prompt():
    serving, _, raw, _ = fixture()
    def broken(*args):
        raise ValueError("private task Bearer local-secret /private/path")
    serving._convert_to_internal_request = broken
    response = asyncio.run(budget._make_endpoint(ChatRequest, "a" * 32)(raw))
    assert response.status_code == 400
    for value in ("private task", "local-secret", "/private/path"):
        assert value not in response.body.decode()


@pytest.mark.parametrize("change", [
    lambda s, p: setattr(s.tokenizer_manager.model_config, "is_multimodal", True),
    lambda s, p: p.update(model="different-model"),
    lambda s, p: p.update(n=2),
    lambda s, p: p.update(lora_path="adapter"),
    lambda s, p: p["messages"][0].update(content=[{"type": "image_url"}]),
])
def test_unsupported_rendering_modes_fail_closed_before_conversion(change):
    serving, payload, raw, calls = fixture()
    change(serving, payload)
    response = asyncio.run(budget._make_endpoint(ChatRequest, "a" * 32)(raw))
    assert response.status_code == 400 and calls == []


def receipt_fixture():
    serving, payload, raw, calls = fixture()
    request = ChatRequest.model_validate(payload)
    original = type(serving)._convert_to_internal_request
    adapted, _ = original(serving, request, raw)
    receipt = budget._count_adapted(serving, request, adapted, "a" * 32)
    receipt["max_tokens"] = payload["max_tokens"]
    calls.clear()
    return serving, request, raw, calls, original, receipt


def test_generation_guard_rechecks_actual_prompt_before_queueing():
    serving, request, raw, calls, original, receipt = receipt_fixture()
    raw.headers = Headers({"authorization": "Bearer local-secret", budget.HEADER: json.dumps(receipt)})
    guarded = budget._guard_conversion(original, "a" * 32)
    adapted, _ = guarded(serving, request, raw)
    assert adapted.sampling_params["max_new_tokens"] == 90 and len(calls) == 1


@pytest.mark.parametrize("change", [
    lambda r: r.update(server_instance_id="b" * 32),
    lambda r: r.update(prompt_sha256="b" * 64),
    lambda r: r.update(prompt_tokens=4),
    lambda r: r.update(context_length=101),
    lambda r: r.update(reserved_tokens=0),
    lambda r: r.update(safety_margin_tokens=True),
    lambda r: r.update(max_tokens=91),
    lambda r: r.update(extra="unused"),
])
def test_changed_instance_template_or_request_receipt_fails_before_generation(change):
    serving, request, raw, _, original, receipt = receipt_fixture()
    change(receipt)
    raw.headers = Headers({"authorization": "Bearer local-secret", budget.HEADER: json.dumps(receipt)})
    with pytest.raises(ValueError, match="SFX context budget receipt"):
        budget._guard_conversion(original, "a" * 32)(serving, request, raw)


def test_same_length_changed_tokenization_also_fails():
    serving, request, raw, _, original, receipt = receipt_fixture()
    raw.headers = Headers({"authorization": "Bearer local-secret", budget.HEADER: json.dumps(receipt)})
    def changed(*args):
        adapted, normalized = original(*args)
        adapted.input_ids = [10, 20, 31]
        return adapted, normalized
    with pytest.raises(ValueError, match="no longer matches"):
        budget._guard_conversion(changed, "a" * 32)(serving, request, raw)


def test_guard_rejects_oversized_completion_even_with_matching_receipt():
    serving, request, raw, _, original, receipt = receipt_fixture()
    request.max_tokens = receipt["max_tokens"] = 94  # 3 prompt + 3 reserved + 94 == context.
    raw.headers = Headers({"authorization": "Bearer local-secret", budget.HEADER: json.dumps(receipt)})
    with pytest.raises(ValueError, match="no longer matches"):
        budget._guard_conversion(original, "a" * 32)(serving, request, raw)


def test_legacy_requests_without_opt_in_are_unchanged():
    serving, request, raw, _, original, _ = receipt_fixture()
    request.max_tokens = 1000
    adapted, _ = budget._guard_conversion(original, "a" * 32)(serving, request, raw)
    assert adapted.sampling_params["max_new_tokens"] == 1000


def test_unsupported_version_fails_before_importing_or_patching_sglang(monkeypatch):
    monkeypatch.setattr(budget.importlib.metadata, "version", lambda name: "0.5.9")
    with pytest.raises(RuntimeError, match="pinned"):
        budget.install()
