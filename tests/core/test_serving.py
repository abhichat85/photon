# tests/core/test_serving.py
import httpx
import respx

from photon.config import BackendConfig, ModelPricing
from photon.core.contract import RouteTarget, ServingBackend
from photon.core.serving import MockServingBackend, VLLMServingBackend

RESP = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


async def test_mock_backend_is_a_serving_backend():
    m = MockServingBackend(canned={"cheap": {"ok": True}})
    assert isinstance(m, ServingBackend)
    resp, latency = await m.generate(RouteTarget(model_id="cheap"), {"messages": []})
    assert resp == {"ok": True}
    assert latency >= 0.0


async def test_mock_records_calls():
    m = MockServingBackend(canned={"cheap": {"ok": True}})
    await m.generate(RouteTarget(model_id="cheap", adapter_id="legal-v3"), {"messages": []})
    assert m.calls[0].model_id == "cheap"
    assert m.calls[0].adapter_id == "legal-v3"


@respx.mock
async def test_vllm_backend_posts_and_injects_adapter_as_model():
    backend = BackendConfig(name="small", base_url="http://s.test/v1", model="qwen-1.5b",
                            pricing=ModelPricing(input_per_1m=0.08, output_per_1m=0.08))
    route = respx.post("http://s.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=RESP))
    async with httpx.AsyncClient() as client:
        b = VLLMServingBackend({"small": backend}, client)
        # adapter_id becomes the served model name (vLLM serves LoRA modules by name)
        resp, latency = await b.generate(
            RouteTarget(model_id="small", adapter_id="legal-v3"), {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert resp["usage"]["prompt_tokens"] == 3
    import json
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "legal-v3"  # adapter name wins when present
