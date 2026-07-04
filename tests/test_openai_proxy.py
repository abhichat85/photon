import httpx
import pytest
import respx

from photon.backends.openai_proxy import BackendError, OpenAIProxy
from photon.config import BackendConfig, ModelPricing

BACKEND = BackendConfig(
    name="big",
    base_url="http://big.test/v1",
    model="qwen-72b",
    pricing=ModelPricing(input_per_1m=0.9, output_per_1m=0.9),
)

OK_RESPONSE = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "model": "qwen-72b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@respx.mock
async def test_proxy_overrides_model_and_returns_json():
    route = respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_RESPONSE)
    )
    async with httpx.AsyncClient() as client:
        proxy = OpenAIProxy(client)
        response, latency_ms = await proxy.chat_completions(
            BACKEND, {"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert response["usage"]["prompt_tokens"] == 10
    assert latency_ms > 0
    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "qwen-72b"  # gateway model name replaced with backend's


@respx.mock
async def test_upstream_4xx_5xx_raises_backend_error():
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    async with httpx.AsyncClient() as client:
        proxy = OpenAIProxy(client)
        with pytest.raises(BackendError):
            await proxy.chat_completions(BACKEND, {"messages": []})


@respx.mock
async def test_connection_error_raises_backend_error():
    respx.post("http://big.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    async with httpx.AsyncClient() as client:
        proxy = OpenAIProxy(client)
        with pytest.raises(BackendError):
            await proxy.chat_completions(BACKEND, {"messages": []})
