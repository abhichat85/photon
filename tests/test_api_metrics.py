# tests/test_api_metrics.py
import httpx
import respx

CHAT_RESPONSE = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "model": "qwen-72b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@respx.mock
def test_metrics_exposes_requests_latency_and_cost(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Photon-Tenant": "mtenant"},  # unique tenant → deterministic sample
    )
    text = client.get("/metrics").text
    assert 'photon_requests_total{backend="big",status="ok",tenant="mtenant"} 1.0' in text
    assert 'photon_cost_usd_total{backend="big",tenant="mtenant"}' in text
    assert "photon_request_latency_seconds_bucket" in text
