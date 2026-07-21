# tests/test_api_chat.py
import json

import httpx
import pytest
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
def test_auto_routes_to_default_backend(client):
    route = respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-Photon-Tenant": "praxiom"},
    )
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"
    assert "X-Photon-Request-Id" in r.headers
    # in-process decision cost surfaced for the §9 selection-overhead benchmark
    assert float(r.headers["X-Photon-Decision-Ms"]) >= 0.0
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "qwen-72b"


@respx.mock
def test_telemetry_records_cost_and_shadow(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-Photon-Tenant": "praxiom"},
    )
    costs = client.get("/photon/v1/costs", params={"tenant": "praxiom"}).json()
    (big,) = costs["backends"]
    assert big["routed_backend"] == "big"
    assert big["requests"] == 1
    assert big["cost_usd"] == pytest.approx((10 * 0.9 + 5 * 0.9) / 1_000_000)
    # short prompt → shadow candidate "small" logged with its estimated cost
    assert big["shadow_est_cost_usd"] == pytest.approx((10 * 0.08 + 5 * 0.08) / 1_000_000)


def test_unknown_model_is_404(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-nonexistent", "messages": []},
    )
    assert r.status_code == 404


@respx.mock
def test_backend_failure_is_502_and_recorded(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Photon-Tenant": "praxiom"},
    )
    assert r.status_code == 502
    decisions = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "praxiom"}
    ).json()["decisions"]
    assert decisions[0]["status"] == "error"


def test_models_endpoint_lists_aliases_and_backends(client):
    r = client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert {"photon-auto", "praxiom-chat", "big", "small"} <= ids
