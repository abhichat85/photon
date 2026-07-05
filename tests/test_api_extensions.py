# tests/test_api_extensions.py
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

COMPLETION_RESPONSE = {
    "id": "cmpl-2",
    "object": "text_completion",
    "model": "qwen-72b",
    "choices": [{"index": 0, "text": "hello world", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}


# --- photon request-extension block ---

@respx.mock
def test_photon_block_stripped_before_upstream(client):
    route = respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "photon-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "photon": {"route": "auto", "quality_bar": 0.8, "latency_slo_ms": 500, "budget": 0.01},
        },
        headers={"X-Photon-Tenant": "px"},
    )
    assert r.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert "photon" not in sent  # never forwarded to vLLM


@respx.mock
def test_photon_fields_recorded_to_telemetry(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client.post(
        "/v1/chat/completions",
        json={
            "model": "photon-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "photon": {"route": "pin", "quality_bar": 0.8, "latency_slo_ms": 500, "budget": 0.01},
        },
        headers={"X-Photon-Tenant": "pxrec"},
    )
    d = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "pxrec"}
    ).json()["decisions"][0]
    assert d["route_mode"] == "pin"
    assert d["quality_bar"] == 0.8
    assert d["latency_slo_ms"] == 500
    assert d["budget"] == 0.01


def test_photon_cascade_rejected_at_ops(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [], "photon": {"route": "cascade"}},
    )
    assert r.status_code == 400
    assert "Core" in r.json()["detail"]


def test_photon_invalid_route_is_422(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [], "photon": {"route": "sideways"}},
    )
    assert r.status_code == 422


def test_photon_block_must_be_object(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "photon-auto", "messages": [], "photon": "not-an-object"},
    )
    assert r.status_code == 422


@respx.mock
def test_route_pin_disables_canary(tmp_path):
    # config with a canary weighted 1.0 (would ALWAYS fire on photon-auto);
    # prove route:pin bypasses it and hits the default backend instead.
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from photon.config import PhotonConfig
    from tests.test_config import VALID

    canary_cfg = PhotonConfig.model_validate(
        {**VALID, "routing": {**VALID["routing"], "canary": {"backend": "small", "weight": 1.0}}}
    )
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    respx.post("http://small.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    app = create_app(
        config=canary_cfg,
        db_path=str(tmp_path / "t.db"),
        registry_db=str(tmp_path / "r.db"),
    )
    with TestClient(app) as c:
        pinned = c.post(
            "/v1/chat/completions",
            json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}],
                  "photon": {"route": "pin"}},
        )
        assert pinned.headers["X-Photon-Backend"] == "big"  # canary bypassed
        auto = c.post(
            "/v1/chat/completions",
            json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert auto.headers["X-Photon-Backend"] == "small"  # canary fires without pin


# --- /v1/completions ---

@respx.mock
def test_completions_routes_and_records(client):
    route = respx.post("http://big.test/v1/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION_RESPONSE)
    )
    r = client.post(
        "/v1/completions",
        json={"model": "photon-auto", "prompt": "say hi"},
        headers={"X-Photon-Tenant": "cmp"},
    )
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "qwen-72b"
    d = client.get("/photon/v1/costs", params={"tenant": "cmp"}).json()["backends"]
    assert d[0]["requests"] == 1


def test_completions_streaming_rejected(client):
    r = client.post(
        "/v1/completions",
        json={"model": "photon-auto", "prompt": "hi", "stream": True},
    )
    assert r.status_code == 400


@respx.mock
def test_completions_backend_error_is_502(client):
    respx.post("http://big.test/v1/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    r = client.post(
        "/v1/completions",
        json={"model": "photon-auto", "prompt": "hi"},
        headers={"X-Photon-Tenant": "cmperr"},
    )
    assert r.status_code == 502
    d = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "cmperr"}
    ).json()["decisions"]
    assert d[0]["status"] == "error"


# --- fleet/status ---

def test_fleet_status_static(client):
    body = client.get("/photon/v1/fleet/status").json()
    names = {b["name"]: b for b in body["backends"]}
    assert names["small"]["quantization"] is None or "quantization" in names["small"]
    assert body["routing"]["default_backend"] == "big"
    assert body["routing"]["shadow_enabled"] is True
    # no probe → no reachable key
    assert "reachable" not in names["big"]


@respx.mock
def test_fleet_status_probe(client):
    respx.get("http://big.test/v1/models").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://small.test/v1/models").mock(side_effect=httpx.ConnectError("down"))
    body = client.get("/photon/v1/fleet/status", params={"probe": "true"}).json()
    names = {b["name"]: b for b in body["backends"]}
    assert names["big"]["reachable"] is True
    assert names["small"]["reachable"] is False


# --- POST /photon/v1/adapters ---

def test_register_adapter_via_http(client):
    r = client.post(
        "/photon/v1/adapters",
        json={"name": "praxiom-intent", "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
              "adapter_path": "/adapters/v1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "praxiom-intent"
    assert body["version"] == 1
    assert body["status"] == "draft"
