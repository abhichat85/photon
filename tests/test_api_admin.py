# tests/test_api_admin.py
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
def test_costs_empty_for_unknown_tenant(client):
    r = client.get("/photon/v1/costs", params={"tenant": "nobody"})
    assert r.status_code == 200
    assert r.json() == {"tenant": "nobody", "since_ts": 0.0, "backends": []}


@respx.mock
def test_decisions_returns_audit_fields(client):
    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client.post(
        "/v1/chat/completions",
        json={"model": "praxiom-chat", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Photon-Tenant": "praxiom"},
    )
    decisions = client.get(
        "/photon/v1/routing/decisions", params={"tenant": "praxiom", "limit": 5}
    ).json()["decisions"]
    assert len(decisions) == 1
    d = decisions[0]
    assert d["requested_model"] == "praxiom-chat"
    assert d["routed_backend"] == "big"
    assert d["backend_model"] == "qwen-72b"
    assert d["status"] == "ok"
