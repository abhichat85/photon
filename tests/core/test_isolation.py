# tests/core/test_isolation.py
"""Adversarial tenant-isolation checks (spec §9 'zero cross-tenant leakage,
tested adversarially'). Guarantees Photon actually enforces at its layer:
per-tenant telemetry/cost/history queries are tenant-scoped and never return
another tenant's rows; a request cannot reach a backend/adapter outside the
configured fleet; and the shadow store persists NO tenant-identifying or prompt
content (so it can't be an exfil surface). Note the shadow store is a global
operator-only plane, not tenant-partitioned — its safety is 'stores nothing
sensitive', not 'partitions per tenant'. GPU-level memory isolation between
LoRA adapters is the serving engine's job (Tier-2), asserted there."""
import httpx
import respx

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "secret-for-A"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@respx.mock
def test_cost_and_decisions_never_leak_across_tenants(client):
    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    # tenant A makes a request
    client.post("/v1/chat/completions",
                json={"model": "photon-auto", "messages": [{"role": "user", "content": "A secret"}]},
                headers={"X-Photon-Tenant": "tenant-A"})
    # tenant B queries their own cost + decisions — must see NOTHING of A
    b_costs = client.get("/photon/v1/costs", params={"tenant": "tenant-B"}).json()
    b_decisions = client.get("/photon/v1/routing/decisions", params={"tenant": "tenant-B"}).json()
    assert b_costs["backends"] == []
    assert b_decisions["decisions"] == []
    # positive controls: A's own cost + decision ARE recorded (so the empty-B
    # assertions can't pass vacuously from broken recording)
    a_costs = client.get("/photon/v1/costs", params={"tenant": "tenant-A"}).json()
    assert a_costs["backends"] and a_costs["backends"][0]["requests"] == 1
    a_decisions = client.get("/photon/v1/routing/decisions", params={"tenant": "tenant-A"}).json()
    assert len(a_decisions["decisions"]) == 1


@respx.mock
def test_history_feature_is_tenant_scoped(client):
    # tenant A's success history must not raise tenant B's accept-rate feature
    store = client.app.state.store
    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    for _ in range(3):
        client.post("/v1/chat/completions",
                    json={"model": "photon-auto", "messages": [{"role": "user", "content": "x"}]},
                    headers={"X-Photon-Tenant": "tenant-A"})
    assert store.recent_ok_rate("tenant-A") == 1.0
    assert store.recent_ok_rate("tenant-B") == 0.0  # B has no history, sees none of A's


def test_unknown_model_cannot_reach_any_backend(client):
    # a request naming a model outside the fleet is refused (404), never proxied
    with respx.mock:
        route = respx.post("http://big.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_RESPONSE))
        r = client.post("/v1/chat/completions",
                        json={"model": "someone-elses-private-model", "messages": []})
    assert r.status_code == 404
    assert route.call_count == 0  # nothing was forwarded upstream


def test_fleet_rejects_adapter_on_unconfigured_base(client):
    # you cannot register an adapter against a base that isn't in the fleet
    r = client.post("/photon/v1/fleet", json={
        "base_models": ["qwen-72b"],
        "adapters": [{"name": "intruder", "base": "not-in-fleet", "pinned": True}],
        "slot_capacity": 3,
    })
    assert r.status_code == 422


@respx.mock
def test_shadow_decisions_carry_no_prompt_content(client):
    # the shadow store must persist ROUTING metadata only — never prompt text,
    # so it can't become a cross-tenant data-exfil surface
    from photon.core.contract import RouteTarget
    from photon.core.policy import PolicyModel
    from photon.core.router import LearnedRouter, ShadowRouter

    class Stub(PolicyModel):
        def predict_acceptable(self, features):
            return 0.99

    learned = LearnedRouter(Stub(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    client.app.state.shadow_router = ShadowRouter(learned, sink=client.app.state.shadow_store.insert)
    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    client.post("/v1/chat/completions",
                json={"model": "photon-auto", "messages": [{"role": "user", "content": "TOP SECRET PROMPT"}]},
                headers={"X-Photon-Tenant": "tenant-A"})
    rows = client.app.state.shadow_store.recent(10)
    assert len(rows) == 1
    serialized = str(rows[0])
    assert "TOP SECRET" not in serialized  # only routing metadata is stored
