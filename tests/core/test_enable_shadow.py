# tests/core/test_enable_shadow.py
import httpx
import respx

from photon.core.router import ShadowRouter

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def test_install_picks_cheapest_and_priciest_and_attaches(client):
    from scripts.enable_shadow import install_shadow_router

    shadow = install_shadow_router(client.app)
    assert isinstance(shadow, ShadowRouter)
    assert client.app.state.shadow_router is shadow
    # VALID fleet: small (0.08) is cheap, big (0.9) is big
    learned = shadow._learned
    assert learned._cascade._cheap.model_id == "small"
    assert learned._cascade._big.model_id == "big"


@respx.mock
def test_installed_shadow_logs_to_durable_store_but_serves_actual(client):
    from scripts.enable_shadow import install_shadow_router

    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    install_shadow_router(client.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Photon-Tenant": "cut"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # untrained policy → served backend unchanged
    rows = client.app.state.shadow_store.recent(10)
    assert len(rows) == 1
    assert rows[0]["actual_backend"] == "big"
