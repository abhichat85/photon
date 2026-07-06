# tests/core/test_shadow_integration.py
import httpx
import respx

from photon.config import PhotonConfig
from photon.core.contract import RouteTarget
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, ShadowRouter

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class StubPolicy(PolicyModel):
    def predict_acceptable(self, features):
        return 0.99  # would always route cheap


@respx.mock
def test_shadow_router_logs_but_does_not_change_backend(tmp_path):
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from tests.test_config import VALID

    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    cfg = PhotonConfig.model_validate(VALID)
    app = create_app(config=cfg, db_path=str(tmp_path / "t.db"), registry_db=str(tmp_path / "r.db"))
    # inject a shadow router that would always pick "cheap"
    logged = []
    learned = LearnedRouter(StubPolicy(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    app.state.shadow_router = ShadowRouter(learned, sink=logged.append)

    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
                   headers={"X-Photon-Tenant": "px"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # UNCHANGED by shadow
    assert len(logged) == 1
    assert logged[0].actual_backend == "big"
    assert logged[0].would_route.model_id == "cheap"


def test_no_shadow_router_is_a_noop(client):
    # default app has no shadow_router set → request path works normally
    import respx as _respx
    import httpx as _httpx
    with _respx.mock:
        _respx.post("http://big.test/v1/chat/completions").mock(return_value=_httpx.Response(200, json=CHAT_RESPONSE))
        r = client.post("/v1/chat/completions",
                        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
