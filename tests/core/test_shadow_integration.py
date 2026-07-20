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


@respx.mock
def test_shadow_router_observes_completions_path_too(tmp_path):
    # the /v1/completions legacy path must also feed the shadow study (no blind spot)
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from tests.test_config import VALID

    completion_response = {
        "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    respx.post("http://big.test/v1/completions").mock(return_value=httpx.Response(200, json=completion_response))
    app = create_app(config=PhotonConfig.model_validate(VALID),
                     db_path=str(tmp_path / "t.db"), registry_db=str(tmp_path / "r.db"))
    logged = []
    learned = LearnedRouter(StubPolicy(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    app.state.shadow_router = ShadowRouter(learned, sink=logged.append)

    with TestClient(app) as c:
        r = c.post("/v1/completions", json={"model": "photon-auto", "prompt": "hi"},
                   headers={"X-Photon-Tenant": "px"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # served backend unchanged
    assert len(logged) == 1
    assert logged[0].would_route.model_id == "cheap"  # counterfactual logged


@respx.mock
def test_shadow_decisions_persist_and_surface_on_admin_endpoint(tmp_path):
    # end-to-end: shadow router → durable ShadowDecisionStore → GET /photon/v1/shadow/decisions
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from tests.test_config import VALID

    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    app = create_app(config=PhotonConfig.model_validate(VALID),
                     db_path=str(tmp_path / "t.db"), registry_db=str(tmp_path / "r.db"),
                     shadow_db=str(tmp_path / "s.db"))
    learned = LearnedRouter(StubPolicy(), 0.6, RouteTarget(model_id="cheap"), RouteTarget(model_id="big"))
    app.state.shadow_router = ShadowRouter(learned, sink=app.state.shadow_store.insert)

    with TestClient(app) as c:
        c.post("/v1/chat/completions",
               json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]})
        body = c.get("/photon/v1/shadow/decisions").json()
    assert body["summary"]["total"] == 1
    assert body["summary"]["would_route_counts"] == {"cheap": 1}
    assert body["decisions"][0]["actual_backend"] == "big"
    assert body["decisions"][0]["would_model"] == "cheap"


@respx.mock
def test_shadow_features_carry_real_tenant_history(tmp_path):
    # the history feature is computed from actual telemetry, not the 0.0 default
    from fastapi.testclient import TestClient

    from photon.api.app import create_app
    from tests.test_config import VALID

    respx.post("http://big.test/v1/chat/completions").mock(return_value=httpx.Response(200, json=CHAT_RESPONSE))
    app = create_app(config=PhotonConfig.model_validate(VALID),
                     db_path=str(tmp_path / "t.db"), registry_db=str(tmp_path / "r.db"),
                     shadow_db=str(tmp_path / "s.db"))

    seen_features = []

    class Recorder:  # duck-typed shadow: capture the features the chat path builds
        def observe(self, actual_backend_name, features, request_id):
            seen_features.append(features)
            return actual_backend_name

    app.state.shadow_router = Recorder()
    with TestClient(app) as c:
        # request 1: no history yet → rate 0.0; it lands an "ok" telemetry row
        c.post("/v1/chat/completions",
               json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
               headers={"X-Photon-Tenant": "hist"})
        # request 2: history now exists → rate reflects the prior ok
        c.post("/v1/chat/completions",
               json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
               headers={"X-Photon-Tenant": "hist"})
    assert seen_features[0].tenant_recent_accept_rate == 0.0
    assert seen_features[1].tenant_recent_accept_rate == 1.0


def test_no_shadow_router_is_a_noop(client):
    # default app has no shadow_router set → request path works normally
    import respx as _respx
    import httpx as _httpx
    with _respx.mock:
        _respx.post("http://big.test/v1/chat/completions").mock(return_value=_httpx.Response(200, json=CHAT_RESPONSE))
        r = client.post("/v1/chat/completions",
                        json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
