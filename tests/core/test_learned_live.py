# tests/core/test_learned_live.py
"""End-to-end proof of the §3 live-swap: when a LearnedRoutingAdapter is
installed as app.state.router, learned decisions actually serve traffic —
through the unchanged /v1/chat/completions API — and route:pin still bypasses
the learned engine. create_app never installs this by default (Tier-3 gate)."""
import httpx
import respx

from photon.config import PhotonConfig
from photon.core.contract import RouteTarget
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, LearnedRoutingAdapter

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class ConfidentPolicy(PolicyModel):
    def predict_acceptable(self, features):
        return 0.99


def _app(tmp_path):
    from photon.api.app import create_app
    from photon.router.static import StaticRouter
    from tests.test_config import VALID

    cfg = PhotonConfig.model_validate(VALID)
    app = create_app(config=cfg, db_path=str(tmp_path / "t.db"),
                     registry_db=str(tmp_path / "r.db"), shadow_db=str(tmp_path / "s.db"))
    learned = LearnedRouter(
        ConfidentPolicy(), 0.6,
        cheap=RouteTarget(model_id="small"), big=RouteTarget(model_id="big"),
    )
    app.state.router = LearnedRoutingAdapter(learned, StaticRouter(cfg), cfg)
    return app


@respx.mock
def test_learned_router_live_serves_learned_choice(tmp_path):
    from fastapi.testclient import TestClient

    respx.post("http://small.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    with TestClient(_app(tmp_path)) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}]},
                   headers={"X-Photon-Tenant": "live"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "small"  # learned choice actually served


@respx.mock
def test_learned_router_live_pin_still_bypasses(tmp_path):
    from fastapi.testclient import TestClient

    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    with TestClient(_app(tmp_path)) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "photon-auto", "messages": [{"role": "user", "content": "hi"}],
                         "photon": {"route": "pin"}},
                   headers={"X-Photon-Tenant": "live"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # pin = audit mode wins over learned
