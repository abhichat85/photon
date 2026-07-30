# tests/india/test_indic_routing.py
"""The arbitrage, end to end: a Hindi request routes to the model that is
cheapest PER CHARACTER, even though it is dearer per token."""
import httpx
import respx

from photon.config import PhotonConfig
from photon.india.routing import IndicAwareRouter
from photon.india.script import Script
from photon.india.token_economics import TokenEfficiencyStore
from photon.router.static import AUTO_MODEL, StaticRouter
from tests.test_config import VALID

CHAT_RESPONSE = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ठीक है"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
HINDI = [{"role": "user", "content": "मुझे जानकारी चाहिए"}]
ENGLISH = [{"role": "user", "content": "I need information"}]


def _router(tmp_path, *, measure=True):
    """VALID fleet: 'big' = 0.9/1M (dear per token), 'small' = 0.08/1M (cheap).
    We measure 'big' as Indic-efficient and 'small' as Indic-hostile."""
    cfg = PhotonConfig.model_validate(VALID)
    store = TokenEfficiencyStore(tmp_path / "te.db")
    if measure:
        # big: 4.0 chars/token on Devanagari → 250 tok/1k chars @0.9 = 2.25e-4
        store.record("big", Script.DEVANAGARI, chars=400, tokens=100)
        # small: 0.25 chars/token (shreds Devanagari) → 4000 tok/1k @0.08 = 3.2e-4
        store.record("small", Script.DEVANAGARI, chars=100, tokens=400)
        # both fine on Latin; small is cheaper per token AND per char there
        store.record("big", Script.LATIN, chars=400, tokens=100)
        store.record("small", Script.LATIN, chars=400, tokens=100)
    return IndicAwareRouter(store, StaticRouter(cfg), cfg), store


def test_hindi_routes_to_indic_efficient_model_despite_higher_token_price(tmp_path):
    router, _ = _router(tmp_path)
    d = router.resolve(AUTO_MODEL, messages=HINDI)
    assert d.backend.name == "big"          # dearer per token, cheaper per character
    assert d.reason == "indic-devanagari"


def test_english_is_left_to_the_static_path(tmp_path):
    # indic_only=True: no override where token pricing isn't structurally unfair
    router, _ = _router(tmp_path)
    d = router.resolve(AUTO_MODEL, messages=ENGLISH)
    assert d.reason == "default"


def test_unmeasured_script_falls_back_to_static(tmp_path):
    router, _ = _router(tmp_path, measure=False)
    d = router.resolve(AUTO_MODEL, messages=HINDI)
    assert d.reason == "default"  # never guesses a ratio it hasn't measured


def test_alias_direct_and_pin_are_never_overridden(tmp_path):
    router, _ = _router(tmp_path)
    assert router.resolve("praxiom-chat", messages=HINDI).reason == "alias"
    assert router.resolve("small", messages=HINDI).reason == "direct"
    assert router.resolve(AUTO_MODEL, allow_canary=False, messages=HINDI).reason == "default"


def test_missing_messages_falls_back_to_static(tmp_path):
    router, _ = _router(tmp_path)
    assert router.resolve(AUTO_MODEL).reason == "default"


@respx.mock
def test_live_request_routes_by_script(tmp_path):
    """Installed as app.state.router, a Hindi request is actually SERVED by the
    Indic-efficient backend through the unchanged OpenAI API."""
    from fastapi.testclient import TestClient

    from photon.api.app import create_app

    cfg = PhotonConfig.model_validate(VALID)
    app = create_app(config=cfg, db_path=str(tmp_path / "t.db"),
                     registry_db=str(tmp_path / "r.db"), shadow_db=str(tmp_path / "s.db"),
                     tokeff_db=str(tmp_path / "te.db"))
    store = app.state.token_efficiency
    store.record("big", Script.DEVANAGARI, chars=400, tokens=100)
    store.record("small", Script.DEVANAGARI, chars=100, tokens=400)
    app.state.router = IndicAwareRouter(store, StaticRouter(cfg), cfg)

    respx.post("http://big.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions",
                   json={"model": "photon-auto", "messages": HINDI},
                   headers={"X-Photon-Tenant": "bharat"})
    assert r.status_code == 200
    assert r.headers["X-Photon-Backend"] == "big"  # script-aware choice served
