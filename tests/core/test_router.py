# tests/core/test_router.py
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, ShadowDecision, ShadowRouter


class StubPolicy(PolicyModel):
    def __init__(self, score):
        super().__init__()
        self._score = score

    def predict_acceptable(self, features):
        return self._score


def _feat(chars=10):
    return RequestFeatures(prompt_chars=chars, message_count=1)


def test_learned_router_decides_cheap_when_confident():
    r = LearnedRouter(
        policy=StubPolicy(0.9), threshold=0.6,
        cheap=RouteTarget(model_id="cheap"), big=RouteTarget(model_id="big"),
    )
    assert r.decide(_feat()).model_id == "cheap"


def test_learned_adapter_is_dropin_for_resolve():
    # §3's promise, exercised: the learned engine behind the SAME resolve()
    # interface. AUTO with features → learned decision mapped to a real backend;
    # alias/direct stay static; reasons are prefixed for the audit log.
    from photon.config import PhotonConfig
    from photon.core.router import LearnedRoutingAdapter
    from tests.test_config import VALID

    cfg = PhotonConfig.model_validate(VALID)
    from photon.router.static import AUTO_MODEL, StaticRouter

    static = StaticRouter(cfg)
    learned = LearnedRouter(
        StubPolicy(0.99), 0.6,
        cheap=RouteTarget(model_id="small"), big=RouteTarget(model_id="big"),
    )
    adapter = LearnedRoutingAdapter(learned, static, cfg)
    assert adapter.wants_features is True

    d = adapter.resolve(AUTO_MODEL, features=_feat())
    assert d.backend.name == "small"
    assert d.reason == "learned-policy-cheap"
    # alias and direct requests are never learned-routed
    assert adapter.resolve("praxiom-chat", features=_feat()).reason == "alias"
    assert adapter.resolve("small", features=_feat()).reason == "direct"


def test_learned_adapter_fails_safe_to_static():
    # pin (allow_canary=False) and missing features both fall back to the
    # static default — the learned path can never be the only path.
    from photon.config import PhotonConfig
    from photon.core.router import LearnedRoutingAdapter
    from photon.router.static import AUTO_MODEL, StaticRouter
    from tests.test_config import VALID

    cfg = PhotonConfig.model_validate(VALID)
    learned = LearnedRouter(
        StubPolicy(0.99), 0.6,
        cheap=RouteTarget(model_id="small"), big=RouteTarget(model_id="big"),
    )
    adapter = LearnedRoutingAdapter(learned, StaticRouter(cfg), cfg)
    pinned = adapter.resolve(AUTO_MODEL, allow_canary=False, features=_feat())
    assert pinned.backend.name == "big" and pinned.reason == "default"
    featureless = adapter.resolve(AUTO_MODEL)  # no features → static
    assert featureless.backend.name == "big" and featureless.reason == "default"


def test_shadow_router_returns_actual_but_logs_counterfactual():
    logged = []
    learned = LearnedRouter(
        policy=StubPolicy(0.9), threshold=0.6,
        cheap=RouteTarget(model_id="cheap"), big=RouteTarget(model_id="big"),
    )
    shadow = ShadowRouter(learned, sink=logged.append)
    # the actual served backend is whatever Ops chose ("big"); shadow must return it unchanged
    served = shadow.observe(actual_backend_name="big", features=_feat(), request_id="r1")
    assert served == "big"  # NEVER overridden in shadow mode
    assert len(logged) == 1
    d: ShadowDecision = logged[0]
    assert d.request_id == "r1"
    assert d.actual_backend == "big"
    assert d.would_route.model_id == "cheap"  # counterfactual only
