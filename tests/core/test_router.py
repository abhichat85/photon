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
