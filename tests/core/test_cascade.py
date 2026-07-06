# tests/core/test_cascade.py
from photon.core.cascade import CascadeController, EscalationSignals
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class StubPolicy(PolicyModel):
    def __init__(self, score: float):
        super().__init__()
        self._score = score

    def predict_acceptable(self, features):
        return self._score


CHEAP = RouteTarget(model_id="qwen-1.5b")
BIG = RouteTarget(model_id="qwen-72b")


def _feat():
    return RequestFeatures(prompt_chars=10, message_count=1)


def test_high_confidence_routes_cheap():
    c = CascadeController(StubPolicy(0.9), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat())
    assert d.target.model_id == "qwen-1.5b"
    assert d.reason == "policy-cheap"


def test_low_confidence_routes_big():
    c = CascadeController(StubPolicy(0.2), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat())
    assert d.target.model_id == "qwen-72b"
    assert d.reason == "policy-escalate"


def test_critic_score_forces_escalation_even_if_policy_confident():
    c = CascadeController(StubPolicy(0.95), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(critic_score=0.1, critic_floor=0.5))
    assert d.target.model_id == "qwen-72b"
    assert d.reason == "critic-escalate"


def test_schema_failure_forces_escalation():
    c = CascadeController(StubPolicy(0.95), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(schema_valid=False))
    assert d.reason == "schema-escalate"


def test_budget_exhausted_forces_cheap():
    c = CascadeController(StubPolicy(0.2), threshold=0.6, cheap=CHEAP, big=BIG)
    d = c.decide(_feat(), signals=EscalationSignals(budget_remaining=0.0))
    assert d.target.model_id == "qwen-1.5b"
    assert d.reason == "budget-cheap"
