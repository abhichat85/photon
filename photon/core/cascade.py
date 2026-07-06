# photon/core/cascade.py
"""Default-to-cheap with escalation. Escalation triggers (in priority order):
per-tenant budget exhausted (forces cheap — hard cost cap wins), a failed schema
/ output signal, Praxiom-1's critic score below a floor (the coupling advantage —
no horizontal router has this signal), then the policy model's own confidence."""
from __future__ import annotations

from pydantic import BaseModel

from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class EscalationSignals(BaseModel):
    critic_score: float | None = None       # Praxiom-1 critic output, if available
    critic_floor: float = 0.5
    schema_valid: bool = True                # False = cheap output failed validation
    budget_remaining: float | None = None    # per-tenant $ left; 0 forces cheap


class CascadeDecision(BaseModel):
    target: RouteTarget
    reason: str
    policy_score: float


class CascadeController:
    def __init__(
        self,
        policy: PolicyModel,
        threshold: float,
        cheap: RouteTarget,
        big: RouteTarget,
    ):
        self._policy = policy
        self._threshold = threshold
        self._cheap = cheap
        self._big = big

    def decide(
        self, features: RequestFeatures, signals: EscalationSignals | None = None
    ) -> CascadeDecision:
        signals = signals or EscalationSignals()
        score = self._policy.predict_acceptable(features)

        # hard cost cap wins over everything
        if signals.budget_remaining is not None and signals.budget_remaining <= 0:
            return CascadeDecision(target=self._cheap, reason="budget-cheap", policy_score=score)
        if not signals.schema_valid:
            return CascadeDecision(target=self._big, reason="schema-escalate", policy_score=score)
        if signals.critic_score is not None and signals.critic_score < signals.critic_floor:
            return CascadeDecision(target=self._big, reason="critic-escalate", policy_score=score)
        if score >= self._threshold:
            return CascadeDecision(target=self._cheap, reason="policy-cheap", policy_score=score)
        return CascadeDecision(target=self._big, reason="policy-escalate", policy_score=score)
