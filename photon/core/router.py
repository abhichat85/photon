# photon/core/router.py
"""LearnedRouter: a RoutingEngine (contract.py) driven by the policy + cascade.
ShadowRouter: wraps it for C0's shadow mode — it OBSERVES each request, logs the
counterfactual RouteTarget the learned engine WOULD have chosen, and returns the
actual backend Ops already selected, unchanged. There is deliberately no code
path from ShadowRouter to the served response: shadow mode cannot alter traffic.
Going live (Phase C1) means swapping ShadowRouter.observe for a decide that Ops
honors — gated on the Tier-3 validation."""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from photon.core.cascade import CascadeController, EscalationSignals
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel


class LearnedRouter:
    def __init__(self, policy: PolicyModel, threshold: float, cheap: RouteTarget, big: RouteTarget):
        self._cascade = CascadeController(policy, threshold, cheap, big)

    def decide(self, features: RequestFeatures, signals: EscalationSignals | None = None) -> RouteTarget:
        return self._cascade.decide(features, signals).target


class ShadowDecision(BaseModel):
    request_id: str
    actual_backend: str
    would_route: RouteTarget
    reason: str


class ShadowRouter:
    def __init__(self, learned: LearnedRouter, sink: Callable[[ShadowDecision], None]):
        self._learned = learned
        self._sink = sink

    def observe(self, actual_backend_name: str, features: RequestFeatures, request_id: str) -> str:
        """Log what the learned router WOULD have chosen; return the actual
        backend unchanged. This is the whole of C0's routing behavior."""
        decision = self._learned._cascade.decide(features)
        self._sink(
            ShadowDecision(
                request_id=request_id,
                actual_backend=actual_backend_name,
                would_route=decision.target,
                reason=decision.reason,
            )
        )
        return actual_backend_name
