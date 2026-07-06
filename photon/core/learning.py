# photon/core/learning.py
"""Offline replay harness: take labeled historical rows (features + whether the
cheap model was acceptable + the two costs), train the policy, then evaluate the
resulting router against the replay — reporting router-regret and cost savings
vs. an always-big baseline. This is what runs nightly on real telemetry; here it
runs on any list of LabeledRow so it is fully testable on synthetic data."""
from __future__ import annotations

from pydantic import BaseModel, Field

from photon.core.cascade import CascadeController
from photon.core.contract import RouteTarget
from photon.core.features import RequestFeatures
from photon.core.policy import PolicyModel
from photon.core.regret import ReplayRow, router_regret


class LabeledRow(BaseModel):
    features: RequestFeatures
    cheap_acceptable: bool
    cheap_cost: float
    big_cost: float


class ReplayReport(BaseModel):
    trained_on: int
    routed_cheap_share: float
    regret: float
    always_big_regret: float
    est_savings_vs_always_big: float
    policy_scores: list[float] = Field(default_factory=list)


class ReplayHarness:
    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold

    def run(self, rows: list[LabeledRow]) -> ReplayReport:
        if not rows:
            return ReplayReport(
                trained_on=0, routed_cheap_share=0.0, regret=0.0,
                always_big_regret=0.0, est_savings_vs_always_big=0.0,
            )
        policy = PolicyModel()
        labels = [int(r.cheap_acceptable) for r in rows]
        # sklearn cannot fit a classifier on a single-class window (all rows
        # acceptable, or none). Real nightly telemetry can produce these. Skip
        # training in that case — the untrained policy fails closed (predicts
        # 0.0), so nothing routes cheap, rather than crashing the nightly job.
        if len(set(labels)) >= 2:
            policy.fit([r.features for r in rows], labels)
        cheap = RouteTarget(model_id="cheap")
        big = RouteTarget(model_id="big")
        controller = CascadeController(policy, self._threshold, cheap, big)

        replay: list[ReplayRow] = []
        always_big: list[ReplayRow] = []
        cheap_count = 0
        chosen_total = 0.0
        big_total = 0.0
        scores: list[float] = []
        for r in rows:
            d = controller.decide(r.features)
            scores.append(d.policy_score)
            oracle = r.cheap_cost if r.cheap_acceptable else r.big_cost
            if d.target.model_id == "cheap":
                cheap_count += 1
                # a cheap route on an unacceptable row "costs" the big price (a retry)
                chosen = r.cheap_cost if r.cheap_acceptable else r.big_cost
            else:
                chosen = r.big_cost
            chosen_total += chosen
            big_total += r.big_cost
            replay.append(ReplayRow(chosen_cost=chosen, cheapest_acceptable_cost=oracle))
            always_big.append(ReplayRow(chosen_cost=r.big_cost, cheapest_acceptable_cost=oracle))

        savings = (big_total - chosen_total) / big_total if big_total else 0.0
        return ReplayReport(
            trained_on=len(rows),
            routed_cheap_share=cheap_count / len(rows),
            regret=router_regret(replay),
            always_big_regret=router_regret(always_big),
            est_savings_vs_always_big=max(0.0, savings),
            policy_scores=scores,
        )
