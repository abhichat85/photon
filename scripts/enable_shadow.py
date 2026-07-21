# scripts/enable_shadow.py
"""Wire the learned router into shadow mode on a running app (Stage 3 of
deploy/PRAXIOM_CUTOVER.md). Log-only: it records counterfactual decisions to the
durable shadow store and never changes which backend serves a request. This is
the switch that starts collecting the Tier-3 data; going LIVE is a separate,
later, gated act (install a LearnedRoutingAdapter as app.state.router)."""
from __future__ import annotations

from photon.core.contract import RouteTarget
from photon.core.policy import PolicyModel
from photon.core.router import LearnedRouter, ShadowRouter


def install_shadow_router(
    app,
    policy: PolicyModel | None = None,
    threshold: float = 0.6,
    cheap_model: str | None = None,
    big_model: str | None = None,
) -> ShadowRouter:
    """Attach a ShadowRouter to `app`, sinking decisions into the app's durable
    shadow store. With no policy provided, an untrained one is used — it fails
    closed (would never route cheap), which is the safe default until a policy
    is trained on collected data. cheap/big default to the fleet's two
    cheapest/most-expensive backends by input price."""
    backends = sorted(app.state.config.backends, key=lambda b: b.pricing.input_per_1m)
    cheap = cheap_model or backends[0].name
    big = big_model or backends[-1].name

    learned = LearnedRouter(
        policy or PolicyModel(),
        threshold,
        cheap=RouteTarget(model_id=cheap),
        big=RouteTarget(model_id=big),
    )
    shadow = ShadowRouter(learned, sink=app.state.shadow_store.insert)
    app.state.shadow_router = shadow
    return shadow
