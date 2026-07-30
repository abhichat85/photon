# photon/india/routing.py
"""Language-aware routing — where the tokenizer arbitrage becomes a decision.

Drop-in for `StaticRouter.resolve()` (same seam as `LearnedRoutingAdapter`).
For `photon-auto` requests it detects the dominant script and picks the backend
with the lowest measured cost-per-1,000-characters FOR THAT SCRIPT — which is
not the same as the lowest cost per token. A model whose tokenizer handles
Devanagari well can be cheaper for Hindi while being dearer per token; per-token
routing picks wrong on exactly the traffic an Indian product serves most.

Fail-safe, matching every other router in this codebase:
- alias / direct / `route: pin` / featureless requests → static path untouched
- a script with no measurement on any backend → static path (never a guess)
- a chosen backend missing from config → static path

DEFAULT OFF. `create_app` installs the static router; enabling this is a
deliberate act, and it should follow the same shadow → canary → full discipline
as the learned router (DECISIONS D11)."""
from __future__ import annotations

from photon.india.script import Script, messages_script
from photon.india.token_economics import TokenEfficiencyStore, cheapest_backend_for_script


class IndicAwareRouter:
    """Routes on measured cost-per-character per script."""

    wants_features = True  # chat handlers pass extracted features when set

    def __init__(self, store: TokenEfficiencyStore, static, config, indic_only: bool = True):
        self._store = store
        self._static = static
        self._config = config
        # indic_only: only override for Indic scripts, where the mis-pricing is
        # structural. Latin traffic keeps the existing (already sane) path.
        self._indic_only = indic_only

    def resolve(self, requested_model: str, allow_canary: bool = True, features=None, messages=None):
        from photon.router.static import AUTO_MODEL, RouteDecision

        if requested_model != AUTO_MODEL or not allow_canary or messages is None:
            return self._static.resolve(requested_model, allow_canary)

        script = messages_script(messages)
        if script is Script.UNKNOWN or (self._indic_only and not script.is_indic):
            return self._static.resolve(requested_model, allow_canary)

        pricing = {b.name: b.pricing for b in self._config.backends}
        choice = cheapest_backend_for_script(self._store, pricing, script)
        if choice is None:
            return self._static.resolve(requested_model, allow_canary)
        try:
            backend = self._config.backend(choice)
        except KeyError:
            return self._static.resolve(requested_model, allow_canary)
        return RouteDecision(backend=backend, reason=f"indic-{script.value}")
