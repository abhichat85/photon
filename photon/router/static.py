# photon/router/static.py
import random

from pydantic import BaseModel

from photon.config import BackendConfig, PhotonConfig

AUTO_MODEL = "photon-auto"


class RouteDecision(BaseModel):
    backend: BackendConfig
    reason: str  # "default" | "alias" | "direct"


class UnknownModelError(KeyError):
    pass


class StaticRouter:
    """Phase 0 router: config-driven static resolution. The learned router
    (Photon Core) replaces this class behind the same resolve() interface."""

    def __init__(self, config: PhotonConfig, rng: random.Random | None = None):
        self._config = config
        self._rng = rng or random.Random()

    def resolve(self, requested_model: str) -> RouteDecision:
        routing = self._config.routing
        if requested_model == AUTO_MODEL:
            canary = routing.canary
            if canary is not None and self._rng.random() < canary.weight:
                return RouteDecision(
                    backend=self._config.backend(canary.backend), reason="canary"
                )
            return RouteDecision(
                backend=self._config.backend(routing.default_backend), reason="default"
            )
        if requested_model in routing.aliases:
            return RouteDecision(
                backend=self._config.backend(routing.aliases[requested_model]), reason="alias"
            )
        for b in self._config.backends:
            if b.name == requested_model or b.model == requested_model:
                return RouteDecision(backend=b, reason="direct")
        raise UnknownModelError(requested_model)
