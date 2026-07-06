# photon/core/contract.py
"""The seam between the Router (selection) and the Fabric (execution). Defined
first so the two subsystems can be built independently and so the Tier-2 GPU
serving engine can drop in behind ServingBackend later without touching either."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RouteTarget(BaseModel):
    """What the Router decides: which base model, which adapter (if any), and the
    decode params to run with."""

    model_id: str
    adapter_id: str | None = None
    decode_params: dict = Field(default_factory=dict)


@runtime_checkable
class RoutingEngine(Protocol):
    """Owned by the Router. Given extracted features, choose a RouteTarget.
    The learned engine and the static fallback both satisfy this."""

    def decide(self, features: "object") -> RouteTarget: ...


@runtime_checkable
class ServingBackend(Protocol):
    """Owned by the Fabric. Execute a RouteTarget. MockServingBackend and
    VLLMServingBackend implement it now; the dense GPU engine implements it later."""

    async def generate(self, target: RouteTarget, payload: dict) -> tuple[dict, float]: ...
