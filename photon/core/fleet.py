"""Fabric control-plane: turn a declarative FleetSpec into a PlacementPlan
(which adapters are resident vs. paged) using pin-priority then insertion order.
This is the placement LOGIC — the IP — independent of the GPU execution that
enacts it (that's the Tier-2 ServingBackend). Advisory in C0, load-bearing once
the dense engine lands."""
from __future__ import annotations

from pydantic import BaseModel


class AdapterSpec(BaseModel):
    name: str
    base: str
    pinned: bool = False


class FleetSpec(BaseModel):
    base_models: list[str]
    adapters: list[AdapterSpec]
    slot_capacity: int  # total resident slots (bases + adapters) on the pool


class PlacementPlan(BaseModel):
    resident_bases: list[str]
    resident_adapters: list[str]
    paged_adapters: list[str]


class FleetManager:
    def plan(self, spec: FleetSpec) -> PlacementPlan:
        bases = set(spec.base_models)
        for a in spec.adapters:
            if a.base not in bases:
                raise ValueError(f"adapter {a.name!r} references unknown base {a.base!r}")

        # bases are always resident; adapters fill remaining slots, pinned first
        adapter_slots = max(0, spec.slot_capacity - len(spec.base_models))
        ordered = sorted(
            spec.adapters, key=lambda a: (not a.pinned,)  # pinned (False sorts first)
        )
        resident = [a.name for a in ordered[:adapter_slots]]
        paged = [a.name for a in ordered[adapter_slots:]]
        return PlacementPlan(
            resident_bases=list(spec.base_models),
            resident_adapters=resident,
            paged_adapters=paged,
        )
