# tests/core/test_fleet.py
import pytest

from photon.core.fleet import AdapterSpec, FleetManager, FleetSpec


def _spec(capacity):
    return FleetSpec(
        base_models=["qwen-1.5b", "qwen-14b"],
        adapters=[
            AdapterSpec(name="legal-v3", base="qwen-1.5b", pinned=True),
            AdapterSpec(name="fin-v2", base="qwen-1.5b", pinned=False),
            AdapterSpec(name="health-v1", base="qwen-14b", pinned=False),
        ],
        slot_capacity=capacity,
    )


def test_plan_places_all_when_capacity_allows():
    plan = FleetManager().plan(_spec(capacity=5))
    assert set(plan.resident_adapters) == {"legal-v3", "fin-v2", "health-v1"}
    assert plan.paged_adapters == []


def test_pinned_adapters_always_resident_under_pressure():
    # only 1 adapter slot beyond the 2 base models → pinned wins
    plan = FleetManager().plan(_spec(capacity=3))
    assert "legal-v3" in plan.resident_adapters  # pinned
    assert set(plan.paged_adapters)  # something got paged out
    assert "legal-v3" not in plan.paged_adapters


def test_plan_rejects_adapter_referencing_unknown_base():
    bad = FleetSpec(
        base_models=["qwen-1.5b"],
        adapters=[AdapterSpec(name="x", base="nonexistent", pinned=False)],
        slot_capacity=5,
    )
    with pytest.raises(ValueError, match="unknown base"):
        FleetManager().plan(bad)
