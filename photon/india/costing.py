# photon/india/costing.py
"""Turn Indian GPU capacity into a quotable ₹ price per 1M tokens.

The India margin thesis is: cheap in-country GPU capacity + language-fair
routing = a per-token price an Indian buyer can actually afford, at a gross
margin that survives Indian price points. That thesis is only real once the
₹/hour on your quote is divided by throughput you MEASURED. This module does
that arithmetic and refuses to do it on placeholders."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from photon.india.inr import InrRate, gpu_cost_per_1m_tokens_inr, with_gst


class ProviderSpec(BaseModel):
    name: str
    region: str = ""
    data_residency: str = ""
    gpu: str = ""
    inr_per_hour: float = 0.0
    measured_tokens_per_second: float = 0.0
    notes: str = ""

    @property
    def is_priced(self) -> bool:
        """A provider is only usable once BOTH a real cost and a real measured
        throughput exist. Placeholders (0.0) are excluded from every comparison
        rather than silently producing a fictional price."""
        return self.inr_per_hour > 0 and self.measured_tokens_per_second > 0


class ProviderBook(BaseModel):
    fx: dict = Field(default_factory=dict)
    gst_rate: float = 0.18
    providers: list[ProviderSpec] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProviderBook":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def rate(self) -> InrRate | None:
        usd_to_inr = self.fx.get("usd_to_inr", 0.0)
        source = self.fx.get("source", "")
        if usd_to_inr > 0 and source and not source.startswith("SET ME"):
            return InrRate(usd_to_inr=usd_to_inr, source=source)
        return None


class ProviderCost(BaseModel):
    name: str
    region: str
    data_residency: str
    inr_per_1m_tokens: float
    inr_per_1m_tokens_with_gst: float


def cost_table(book: ProviderBook) -> list[ProviderCost]:
    """₹ per 1M tokens for every FULLY-PRICED provider, cheapest first.
    Unpriced (placeholder) entries are omitted — not defaulted."""
    rows: list[ProviderCost] = []
    for p in book.providers:
        if not p.is_priced:
            continue
        net = gpu_cost_per_1m_tokens_inr(p.inr_per_hour, p.measured_tokens_per_second)
        if net is None:
            continue
        _, _, gross = with_gst(net, book.gst_rate)
        rows.append(ProviderCost(
            name=p.name, region=p.region, data_residency=p.data_residency,
            inr_per_1m_tokens=net, inr_per_1m_tokens_with_gst=gross,
        ))
    return sorted(rows, key=lambda r: r.inr_per_1m_tokens)


def rent_vs_own_breakeven_hours(
    rent_inr_per_hour: float, own_inr_per_hour_amortised: float
) -> float | None:
    """Hours of monthly utilisation beyond which owning beats renting, as a
    share of the 730-hour month. Returns None when owning is never cheaper
    (or inputs are unusable) — an honest 'no', not a misleading number."""
    if rent_inr_per_hour <= 0 or own_inr_per_hour_amortised <= 0:
        return None
    if own_inr_per_hour_amortised >= rent_inr_per_hour:
        return None
    # Owning bills all 730 hours regardless of use; renting bills only usage.
    # Own wins when: own_rate * 730 <= rent_rate * used_hours
    return (own_inr_per_hour_amortised * 730.0) / rent_inr_per_hour
