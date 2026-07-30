# photon/india/inr.py
"""Rupee-native cost accounting.

An Indian enterprise budgets, approves and audits in ₹ with GST — not in USD.
Reporting inference cost in dollars pushes the conversion (and the tax
treatment) onto the customer's finance team and makes the numbers unusable in a
procurement conversation. This module makes ₹ a first-class output.

Two disciplines, both deliberate:
- **A rate must carry provenance.** `InrRate` requires a `source`. An FX rate
  without one is a guess with a decimal point, and cost reports built on it are
  unauditable.
- **GPU cost is DERIVED, never quoted.** `gpu_cost_per_1m_tokens_inr` takes the
  instance's ₹/hour and its MEASURED tokens/second. Vendor throughput claims are
  not inputs. This is the project's 'benchmark precedes the number' rule applied
  to unit economics — and it is what makes cheap Indian GPU capacity legible as
  a per-token price you can quote a customer."""
from __future__ import annotations

from pydantic import BaseModel, Field

# GST on SaaS / cloud / online information services in India. Overridable —
# rates change and classification can differ; this is the common case, not law.
GST_RATE_SERVICES = 0.18


class InrRate(BaseModel):
    """USD→INR with mandatory provenance. Set from your own reference (RBI
    reference rate, your bank's card rate, your cloud invoice's applied rate)."""

    usd_to_inr: float = Field(gt=0)
    source: str = Field(min_length=1)  # e.g. "RBI reference rate 2026-07-01"


def to_inr(usd: float, rate: InrRate) -> float:
    return usd * rate.usd_to_inr


def with_gst(net_inr: float, rate: float = GST_RATE_SERVICES) -> tuple[float, float, float]:
    """(net, gst, gross) — the three numbers an Indian invoice must show
    separately. Input-tax-credit eligibility depends on the split being visible."""
    gst = net_inr * rate
    return net_inr, gst, net_inr + gst


def format_inr(amount: float) -> str:
    """Indian digit grouping (₹12,34,567.00): the last three digits, then
    two-digit groups (thousand → lakh → crore). Western grouping reads as
    wrong to an Indian finance reader."""
    negative = amount < 0
    whole, frac = divmod(round(abs(amount) * 100), 100)
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    out = f"₹{s}.{frac:02d}"
    return f"-{out}" if negative else out


def gpu_cost_per_1m_tokens_inr(inr_per_hour: float, tokens_per_second: float) -> float | None:
    """₹ per 1M tokens, derived from what the instance costs and what it was
    MEASURED to produce. Returns None for non-positive throughput — an
    unmeasured box has no cost per token, and inventing one would be the exact
    failure this codebase refuses elsewhere."""
    if tokens_per_second <= 0 or inr_per_hour < 0:
        return None
    tokens_per_hour = tokens_per_second * 3600.0
    return inr_per_hour / (tokens_per_hour / 1_000_000.0)
