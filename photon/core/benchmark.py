# photon/core/benchmark.py
"""The §9 benchmark harness — turns the spec's Fabric/Router DoD numbers from
claims into measurements. Pure math over latency/cost samples here (fully
unit-tested); scripts/benchmark.py drives real load against a live gateway and
feeds results into these functions. The point: 'benchmark precedes the number,
always' (parent spec anti-goal #7) is now executable, not aspirational."""
from __future__ import annotations

from pydantic import BaseModel


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0,1]). Empty → 0.0."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if p <= 0:
        return ordered[0]
    if p >= 1:
        return ordered[-1]
    import math

    rank = math.ceil(p * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


class RoutingTaxResult(BaseModel):
    baseline_p95_ms: float
    routed_p95_ms: float
    tax_pct: float          # extra latency added by the routing layer
    within_target: bool     # tax <= target (default 15%, spec §9)


def routing_tax(
    baseline_latencies_ms: list[float],
    routed_latencies_ms: list[float],
    target_pct: float = 15.0,
) -> RoutingTaxResult:
    """The 'routing tax must be near-zero when routing isn't used' DoD: how much
    p95 latency the gateway adds vs hitting the backend directly."""
    base = percentile(baseline_latencies_ms, 0.95)
    routed = percentile(routed_latencies_ms, 0.95)
    tax = ((routed - base) / base * 100.0) if base > 0 else 0.0
    return RoutingTaxResult(
        baseline_p95_ms=base, routed_p95_ms=routed, tax_pct=tax,
        within_target=tax <= target_pct,
    )


class SelectionOverheadResult(BaseModel):
    p95_ms: float
    within_target: bool  # <= target (default 3ms, spec §9 Fabric DoD)


def selection_overhead(overhead_samples_ms: list[float], target_ms: float = 3.0) -> SelectionOverheadResult:
    """Adapter/model selection overhead — the router's own decision cost."""
    p95 = percentile(overhead_samples_ms, 0.95)
    return SelectionOverheadResult(p95_ms=p95, within_target=p95 <= target_ms)


class CostReductionResult(BaseModel):
    baseline_cost_usd: float
    measured_cost_usd: float
    reduction_factor: float   # baseline / measured
    within_target: bool       # factor >= target (default 3x, spec §9 Phase 0)


def cost_reduction(
    baseline_cost_usd: float, measured_cost_usd: float, target_factor: float = 3.0
) -> CostReductionResult:
    factor = (baseline_cost_usd / measured_cost_usd) if measured_cost_usd > 0 else 0.0
    return CostReductionResult(
        baseline_cost_usd=baseline_cost_usd, measured_cost_usd=measured_cost_usd,
        reduction_factor=factor, within_target=factor >= target_factor,
    )


class BenchmarkReport(BaseModel):
    routing_tax: RoutingTaxResult
    selection_overhead: SelectionOverheadResult
    cost_reduction: CostReductionResult
    all_targets_met: bool


def build_report(
    baseline_latencies_ms: list[float],
    routed_latencies_ms: list[float],
    selection_overhead_ms: list[float],
    baseline_cost_usd: float,
    measured_cost_usd: float,
) -> BenchmarkReport:
    tax = routing_tax(baseline_latencies_ms, routed_latencies_ms)
    overhead = selection_overhead(selection_overhead_ms)
    cost = cost_reduction(baseline_cost_usd, measured_cost_usd)
    return BenchmarkReport(
        routing_tax=tax, selection_overhead=overhead, cost_reduction=cost,
        all_targets_met=tax.within_target and overhead.within_target and cost.within_target,
    )
