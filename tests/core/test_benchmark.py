# tests/core/test_benchmark.py
import pytest

from photon.core.benchmark import (
    build_report,
    cost_reduction,
    percentile,
    routing_tax,
    selection_overhead,
)


def test_percentile_nearest_rank():
    s = [10, 20, 30, 40, 50]
    assert percentile(s, 0.95) == 50
    assert percentile(s, 0.5) == 30
    assert percentile([], 0.95) == 0.0
    assert percentile([7], 0.95) == 7


def test_routing_tax_within_and_over_target():
    # +10% p95 → within 15% target
    ok = routing_tax([100] * 10, [110] * 10)
    assert ok.tax_pct == pytest.approx(10.0)
    assert ok.within_target is True
    # +25% → over target
    bad = routing_tax([100] * 10, [125] * 10)
    assert bad.within_target is False


def test_selection_overhead_target():
    assert selection_overhead([1.0, 2.0, 2.5]).within_target is True
    assert selection_overhead([5.0, 6.0]).within_target is False


def test_cost_reduction_factor():
    r = cost_reduction(baseline_cost_usd=0.030, measured_cost_usd=0.010)
    assert r.reduction_factor == pytest.approx(3.0)
    assert r.within_target is True
    assert cost_reduction(0.030, 0.020).within_target is False  # only 1.5x


def test_build_report_all_targets_met():
    report = build_report(
        baseline_latencies_ms=[100] * 20,
        routed_latencies_ms=[108] * 20,        # +8% tax
        selection_overhead_ms=[1.5] * 20,      # under 3ms
        baseline_cost_usd=0.030,
        measured_cost_usd=0.008,               # 3.75x
    )
    assert report.all_targets_met is True


def test_build_report_flags_a_missed_target():
    report = build_report(
        baseline_latencies_ms=[100] * 20,
        routed_latencies_ms=[100] * 20,
        selection_overhead_ms=[10.0] * 20,     # BLOWS the 3ms overhead target
        baseline_cost_usd=0.030,
        measured_cost_usd=0.008,
    )
    assert report.selection_overhead.within_target is False
    assert report.all_targets_met is False
