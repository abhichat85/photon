# tests/test_drift.py
from photon.evals.golden import CaseResult
from photon.evals.runner import EvalReport
from scripts.drift_check import compute_pass_rate


def _report(passed: int, total: int) -> EvalReport:
    results = [
        CaseResult(case_id=f"c{i}", passed=(i < passed), failures=[], latency_ms=1.0)
        for i in range(total)
    ]
    return EvalReport(golden_set="t", total=total, passed=passed, results=results)


def test_pass_rate_all_pass():
    assert compute_pass_rate(_report(5, 5)) == 1.0


def test_pass_rate_partial():
    assert compute_pass_rate(_report(3, 4)) == 0.75


def test_pass_rate_empty_is_zero():
    assert compute_pass_rate(_report(0, 0)) == 0.0


def test_golden_pass_rate_gauge_registered():
    # the gauge the drift alert queries must exist in the metrics module
    from photon.api import metrics

    assert hasattr(metrics, "GOLDEN_PASS_RATE")
    metrics.set_golden_pass_rate("praxiom-core", 0.9)  # must not raise
    text = _metrics_text()
    assert 'photon_golden_pass_rate{golden_set="praxiom-core"} 0.9' in text


def _metrics_text() -> str:
    from prometheus_client import generate_latest

    return generate_latest().decode()
