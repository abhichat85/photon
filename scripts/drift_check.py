# scripts/drift_check.py
"""Continuous model-quality drift check (WS-B).

Runs the golden set against a live gateway, updates the photon_golden_pass_rate
gauge, and (if PUSHGATEWAY_URL is set) pushes it to a Prometheus Pushgateway so
the PhotonGoldenQualityDrift alert can fire. Exits non-zero if the pass rate is
below --min-pass-rate, so it doubles as a cron-friendly canary.

Schedule this (k8s CronJob / cron) every N minutes against production traffic's
gateway. This is distinct from the promotion gate (registry.promote): the gate
is point-in-time at deploy; this is ongoing in production.

Usage:
    python -m scripts.drift_check <gateway_url> <golden.yaml> [--min-pass-rate 0.95]
"""
import asyncio
import os
import sys

from photon.api.metrics import GOLDEN_PASS_RATE, set_golden_pass_rate
from photon.evals.golden import GoldenSet
from photon.evals.runner import EvalReport, run_golden_set


def compute_pass_rate(report: EvalReport) -> float:
    if report.total == 0:
        return 0.0
    return report.passed / report.total


def _push_if_configured(golden_set: str) -> None:
    url = os.environ.get("PUSHGATEWAY_URL")
    if not url:
        return
    from prometheus_client import push_to_gateway
    from prometheus_client.registry import REGISTRY

    push_to_gateway(url, job=f"photon-drift-{golden_set}", registry=REGISTRY)


def main() -> None:
    argv = sys.argv[1:]
    gateway_url, golden_path = argv[0], argv[1]
    min_rate = (
        float(argv[argv.index("--min-pass-rate") + 1])
        if "--min-pass-rate" in argv
        else 0.95
    )
    golden = GoldenSet.from_yaml(golden_path)
    report = asyncio.run(run_golden_set(gateway_url, golden))
    rate = compute_pass_rate(report)
    set_golden_pass_rate(golden.name, rate)
    _push_if_configured(golden.name)
    print(f"golden_set={golden.name} pass_rate={rate:.3f} threshold={min_rate:.3f}")
    if rate < min_rate:
        sys.exit(1)


if __name__ == "__main__":
    main()
