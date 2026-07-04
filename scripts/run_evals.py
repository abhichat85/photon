# scripts/run_evals.py
"""Run a golden set against a gateway; exit 1 if pass rate < --min-pass-rate.

Usage: python -m scripts.run_evals <gateway_url> <golden.yaml> \
           [--min-pass-rate 1.0] [--model <backend-or-alias>]
"""
import asyncio
import json
import sys

from photon.evals.golden import GoldenSet
from photon.evals.runner import run_golden_set


def main() -> None:
    argv = sys.argv[1:]
    gateway_url, golden_path = argv[0], argv[1]
    min_rate = float(argv[argv.index("--min-pass-rate") + 1]) if "--min-pass-rate" in argv else 1.0
    golden = GoldenSet.from_yaml(golden_path)
    if "--model" in argv:  # evaluate a candidate backend without editing the YAML
        golden = golden.model_copy(update={"model": argv[argv.index("--model") + 1]})
    report = asyncio.run(run_golden_set(gateway_url, golden))
    print(json.dumps(report.model_dump(), indent=2))
    if report.total == 0 or report.passed / report.total < min_rate:
        sys.exit(1)


if __name__ == "__main__":
    main()
