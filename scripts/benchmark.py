# scripts/benchmark.py
"""Drive real load against a live Photon gateway (and, for baseline, a backend
directly), then print the §9 DoD report. Run this once you have a GPU box up
(deploy/GPU_SETUP.md). It is the single command that answers 'did we hit the
spec's numbers?' — exit 1 if any target is missed, so it doubles as a CI/perf gate.

Usage:
    python -m scripts.benchmark \
        --gateway http://localhost:8080 \
        --backend http://localhost:8000/v1 \
        --requests 200 \
        --baseline-cost-usd 0.030 --measured-cost-usd 0.010
"""
import argparse
import asyncio
import json
import sys
import time

import httpx

from photon.core.benchmark import build_report

_PAYLOAD = {"model": "photon-auto", "messages": [{"role": "user", "content": "Say hi in 5 words."}]}
_BACKEND_PAYLOAD = {"messages": _PAYLOAD["messages"], "max_tokens": 16}


async def _time_calls(client: httpx.AsyncClient, url: str, payload: dict, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        started = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # a failed sample is not a latency sample
            print(f"  request failed: {exc}", file=sys.stderr)
            continue
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies


async def _run(args) -> int:
    async with httpx.AsyncClient() as client:
        print(f"warming up + timing {args.requests} baseline calls (direct backend)...")
        baseline = await _time_calls(
            client, f"{args.backend}/chat/completions", {**_BACKEND_PAYLOAD, "model": args.backend_model}, args.requests
        )
        print(f"timing {args.requests} routed calls (through Photon)...")
        routed = await _time_calls(client, f"{args.gateway}/v1/chat/completions", _PAYLOAD, args.requests)

    # selection overhead = the gateway's own decision cost, approximated as the
    # per-request delta of routed vs baseline (both hit the same backend). A
    # true in-process measurement is available once the router exposes timing;
    # this end-to-end delta is the honest black-box proxy.
    n = min(len(baseline), len(routed))
    overhead = [max(0.0, routed[i] - baseline[i]) for i in range(n)]

    report = build_report(
        baseline_latencies_ms=baseline,
        routed_latencies_ms=routed,
        selection_overhead_ms=overhead,
        baseline_cost_usd=args.baseline_cost_usd,
        measured_cost_usd=args.measured_cost_usd,
    )
    print(json.dumps(report.model_dump(), indent=2))
    if not report.all_targets_met:
        print("\nFAIL: one or more §9 targets missed", file=sys.stderr)
        return 1
    print("\nPASS: all §9 targets met")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--backend", required=True, help="a backend's OpenAI root, e.g. http://localhost:8000/v1")
    ap.add_argument("--backend-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--baseline-cost-usd", type=float, required=True)
    ap.add_argument("--measured-cost-usd", type=float, required=True)
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
