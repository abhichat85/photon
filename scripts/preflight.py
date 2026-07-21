# scripts/preflight.py
"""Preflight for the solo GPU run (deploy/GPU_SETUP.md): confirm every wire is
connected BEFORE you spend time benchmarking. Checks the gateway is up, the
vLLM backend is reachable with runtime-LoRA enabled, and the fleet-enact +
pipeline paths respond. Exit 0 = ready to benchmark.

    python -m scripts.preflight --gateway http://localhost:8080 --vllm http://localhost:8000/v1
"""
import argparse
import sys

import httpx


def _check(name: str, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 — preflight reports, never raises
        ok, detail = False, str(exc)
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}: {detail}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--vllm", required=True, help="vLLM OpenAI root, e.g. http://localhost:8000/v1")
    args = ap.parse_args()

    results = []
    with httpx.Client(timeout=10.0) as c:
        results.append(_check(
            "gateway /v1/models",
            lambda: (c.get(f"{args.gateway}/v1/models").status_code == 200, "reachable"),
        ))
        results.append(_check(
            "vLLM /v1/models",
            lambda: (
                c.get(f"{args.vllm}/models").status_code == 200,
                f"serving {[m['id'] for m in c.get(f'{args.vllm}/models').json().get('data', [])]}",
            ),
        ))

        def _lora_enabled():
            # a bad-path load returns 4xx (enabled) vs 404 (endpoint absent = flag off)
            r = c.post(f"{args.vllm}/load_lora_adapter",
                       json={"lora_name": "__preflight__", "lora_path": "/does/not/exist"})
            if r.status_code == 404:
                return False, "endpoint missing — start vLLM with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True --enable-lora"
            return True, f"runtime-LoRA endpoint present (status {r.status_code} on bad path is expected)"

        results.append(_check("vLLM runtime-LoRA enabled", _lora_enabled))
        results.append(_check(
            "gateway /photon/v1/fleet/status",
            lambda: (c.get(f"{args.gateway}/photon/v1/fleet/status").status_code == 200, "reachable"),
        ))
        results.append(_check(
            "gateway /photon/v1/pipelines",
            lambda: (c.get(f"{args.gateway}/photon/v1/pipelines").status_code == 200, "reachable"),
        ))

    if all(results):
        print("\nPREFLIGHT PASS — ready to enact a fleet and run scripts/benchmark.py")
        sys.exit(0)
    print("\nPREFLIGHT FAIL — fix the ✗ items above before benchmarking", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
