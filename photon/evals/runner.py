# photon/evals/runner.py
import time

import httpx
from pydantic import BaseModel

from photon.evals.golden import CaseResult, GoldenSet, check_case


class EvalReport(BaseModel):
    golden_set: str
    total: int
    passed: int
    results: list[CaseResult]


async def run_golden_set(
    gateway_url: str, golden: GoldenSet, tenant: str = "evals"
) -> EvalReport:
    results: list[CaseResult] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for case in golden.cases:
            started = time.perf_counter()
            resp = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={"model": golden.model, "messages": case.messages},
                headers={"X-Photon-Tenant": tenant},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            if resp.status_code != 200:
                results.append(
                    CaseResult(
                        case_id=case.id,
                        passed=False,
                        failures=[f"http {resp.status_code}"],
                        latency_ms=latency_ms,
                    )
                )
                continue
            text = resp.json()["choices"][0]["message"]["content"]
            results.append(check_case(case, text, latency_ms))
    return EvalReport(
        golden_set=golden.name,
        total=len(results),
        passed=sum(r.passed for r in results),
        results=results,
    )
