# photon/evals/golden.py
from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    id: str
    messages: list[dict]
    must_contain: list[str] = Field(default_factory=list)  # case-insensitive
    must_not_contain: list[str] = Field(default_factory=list)
    must_match: str | None = None  # regex, case-insensitive, dotall
    max_latency_ms: float | None = None


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    failures: list[str]
    latency_ms: float


class GoldenSet(BaseModel):
    name: str
    model: str = "photon-auto"
    cases: list[GoldenCase]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoldenSet":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


def check_case(case: GoldenCase, text: str, latency_ms: float) -> CaseResult:
    failures: list[str] = []
    lowered = text.lower()
    for s in case.must_contain:
        if s.lower() not in lowered:
            failures.append(f"missing required substring: {s!r}")
    for s in case.must_not_contain:
        if s.lower() in lowered:
            failures.append(f"contains forbidden substring: {s!r}")
    if case.must_match and not re.search(case.must_match, text, re.IGNORECASE | re.DOTALL):
        failures.append(f"regex not matched: {case.must_match!r}")
    if case.max_latency_ms is not None and latency_ms > case.max_latency_ms:
        failures.append(f"latency {latency_ms:.0f}ms > budget {case.max_latency_ms:.0f}ms")
    return CaseResult(
        case_id=case.id, passed=not failures, failures=failures, latency_ms=latency_ms
    )
