# photon/core/pipeline.py
"""Pipeline orchestrator — the CPU control-plane of spec §4.1's 'pipeline
orchestrator' and the execution engine behind §6's POST /photon/v1/pipelines/{id}.

C0 scope, stated honestly: a SEQUENTIAL chain (a linear DAG) where each stage is
independently routable via a RouteTarget and the latency budget is enforced
END-TO-END across stages, not per-call — a request is aborted between stages
once the budget is spent. Each stage's output is appended to the running message
context for the next stage (Praxiom-1's parse → causal → simulate → critique →
generate shape). What is deliberately NOT here (Tier 2, needs the GPU engine):
branching DAGs, cross-stage KV/prefix reuse on the GPU, and avoiding the
serialize-transfer-deserialize hop between stages — those require owning the
serving engine's memory, which is the hire's work behind ServingBackend."""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from photon.core.contract import RouteTarget, ServingBackend


class StageSpec(BaseModel):
    name: str
    target: RouteTarget


class PipelineSpec(BaseModel):
    id: str
    stages: list[StageSpec] = Field(min_length=1)
    latency_budget_ms: float | None = None  # end-to-end, across ALL stages


class StageOutput(BaseModel):
    name: str
    latency_ms: float
    content: str


class PipelineResult(BaseModel):
    pipeline_id: str
    completed: bool
    aborted_at_stage: str | None = None
    total_latency_ms: float
    stage_outputs: list[StageOutput] = Field(default_factory=list)


def _content_of(response: dict) -> str:
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return message.get("content") or ""


class PipelineOrchestrator:
    def __init__(self, backend: ServingBackend):
        self._backend = backend

    async def execute(self, spec: PipelineSpec, payload: dict) -> PipelineResult:
        messages = list(payload.get("messages", []))
        outputs: list[StageOutput] = []
        started = time.perf_counter()

        for stage in spec.stages:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if spec.latency_budget_ms is not None and elapsed_ms >= spec.latency_budget_ms:
                return PipelineResult(
                    pipeline_id=spec.id,
                    completed=False,
                    aborted_at_stage=stage.name,
                    total_latency_ms=elapsed_ms,
                    stage_outputs=outputs,
                )
            response, latency_ms = await self._backend.generate(
                stage.target, {"messages": messages}
            )
            content = _content_of(response)
            outputs.append(StageOutput(name=stage.name, latency_ms=latency_ms, content=content))
            # chain: this stage's output becomes context for the next stage
            messages = messages + [{"role": "assistant", "content": content}]

        return PipelineResult(
            pipeline_id=spec.id,
            completed=True,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            stage_outputs=outputs,
        )
