# photon/api/pipelines.py
"""Spec §6: POST /photon/v1/pipelines/{id} — execute a registered DAG with
per-stage routing and an end-to-end latency budget. Registration is in-memory
and per-process (config-like, re-registered on boot — same posture as the
fleet plan; see DECISIONS.md). Execution runs against app.state.serving_backend
(a thin adapter-aware vLLM backend by default; the Tier-2 dense engine drops in
behind the same interface)."""
from fastapi import APIRouter, HTTPException, Request

from photon.core.pipeline import PipelineOrchestrator, PipelineSpec

pipelines_router = APIRouter(prefix="/photon/v1")


@pipelines_router.post("/pipelines")
async def register_pipeline(request: Request, spec: PipelineSpec):
    request.app.state.pipelines[spec.id] = spec
    return spec.model_dump()


@pipelines_router.get("/pipelines")
async def list_pipelines(request: Request):
    return {"pipelines": sorted(request.app.state.pipelines.keys())}


@pipelines_router.post("/pipelines/{pipeline_id}")
async def execute_pipeline(request: Request, pipeline_id: str):
    spec = request.app.state.pipelines.get(pipeline_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown pipeline {pipeline_id!r}")
    payload = await request.json()
    orchestrator = PipelineOrchestrator(request.app.state.serving_backend)
    result = await orchestrator.execute(spec, payload)
    return result.model_dump()
