# photon/api/chat.py
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from photon.api import metrics
from photon.backends.openai_proxy import BackendError
from photon.costs import compute_cost_usd
from photon.router.static import AUTO_MODEL, UnknownModelError
from photon.telemetry.records import RequestRecord

chat_router = APIRouter()


@chat_router.get("/v1/models")
async def list_models(request: Request):
    config = request.app.state.config
    ids = [AUTO_MODEL, *config.routing.aliases.keys(), *(b.name for b in config.backends)]
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


@chat_router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    tenant = request.headers.get("x-photon-tenant", "default")
    requested_model = payload.get("model", AUTO_MODEL)
    request_id = uuid.uuid4().hex
    state = request.app.state

    try:
        decision = state.router.resolve(requested_model)
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"unknown model {requested_model!r}")

    backend = decision.backend
    record = RequestRecord(
        request_id=request_id,
        tenant=tenant,
        ts=time.time(),
        requested_model=requested_model,
        routed_backend=backend.name,
        backend_model=backend.model,
        status="error",  # flipped to "ok" on success
        latency_ms=0.0,
    )

    if payload.get("stream") is True:
        return await _stream_chat(state, backend, payload, record)

    try:
        response, latency_ms = await state.proxy.chat_completions(backend, payload)
    except BackendError as exc:
        state.store.insert(record)
        metrics.observe(record)
        raise HTTPException(status_code=502, detail=str(exc))

    record.status = "ok"
    record.latency_ms = latency_ms
    usage = response.get("usage") or {}
    record.prompt_tokens = usage.get("prompt_tokens")
    record.completion_tokens = usage.get("completion_tokens")
    if record.prompt_tokens is not None and record.completion_tokens is not None:
        record.cost_usd = compute_cost_usd(
            backend.pricing, record.prompt_tokens, record.completion_tokens
        )
        shadow_name = state.shadow.candidate(backend.name, payload.get("messages", []))
        if shadow_name is not None:
            shadow_backend = state.config.backend(shadow_name)
            record.shadow_backend = shadow_name
            record.shadow_est_cost_usd = compute_cost_usd(
                shadow_backend.pricing, record.prompt_tokens, record.completion_tokens
            )
    state.store.insert(record)
    metrics.observe(record)

    return JSONResponse(
        response,
        headers={"X-Photon-Request-Id": request_id, "X-Photon-Backend": backend.name},
    )


async def _stream_chat(state, backend, payload, record):
    body = {**payload, "model": backend.model}
    opts = payload.get("stream_options")
    body["stream_options"] = {**(opts if isinstance(opts, dict) else {}), "include_usage": True}
    req = state.http.build_request(
        "POST", f"{backend.base_url}/chat/completions", json=body
    )
    try:
        upstream = await state.http.send(req, stream=True)
    except Exception as exc:
        state.store.insert(record)
        metrics.observe(record)
        raise HTTPException(status_code=502, detail=f"{backend.name}: {exc}")
    if upstream.status_code >= 400:
        await upstream.aclose()
        state.store.insert(record)
        metrics.observe(record)
        raise HTTPException(
            status_code=502,
            detail=f"{backend.name}: upstream status {upstream.status_code}",
        )

    started = time.perf_counter()

    async def relay():
        usage: dict = {}
        completed = False
        try:
            async for line in upstream.aiter_lines():
                if line.startswith("data: ") and '"usage"' in line:
                    try:
                        chunk = json.loads(line[len("data: "):])
                        if isinstance(chunk, dict) and chunk.get("usage"):
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        pass
                yield (line + "\n").encode()
            completed = True
        finally:
            await upstream.aclose()
            record.status = "ok" if completed else "error"
            record.latency_ms = (time.perf_counter() - started) * 1000
            record.prompt_tokens = usage.get("prompt_tokens")
            record.completion_tokens = usage.get("completion_tokens")
            if record.prompt_tokens is not None and record.completion_tokens is not None:
                record.cost_usd = compute_cost_usd(
                    backend.pricing, record.prompt_tokens, record.completion_tokens
                )
                shadow_name = state.shadow.candidate(backend.name, payload.get("messages", []))
                if shadow_name is not None:
                    shadow_backend = state.config.backend(shadow_name)
                    record.shadow_backend = shadow_name
                    record.shadow_est_cost_usd = compute_cost_usd(
                        shadow_backend.pricing, record.prompt_tokens, record.completion_tokens
                    )
            state.store.insert(record)
            metrics.observe(record)

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={
            "X-Photon-Request-Id": record.request_id,
            "X-Photon-Backend": backend.name,
        },
    )
