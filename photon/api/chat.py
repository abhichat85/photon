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


def _record_token_efficiency(state, backend_name: str, messages: list, prompt_tokens) -> None:
    """Fold this request into the (backend, script) tokenizer-efficiency ledger.
    This is what makes language-fair cost accounting possible for Indic traffic —
    the ratios are measured from real usage, never assumed. No-op if the store
    isn't configured or the request had no usable prompt/token counts."""
    store = getattr(state, "token_efficiency", None)
    if store is None or not prompt_tokens:
        return
    from photon.india.script import messages_script

    chars = sum(m.get("content", "") and len(m["content"]) or 0
                for m in messages if isinstance(m.get("content"), str))
    if chars:
        store.record(backend_name, messages_script(messages), chars, prompt_tokens)


def parse_photon_block(payload: dict) -> dict:
    """Pop and validate the optional `photon` request-extension block (spec §6),
    mutating `payload` so it never reaches the upstream vLLM (which rejects
    unknown fields). Returns a normalized dict with route/quality_bar/
    latency_slo_ms/budget. At Ops, `route: pin` disables canary and `cascade`
    is rejected (Core-only); the quality/latency/budget fields are recorded for
    the future learned router, not enforced."""
    block = payload.pop("photon", None)
    if block is not None and not isinstance(block, dict):
        raise HTTPException(status_code=422, detail="`photon` block must be an object")
    block = block or {}
    route = block.get("route", "auto")
    if route not in ("auto", "pin", "cascade"):
        raise HTTPException(status_code=422, detail=f"invalid photon.route {route!r}")
    if route == "cascade":
        raise HTTPException(
            status_code=400,
            detail="photon.route 'cascade' requires Photon Core (not available in Ops)",
        )
    return {
        "route": route,
        "quality_bar": block.get("quality_bar"),
        "latency_slo_ms": block.get("latency_slo_ms"),
        "budget": block.get("budget"),
    }


@chat_router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    tenant = request.headers.get("x-photon-tenant", "default")
    requested_model = payload.get("model", AUTO_MODEL)
    request_id = uuid.uuid4().hex
    state = request.app.state

    photon = parse_photon_block(payload)  # also strips `photon` from payload

    decide_started = time.perf_counter()
    try:
        if getattr(state.router, "wants_features", False):
            from photon.core.features import extract_features

            route_feats = extract_features(
                messages=payload.get("messages", []),
                tenant=tenant,
                route_hint=photon["route"],
                tenant_recent_accept_rate=state.store.recent_ok_rate(tenant),
            )
            decision = state.router.resolve(
                requested_model,
                allow_canary=(photon["route"] != "pin"),
                features=route_feats,
            )
        else:
            decision = state.router.resolve(
                requested_model, allow_canary=(photon["route"] != "pin")
            )
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"unknown model {requested_model!r}")
    # in-process routing-decision cost — the honest source for the §9 <3ms
    # selection-overhead DoD (benchmark.py reads this header, no black-box guess)
    decide_ms = (time.perf_counter() - decide_started) * 1000

    backend = decision.backend

    shadow = getattr(state, "shadow_router", None)
    if shadow is not None:
        from photon.core.features import extract_features

        feats = extract_features(
            messages=payload.get("messages", []),
            tenant=tenant,
            route_hint=photon["route"],
            tenant_recent_accept_rate=state.store.recent_ok_rate(tenant),
        )
        shadow.observe(actual_backend_name=backend.name, features=feats, request_id=request_id)

    record = RequestRecord(
        request_id=request_id,
        tenant=tenant,
        ts=time.time(),
        requested_model=requested_model,
        routed_backend=backend.name,
        backend_model=backend.model,
        status="error",  # flipped to "ok" on success
        latency_ms=0.0,
        route_mode=photon["route"],
        quality_bar=photon["quality_bar"],
        latency_slo_ms=photon["latency_slo_ms"],
        budget=photon["budget"],
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
    _record_token_efficiency(state, backend.name, payload.get("messages", []), record.prompt_tokens)
    state.store.insert(record)
    metrics.observe(record)

    return JSONResponse(
        response,
        headers={
            "X-Photon-Request-Id": request_id,
            "X-Photon-Backend": backend.name,
            "X-Photon-Decision-Ms": f"{decide_ms:.4f}",
        },
    )


@chat_router.post("/v1/completions")
async def completions(request: Request):
    """Legacy (non-chat) completions. Non-streaming only — use
    /v1/chat/completions for SSE streaming."""
    payload = await request.json()
    if payload.get("stream") is True:
        raise HTTPException(
            status_code=400,
            detail="streaming not supported on /v1/completions; use /v1/chat/completions",
        )
    tenant = request.headers.get("x-photon-tenant", "default")
    requested_model = payload.get("model", AUTO_MODEL)
    request_id = uuid.uuid4().hex
    state = request.app.state

    photon = parse_photon_block(payload)
    try:
        if getattr(state.router, "wants_features", False):
            from photon.core.features import extract_features

            _p = payload.get("prompt")
            _pm = [{"role": "user", "content": _p}] if isinstance(_p, str) else []
            route_feats = extract_features(
                messages=_pm,
                tenant=tenant,
                route_hint=photon["route"],
                tenant_recent_accept_rate=state.store.recent_ok_rate(tenant),
            )
            decision = state.router.resolve(
                requested_model,
                allow_canary=(photon["route"] != "pin"),
                features=route_feats,
            )
        else:
            decision = state.router.resolve(
                requested_model, allow_canary=(photon["route"] != "pin")
            )
    except UnknownModelError:
        raise HTTPException(status_code=404, detail=f"unknown model {requested_model!r}")

    backend = decision.backend

    shadow = getattr(state, "shadow_router", None)
    if shadow is not None:
        from photon.core.features import extract_features

        prompt = payload.get("prompt")
        pseudo_messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else []
        )
        feats = extract_features(
            messages=pseudo_messages,
            tenant=tenant,
            route_hint=photon["route"],
            tenant_recent_accept_rate=state.store.recent_ok_rate(tenant),
        )
        shadow.observe(actual_backend_name=backend.name, features=feats, request_id=request_id)

    record = RequestRecord(
        request_id=request_id,
        tenant=tenant,
        ts=time.time(),
        requested_model=requested_model,
        routed_backend=backend.name,
        backend_model=backend.model,
        status="error",
        latency_ms=0.0,
        route_mode=photon["route"],
        quality_bar=photon["quality_bar"],
        latency_slo_ms=photon["latency_slo_ms"],
        budget=photon["budget"],
    )

    try:
        response, latency_ms = await state.proxy.completions(backend, payload)
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
        # synthesize a message list from the prompt so the shadow char-heuristic applies
        prompt = payload.get("prompt")
        pseudo_messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else []
        )
        shadow_name = state.shadow.candidate(backend.name, pseudo_messages)
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
