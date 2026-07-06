# photon/api/admin.py
import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from photon.core.fleet import FleetSpec

admin_router = APIRouter(prefix="/photon/v1")


@admin_router.get("/costs")
async def costs(request: Request, tenant: str = "default", since_ts: float = 0.0):
    return {
        "tenant": tenant,
        "since_ts": since_ts,
        "backends": request.app.state.store.cost_summary(tenant, since_ts),
    }


@admin_router.get("/routing/decisions")
async def routing_decisions(request: Request, tenant: str = "default", limit: int = 100):
    return {
        "tenant": tenant,
        "decisions": request.app.state.store.recent_decisions(tenant, limit),
    }


@admin_router.get("/fleet/status")
async def fleet_status(request: Request, probe: bool = False):
    """Configured fleet + routing policy. At Ops the fleet is static (from the
    YAML config); Fabric's dynamic residency/memory/replica state is Core.
    With ?probe=true, best-effort pings each backend's /models for reachability."""
    config = request.app.state.config
    routing = config.routing
    backends = []
    for b in config.backends:
        entry = {
            "name": b.name,
            "model": b.model,
            "base_url": b.base_url,
            "quantization": b.quantization,
            "pricing": b.pricing.model_dump(),
        }
        if probe:
            entry["reachable"] = await _probe(request.app.state.http, b.base_url)
        backends.append(entry)
    plan = request.app.state.fleet_plan
    return {
        "backends": backends,
        "routing": {
            "default_backend": routing.default_backend,
            "aliases": routing.aliases,
            "canary": routing.canary.model_dump() if routing.canary else None,
            "shadow_enabled": routing.shadow.enabled,
        },
        "residency": plan.model_dump() if plan else None,
    }


async def _probe(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        resp = await client.get(f"{base_url}/models", timeout=3.0)
        return resp.status_code < 400
    except httpx.HTTPError:
        return False


class AdapterRegistration(BaseModel):
    name: str
    base_model: str
    adapter_path: str | None = None
    eval_report: str | None = None
    promote: bool = False


@admin_router.post("/adapters")
async def register_adapter(request: Request, body: AdapterRegistration):
    """Register a fine-tuned adapter (HTTP equivalent of scripts/register_adapter.py).
    If promote=true, promotion is gated on a passing eval_report (see WS-B)."""
    registry = request.app.state.registry
    mv = registry.register(
        body.name, body.base_model, body.adapter_path, body.eval_report
    )
    if body.promote:
        try:
            registry.promote(body.name, mv.version)
        except ValueError as exc:  # gate rejection (WS-B)
            raise HTTPException(status_code=422, detail=str(exc))
        mv = registry.get(body.name, mv.version)
    return mv.model_dump()


@admin_router.post("/fleet")
async def apply_fleet(request: Request, spec: FleetSpec):
    try:
        plan = request.app.state.fleet_manager.plan(spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    request.app.state.fleet_plan = plan
    return plan.model_dump()
