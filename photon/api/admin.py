# photon/api/admin.py
from fastapi import APIRouter, Request

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
