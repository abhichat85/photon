# photon/telemetry/records.py
from pydantic import BaseModel


class RequestRecord(BaseModel):
    request_id: str
    tenant: str
    ts: float  # unix seconds
    requested_model: str  # what the client asked for (may be "photon-auto")
    routed_backend: str  # backend name the router chose
    backend_model: str  # concrete model id sent upstream
    status: str  # "ok" | "error"
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    # Shadow routing: what a cheaper route WOULD have cost. Log-only.
    shadow_backend: str | None = None
    shadow_est_cost_usd: float | None = None
