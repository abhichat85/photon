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
    # `photon` request-extension block (spec §6). At Ops these are RECORDED for
    # the future learned router's training data, not enforced — enforcement of
    # quality_bar/latency_slo_ms/budget is Photon Core. route_mode is honored
    # now (pin disables canary; cascade is rejected upstream at the API layer).
    route_mode: str | None = None  # "auto" | "pin"
    quality_bar: float | None = None
    latency_slo_ms: float | None = None
    budget: float | None = None
