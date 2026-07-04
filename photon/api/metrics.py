# photon/api/metrics.py
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from photon.telemetry.records import RequestRecord

REQUESTS = Counter(
    "photon_requests_total", "Requests through the gateway", ["backend", "status", "tenant"]
)
LATENCY = Histogram(
    "photon_request_latency_seconds", "Upstream request latency", ["backend"]
)
COST = Counter(
    "photon_cost_usd_total", "Accumulated inference cost in USD", ["backend", "tenant"]
)

metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe(record: RequestRecord) -> None:
    """Mirror every telemetry insert into Prometheus. Called wherever
    TelemetryStore.insert() is called — the SQLite row is the audit truth,
    the metric is the operational signal."""
    REQUESTS.labels(record.routed_backend, record.status, record.tenant).inc()
    if record.status == "ok":
        LATENCY.labels(record.routed_backend).observe(record.latency_ms / 1000)
    if record.cost_usd:
        COST.labels(record.routed_backend, record.tenant).inc(record.cost_usd)
