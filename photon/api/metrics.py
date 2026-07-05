# photon/api/metrics.py
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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
# Model-quality drift signal: latest golden-set pass rate, updated by the
# periodic drift check (scripts/drift_check.py). The PhotonGoldenQualityDrift
# alert fires when this stays below threshold — this is continuous quality
# monitoring, distinct from the point-in-time promotion gate.
GOLDEN_PASS_RATE = Gauge(
    "photon_golden_pass_rate", "Latest golden-set pass rate", ["golden_set"]
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


def set_golden_pass_rate(golden_set: str, rate: float) -> None:
    """Update the golden-set pass-rate gauge (called by the drift check)."""
    GOLDEN_PASS_RATE.labels(golden_set).set(rate)
