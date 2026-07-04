# tests/test_cost_report.py
import time

from photon.telemetry.records import RequestRecord
from photon.telemetry.store import TelemetryStore
from scripts.cost_report import build_report


def test_report_computes_shadow_savings(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    store.insert(
        RequestRecord(
            request_id="a",
            tenant="praxiom",
            ts=time.time(),
            requested_model="photon-auto",
            routed_backend="big",
            backend_model="qwen-72b",
            status="ok",
            latency_ms=100.0,
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.010,
            shadow_backend="small",
            shadow_est_cost_usd=0.001,
        )
    )
    store.insert(
        RequestRecord(
            request_id="b",
            tenant="praxiom",
            ts=time.time(),
            requested_model="photon-auto",
            routed_backend="big",
            backend_model="qwen-72b",
            status="ok",
            latency_ms=100.0,
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.010,
            # long prompt: no shadow candidate
        )
    )
    report = build_report(store, tenant="praxiom")
    assert report["total_cost_usd"] == 0.020
    assert report["shadow"]["candidate_requests"] == 1
    assert report["shadow"]["candidate_share"] == 0.5
    # savings if candidate had served those requests: 0.010 - 0.001
    assert report["shadow"]["est_savings_usd"] == 0.009
