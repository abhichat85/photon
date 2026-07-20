# tests/test_telemetry_store.py
import time

import pytest

from photon.telemetry.records import RequestRecord
from photon.telemetry.store import TelemetryStore


def make_record(**overrides) -> RequestRecord:
    base = dict(
        request_id="req1",
        tenant="praxiom",
        ts=time.time(),
        requested_model="photon-auto",
        routed_backend="big",
        backend_model="qwen-72b",
        status="ok",
        latency_ms=120.5,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0000135,
        shadow_backend="small",
        shadow_est_cost_usd=0.0000012,
    )
    base.update(overrides)
    return RequestRecord(**base)


@pytest.fixture
def store(tmp_path) -> TelemetryStore:
    return TelemetryStore(tmp_path / "t.db")


def test_insert_and_read_back(store):
    store.insert(make_record())
    rows = store.recent_decisions("praxiom", limit=10)
    assert len(rows) == 1
    assert rows[0]["routed_backend"] == "big"
    assert rows[0]["shadow_backend"] == "small"


def test_cost_summary_groups_by_backend(store):
    store.insert(make_record(request_id="a", routed_backend="big", cost_usd=0.002))
    store.insert(make_record(request_id="b", routed_backend="big", cost_usd=0.003))
    store.insert(make_record(request_id="c", routed_backend="small", cost_usd=0.001, shadow_backend=None, shadow_est_cost_usd=None))
    summary = {row["routed_backend"]: row for row in store.cost_summary("praxiom")}
    assert summary["big"]["requests"] == 2
    assert summary["big"]["cost_usd"] == pytest.approx(0.005)
    assert summary["small"]["requests"] == 1


def test_cost_summary_excludes_errors_and_other_tenants(store):
    store.insert(make_record(request_id="a", status="error"))
    store.insert(make_record(request_id="b", tenant="other"))
    assert store.cost_summary("praxiom") == []


def test_recent_decisions_orders_newest_first_and_limits(store):
    store.insert(make_record(request_id="old", ts=100.0))
    store.insert(make_record(request_id="new", ts=200.0))
    rows = store.recent_decisions("praxiom", limit=1)
    assert len(rows) == 1
    assert rows[0]["request_id"] == "new"


def test_recent_ok_rate_over_window(store):
    store.insert(make_record(request_id="a", status="ok"))
    store.insert(make_record(request_id="b", status="ok"))
    store.insert(make_record(request_id="c", status="error"))
    assert store.recent_ok_rate("praxiom") == pytest.approx(2 / 3)


def test_recent_ok_rate_no_history_is_zero(store):
    assert store.recent_ok_rate("nobody") == 0.0


def test_recent_ok_rate_respects_window_limit(store):
    # oldest is an error, but with limit=2 only the two newest (ok) rows count
    store.insert(make_record(request_id="a", status="error", ts=100.0))
    store.insert(make_record(request_id="b", status="ok", ts=200.0))
    store.insert(make_record(request_id="c", status="ok", ts=300.0))
    assert store.recent_ok_rate("praxiom", limit=2) == 1.0
