# tests/test_observability.py
import json
import logging

from photon.observability import (
    JsonFormatter,
    init_sentry,
    init_tracing,
    setup_logging,
)


def test_json_formatter_emits_valid_json_with_fields():
    rec = logging.LogRecord("photon.access", logging.INFO, __file__, 1, "request", (), None)
    rec.fields = {"path": "/v1/models", "status": 200}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "request"
    assert out["path"] == "/v1/models"
    assert out["status"] == 200
    assert out["level"] == "INFO"


def test_setup_logging_is_idempotent():
    setup_logging(json_logs=True)
    setup_logging(json_logs=True)
    assert len(logging.getLogger().handlers) == 1  # not stacked


def test_init_sentry_noop_without_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_sentry() is False


def test_init_tracing_noop_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert init_tracing(app=None) is False


def test_access_log_emitted_per_request(client, caplog):
    with caplog.at_level(logging.INFO, logger="photon.access"):
        r = client.get("/v1/models")
    assert r.status_code == 200
    access_records = [rec for rec in caplog.records if rec.name == "photon.access"]
    assert access_records
    fields = access_records[-1].fields
    assert fields["path"] == "/v1/models"
    assert fields["status"] == 200
