"""Observability wiring: structured logging, optional OTel tracing, optional
Sentry error tracking. Tracing and Sentry are GATED on env + package presence
and are safe no-ops otherwise — so dev/test stay dependency-light while
production can opt in without code changes.

Env switches:
  PHOTON_LOG_JSON=1                 → JSON logs (default: plain text)
  OTEL_EXPORTER_OTLP_ENDPOINT=...   → enable OTel FastAPI+httpx instrumentation
  SENTRY_DSN=...                    → enable Sentry error tracking
"""
from __future__ import annotations

import json
import logging
import os
import time

ACCESS_LOGGER = "photon.access"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # merge structured extras attached via logger.info(..., extra={"fields": {...}})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(json_logs: bool | None = None) -> None:
    """Configure the root logger. Idempotent — replaces existing handlers so
    repeated create_app() calls (tests) don't stack duplicates."""
    if json_logs is None:
        json_logs = os.environ.get("PHOTON_LOG_JSON") == "1"
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def init_sentry() -> bool:
    """Enable Sentry if SENTRY_DSN is set and sentry_sdk is installed.
    Returns True if initialized, else False (safe no-op)."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger("photon").warning(
            "SENTRY_DSN set but sentry_sdk not installed; skipping"
        )
        return False
    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0)
    return True


def init_tracing(app) -> bool:
    """Enable OTel FastAPI + httpx instrumentation if OTEL_EXPORTER_OTLP_ENDPOINT
    is set and the instrumentation packages are installed. Returns True if
    initialized, else False (safe no-op)."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        logging.getLogger("photon").warning(
            "OTEL endpoint set but instrumentation packages not installed; skipping"
        )
        return False
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return True


class RequestLogMiddleware:
    """ASGI middleware emitting one structured access-log record per HTTP
    request — the log line Loki/promtail ships for aggregation."""

    def __init__(self, app):
        self.app = app
        self._log = logging.getLogger(ACCESS_LOGGER)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            headers = dict(scope.get("headers") or [])
            tenant = headers.get(b"x-photon-tenant", b"default").decode()
            self._log.info(
                "request",
                extra={
                    "fields": {
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status["code"],
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "tenant": tenant,
                    }
                },
            )
