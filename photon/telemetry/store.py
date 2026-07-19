# photon/telemetry/store.py
import sqlite3
from pathlib import Path

from photon.telemetry.records import RequestRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    ts REAL NOT NULL,
    requested_model TEXT NOT NULL,
    routed_backend TEXT NOT NULL,
    backend_model TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    shadow_backend TEXT,
    shadow_est_cost_usd REAL,
    route_mode TEXT,
    quality_bar REAL,
    latency_slo_ms REAL,
    budget REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_tenant_ts ON requests (tenant, ts);
"""

# Columns added after the initial schema shipped. Applied idempotently on open
# so pre-existing SQLite files gain them without a manual migration step.
_MIGRATIONS = {
    "route_mode": "TEXT",
    "quality_bar": "REAL",
    "latency_slo_ms": "REAL",
    "budget": "REAL",
}


class TelemetryStore:
    """SQLite-backed request log. Opens a connection per operation so it is
    safe from any thread (FastAPI runs sync work in a threadpool). Phase 0
    volume is low enough that per-call connect + WAL is the simple, correct
    choice; revisit only if profiling says otherwise."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(requests)")}
            for col, col_type in _MIGRATIONS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {col_type}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, record: RequestRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO requests (
                    request_id, tenant, ts, requested_model, routed_backend,
                    backend_model, status, latency_ms, prompt_tokens,
                    completion_tokens, cost_usd, shadow_backend, shadow_est_cost_usd,
                    route_mode, quality_bar, latency_slo_ms, budget
                ) VALUES (
                    :request_id, :tenant, :ts, :requested_model, :routed_backend,
                    :backend_model, :status, :latency_ms, :prompt_tokens,
                    :completion_tokens, :cost_usd, :shadow_backend, :shadow_est_cost_usd,
                    :route_mode, :quality_bar, :latency_slo_ms, :budget
                )
                """,
                record.model_dump(),
            )

    def cost_summary(self, tenant: str, since_ts: float = 0.0) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT routed_backend,
                       COUNT(*) AS requests,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(SUM(shadow_est_cost_usd), 0.0) AS shadow_est_cost_usd
                FROM requests
                WHERE tenant = ? AND ts >= ? AND status = 'ok'
                GROUP BY routed_backend
                ORDER BY routed_backend
                """,
                (tenant, since_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_decisions(self, tenant: str, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM requests WHERE tenant = ? ORDER BY ts DESC LIMIT ?",
                (tenant, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_ok_rate(self, tenant: str, limit: int = 200) -> float:
        """Tenant history feature for the routing policy: share of the tenant's
        most recent `limit` requests that succeeded. A PROXY for acceptance —
        real quality labels arrive with the Tier-3 study; until then success
        rate is the honest signal we actually have. Returns 0.0 with no history
        (consistent with the policy model's fail-closed posture)."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT AVG(CASE WHEN status = 'ok' THEN 1.0 ELSE 0.0 END)
                   FROM (SELECT status FROM requests
                         WHERE tenant = ? ORDER BY ts DESC LIMIT ?)""",
                (tenant, limit),
            ).fetchone()
        return float(row[0]) if row[0] is not None else 0.0
