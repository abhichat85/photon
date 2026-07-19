# photon/core/shadow_store.py
"""Durable persistence for shadow-router decisions. Without this, the Tier-3
routing study has nowhere to write its data: ShadowRouter only hands decisions
to a sink, and an in-memory sink dies with the process. Same SQLite pattern as
the telemetry and registry stores (WAL, per-call connection, single-replica —
DECISIONS.md D1/D3 apply). `insert` satisfies the ShadowRouter sink signature
directly: ShadowRouter(learned, sink=shadow_store.insert)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from photon.core.router import ShadowDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_decisions (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    actual_backend TEXT NOT NULL,
    would_model TEXT NOT NULL,
    would_adapter TEXT,
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_ts ON shadow_decisions (ts);
"""


class ShadowDecisionStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, decision: ShadowDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO shadow_decisions
                   (request_id, ts, actual_backend, would_model, would_adapter, reason)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    decision.request_id,
                    time.time(),
                    decision.actual_backend,
                    decision.would_route.model_id,
                    decision.would_route.adapter_id,
                    decision.reason,
                ),
            )

    def recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_decisions ORDER BY ts DESC, request_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        """Agreement share (learned would-route == actual) and would-route
        distribution — the first-glance readout of the shadow study."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
            if not total:
                return {"total": 0, "agreement_share": 0.0, "would_route_counts": {}}
            agree = conn.execute(
                "SELECT COUNT(*) FROM shadow_decisions WHERE would_model = actual_backend"
            ).fetchone()[0]
            counts = dict(
                conn.execute(
                    "SELECT would_model, COUNT(*) FROM shadow_decisions GROUP BY would_model"
                ).fetchall()
            )
        return {
            "total": total,
            "agreement_share": agree / total,
            "would_route_counts": counts,
        }
