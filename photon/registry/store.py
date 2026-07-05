# photon/registry/store.py
import json
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel


def eval_pass_rate(eval_report: str | None) -> float | None:
    """Parse an attached eval report (JSON with `passed`/`total`) into a pass
    rate. Returns None when the report is missing, malformed, or has total==0 —
    the promotion gate treats None as 'unverified' and refuses to promote."""
    if not eval_report:
        return None
    try:
        data = json.loads(eval_report)
        total = data["total"]
        if not total:
            return None
        return data["passed"] / total
    except (json.JSONDecodeError, KeyError, TypeError, ZeroDivisionError):
        return None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    base_model TEXT NOT NULL,
    adapter_path TEXT,
    status TEXT NOT NULL DEFAULT 'draft',  -- draft | candidate | production | retired
    eval_report TEXT,
    created_ts REAL NOT NULL,
    PRIMARY KEY (name, version)
);
"""


class ModelVersion(BaseModel):
    name: str
    version: int
    base_model: str
    adapter_path: str | None
    status: str
    eval_report: str | None
    created_ts: float


class RegistryStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def register(
        self,
        name: str,
        base_model: str,
        adapter_path: str | None,
        eval_report: str | None = None,
    ) -> ModelVersion:
        with self._connect() as conn:
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM model_versions WHERE name = ?",
                (name,),
            ).fetchone()[0]
            row = {
                "name": name,
                "version": version,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "status": "draft",
                "eval_report": eval_report,
                "created_ts": time.time(),
            }
            conn.execute(
                """INSERT INTO model_versions
                   (name, version, base_model, adapter_path, status, eval_report, created_ts)
                   VALUES (:name, :version, :base_model, :adapter_path, :status,
                           :eval_report, :created_ts)""",
                row,
            )
        return ModelVersion(**row)

    def get(self, name: str, version: int) -> ModelVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
        return ModelVersion(**dict(row)) if row else None

    def promote(
        self,
        name: str,
        version: int,
        min_pass_rate: float = 1.0,
        force: bool = False,
    ) -> None:
        """Promote a version to production, demoting the current one to retired.

        Quality gate (WS-B): unless force=True, the version MUST carry an
        eval_report whose pass rate is >= min_pass_rate. This is what makes the
        golden-set gate a real gate rather than a convention — a fine-tune with
        no eval, or a failing one, cannot reach production traffic through this
        method. Use force=True only for deliberate operational overrides."""
        mv = self.get(name, version)
        if mv is None:
            raise KeyError((name, version))
        if not force:
            rate = eval_pass_rate(mv.eval_report)
            if rate is None:
                raise ValueError(
                    f"cannot promote {name} v{version} without a valid eval_report; "
                    "attach one (run the golden gate) or promote with force=True"
                )
            if rate < min_pass_rate:
                raise ValueError(
                    f"{name} v{version} eval pass rate {rate:.3f} < required "
                    f"{min_pass_rate:.3f}; not promoting"
                )
        with self._connect() as conn:
            conn.execute(
                "UPDATE model_versions SET status = 'retired' "
                "WHERE name = ? AND status = 'production'",
                (name,),
            )
            conn.execute(
                "UPDATE model_versions SET status = 'production' "
                "WHERE name = ? AND version = ?",
                (name, version),
            )

    def attach_eval(self, name: str, version: int, eval_report: str) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE model_versions SET eval_report = ? WHERE name = ? AND version = ?",
                (eval_report, name, version),
            )
            if cur.rowcount == 0:
                raise KeyError((name, version))

    def production(self, name: str) -> ModelVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE name = ? AND status = 'production'",
                (name,),
            ).fetchone()
        return ModelVersion(**dict(row)) if row else None

    def history(self, name: str) -> list[ModelVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE name = ? ORDER BY version",
                (name,),
            ).fetchall()
        return [ModelVersion(**dict(r)) for r in rows]
