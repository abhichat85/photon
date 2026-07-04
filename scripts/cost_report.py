# scripts/cost_report.py
"""Phase 0 shadow-study report: actual spend vs counterfactual cheap-route
spend. Usage: python -m scripts.cost_report <db_path> [tenant] [since_ts]"""

import sqlite3
import sys

from photon.telemetry.store import TelemetryStore


def build_report(store: TelemetryStore, tenant: str, since_ts: float = 0.0) -> dict:
    with store._connect() as conn:  # noqa: SLF001 — report is a privileged consumer
        row = conn.execute(
            """
            SELECT COUNT(*) AS requests,
                   COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                   SUM(CASE WHEN shadow_backend IS NOT NULL THEN 1 ELSE 0 END)
                       AS candidate_requests,
                   COALESCE(SUM(CASE WHEN shadow_backend IS NOT NULL
                                     THEN cost_usd ELSE 0 END), 0.0)
                       AS candidate_actual_cost,
                   COALESCE(SUM(shadow_est_cost_usd), 0.0) AS candidate_shadow_cost
            FROM requests
            WHERE tenant = ? AND ts >= ? AND status = 'ok'
            """,
            (tenant, since_ts),
        ).fetchone()

    requests = row["requests"]
    candidate_requests = row["candidate_requests"] or 0
    return {
        "tenant": tenant,
        "requests": requests,
        "total_cost_usd": round(row["total_cost_usd"], 6),
        "per_backend": store.cost_summary(tenant, since_ts),
        "shadow": {
            "candidate_requests": candidate_requests,
            "candidate_share": (candidate_requests / requests) if requests else 0.0,
            "est_savings_usd": round(
                row["candidate_actual_cost"] - row["candidate_shadow_cost"], 6
            ),
        },
    }


def main() -> None:
    db_path = sys.argv[1]
    tenant = sys.argv[2] if len(sys.argv) > 2 else "default"
    since_ts = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    report = build_report(TelemetryStore(db_path), tenant, since_ts)
    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
