# photon/core/regret.py
"""Router regret: mean excess cost vs. an oracle that always picks the cheapest
model that would have been acceptable in hindsight. The north-star internal
metric for the learning loop — lower is better, 0 is oracle-optimal."""
from __future__ import annotations

from pydantic import BaseModel


class ReplayRow(BaseModel):
    chosen_cost: float               # what the router's decision actually cost
    cheapest_acceptable_cost: float  # oracle-in-hindsight cost


def router_regret(rows: list[ReplayRow]) -> float:
    if not rows:
        return 0.0
    excess = [max(0.0, r.chosen_cost - r.cheapest_acceptable_cost) for r in rows]
    return sum(excess) / len(rows)
