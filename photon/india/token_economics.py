# photon/india/token_economics.py
"""Language-fair cost accounting — the economic core of Photon-for-India.

THE PROBLEM. LLM pricing is per token, but a token is not a constant amount of
meaning. A Latin-optimised BPE fragments Devanagari, Tamil, Telugu and the other
Indic scripts into far more tokens per unit of semantic content than English.
The consequence is structural, not cosmetic: the same question asked in Hindi
can cost several times what it costs in English, on the same model, for the same
answer. Any cost model denominated purely in tokens systematically mis-prices
Indian traffic — and any router that optimises $/token will keep choosing the
wrong model for Indic requests.

THE FIX. Measure chars-per-token per (backend, script) from real traffic, then
compare models in a language-fair unit: cost per 1,000 characters of content.
Under that unit a model with a HIGHER token price can be genuinely cheaper for
Hindi, because its tokenizer emits fewer tokens for the same text. That
inversion is the arbitrage; `cheapest_backend_for_script` is where it is taken.

DISCIPLINE. Nothing here assumes a ratio. `chars_per_token` returns None for a
(backend, script) pair that has never been measured, and unmeasured backends are
ineligible for selection — the project's 'benchmark precedes the number' rule
applied to tokenizer economics. Ratios accumulate from live usage; the numbers
are yours, measured on your traffic, not vendor claims."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from photon.config import ModelPricing
from photon.india.script import Script

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_efficiency (
    backend TEXT NOT NULL,
    script TEXT NOT NULL,
    chars INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    samples INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (backend, script)
);
"""


class TokenEfficiencyStore:
    """Accumulating chars/tokens per (backend, script). Same SQLite posture as
    the rest of the stores (WAL, per-call connection, single replica — D1/D3)."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, backend: str, script: Script, chars: int, tokens: int) -> None:
        """Fold one observation in. Zero/negative token counts are ignored —
        they'd corrupt the ratio and mean nothing."""
        if tokens <= 0 or chars <= 0:
            return
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO token_efficiency (backend, script, chars, tokens, samples)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(backend, script) DO UPDATE SET
                     chars = chars + excluded.chars,
                     tokens = tokens + excluded.tokens,
                     samples = samples + 1""",
                (backend, script.value, chars, tokens),
            )

    def chars_per_token(self, backend: str, script: Script) -> float | None:
        """Measured efficiency, or None if never observed. None means 'we do
        not know' — never a default, never a guess."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chars, tokens FROM token_efficiency WHERE backend = ? AND script = ?",
                (backend, script.value),
            ).fetchone()
        if row is None or not row["tokens"]:
            return None
        return row["chars"] / row["tokens"]

    def summary(self, backend: str) -> list[dict]:
        """Per-script efficiency for one backend, with the Indic penalty made
        explicit: how many times more tokens this script costs per character
        than Latin on the same model (1.0 = parity, 3.0 = three times worse)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT script, chars, tokens, samples FROM token_efficiency WHERE backend = ?",
                (backend,),
            ).fetchall()
        latin = next((r for r in rows if r["script"] == Script.LATIN.value), None)
        latin_cpt = (latin["chars"] / latin["tokens"]) if latin and latin["tokens"] else None
        out = []
        for r in rows:
            cpt = r["chars"] / r["tokens"] if r["tokens"] else None
            out.append({
                "script": r["script"],
                "chars_per_token": cpt,
                "samples": r["samples"],
                "indic_penalty_vs_latin": (latin_cpt / cpt) if (cpt and latin_cpt) else None,
            })
        return out


def cost_per_1k_chars(pricing: ModelPricing, chars_per_token: float) -> float | None:
    """The language-fair unit: USD to process 1,000 characters of content on
    this backend, given its measured tokenizer efficiency for the script in
    question. Uses input pricing (the prompt side is where script dominates)."""
    if chars_per_token <= 0:
        return None
    tokens = 1000.0 / chars_per_token
    return tokens * pricing.input_per_1m / 1_000_000


def cheapest_backend_for_script(
    store: TokenEfficiencyStore,
    pricing: dict[str, ModelPricing],
    script: Script,
) -> str | None:
    """Cheapest backend for this script by cost-per-1k-characters — the
    comparison that can invert per-token ranking for Indic input. Backends with
    no measurement for this script are ineligible (we don't guess). Returns None
    if nothing has been measured yet."""
    best_name, best_cost = None, None
    for name, price in pricing.items():
        cpt = store.chars_per_token(name, script)
        if cpt is None:
            continue
        cost = cost_per_1k_chars(price, cpt)
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_name, best_cost = name, cost
    return best_name
