# photon/finetune/dataprep.py
"""Customer-data preparation for fine-tuning: validate chat-format JSONL,
scrub obvious PII, dedupe, and produce deterministic train/eval splits.

The regex scrub is a BASELINE. Enterprise deployments layer domain-specific
scrubbing (e.g. Microsoft Presidio) on top per engagement — buy, don't build."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import BaseModel

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}")

VALID_ROLES = {"system", "user", "assistant"}


class PrepReport(BaseModel):
    total: int
    kept: int
    dropped_invalid: int
    dropped_duplicate: int
    train: int
    eval: int


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def validate_example(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES:
            return False
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
    return msgs[-1]["role"] == "assistant"


def prepare(
    in_path: str | Path,
    out_dir: str | Path,
    eval_fraction: float = 0.05,
    seed: int = 13,
) -> PrepReport:
    total = dropped_invalid = dropped_duplicate = 0
    kept: list[dict] = []
    seen: set[str] = set()

    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped_invalid += 1
                continue
            if not validate_example(obj):
                dropped_invalid += 1
                continue
            for m in obj["messages"]:
                m["content"] = scrub_pii(m["content"])
            key = json.dumps(obj["messages"], sort_keys=True)
            if key in seen:
                dropped_duplicate += 1
                continue
            seen.add(key)
            kept.append(obj)

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_eval = int(len(kept) * eval_fraction)
    if eval_fraction > 0 and n_eval == 0 and len(kept) > 1:
        n_eval = 1
    eval_set, train_set = kept[:n_eval], kept[n_eval:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_set), ("eval.jsonl", eval_set)):
        with open(out / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return PrepReport(
        total=total,
        kept=len(kept),
        dropped_invalid=dropped_invalid,
        dropped_duplicate=dropped_duplicate,
        train=len(train_set),
        eval=len(eval_set),
    )


if __name__ == "__main__":
    import sys

    print(prepare(sys.argv[1], sys.argv[2]).model_dump())
