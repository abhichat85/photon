# photon/india/script.py
"""Script detection for Indian-language routing.

Why this exists in an inference engine: token-based pricing is denominated in a
unit whose cost-per-meaning varies enormously by script. A Latin-optimised BPE
splits Devanagari or Tamil into far more tokens per unit of semantic content
than English, so the SAME question costs multiples more in Hindi than in
English on the same model. Routing and cost accounting that ignore script
systematically mis-price Indian traffic.

Detection is by Unicode block over letter characters only (digits/punctuation/
whitespace are ignored), and the dominant script wins — which is the right
behaviour for code-mixed 'Hinglish' text, the common case in Indian products.
No model, no dependency, microseconds: safe on the hot routing path."""
from __future__ import annotations

from enum import Enum

# Unicode blocks for the scripts of the Indian constitution's scheduled
# languages that have distinct blocks, plus Latin (English/romanised input).
_BLOCKS: list[tuple[str, int, int]] = [
    ("DEVANAGARI", 0x0900, 0x097F),  # Hindi, Marathi, Nepali, Sanskrit, Konkani
    ("BENGALI", 0x0980, 0x09FF),     # Bengali, Assamese
    ("GURMUKHI", 0x0A00, 0x0A7F),    # Punjabi
    ("GUJARATI", 0x0A80, 0x0AFF),
    ("ODIA", 0x0B00, 0x0B7F),
    ("TAMIL", 0x0B80, 0x0BFF),
    ("TELUGU", 0x0C00, 0x0C7F),
    ("KANNADA", 0x0C80, 0x0CFF),
    ("MALAYALAM", 0x0D00, 0x0D7F),
]


class Script(str, Enum):
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    GURMUKHI = "gurmukhi"
    GUJARATI = "gujarati"
    ODIA = "odia"
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    LATIN = "latin"
    UNKNOWN = "unknown"

    @property
    def is_indic(self) -> bool:
        return self not in (Script.LATIN, Script.UNKNOWN)


def _script_of_char(ch: str) -> Script | None:
    """Script of a single character, or None if it isn't a letter we score."""
    code = ord(ch)
    for name, lo, hi in _BLOCKS:
        if lo <= code <= hi:
            return Script[name]
    if ch.isalpha() and code < 0x0250:  # Basic Latin + Latin-1/Extended-A
        return Script.LATIN
    return None


def script_mix(text: str) -> dict[Script, float]:
    """Proportion of scored letters belonging to each script. Empty for text
    with no letters (digits/punctuation only)."""
    counts: dict[Script, int] = {}
    total = 0
    for ch in text:
        script = _script_of_char(ch)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if not total:
        return {}
    return {s: n / total for s, n in counts.items()}


def detect_script(text: str) -> Script:
    """Dominant script of `text` — the routing-relevant answer for code-mixed
    input. UNKNOWN when there are no letters to judge by."""
    mix = script_mix(text)
    if not mix:
        return Script.UNKNOWN
    return max(mix.items(), key=lambda kv: kv[1])[0]


def messages_script(messages: list[dict]) -> Script:
    """Dominant script across an OpenAI-format message list (string content
    only; multimodal parts are skipped, matching the rest of the codebase)."""
    text = " ".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    )
    return detect_script(text)
