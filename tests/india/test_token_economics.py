# tests/india/test_token_economics.py
import pytest

from photon.config import ModelPricing
from photon.india.script import Script
from photon.india.token_economics import (
    TokenEfficiencyStore,
    cost_per_1k_chars,
    cheapest_backend_for_script,
)


@pytest.fixture
def store(tmp_path) -> TokenEfficiencyStore:
    return TokenEfficiencyStore(tmp_path / "tokeff.db")


def test_records_and_averages_chars_per_token(store):
    # 400 chars → 100 tokens = 4.0 chars/token (typical English efficiency)
    store.record("big", Script.LATIN, chars=400, tokens=100)
    store.record("big", Script.LATIN, chars=200, tokens=50)
    assert store.chars_per_token("big", Script.LATIN) == pytest.approx(4.0)


def test_unmeasured_pair_returns_none_not_a_guess(store):
    # the whole point: we never invent a ratio we haven't measured
    assert store.chars_per_token("big", Script.DEVANAGARI) is None


def test_indic_penalty_is_visible_in_measurements(store):
    # same model: English 4 chars/token, Hindi 1.25 → Hindi costs 3.2x per char
    store.record("big", Script.LATIN, chars=400, tokens=100)
    store.record("big", Script.DEVANAGARI, chars=250, tokens=200)
    latin = store.chars_per_token("big", Script.LATIN)
    hindi = store.chars_per_token("big", Script.DEVANAGARI)
    assert latin / hindi == pytest.approx(3.2)


def test_cost_per_1k_chars_uses_measured_efficiency():
    pricing = ModelPricing(input_per_1m=1.0, output_per_1m=1.0)
    # at 4 chars/token, 1000 chars = 250 tokens = 250/1e6 * $1.0
    assert cost_per_1k_chars(pricing, chars_per_token=4.0) == pytest.approx(250 / 1e6)
    # at 1.25 chars/token, 1000 chars = 800 tokens → 3.2x the cost
    assert cost_per_1k_chars(pricing, chars_per_token=1.25) == pytest.approx(800 / 1e6)


def test_cost_per_1k_chars_rejects_nonpositive_efficiency():
    pricing = ModelPricing(input_per_1m=1.0, output_per_1m=1.0)
    assert cost_per_1k_chars(pricing, chars_per_token=0.0) is None


def test_cheapest_for_script_flips_with_tokenizer_quality(store):
    """The arbitrage this whole module exists for: a model that is MORE
    expensive per token can be CHEAPER for Hindi if its tokenizer is
    Indic-efficient. Per-token pricing alone picks the wrong model."""
    pricing = {
        "latin-optimised": ModelPricing(input_per_1m=0.50, output_per_1m=0.50),
        "indic-optimised": ModelPricing(input_per_1m=0.90, output_per_1m=0.90),
    }
    # English: both tokenize well; the cheaper-per-token model wins
    store.record("latin-optimised", Script.LATIN, chars=400, tokens=100)   # 4.0
    store.record("indic-optimised", Script.LATIN, chars=400, tokens=100)   # 4.0
    assert cheapest_backend_for_script(store, pricing, Script.LATIN) == "latin-optimised"

    # Hindi: the Latin-optimised model shreds Devanagari (1.0 c/t) while the
    # Indic one holds 3.0 c/t → despite 1.8x the token price, it wins on cost.
    store.record("latin-optimised", Script.DEVANAGARI, chars=100, tokens=100)  # 1.0
    store.record("indic-optimised", Script.DEVANAGARI, chars=300, tokens=100)  # 3.0
    assert cheapest_backend_for_script(store, pricing, Script.DEVANAGARI) == "indic-optimised"


def test_cheapest_ignores_unmeasured_backends(store):
    pricing = {
        "measured": ModelPricing(input_per_1m=1.0, output_per_1m=1.0),
        "unmeasured": ModelPricing(input_per_1m=0.01, output_per_1m=0.01),
    }
    store.record("measured", Script.TAMIL, chars=300, tokens=100)
    # 'unmeasured' looks cheap per token but has no measurement → not eligible
    assert cheapest_backend_for_script(store, pricing, Script.TAMIL) == "measured"


def test_cheapest_returns_none_when_nothing_measured(store):
    pricing = {"a": ModelPricing(input_per_1m=1.0, output_per_1m=1.0)}
    assert cheapest_backend_for_script(store, pricing, Script.ODIA) is None


def test_summary_reports_penalty_vs_latin(store):
    store.record("big", Script.LATIN, chars=400, tokens=100)       # 4.0
    store.record("big", Script.DEVANAGARI, chars=200, tokens=100)  # 2.0
    rows = {r["script"]: r for r in store.summary("big")}
    assert rows["devanagari"]["chars_per_token"] == pytest.approx(2.0)
    assert rows["devanagari"]["indic_penalty_vs_latin"] == pytest.approx(2.0)
    assert rows["latin"]["indic_penalty_vs_latin"] == pytest.approx(1.0)
