# tests/india/test_costing.py
import pytest
import yaml

from photon.india.costing import (
    ProviderBook,
    ProviderSpec,
    cost_table,
    rent_vs_own_breakeven_hours,
)

_BOOK = {
    "fx": {"usd_to_inr": 83.5, "source": "RBI reference 2026-07-01"},
    "gst_rate": 0.18,
    "providers": [
        {"name": "indian-cheap", "region": "in-north-1", "data_residency": "india",
         "inr_per_hour": 150.0, "measured_tokens_per_second": 500.0},
        {"name": "hyperscaler", "region": "ap-south-1", "data_residency": "india",
         "inr_per_hour": 400.0, "measured_tokens_per_second": 500.0},
        {"name": "unpriced-placeholder", "region": "x", "inr_per_hour": 0.0,
         "measured_tokens_per_second": 0.0},
    ],
}


def test_placeholders_are_excluded_never_defaulted():
    book = ProviderBook.model_validate(_BOOK)
    names = [r.name for r in cost_table(book)]
    assert "unpriced-placeholder" not in names  # no fictional price invented
    assert names == ["indian-cheap", "hyperscaler"]  # cheapest first


def test_cost_is_derived_from_measured_throughput_with_gst():
    book = ProviderBook.model_validate(_BOOK)
    cheap = cost_table(book)[0]
    # 150 ₹/hr ÷ 1.8M tokens/hr = ₹83.33 per 1M tokens
    assert cheap.inr_per_1m_tokens == pytest.approx(150.0 / 1.8, rel=1e-6)
    assert cheap.inr_per_1m_tokens_with_gst == pytest.approx(150.0 / 1.8 * 1.18, rel=1e-6)


def test_is_priced_requires_both_cost_and_measurement():
    assert ProviderSpec(name="a", inr_per_hour=150, measured_tokens_per_second=500).is_priced
    assert not ProviderSpec(name="b", inr_per_hour=150, measured_tokens_per_second=0).is_priced
    assert not ProviderSpec(name="c", inr_per_hour=0, measured_tokens_per_second=500).is_priced


def test_rate_requires_real_provenance():
    book = ProviderBook.model_validate(_BOOK)
    assert book.rate().usd_to_inr == 83.5
    placeholder = ProviderBook.model_validate(
        {**_BOOK, "fx": {"usd_to_inr": 0.0, "source": "SET ME — e.g. ..."}}
    )
    assert placeholder.rate() is None  # unset FX yields no rate, not a default


def test_example_config_ships_as_placeholders_only():
    # the shipped example must never contain quotable numbers
    book = ProviderBook.from_yaml("config/india_providers.example.yaml")
    assert cost_table(book) == []
    assert book.rate() is None


def test_rent_vs_own_breakeven():
    # own at ₹50/hr amortised vs rent at ₹150/hr → own wins above ~243 hrs/month
    hours = rent_vs_own_breakeven_hours(rent_inr_per_hour=150.0, own_inr_per_hour_amortised=50.0)
    assert hours == pytest.approx(50.0 * 730 / 150.0)
    # owning never cheaper → honest None
    assert rent_vs_own_breakeven_hours(100.0, 100.0) is None
    assert rent_vs_own_breakeven_hours(0.0, 50.0) is None
