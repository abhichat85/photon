# tests/india/test_inr.py
import pytest

from photon.india.inr import (
    GST_RATE_SERVICES,
    InrRate,
    format_inr,
    gpu_cost_per_1m_tokens_inr,
    to_inr,
    with_gst,
)


def test_usd_to_inr_uses_configured_rate_not_a_hardcoded_one():
    rate = InrRate(usd_to_inr=83.5, source="RBI reference 2026-07-01")
    assert to_inr(10.0, rate) == pytest.approx(835.0)
    assert rate.source  # provenance is mandatory — a rate without a source is a guess


def test_gst_default_is_18_percent_services():
    assert GST_RATE_SERVICES == 0.18
    net, gst, gross = with_gst(1000.0)
    assert net == 1000.0
    assert gst == pytest.approx(180.0)
    assert gross == pytest.approx(1180.0)


def test_gst_rate_is_overridable():
    _, gst, gross = with_gst(1000.0, rate=0.05)
    assert gst == pytest.approx(50.0)
    assert gross == pytest.approx(1050.0)


def test_format_inr_uses_indian_digit_grouping():
    # Indian numbering: thousands, then lakhs (2-digit groups) — 12,34,567
    assert format_inr(1234567) == "₹12,34,567.00"
    assert format_inr(100000) == "₹1,00,000.00"
    assert format_inr(999) == "₹999.00"
    assert format_inr(1234.5) == "₹1,234.50"


def test_format_inr_handles_zero_and_negative():
    assert format_inr(0) == "₹0.00"
    assert format_inr(-4500) == "-₹4,500.00"


def test_gpu_cost_per_1m_tokens_derives_from_measured_throughput():
    # ₹150/hr instance measured at 500 tokens/sec → 1.8M tokens/hr
    # cost per 1M tokens = 150 / 1.8 = ₹83.33
    cost = gpu_cost_per_1m_tokens_inr(inr_per_hour=150.0, tokens_per_second=500.0)
    assert cost == pytest.approx(150.0 / 1.8, rel=1e-6)


def test_gpu_cost_rejects_zero_throughput():
    # you cannot derive a cost from an unmeasured throughput — no silent divide
    assert gpu_cost_per_1m_tokens_inr(inr_per_hour=150.0, tokens_per_second=0.0) is None
