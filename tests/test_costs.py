# tests/test_costs.py
import pytest

from photon.config import ModelPricing
from photon.costs import compute_cost_usd


def test_cost_is_tokens_times_price_per_million():
    pricing = ModelPricing(input_per_1m=0.9, output_per_1m=2.7)
    # 10 input + 5 output tokens
    assert compute_cost_usd(pricing, 10, 5) == pytest.approx((10 * 0.9 + 5 * 2.7) / 1_000_000)


def test_zero_tokens_costs_zero():
    pricing = ModelPricing(input_per_1m=0.9, output_per_1m=2.7)
    assert compute_cost_usd(pricing, 0, 0) == 0.0
