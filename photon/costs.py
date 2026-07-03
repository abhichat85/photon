# photon/costs.py
from photon.config import ModelPricing


def compute_cost_usd(pricing: ModelPricing, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m
    ) / 1_000_000
