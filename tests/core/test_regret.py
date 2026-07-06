# tests/core/test_regret.py
import pytest

from photon.core.regret import ReplayRow, router_regret


def test_zero_regret_when_router_matches_oracle():
    # oracle picks the cheapest acceptable; router did too
    rows = [
        ReplayRow(chosen_cost=0.001, cheapest_acceptable_cost=0.001),
        ReplayRow(chosen_cost=0.010, cheapest_acceptable_cost=0.010),
    ]
    assert router_regret(rows) == 0.0


def test_regret_is_mean_excess_cost():
    rows = [
        ReplayRow(chosen_cost=0.010, cheapest_acceptable_cost=0.001),  # excess 0.009
        ReplayRow(chosen_cost=0.005, cheapest_acceptable_cost=0.005),  # excess 0.0
    ]
    assert router_regret(rows) == pytest.approx(0.0045)


def test_empty_replay_is_zero_regret():
    assert router_regret([]) == 0.0
