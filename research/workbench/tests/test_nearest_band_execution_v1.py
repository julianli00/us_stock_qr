from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import nearest_band_execution_v1 as rules
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_known_projection_stops_at_band_edge_instead_of_center():
    current = np.array([0.60, 0.38, 0.0])
    center = np.array([0.49, 0.49, 0.0])
    target, proof = rules.project(current, center)
    np.testing.assert_allclose(target, [0.515, 0.465, 0], rtol=0, atol=1e-12)
    assert proof["squared_weight_movement"] < proof["original_center_squared_movement"]
    assert proof["minimum_fee_or_future_cost_claim"] is False
    np.testing.assert_allclose(target, rules.independent_projection(current, center), atol=1e-12)


def test_projection_restores_budget_without_introducing_unused_assets():
    current = np.array([0.50, 0.47, 0.0])
    desired = np.array([0.49, 0.49, 0.0])
    target, _ = rules.project(current, desired)
    np.testing.assert_allclose(target, [0.505, 0.475, 0], rtol=0, atol=1e-12)
    assert target.sum() == pytest.approx(0.98)
    assert target[-1] == 0


def test_projection_of_feasible_weights_is_identity():
    current = np.array([0.50, 0.48])
    target, _ = rules.project(current, np.array([0.49, 0.49]))
    np.testing.assert_allclose(target, current, rtol=0, atol=1e-12)


def test_random_bisection_and_breakpoint_oracles_agree():
    rng = np.random.default_rng(42)
    for _ in range(40):
        desired = rng.dirichlet(np.ones(7)) * 0.98
        current = rng.dirichlet(np.ones(8))[:7]
        target, proof = rules.project(current, desired)
        assert abs(target.sum() - 0.98) < 1e-12
        assert np.max(abs(target - desired)) <= 0.025 + 1e-12
        assert proof["independent_breakpoint_maximum_difference"] <= 1e-12


@pytest.mark.parametrize("damage", ["negative", "nan", "leverage", "budget", "shape"])
def test_invalid_projection_input_does_not_fall_back(damage):
    current, desired = np.array([0.50, 0.48]), np.array([0.49, 0.49])
    if damage == "negative":
        current[0] = -0.1
    elif damage == "nan":
        current[0] = np.nan
    elif damage == "leverage":
        current[0] = 0.9
    elif damage == "budget":
        desired[0] = 0.6
    else:
        desired = desired[:1]
    with pytest.raises(QuantError):
        rules.project(current, desired)


def test_original_fixed_fee_engine_not_cheapest_algebraic_root_is_preserved():
    nav, values, rate = 10000.0, np.array([3000.0, 3000.0, 3800.0]), 0.0005
    cost = (2 + rate * (3800 - 0.23 * nav)) / (1 - 0.23 * rate)
    post = nav - cost
    proposed = np.array([0.375 * post, 0.605 * post - 3800, 3800.0])
    target = proposed / post
    algebraic = rate * abs(proposed - values).sum() + np.count_nonzero(
        abs(proposed - values) > 1e-6
    )
    actual, cash, charged, _, orders = rebalance(values, 200.0, target, 5.0, 1.0)
    assert post + algebraic == pytest.approx(nav)
    assert orders == 3 and charged > algebraic
    assert actual.sum() + cash + charged == pytest.approx(nav)
    assert cash >= 0


def fixture():
    dates = sessions("2019-01-02", "2020-04-03")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0001 * t + 0.002 * np.sin(t / (8 + i)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )
    desired = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    for day, share in (
        ("2019-12-31", 0.20),
        ("2020-01-31", 0.40),
        ("2020-02-28", 0.20),
        ("2020-03-31", 0.40),
    ):
        desired.loc[pd.Timestamp(day)] = 0
        desired.loc[pd.Timestamp(day), "SOXX"] = share
        desired.loc[pd.Timestamp(day), "GLD"] = 0.98 - share
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    return data, desired


@pytest.mark.parametrize("delay,cost", [(1, 5.0), (2, 20.0)])
def test_initial_center_then_monthly_projected_targets_use_original_delays(delay, cost):
    data, desired = fixture()
    result, issued, decisions = rules.run(
        data, desired, read_json(rules.POLICY), "2020-01-08", "2020-04-03", cost, delay
    )
    anchor = previous_session("2020-01-08")
    np.testing.assert_allclose(issued.loc[anchor], desired.ffill().loc[anchor], atol=1e-12)
    assert len(decisions) == 4 and decisions[0]["reason"] == "initial_capital"
    assert (result.frame["cash"] >= 0).all() and (result.frame["gross_exposure"] <= 1).all()
    for row in decisions:
        day = pd.Timestamp(row["signal_session"])
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        assert row["execution_session"] == str(execution.date())
        if row["reason"] == "nearest_band":
            actual = result.weights.loc[day].to_numpy()
            oracle = rules.independent_projection(actual, desired.loc[day].to_numpy())
            np.testing.assert_allclose(issued.loc[day], oracle, rtol=0, atol=1e-12)
            assert row["projection_certificate"]["minimum_fee_or_future_cost_claim"] is False
    assert len(issued.dropna()) == len([row for row in decisions if row["reason"] != "hold"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("weight_band", 0.05),
        ("cash_reserve", 0.01),
        ("commission_per_order", 0.0),
        ("projection_iterations", 50),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_fixed_execution_law_cannot_be_tuned(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
