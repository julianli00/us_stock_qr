from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import ordered_path_growth_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def path_from_returns(values):
    stock = np.r_[1.0, np.cumprod(values)]
    return np.column_stack([stock, np.ones(len(stock)), np.ones(len(stock))])


def test_peak_drawdown_is_not_initial_capital_floor():
    paths = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.5, 1.0, 1.0]])
    weights = np.array([0.98, 0.0, 0.0])
    _, a, b, _, _, _ = rules.problem(paths)
    wealth = 0.02 + paths @ weights
    assert wealth.min() >= 0.85
    assert rules.drawdown(paths, weights)[0] > 0.24
    assert (a @ weights > b).any()


def test_same_daily_multiset_and_endpoint_have_different_ordered_risk_budgets():
    alternate = path_from_returns([1.3, 0.8, 1.3, 0.8])
    clustered = path_from_returns([1.3, 1.3, 0.8, 0.8])
    assert alternate[-1, 0] == pytest.approx(clustered[-1, 0])
    first, a = rules.allocate(alternate)
    second, b = rules.allocate(clustered)
    assert first[0] > second[0] + 0.3
    for weights, record, paths in ((first, a, alternate), (second, b, clustered)):
        rules.verify_plan(weights, record, paths)
        assert rules.drawdown(paths, weights)[0] <= 0.15 + rules.FEAS_TOL
        assert weights.sum() == pytest.approx(0.98)
        assert weights[0] >= 0.147 - rules.FEAS_TOL


def test_flat_tie_prefers_bills_with_genuine_stock_floor():
    paths = np.ones((6, 3))
    weights, record = rules.allocate(paths)
    np.testing.assert_allclose(weights, [0.147, 0, 0.833], atol=1e-8)
    assert all(rules.scatter(weights)[name] > 0 for name in rules.CORE)
    rules.verify_plan(weights, record, paths)


def test_infeasible_family_floor_stops_without_cash_fallback():
    paths = np.array([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(QuantError, match="without fallback"):
        rules.allocate(paths)


def test_sparse_dual_corruption_is_detected():
    paths = path_from_returns([1.3, 1.3, 0.8, 0.8])
    weights, record = rules.allocate(paths)
    corrupted = copy.deepcopy(record)
    sparse = corrupted["primary_solution"]["nonzero_inequality_dual"]
    assert sparse
    sparse[0]["value"] += 0.1
    with pytest.raises(QuantError):
        rules.verify_plan(weights, corrupted, paths)


@pytest.mark.parametrize("kind", ["nan", "zero", "wrong_initial", "wrong_width"])
def test_bad_path_data_never_produces_a_portfolio(kind):
    paths = np.ones((5, 3))
    if kind == "nan":
        paths[2, 0] = np.nan
    elif kind == "zero":
        paths[2, 0] = 0
    elif kind == "wrong_initial":
        paths[0, 0] = 1.1
    else:
        paths = paths[:, :2]
    with pytest.raises(QuantError):
        rules.allocate(paths)


def test_constant_units_and_prefix_invariance_preserve_full_warmup():
    dates = sessions("2015-08-10", "2016-10-05")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0002 * (i + 1) * t + 0.002 * np.sin(t / (i + 4)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )
    data = MarketData(
        close * 0.9999,
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(data, policy)
    target = full[rules.CANDIDATES[0]["id"]]
    assert target.iloc[:252].isna().all().all()
    assert target.dropna().index.strftime("%Y-%m-%d").tolist() == ["2016-08-31", "2016-09-30"]
    assert all(row["ordered_peak_trough_constraints"] == 31878 for row in audits)
    end = pd.Timestamp("2016-08-31")
    sliced = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    prefix, _ = rules.build_targets(sliced, policy)
    pd.testing.assert_frame_equal(prefix[rules.CANDIDATES[0]["id"]], target.loc[:end])
    np.testing.assert_allclose(
        rules.observed_paths(close, end), rules.independent_paths(close, end), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_return_sessions", 126),
        ("training_maximum_drawdown", 0.2),
        ("stock_target_minimum", 0.01),
        ("cash_reserve", 0.0),
        ("actual_weight_trade_band", 0.01),
    ],
)
def test_fixed_research_contract_is_not_retuned(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
