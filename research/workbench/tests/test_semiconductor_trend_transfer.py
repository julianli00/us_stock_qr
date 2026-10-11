from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import semiconductor_trend_transfer as trend
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import read_json
from us_quant.trend_factor_guard import gated_target

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/semiconductor-trend-transfer.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "IEF", "GLD", "BIL", *trend.FACTORS, "SOXX"))


def source(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for index, day in enumerate(frame.index):
        if index >= 63 and is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, list(trend.FACTORS)] = 0.98 * 0.6 / 8
            frame.loc[day, "SOXX"] = 0.98 * 0.6 / 2
            frame.loc[day, "GLD"] = 0.98 * 0.4
    return frame


def test_equity_signal_compounds_actual_contemporaneous_fixed_weight_returns(market):
    actual = trend.equity_signal(market.close)
    returns = market.close.pct_change(fill_method=None).iloc[1:]
    expected = (1 + returns["SOXX"] * 0.5 + returns[list(trend.FACTORS)].sum(axis=1) * 0.125).cumprod()
    assert actual.iloc[0] == 1.0
    np.testing.assert_allclose(actual.iloc[1:], expected, rtol=0, atol=1e-14)
    assert actual.index.equals(market.close.index)


@pytest.mark.parametrize("state", [(True, True), (False, True), (True, False), (False, False)])
def test_component_gate_reuses_old_helper_and_extends_only_its_equity_component_to_soxx(market, state):
    monthly = source(market).dropna(how="all").iloc[0]
    expected = gated_target(monthly, state)
    if not state[0]:
        expected["BIL"] += expected["SOXX"]
        expected["SOXX"] = 0
    pd.testing.assert_series_equal(trend.gate_components(monthly, state), expected)
    actual = trend.gate_components(monthly, state)
    assert actual.sum() == pytest.approx(0.98)
    if not state[0]:
        assert actual.loc[[*trend.FACTORS, "SOXX"]].eq(0).all()
    if not state[1]:
        assert actual["GLD"] == 0


def test_flat_price_equality_is_risk_off_not_a_passing_trend(market, policy):
    close = market.close.copy()
    close[:] = 100.0
    data = replace(market, close=close, raw_close=close.copy(), open=close.copy())
    frames, _ = trend.build_from_baseline(data, source(data), policy)
    for frame in frames.values():
        active = frame.dropna(how="all")
        assert not active.empty
        assert active["BIL"].eq(0.98).all()
        assert active.drop(columns="BIL").eq(0).all().all()


def test_relative_trends_use_three_fixed_signs_against_actual_bills(market, policy):
    frames, _ = trend.build_from_baseline(market, source(market), policy)
    monthly = source(market)
    levels = pd.DataFrame({"equity": trend.equity_signal(market.close), "gold": market.close["GLD"]})
    active = frames[trend.CANDIDATES[1]["id"]].dropna(how="all")
    assert all(is_month_end(day) for day in active.index)
    for day in active.index:
        index = market.close.index.get_loc(day)
        fractions = [
            sum(levels[name].iloc[index] / levels[name].iloc[index - lookback] - 1
                > market.close["BIL"].iloc[index] / market.close["BIL"].iloc[index - lookback] - 1
                for lookback in (63, 126, 252)) / 3
            for name in ("equity", "gold")
        ]
        np.testing.assert_allclose(active.loc[day, [*trend.FACTORS, "SOXX"]],
                                   monthly.loc[day, [*trend.FACTORS, "SOXX"]] * fractions[0], atol=1e-12)
        assert active.loc[day, "GLD"] == pytest.approx(monthly.loc[day, "GLD"] * fractions[1])
        assert active.loc[day].sum() == pytest.approx(0.98)


def test_future_prices_do_not_change_earlier_index_states_or_targets(market, policy):
    monthly = source(market)
    before, _ = trend.build_from_baseline(market, monthly, policy)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multiplier = np.linspace(1, 2, later.sum())
    close.loc[later, "SOXX"] *= multiplier
    opening.loc[later, "SOXX"] *= multiplier
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, _ = trend.build_from_baseline(changed, monthly, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])
    pd.testing.assert_series_equal(trend.equity_signal(market.close).loc[:boundary],
                                   trend.equity_signal(changed.close).loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_both_independent_seeded_windows_keep_funded_v2_and_bt(market, policy, cost, delay):
    frames, _ = trend.build_from_baseline(market, source(market), policy)
    for frame in frames.values():
        targets = seed_window(frame, "2016-10-06")
        own = simulate(market, targets, "2016-10-06", "2017-06-30",
                       initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        independent = independent_equity(market, targets, "2016-10-06", "2017-06-30",
                                         capital=10000, cost_bps=cost, commission=1, delay=delay)
        np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0
        if delay == 2:
            assert own.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "empty"])
def test_invalid_original_source_has_no_safe_signal_fallback(market, policy, problem):
    monthly = source(market)
    day = monthly.dropna(how="all").index[0]
    if problem == "partial":
        monthly.iloc[0, 0] = 0.1
    elif problem == "negative":
        monthly.loc[day, "BIL"] = -0.1
    elif problem == "budget":
        monthly.loc[day] *= 2
    else:
        monthly[:] = np.nan
    with pytest.raises(QuantError):
        trend.build_from_baseline(market, monthly, policy)


@pytest.mark.parametrize("field,value", [("sma_observations", 150), ("relative_trend_sessions", [126]),
                                        ("new_configurations", 10), ("new_economic_factor_definitions", 1),
                                        ("factor_symbols", ["SOXX"]), ("cash_reserve", 0.0)])
def test_prior_horizons_and_source_budget_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        trend.validate(policy)
