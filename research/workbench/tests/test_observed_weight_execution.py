from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import observed_weight_execution as execution
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, previous_session, next_session
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/observed-weight-execution.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "IEF", "TLT", "GLD", "BIL", *execution.FACTORS, "XLK", "SOXX"))


def source(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for position, day in enumerate(frame.index):
        if position >= 252 and is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, list(execution.FACTORS)] = 0.98 * 0.6 / 8
            frame.loc[day, "SOXX"] = 0.98 * 0.6 / 2
            frame.loc[day, "GLD"] = 0.98 * 0.4
    return frame


def constant(data):
    close = data.close.copy()
    close[:] = 100.0
    return replace(data, close=close, open=close.copy(), raw_close=close.copy())


def test_reused_band_has_no_leverage_or_volatility_override(market, policy):
    candidate, _ = execution.pure_band(policy)
    assert candidate["weight_band"] == 0.025
    assert candidate["qld_cap"] is candidate["volatility_target"] is None
    desired = source(market).dropna(how="all").iloc[0]
    current = desired.copy()
    covariance = pd.DataFrame(0.0, index=current.index, columns=current.index)
    current["SOXX"] += 0.025
    current["GLD"] -= 0.025
    assert execution.prior.rebalance_reason(current, desired, covariance, candidate) == "observed_weight_gap"
    current["SOXX"] -= 0.001
    current["GLD"] += 0.001
    assert execution.prior.rebalance_reason(current, desired, covariance, candidate) == "hold"


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_initial_deployment_always_executes_and_constant_holdings_skip_monthly_noise(market, policy, cost, delay):
    data = constant(market)
    own, issued, decisions = execution.run(data, source(data), execution.CANDIDATES[0], policy,
                                           "2016-10-06", "2017-06-30", cost, delay)
    assert len(issued.dropna(how="all")) == 1
    assert decisions[0]["reason"] == "initial_capital"
    assert issued.dropna(how="all").index[0] == previous_session(pd.Timestamp("2016-10-06"))
    assert all(row["reason"] == "hold" for row in decisions[1:])
    first_fill = pd.Timestamp("2016-10-06") if delay == 1 else next_session(pd.Timestamp("2016-10-06"))
    assert own.frame.loc[first_fill, "orders"] > 0
    assert own.frame.loc[own.frame.index != first_fill, "orders"].eq(0).all()
    if delay == 2:
        assert own.frame.iloc[0]["equity"] == 10000


def test_actual_drift_can_trigger_even_when_last_target_is_unchanged(market, policy):
    data = constant(market)
    close = data.close.copy()
    close.loc[close.index >= "2016-11-01", "SOXX"] *= 1.5
    changed = replace(data, close=close, open=close.copy(), raw_close=close.copy())
    baseline = source(changed)
    _, issued, decisions = execution.run(changed, baseline, execution.CANDIDATES[0], policy,
                                         "2016-10-06", "2017-06-30", 5.0, 1)
    assert any(row["reason"] == "observed_weight_gap" for row in decisions)
    desired = baseline.dropna(how="all").iloc[0]
    for _, target in issued.dropna(how="all").iterrows():
        np.testing.assert_allclose(target, desired, rtol=0, atol=1e-12)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_issued_original_targets_replay_through_funded_v2_and_independent_bt(market, policy, cost, delay):
    own, issued, _ = execution.run(market, source(market), execution.CANDIDATES[0], policy,
                                   "2016-10-06", "2017-06-30", cost, delay)
    replay = simulate(market, issued, "2016-10-06", "2017-06-30",
                      initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, issued, "2016-10-06", "2017-06-30",
                                 capital=10000, cost_bps=cost, commission=1, delay=delay)
    pd.testing.assert_frame_equal(own.frame, replay.frame)
    pd.testing.assert_frame_equal(own.weights, replay.weights)
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].min() >= 0


def test_future_prices_cannot_change_prior_fills_or_closing_weight_decisions(market, policy):
    baseline = source(market)
    before, before_issued, _ = execution.run(market, baseline, execution.CANDIDATES[0], policy,
                                            "2016-10-06", "2017-06-30", 20.0, 2)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multiplier = np.linspace(1, 1.5, later.sum())
    close.loc[later, "SOXX"] *= multiplier
    opening.loc[later, "SOXX"] *= multiplier
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, after_issued, _ = execution.run(changed, baseline, execution.CANDIDATES[0], policy,
                                          "2016-10-06", "2017-06-30", 20.0, 2)
    pd.testing.assert_frame_equal(before.frame.loc[:boundary], after.frame.loc[:boundary])
    pd.testing.assert_frame_equal(before_issued.loc[:boundary], after_issued.loc[:boundary])


def test_each_window_restarts_capital_and_uses_its_own_known_source_seed(market, policy):
    baseline = source(market)
    for start in ("2016-10-06", "2017-01-06"):
        result, issued, decisions = execution.run(market, baseline, execution.CANDIDATES[0], policy,
                                                  start, "2017-06-30", 20.0, 2)
        assert result.frame.iloc[0]["equity"] == 10000
        anchor = previous_session(pd.Timestamp(start))
        assert decisions[0]["signal_session"] == str(anchor.date())
        pd.testing.assert_series_equal(issued.loc[anchor], baseline.loc[:anchor].dropna(how="all").iloc[-1],
                                       check_names=False)


def test_end_of_window_request_is_retained_not_filled_or_charged_retroactively(market, policy):
    data = constant(market)
    baseline = source(data)
    day = pd.Timestamp("2017-06-30")
    baseline.loc[day, "BIL"] += 0.2
    baseline.loc[day, "SOXX"] -= 0.2
    result, issued, decisions = execution.run(data, baseline, execution.CANDIDATES[0], policy,
                                              "2016-10-06", "2017-06-30", 20.0, 2)
    assert not issued.loc[day].isna().any()
    assert pd.Timestamp(decisions[-1]["execution_session"]) > day
    assert result.frame.loc[day, "orders"] == 0
    assert result.frame.loc[day, "cost"] == 0


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "empty"])
def test_bad_original_source_does_not_fall_back_to_no_trade(market, policy, problem):
    baseline = source(market)
    day = baseline.dropna(how="all").index[0]
    if problem == "partial":
        baseline.iloc[0, 0] = 0.1
    elif problem == "negative":
        baseline.loc[day, "GLD"] = -0.1
    elif problem == "budget":
        baseline.loc[day] *= 2
    else:
        baseline[:] = np.nan
    with pytest.raises(QuantError):
        execution.run(market, baseline, execution.CANDIDATES[0], policy, "2016-10-06", "2017-06-30", 5.0, 1)


@pytest.mark.parametrize("field,value", [("actual_weight_trade_band", 0.05), ("capital_usd", 100000),
                                        ("commission_per_order", 0.0), ("cash_reserve", 0.0),
                                        ("new_economic_factor_definitions", 1), ("new_configurations", 10)])
def test_original_band_and_costs_cannot_be_tuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        execution.validate(policy)


def test_finite_generator_refuses_an_unbound_execution_or_source_helper():
    spec = {"id": execution.CANDIDATES[0]["id"], "configuration": execution.CANDIDATES[0],
            "frozen_files": {"src/us_quant/observed_weight_execution.py": "0" * 64,
                             "config/observed-weight-execution.json": "0" * 64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
