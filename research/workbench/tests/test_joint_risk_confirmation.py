from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import joint_risk_confirmation as joint
from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


def inputs():
    dates = pd.bdate_range("2026-09-28", periods=7)
    monthly = pd.DataFrame(np.nan, index=dates, columns=["SPY", "GLD", "BIL"])
    monthly.iloc[0] = [0.49, 0.49, 0]
    monthly.iloc[-1] = [0.40, 0.58, 0]
    terms = pd.DataFrame({"VIX": [18, 22, 18, 22, 22, 18, 22], "VIX3M": 20}, index=dates)
    dollar = pd.Series([0.01, -0.01], index=[dates[0], dates[-1]])
    return monthly, terms, dollar


def test_daily_risk_off_requires_both_not_either_input():
    monthly, terms, dollar = inputs()
    actual = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    active = actual.dropna(how="all")
    assert actual.iloc[0]["BIL"] == 0
    assert actual.iloc[1]["BIL"] == pytest.approx(0.49)
    assert actual.iloc[2]["BIL"] == 0
    assert actual.iloc[-1]["BIL"] == 0
    assert active.sum(axis=1).eq(0.98).all()
    dollar[:] = -0.01
    result = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    assert result.dropna(how="all")["BIL"].eq(0).all()
    assert len(result.dropna(how="all")) == len(monthly.dropna(how="all"))


def test_monthly_confirmation_does_not_admit_later_daily_pressure():
    monthly, terms, dollar = inputs()
    actual = joint.confirmation_targets(monthly, terms, dollar, "completed_month")
    pd.testing.assert_frame_equal(actual, monthly)
    assert actual.iloc[1:-1].isna().all().all()


def test_ratio_equality_triggers_only_when_dollar_strictly_positive():
    monthly, terms, dollar = inputs()
    terms["VIX"] = terms["VIX3M"]
    dollar[:] = 0
    result = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    pd.testing.assert_frame_equal(result, monthly)
    dollar.iloc[0] = 0.01
    result = joint.confirmation_targets(monthly, terms, dollar, "completed_month")
    assert result.iloc[0]["BIL"] == pytest.approx(0.49)
    assert result.iloc[-1]["BIL"] == 0


def test_no_event_is_empty_not_an_invented_daily_rebalance():
    monthly, terms, dollar = inputs()
    terms["VIX"] = 22
    dollar[:] = 0.01
    result = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    assert result.iloc[1:-1].isna().all().all()
    assert len(result.dropna(how="all")) == 2


def test_future_source_and_dollar_updates_cannot_change_earlier_targets():
    monthly, terms, dollar = inputs()
    original = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    monthly.iloc[-1] = [0.29, 0.49, 0.20]
    terms.iloc[-1, 0] = 100
    dollar.iloc[-1] = 0.5
    after = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    pd.testing.assert_frame_equal(original.iloc[:-1], after.iloc[:-1])
    assert after.iloc[-1]["BIL"] == pytest.approx(0.59)


@pytest.mark.parametrize(
    "problem", ["partial", "bad_budget", "negative", "missing_daily", "nan_daily", "bad_level", "missing_dollar", "unknown_schedule"],
)
def test_incomplete_risk_or_source_inputs_are_explicit_failures(problem):
    monthly, terms, dollar = inputs()
    schedule = "daily_option_state"
    if problem == "partial":
        monthly.iloc[1, 0] = 0.1
    elif problem == "bad_budget":
        monthly.iloc[0] *= 2
    elif problem == "negative":
        monthly.iloc[0, 2] = -0.1
    elif problem == "missing_daily":
        terms = terms.iloc[1:]
    elif problem == "nan_daily":
        terms.iloc[0, 0] = np.nan
    elif problem == "bad_level":
        terms.iloc[0, 0] = 0
    elif problem == "missing_dollar":
        dollar = dollar.iloc[1:]
    else:
        schedule = "selected_from_returns"
    with pytest.raises(QuantError):
        joint.confirmation_targets(monthly, terms, dollar, schedule)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_daily_reversals_keep_dated_delayed_targets_and_independent_funded_accounts(market_factory, cost, delay):
    data = market_factory("2020-01-02", "2020-04-30", ("SPY", "GLD", "BIL"))
    monthly = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    monthly.iloc[0] = [0.49, 0.49, 0]
    monthly.loc["2020-01-31"] = [0.45, 0.53, 0]
    terms = pd.DataFrame({"VIX": 18.0, "VIX3M": 20.0}, index=data.close.index)
    terms.loc["2020-02-03", "VIX"] = 22
    dollar = pd.Series(0.01, index=monthly.dropna(how="all").index)
    signals = joint.confirmation_targets(monthly, terms, dollar, "daily_option_state")
    assert signals.loc["2020-02-03", "BIL"] == pytest.approx(0.49)
    assert signals.loc["2020-02-04", "BIL"] == 0
    targets = seed_window(signals, "2020-01-03")
    own = simulate(data, targets, "2020-01-03", "2020-04-30",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(data, targets, "2020-01-03", "2020-04-30",
                                 capital=10000, cost_bps=cost, commission=1, delay=delay)
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].min() >= 0
    if delay == 2:
        assert own.frame.iloc[0]["orders"] == 0
        assert own.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize(
    "field,value", [("option_ratio_risk_threshold", 1.1), ("risk_off_investment_scale", 0.25),
                    ("new_configurations", 10), ("new_economic_factor_definitions", 1),
                    ("factor_symbols", ["SOXX"]), ("cash_reserve", 0)],
)
def test_no_post_result_ratio_scale_factor_or_cost_scope_changes(field, value):
    policy = read_json(ROOT / "config/joint-risk-confirmation.json")
    policy[field] = value
    with pytest.raises(QuantError):
        joint.validate(policy)
