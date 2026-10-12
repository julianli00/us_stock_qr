from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import semiconductor_portfolio_protection as protection
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.growth_portfolio_protection import protect_target
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/semiconductor-portfolio-protection.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2018-06-29",
        ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *protection.FACTORS, "XLK", "SOXX"),
    )


def baseline(data, bil=0.0):
    row = pd.Series(0.0, index=data.close.columns)
    row.loc[list(protection.FACTORS)] = 0.98 * 0.50 / 8
    row["SOXX"], row["GLD"] = 0.98 * 0.50 / 2, 0.98 * 0.50
    row *= (0.98 - bil) / 0.98
    row["BIL"] = bil
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for index, day in enumerate(data.close.index):
        if index >= 63 and is_month_end(day):
            frame.loc[day] = row
    return frame


@pytest.mark.parametrize("nav", [10000.0, 9500.0, 9000.0, 8500.0, 8000.0])
def test_zero_bil_source_reuses_exact_frozen_floor_helper(market, nav):
    row = baseline(market).dropna(how="all").iloc[0]
    actual = protection.protected_target(row, nav, 10000)
    expected = protect_target(row, nav, 10000, 0.15)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("bil", [0.0, 0.49, 0.98])
def test_existing_bills_are_never_removed_and_only_risk_cushion_is_scaled(market, bil):
    row = baseline(market, bil).dropna(how="all").iloc[0]
    target = protection.protected_target(row, 9500, 10000)
    scale = (9500 - 8500) / (0.15 * 9500)
    np.testing.assert_allclose(target.drop("BIL"), row.drop("BIL") * scale, atol=1e-15)
    assert target["BIL"] == pytest.approx(bil + (0.98 - bil) * (1 - scale))
    assert target["BIL"] >= bil
    assert target.sum() == pytest.approx(0.98)


@pytest.mark.parametrize("problem", ["negative", "partial", "budget", "bad_nav", "bad_peak"])
def test_invalid_source_or_wealth_cannot_be_silently_repaired(market, problem):
    row, nav, peak = baseline(market).dropna(how="all").iloc[0], 9500, 10000
    if problem == "negative":
        row["BIL"] = -1
    elif problem == "partial":
        row["SOXX"] = np.nan
    elif problem == "budget":
        row *= 2
    elif problem == "bad_nav":
        nav = 0
    else:
        peak = 9000
    with pytest.raises(QuantError):
        protection.protected_target(row, nav, peak)


@pytest.mark.parametrize("bil", [0.0, 0.49])
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_cost_dependent_feedback_preserves_funded_account_and_independent_bt(market, policy, bil, cost, delay):
    own, targets, requests, states = protection.run(
        market, baseline(market, bil), protection.CANDIDATES[int(bil > 0)], policy,
        "2016-10-06", "2017-06-30", cost, delay,
    )
    independent = independent_equity(
        market, targets, "2016-10-06", "2017-06-30",
        capital=10000, cost_bps=cost, commission=1, delay=delay,
    )
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].min() >= 0
    assert requests[0]["known_nav_usd"] == requests[0]["known_peak_usd"] == 10000
    assert requests[0]["protected_bil_weight"] == pytest.approx(bil)
    assert len(states) == len(own.frame)
    for before, after in zip(requests, requests[1:]):
        assert before["execution_session"] <= after["signal_session"]
    active = targets.dropna(how="all")
    np.testing.assert_allclose(active.sum(axis=1), 0.98, atol=1e-12)
    if delay == 2:
        assert own.frame.iloc[0]["orders"] == 0
        assert own.frame.iloc[0]["equity"] == 10000


def test_future_prices_do_not_change_earlier_targets_peak_or_pending_orders(market, policy):
    cutoff = pd.Timestamp("2017-06-30")
    source = baseline(market, 0.49)
    original = protection.run(
        market, source, protection.CANDIDATES[1], policy, "2016-10-06", "2018-01-31", 20, 2,
    )
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    multiplier = np.linspace(1, 1.5, later.sum())
    close.loc[later, "SOXX"] *= multiplier
    opening.loc[later, "SOXX"] *= multiplier
    changed = replace(market, close=close, raw_close=close.copy(), open=opening)
    after = protection.run(changed, source, protection.CANDIDATES[1], policy, "2016-10-06", "2018-01-31", 20, 2)
    pd.testing.assert_frame_equal(original[0].frame.loc[:cutoff], after[0].frame.loc[:cutoff])
    pd.testing.assert_frame_equal(original[1].loc[:cutoff], after[1].loc[:cutoff])
    assert [row for row in original[2] if row["signal_session"] <= str(cutoff.date())] == [
        row for row in after[2] if row["signal_session"] <= str(cutoff.date())
    ]


def test_monthly_update_during_pending_request_is_retained_and_gap_can_breach_floor(market, policy):
    end = "2016-11-10"
    close, opening = market.close.loc[:end].copy(), market.open.loc[:end].copy()
    later = close.index > pd.Timestamp("2016-10-05")
    anchor = close.loc["2016-10-05"]
    close.loc[later] = np.outer(1.0002 ** np.arange(1, later.sum() + 1), anchor)
    opening.loc[later] = np.outer(1.0002 ** np.arange(later.sum()), anchor)
    shocked = close.index >= pd.Timestamp("2016-10-28")
    for symbol in (*protection.FACTORS, "SOXX", "GLD"):
        close.loc[shocked, symbol] *= 0.8
        opening.loc[shocked, symbol] *= 0.8
    data = replace(
        market, close=close, open=opening, raw_close=close.copy(),
        volume=market.volume.loc[:end], risk_free=market.risk_free.loc[:end],
    )
    own, targets, requests, _ = protection.run(
        data, baseline(data), protection.CANDIDATES[0], policy, "2016-10-06", end, 20, 2,
    )
    assert next(row for row in requests if row["signal_session"] == "2016-10-28")["execution_session"] == "2016-11-01"
    assert next(row for row in requests if row["signal_session"] == "2016-11-01")["reason"] == "monthly_target"
    assert targets.loc["2016-10-31"].isna().all()
    equity = np.r_[10000, own.frame["equity"].to_numpy()]
    assert float((1 - equity / np.maximum.accumulate(equity)).max()) > 0.15


@pytest.mark.parametrize(
    "field,value",
    [
        ("drawdown_target", 0.12), ("actual_weight_trade_band", 0.01),
        ("new_configurations", 4), ("new_economic_factor_definitions", 1),
        ("cash_reserve", 0), ("factor_symbols", ["SOXX"]),
    ],
)
def test_protection_parameters_cannot_be_changed_after_outcomes(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        protection.validate(policy)
