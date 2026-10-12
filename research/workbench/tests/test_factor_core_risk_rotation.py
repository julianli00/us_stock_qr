from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import factor_core_risk_rotation as core
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-core-risk-rotation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "IEF", "TLT", "GLD", "BIL", *core.FACTORS, "XLK", "SOXX"))


def original(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for position, day in enumerate(frame.index):
        if position >= 252 and is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, list(core.FACTORS)] = 0.196 / 4
            frame.loc[day, "SOXX"] = frame.loc[day, "SPY"] = 0.098
            frame.loc[day, "GLD"] = 0.392
            frame.loc[day, "BIL"] = 0.196
    return frame


@pytest.mark.parametrize("method", ["sample_volatility", "semideviation"])
def test_all_four_bounded_shares_and_prior_risk_helpers_remain(market, method):
    history = market.close.loc[:, list(core.FACTORS)]
    shares, risks = core.factor_risk_shares(history, market.risk_free, method)
    assert shares.index.tolist() == list(core.FACTORS)
    assert shares.sum() == pytest.approx(1.0, abs=1e-12)
    assert shares.between(0.15, 0.35).all()
    assert risks.gt(0).all()
    if method == "sample_volatility":
        pd.testing.assert_series_equal(risks, history.pct_change(fill_method=None).iloc[1:].tail(63).std(ddof=1))
    else:
        daily = history.pct_change(fill_method=None).iloc[1:].tail(63)
        for name in core.FACTORS:
            excess = daily[name] - market.risk_free.loc[daily.index]
            assert risks[name] == pytest.approx(np.sqrt(np.minimum(excess, 0).pow(2).mean()))


def test_only_style_core_changes_not_equity_gold_or_satellite_budgets(market, policy):
    source = original(market)
    frames, _ = core.build_from_source(market, source, policy)
    for frame in frames.values():
        assert frame.isna().equals(source.isna())
        pd.testing.assert_frame_equal(frame.drop(columns=list(core.FACTORS)),
                                      source.drop(columns=list(core.FACTORS)))
        active = frame.dropna(how="all")
        np.testing.assert_allclose(active.loc[:, list(core.FACTORS)].sum(axis=1), 0.196, rtol=0, atol=1e-12)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_zero_risks_fail_without_equal_weight_or_cash_fallback(market):
    history = market.close.loc[:, list(core.FACTORS)].copy() * 0 + 100.0
    rates = market.risk_free * 0
    for method in ("sample_volatility", "semideviation"):
        with pytest.raises(QuantError):
            core.factor_risk_shares(history, rates, method)


def test_future_factor_prices_cannot_change_past_risk_allocations(market, policy):
    source = original(market)
    before, _ = core.build_from_source(market, source, policy)
    close = market.close.copy()
    boundary = pd.Timestamp("2017-01-31")
    later = close.index > boundary
    close.loc[later, "MTUM"] *= np.linspace(1, 1.5, later.sum())
    changed = replace(market, close=close, raw_close=close.copy())
    after, _ = core.build_from_source(changed, source, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_band_and_funded_bt_accounting_replay_new_core_targets(market, policy, cost, delay):
    desired, _ = core.build_from_source(market, original(market), policy)
    for candidate in core.CANDIDATES:
        own, issued, _ = core.run(market, desired[candidate["id"]], candidate, policy,
                                  "2016-10-06", "2017-06-30", cost, delay)
        replay = simulate(market, issued, "2016-10-06", "2017-06-30",
                          initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        separate = independent_equity(market, issued, "2016-10-06", "2017-06-30",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
        pd.testing.assert_frame_equal(own.frame, replay.frame)
        np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "zero_core", "empty"])
def test_bad_original_source_is_not_a_safe_risk_default(market, policy, problem):
    source = original(market)
    day = source.dropna(how="all").index[0]
    if problem == "partial":
        source.iloc[0, 0] = 0.1
    elif problem == "negative":
        source.loc[day, "MTUM"] = -0.1
    elif problem == "budget":
        source.loc[day] *= 2
    elif problem == "zero_core":
        source.loc[day, list(core.FACTORS)] = 0.0
        source.loc[day, "BIL"] += 0.196
    else:
        source[:] = np.nan
    with pytest.raises(QuantError):
        core.build_from_source(market, source, policy)


@pytest.mark.parametrize("field,value", [("risk_sessions", 126), ("factor_share_min", 0.10),
                                        ("factor_share_max", 0.50), ("factor_symbols", ["MTUM"]),
                                        ("actual_weight_trade_band", 0.05), ("new_economic_factor_definitions", 1)])
def test_no_retuned_risk_windows_factor_subsets_bounds_or_execution(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        core.validate(policy)


def test_finite_replay_requires_all_parent_and_risk_helper_dependencies():
    spec = {"id": core.CANDIDATES[0]["id"], "configuration": core.CANDIDATES[0],
            "frozen_files": {"src/us_quant/factor_core_risk_rotation.py": "0"*64,
                             "config/factor-core-risk-rotation.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
