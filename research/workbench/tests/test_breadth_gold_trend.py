from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import breadth_gold_trend as gold
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator
from us_quant.trend_factor_guard import gated_target

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/breadth-gold-trend.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "IEF", "TLT", "GLD", "BIL", *gold.FACTORS, "XLK", "SOXX"))


def sources(data):
    result = {}
    for candidate, scale in zip(gold.CANDIDATES, (0.5, 0.0), strict=True):
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for position, day in enumerate(frame.index):
            if position >= 199 and is_month_end(day):
                frame.loc[day] = 0.0
                frame.loc[day, list(gold.FACTORS)] = 0.98 * 0.6 * scale / 8
                frame.loc[day, "SOXX"] = 0.98 * 0.6 * scale / 2
                frame.loc[day, "GLD"] = 0.98 * 0.4
                frame.loc[day, "BIL"] = 0.98 * 0.6 * (1 - scale)
        result[candidate["source_id"]] = frame
    return result


def test_gold_equality_is_off_and_warmup_is_not_invented(market):
    close = market.close["GLD"] * 0 + 100.0
    states = gold.gold_states(close)
    assert states.iloc[:199].isna().all()
    assert states.iloc[199:].eq(0).all()
    rising = pd.Series(100 * np.exp(np.arange(len(close)) * 0.001), index=close.index)
    assert gold.gold_states(rising).iloc[199:].eq(1).all()


@pytest.mark.parametrize("drift", [-0.001, 0.0, 0.001])
def test_only_gold_is_removed_and_unchanged_old_helper_is_reused(market, policy, drift):
    close = market.close.copy()
    close["GLD"] = 100 * np.exp(np.arange(len(close)) * drift)
    data = replace(market, close=close, raw_close=close.copy())
    original = sources(data)
    frames, _ = gold.build_from_sources(data, original, policy)
    for candidate in gold.CANDIDATES:
        parent = original[candidate["source_id"]]
        actual = frames[candidate["id"]]
        assert actual.index.equals(parent.index)
        assert actual.columns.equals(parent.columns)
        assert actual.isna().equals(parent.isna())
        pd.testing.assert_frame_equal(actual.drop(columns=["GLD", "BIL"]), parent.drop(columns=["GLD", "BIL"]))
        for day in actual.dropna(how="all").index:
            pd.testing.assert_series_equal(actual.loc[day], gated_target(parent.loc[day], (True, drift > 0)))
        np.testing.assert_allclose(actual.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_no_daily_decisions_when_gold_state_changes_between_source_months(market, policy):
    original = sources(market)
    frames, _ = gold.build_from_sources(market, original, policy)
    for frame in frames.values():
        assert all(is_month_end(day) for day in frame.dropna(how="all").index)


def test_future_gold_prices_cannot_change_prior_targets(market, policy):
    original = sources(market)
    before, _ = gold.build_from_sources(market, original, policy)
    boundary = pd.Timestamp("2017-01-31")
    close = market.close.copy()
    later = close.index > boundary
    close.loc[later, "GLD"] *= np.linspace(1, 2, later.sum())
    changed = replace(market, close=close, raw_close=close.copy())
    after, _ = gold.build_from_sources(changed, original, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_complete_original_seeded_accounting_with_gold_removal(market, policy, cost, delay):
    frames, _ = gold.build_from_sources(market, sources(market), policy)
    for frame in frames.values():
        targets = seed_window(frame, "2016-10-06")
        own = simulate(market, targets, "2016-10-06", "2017-06-30",
                       initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        separate = independent_equity(market, targets, "2016-10-06", "2017-06-30",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
        np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0
        if delay == 2:
            assert own.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "empty", "missing_source", "early"])
def test_original_source_failures_cannot_become_safe_gold_cash_fallback(market, policy, problem):
    original = sources(market)
    frame = original[gold.CANDIDATES[0]["source_id"]]
    day = frame.dropna(how="all").index[0]
    if problem == "partial":
        frame.iloc[0, 0] = 0.1
    elif problem == "negative":
        frame.loc[day, "GLD"] = -0.1
    elif problem == "budget":
        frame.loc[day] *= 2
    elif problem == "empty":
        frame[:] = np.nan
    elif problem == "early":
        early = next(day for day in frame.index[:190] if is_month_end(day))
        frame.loc[early] = frame.loc[day]
    else:
        del original[gold.CANDIDATES[0]["source_id"]]
    with pytest.raises(QuantError):
        gold.build_from_sources(market, original, policy)


@pytest.mark.parametrize("field,value", [("gold_sma_observations", 150), ("gold_signal_asset", "BIL"),
                                        ("decision_clock", "daily"), ("cash_reserve", 0.0),
                                        ("new_economic_factor_definitions", 1), ("new_configurations", 10)])
def test_gold_rule_source_clock_and_factor_counts_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        gold.validate(policy)


def test_finite_replay_rejects_missing_parent_or_gold_helper_dependencies():
    spec = {"id": gold.CANDIDATES[0]["id"], "configuration": gold.CANDIDATES[0],
            "frozen_files": {"src/us_quant/breadth_gold_trend.py": "0" * 64,
                             "config/breadth-gold-trend.json": "0" * 64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
