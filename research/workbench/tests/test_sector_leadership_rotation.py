from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import sector_leadership_rotation as leadership
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/sector-leadership-rotation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "IEF", "TLT", "GLD", "BIL", *leadership.FACTORS, "XLK", "SOXX"))


def original(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for position, day in enumerate(frame.index):
        if position >= 252 and is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, list(leadership.FACTORS)] = 0.196 / 4
            frame.loc[day, "SOXX"] = 0.196
            frame.loc[day, "GLD"] = 0.392
            frame.loc[day, "BIL"] = 0.196
    return frame


def test_equality_routes_to_spy_and_keeps_complete_warmup(market):
    close = market.close.loc[:, ["SOXX", "SPY"]].copy()
    close["SOXX"] = close["SPY"]
    for candidate in leadership.CANDIDATES:
        shares = leadership.leadership_shares(close, candidate["method"])
        assert shares.iloc[:252].isna().all()
        assert shares.iloc[252:].eq(0).all()
    close["SOXX"] *= np.exp(np.arange(len(close)) * 0.001)
    for candidate in leadership.CANDIDATES:
        assert leadership.leadership_shares(close, candidate["method"]).iloc[252:].eq(1).all()


def test_three_strict_equal_horizon_votes_not_fitted_continuous_weights(market):
    close = market.close.loc[:, ["SOXX", "SPY"]]
    shares = leadership.leadership_shares(close, "equal_relative_horizon_votes")
    for position in range(252, len(close)):
        votes = sum(close["SOXX"].iloc[position] / close["SOXX"].iloc[position-horizon]
                    > close["SPY"].iloc[position] / close["SPY"].iloc[position-horizon]
                    for horizon in (63, 126, 252))
        assert shares.iloc[position] == votes / 3


def test_only_satellite_moves_and_combined_equity_gold_bill_budgets_stay_exact(market, policy):
    source = original(market)
    frames, _ = leadership.build_from_source(market, source, policy)
    unchanged = source.columns.difference(["SOXX", "SPY"])
    for frame in frames.values():
        assert frame.isna().equals(source.isna())
        pd.testing.assert_frame_equal(frame.loc[:, unchanged], source.loc[:, unchanged])
        active = frame.dropna(how="all")
        np.testing.assert_allclose(active[["SOXX", "SPY"]].sum(axis=1), 0.196, rtol=0, atol=1e-12)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_future_relative_prices_do_not_change_past_allocations(market, policy):
    source = original(market)
    before, _ = leadership.build_from_source(market, source, policy)
    close = market.close.copy()
    boundary = pd.Timestamp("2017-01-31")
    later = close.index > boundary
    close.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    changed = replace(market, close=close, raw_close=close.copy())
    after, _ = leadership.build_from_source(changed, source, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_unmodified_band_replays_funded_and_bt_accounts_on_switched_targets(market, policy, cost, delay):
    desired, _ = leadership.build_from_source(market, original(market), policy)
    for candidate in leadership.CANDIDATES:
        own, issued, _ = leadership.run(market, desired[candidate["id"]], candidate, policy,
                                        "2016-10-06", "2017-06-30", cost, delay)
        replay = simulate(market, issued, "2016-10-06", "2017-06-30",
                          initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        separate = independent_equity(market, issued, "2016-10-06", "2017-06-30",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
        pd.testing.assert_frame_equal(own.frame, replay.frame)
        np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "already_spy", "empty"])
def test_invalid_source_does_not_become_benchmark_fallback(market, policy, problem):
    source = original(market)
    day = source.dropna(how="all").index[0]
    if problem == "partial":
        source.iloc[0, 0] = 0.1
    elif problem == "negative":
        source.loc[day, "SOXX"] = -0.1
    elif problem == "budget":
        source.loc[day] *= 2
    elif problem == "already_spy":
        source.loc[day, "SPY"] = 0.01
        source.loc[day, "SOXX"] -= 0.01
    else:
        source[:] = np.nan
    with pytest.raises(QuantError):
        leadership.build_from_source(market, source, policy)


@pytest.mark.parametrize("field,value", [("winner_sessions", 63), ("relative_horizon_sessions", [126]),
                                        ("fallback_equity_asset", "BIL"), ("actual_weight_trade_band", 0.05),
                                        ("source_id", "another_source"), ("new_economic_factor_definitions", 1)])
def test_no_result_driven_rotation_horizons_or_execution_changes(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        leadership.validate(policy)


def test_finite_replay_refuses_unbound_source_or_execution_helper():
    spec = {"id": leadership.CANDIDATES[0]["id"], "configuration": leadership.CANDIDATES[0],
            "frozen_files": {"src/us_quant/sector_leadership_rotation.py": "0"*64,
                             "config/sector-leadership-rotation.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
