from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import factor_core_momentum as core
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-core-momentum.json")


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


def test_lexical_ties_and_negative_scores_keep_all_four_fixed_rank_shares():
    scores = pd.Series(-1.0, index=core.FACTORS)
    actual = core.ranked_shares(scores)
    assert actual.to_dict() == {"MTUM":0.35,"VLUE":0.15,"QUAL":0.30,"USMV":0.20}
    assert actual.sum() == pytest.approx(1.0)
    scores["VLUE"] = 1.0
    assert core.ranked_shares(scores)["VLUE"] == 0.35


def test_only_core_ranks_change_not_budget_or_industry_gold_cash_rules(market, policy):
    source = original(market)
    frames, _ = core.build_from_source(market, source, policy)
    for frame in frames.values():
        assert frame.isna().equals(source.isna())
        pd.testing.assert_frame_equal(frame.drop(columns=list(core.FACTORS)),
                                      source.drop(columns=list(core.FACTORS)))
        active = frame.dropna(how="all")
        np.testing.assert_allclose(active.loc[:,list(core.FACTORS)].sum(axis=1),0.196,rtol=0,atol=1e-12)
        shares = active.loc[:,list(core.FACTORS)]/0.196
        np.testing.assert_allclose(np.sort(shares.to_numpy(),axis=1),
                                   np.tile([0.15,0.20,0.30,0.35],(len(active),1)),rtol=0,atol=1e-12)


def test_past_rankings_do_not_use_future_prices_or_rates(market, policy):
    source = original(market)
    before, _ = core.build_from_source(market, source, policy)
    boundary = pd.Timestamp("2017-01-31")
    close, rates = market.close.copy(), market.risk_free.copy()
    later = close.index > boundary
    close.loc[later,"MTUM"] *= np.linspace(1,1.5,later.sum())
    rates.loc[later] += 0.0001
    changed = replace(market,close=close,raw_close=close.copy(),risk_free=rates)
    after, _ = core.build_from_source(changed, source, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary],after[identifier].loc[:boundary])


def test_rank_deficient_spy_regression_fails_without_valid_score_fallback(market, policy):
    close = market.close.copy()
    close["SPY"] = 100.0
    rates = market.risk_free*0
    changed = replace(market,close=close,raw_close=close.copy(),risk_free=rates)
    with pytest.raises(QuantError,match="rank deficient"):
        core.build_from_source(changed,original(changed),policy)


@pytest.mark.parametrize("cost,delay",[(5.0,1),(20.0,2)])
def test_original_band_and_funded_bt_execution_replays_rank_targets(market, policy, cost, delay):
    desired, _ = core.build_from_source(market,original(market),policy)
    for candidate in core.CANDIDATES:
        own,issued,_ = core.run(market,desired[candidate["id"]],candidate,policy,
                                "2016-10-06","2017-06-30",cost,delay)
        replay = simulate(market,issued,"2016-10-06","2017-06-30",
                          initial_capital=10000,cost_bps=cost,commission=1,delay=delay)
        separate = independent_equity(market,issued,"2016-10-06","2017-06-30",
                                     capital=10000,cost_bps=cost,commission=1,delay=delay)
        pd.testing.assert_frame_equal(own.frame,replay.frame)
        np.testing.assert_allclose(own.frame["equity"],separate["equity"],rtol=0,atol=1e-8)
        assert own.frame["cash"].min() >= 0


@pytest.mark.parametrize("problem",["partial","negative","budget","zero_core","empty"])
def test_bad_source_does_not_become_equal_rank_default(market, policy, problem):
    source = original(market)
    day = source.dropna(how="all").index[0]
    if problem == "partial":
        source.iloc[0,0] = 0.1
    elif problem == "negative":
        source.loc[day,"MTUM"] = -0.1
    elif problem == "budget":
        source.loc[day] *= 2
    elif problem == "zero_core":
        source.loc[day,list(core.FACTORS)] = 0.0
        source.loc[day,"BIL"] += 0.196
    else:
        source[:] = np.nan
    with pytest.raises(QuantError):
        core.build_from_source(market,source,policy)


@pytest.mark.parametrize("field,value",[("ranked_factor_shares",[0.4,0.3,0.2,0.1]),
                                        ("factor_symbols",["MTUM"]),("actual_weight_trade_band",0.05),
                                        ("cash_reserve",0.0),("new_economic_factor_definitions",1)])
def test_no_post_result_rank_shares_factor_subset_or_execution_changes(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        core.validate(policy)


def test_finite_replay_requires_original_scores_and_complete_source_helpers():
    spec = {"id":core.CANDIDATES[0]["id"],"configuration":core.CANDIDATES[0],
            "frozen_files":{"src/us_quant/factor_core_momentum.py":"0"*64,
                            "config/factor-core-momentum.json":"0"*64}}
    with pytest.raises(QuantError,match="frozen helper dependency"):
        registered_generator(spec,ROOT)
