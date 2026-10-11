from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import activity_gold_trend as gold
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/activity-gold-trend.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *gold.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def source_frames(data):
    frames = {}
    for candidate in gold.CANDIDATES:
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for position, day in enumerate(frame.index):
            if position >= 63 and is_month_end(day) and str(day.date()) not in gold.parent.PAUSED:
                frame.loc[day] = 0.0
                frame.loc[day, list(gold.FACTORS)] = 0.98*0.6/8
                frame.loc[day, "SOXX"] = 0.98*0.6/2
                frame.loc[day, "GLD"] = 0.98*0.4
        frames[candidate["source_id"]] = frame
    return frames


def test_every_original_equity_target_is_exact_and_all_removed_gold_goes_only_bil(market, policy):
    source = source_frames(market)
    frames, _ = gold.build_from_sources(market, source, policy)
    for candidate in gold.CANDIDATES:
        target, original = frames[candidate["id"]], source[candidate["source_id"]]
        active = target.dropna(how="all").index
        nongold = original.columns.difference(["GLD", "BIL"])
        pd.testing.assert_frame_equal(target.loc[active, nongold], original.loc[active, nongold])
        pd.testing.assert_series_equal(target.loc[active, "GLD"] + target.loc[active, "BIL"],
                                       original.loc[active, "GLD"] + original.loc[active, "BIL"], check_names=False)
        np.testing.assert_allclose(target.loc[active].sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert target.loc[list(gold.parent.PAUSED)].isna().all().all()


def test_existing200_observation_equality_is_off_not_a_tuned_boundary(market, policy):
    close, opening = market.close.copy(), market.open.copy()
    close["GLD"], opening["GLD"] = 100.0, 100.0
    data = replace(market, close=close, open=opening, raw_close=close.copy())
    frames, _ = gold.build_from_sources(data, source_frames(data), policy)
    for target in frames.values():
        assert target.dropna(how="all")["GLD"].eq(0).all()
        assert target.dropna(how="all")["BIL"].eq(0.98*0.4).all()


def test_gold_warmup_does_not_invent_early_targets_or_remove_formal_return_days(market, policy):
    source = source_frames(market)
    frames, diagnostics = gold.build_from_sources(market, source, policy)
    for candidate in gold.CANDIDATES:
        target, original = frames[candidate["id"]], source[candidate["source_id"]]
        expected_warmup = original.dropna(how="all").index[original.dropna(how="all").index < market.close.index[199]]
        assert len(expected_warmup) == 6
        assert target.loc[expected_warmup].isna().all().all()
        assert target.iloc[:199].isna().all().all()
        assert len(original.dropna(how="all")) == 126
        assert len(target.dropna(how="all")) == 120
        actual_warmup = [row["session"] for row in diagnostics if row["candidate_id"] == candidate["id"]
                         and row["status"] == "unissued_pre_window_gold200warmup"]
        assert actual_warmup == expected_warmup.strftime("%Y-%m-%d").tolist()
        assert all(pd.Timestamp(day) < pd.Timestamp("2016-10-06") for day in actual_warmup)
        assert target.index.equals(market.close.index)
        assert all(is_month_end(day) for day in target.dropna(how="all").index)


def test_known_macro_pause_cannot_become_a_gold_only_daily_update(market, policy):
    frames, _ = gold.build_from_sources(market, source_frames(market), policy)
    for target in frames.values():
        assert target.loc[list(gold.parent.PAUSED)].isna().all().all()
        nonmonths = target.index[[not is_month_end(day) for day in target.index]]
        assert target.loc[nonmonths].isna().all().all()


def test_future_gold_prices_cannot_change_an_earlier_gate_or_source_target(market, policy):
    source = source_frames(market)
    before, _ = gold.build_from_sources(market, source, policy)
    boundary = pd.Timestamp("2021-12-31")
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > boundary
    multiplier = np.linspace(1.0, 1.5, later.sum())
    closing.loc[later, "GLD"] *= multiplier
    opening.loc[later, "GLD"] *= multiplier
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    after, _ = gold.build_from_sources(changed, source, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("problem", ["missing_source", "partial", "negative", "budget", "empty", "filled_macro_pause"])
def test_source_errors_cannot_be_silently_filled_or_relabelled_gold_warmup(market, policy, problem):
    sources = source_frames(market)
    identifier = gold.CANDIDATES[0]["source_id"]
    frame = sources[identifier]
    day = frame.dropna(how="all").index[-1]
    if problem == "missing_source":
        sources.pop(identifier)
    elif problem == "partial":
        frame.loc[day, "GLD"] = np.nan
    elif problem == "negative":
        frame.loc[day, "GLD"] = -0.1
    elif problem == "budget":
        frame.loc[day] *= 2
    elif problem == "empty":
        frame[:] = np.nan
    else:
        frame.loc[gold.parent.PAUSED[0]] = frame.loc[day]
    with pytest.raises(QuantError):
        gold.build_from_sources(market, sources, policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_full_fresh_capital_and_known_pause_paths_match_funded_and_independent_bt(market, policy, cost, delay):
    target = gold.build_from_sources(market, source_frames(market), policy)[0][gold.CANDIDATES[0]["id"]]
    seeded = seed_window(target, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    independent = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].ge(0).all()
    for day in gold.parent.PAUSED:
        assert seeded.loc[day].isna().all()
        assert own.frame.loc[day, "gross_exposure"] > 0


@pytest.mark.parametrize("field,value", [("gold_sma_observations", 126), ("gold_signal_asset", "SOXX"),
                                        ("decision_clock", "daily"), ("cash_reserve", 0.0),
                                        ("gold_warmup_policy", "fill_bil"), ("new_economic_factor_definitions", 2)])
def test_existing_window_clock_source_and_funding_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        gold.validate(policy)


def test_finite_generator_requires_both_complete_registered_source_graphs():
    spec = {"id":gold.CANDIDATES[0]["id"], "configuration":gold.CANDIDATES[0],
            "frozen_files":{"src/us_quant/activity_gold_trend.py":"0"*64,
                            "config/activity-gold-trend.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
