from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import sector_breadth_risk as breadth
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
    return read_json(ROOT / "config/sector-breadth-risk.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "IEF", "TLT", "GLD", "BIL", *breadth.FACTORS, "XLK", "SOXX"))


def source(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for position, day in enumerate(frame.index):
        if position >= 63 and is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, list(breadth.FACTORS)] = 0.98 * 0.6 / 8
            frame.loc[day, "SOXX"] = 0.98 * 0.6 / 2
            frame.loc[day, "GLD"] = 0.98 * 0.4
    return frame


def signals(data, positive):
    result = pd.DataFrame(100.0, index=data.close.index, columns=breadth.SECTORS)
    time = np.arange(len(result))
    for name in result.columns[:positive]:
        result[name] = 100 * np.exp(time * 0.001)
    return result


def test_no_signal_before200_actual_observations_and_equality_is_off(market):
    close = signals(market, 0)
    counts = breadth.breadth_counts(close)
    assert counts.iloc[:199].isna().all()
    assert counts.iloc[199:].eq(0).all()
    assert counts.index.equals(close.index)


@pytest.mark.parametrize("positive", [0, 4, 5, 9])
def test_structural_strict_majority_scales_only_equity_and_leaves_gold_exact(market, policy, positive):
    original = source(market)
    frames, diagnostics = breadth.build_from_source(market, original, signals(market, positive), policy)
    for candidate in breadth.CANDIDATES:
        active = frames[candidate["id"]].dropna(how="all")
        scale = 1.0 if positive >= 5 else candidate["weak_equity_scale"]
        for day in active.index:
            np.testing.assert_allclose(active.loc[day, [*breadth.FACTORS, "SOXX"]],
                                       original.loc[day, [*breadth.FACTORS, "SOXX"]] * scale, atol=1e-12)
            assert active.loc[day, "GLD"] == original.loc[day, "GLD"]
            assert active.loc[day, "BIL"] == pytest.approx(0.98 * 0.6 * (1 - scale))
        assert all(is_month_end(day) for day in active.index)
        assert all(market.close.index.get_loc(day) >= 199 for day in active.index)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert len(diagnostics) == sum(len(frame.dropna(how="all")) for frame in frames.values())


def test_no_price_or_source_target_future_information_changes_prior_breadth(market, policy):
    close = signals(market, 4)
    original = source(market)
    before, _ = breadth.build_from_source(market, original, close, policy)
    boundary = pd.Timestamp("2017-01-31")
    changed = close.copy()
    changed.loc[changed.index > boundary] *= 2
    after, _ = breadth.build_from_source(market, original, changed, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])
    pd.testing.assert_series_equal(breadth.breadth_counts(close).loc[:boundary],
                                   breadth.breadth_counts(changed).loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_delayed_funded_and_bt_accounts_agree(market, policy, cost, delay):
    frames, _ = breadth.build_from_source(market, source(market), signals(market, 4), policy)
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


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "nonpositive", "date", "short"])
def test_incomplete_sector_signal_is_not_weak_breadth_fallback(market, policy, problem):
    close = signals(market, 9)
    if problem == "missing":
        close = close.drop(columns="XLK")
    elif problem == "nonfinite":
        close.iloc[0, 0] = np.nan
    elif problem == "nonpositive":
        close.iloc[0, 0] = 0
    elif problem == "date":
        close.index = pd.DatetimeIndex([close.index[0]] * len(close))
    else:
        close = close.iloc[:199]
    with pytest.raises(QuantError):
        breadth.build_from_source(market, source(market), close, policy)


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "empty"])
def test_invalid_original_targets_fail_explicitly(market, policy, problem):
    original = source(market)
    day = original.dropna(how="all").index[0]
    if problem == "partial":
        original.iloc[0, 0] = 0.1
    elif problem == "negative":
        original.loc[day, "BIL"] = -0.1
    elif problem == "budget":
        original.loc[day] *= 2
    else:
        original[:] = np.nan
    with pytest.raises(QuantError):
        breadth.build_from_source(market, original, signals(market, 9), policy)


@pytest.mark.parametrize("field,value", [("sma_observations", 150), ("signal_sector_symbols", ["XLK"]),
                                        ("strong_breadth", "two_thirds"), ("cash_reserve", 0.0),
                                        ("new_economic_factor_definitions", 1), ("new_configurations", 10)])
def test_no_post_result_retuning_or_new_factor_claim(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        breadth.validate(policy)


def test_finite_generator_refuses_an_incomplete_signal_dependency_graph():
    spec = {"id": breadth.CANDIDATES[0]["id"], "configuration": breadth.CANDIDATES[0],
            "frozen_files": {"src/us_quant/sector_breadth_risk.py": "0" * 64,
                             "config/sector-breadth-risk.json": "0" * 64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)


@pytest.mark.parametrize("problem", ["separate_xlk", "changed_core", "traded_xlk"])
def test_separate_signal_xlk_version_never_replaces_portfolio_prices(market, policy, monkeypatch, problem):
    original = source(market)
    close, opening = market.close.copy(), market.open.copy()
    close["XLK"] *= 1.00001
    opening["XLK"] *= 1.00001
    signal_data = replace(market, close=close, open=opening, raw_close=close.copy(), volume=market.volume.copy())
    for name in breadth.SECTORS:
        if name not in signal_data.close:
            for frame in (signal_data.close, signal_data.open, signal_data.raw_close):
                frame[name] = 100.0
            signal_data.volume[name] = 1000000.0
    if problem == "changed_core":
        signal_data.close["MTUM"] *= 1.00001
    elif problem == "traded_xlk":
        days = original.dropna(how="all").index
        original.loc[days, "XLK"] = 0.1
        original.loc[days, "GLD"] -= 0.1
    before = market.close.copy()
    monkeypatch.setattr(breadth, "safe_file", lambda *_args: ROOT)
    monkeypatch.setattr(breadth, "source_targets", lambda *_args: {"four_factor_semiconductor_sector50": original})
    monkeypatch.setattr(breadth, "signal_market", lambda *_args: signal_data)
    if problem == "separate_xlk":
        actual, selected = breadth.inputs(market, policy)
        pd.testing.assert_frame_equal(actual, original)
        pd.testing.assert_frame_equal(selected, signal_data.close.loc[:, list(breadth.SECTORS)])
    else:
        with pytest.raises(QuantError):
            breadth.inputs(market, policy)
    pd.testing.assert_frame_equal(market.close, before)


@pytest.mark.parametrize("field,value", [("signal_xlk_source", "portfolio_snapshot"),
                                        ("portfolio_xlk_target", "trade_both_versions")])
def test_price_version_choice_is_frozen_before_outcomes(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        breadth.validate(policy)
