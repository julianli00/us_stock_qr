from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import breadth_gold_relative as gold
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
    return read_json(ROOT / "config/breadth-gold-relative.json")


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


def reference(data, original):
    raw = original[gold.CANDIDATES[0]["source_id"]]["GLD"]
    relative = pd.Series(np.nan, index=raw.index)
    for position, day in enumerate(raw.index):
        if position >= 252 and not pd.isna(raw.loc[day]):
            passed = sum(
                data.close["GLD"].iloc[position] / data.close["GLD"].iloc[position - lookback] - 1
                > data.close["BIL"].iloc[position] / data.close["BIL"].iloc[position - lookback] - 1
                for lookback in (63, 126, 252)
            )
            relative.loc[day] = raw.loc[day] * (passed / 3)
    return pd.DataFrame({"original_gold": raw, "relative_gold": relative})


@pytest.mark.parametrize("fraction", [0.0, 1 / 3, 2 / 3, 1.0])
def test_copy_only_prior_gold_component_and_keep_every_equity_target(market, policy, fraction):
    original = sources(market)
    ref = reference(market, original)
    ref.loc[ref["relative_gold"].notna(), "relative_gold"] = ref.loc[ref["relative_gold"].notna(), "original_gold"] * fraction
    frames, _ = gold.build_from_sources(market, original, ref, policy)
    for candidate in gold.CANDIDATES:
        parent = original[candidate["source_id"]]
        actual = frames[candidate["id"]]
        for day in actual.dropna(how="all").index:
            pd.testing.assert_series_equal(actual.loc[day].drop(["GLD", "BIL"]), parent.loc[day].drop(["GLD", "BIL"]))
            assert actual.loc[day, "GLD"] == ref.loc[day, "relative_gold"]
            assert actual.loc[day, "BIL"] == pytest.approx(parent.loc[day, "BIL"] + parent.loc[day, "GLD"] * (1 - fraction))
        np.testing.assert_allclose(actual.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_actual_bill_equality_is_zero_relative_gold_not_positive_momentum(market, policy):
    close = market.close.copy()
    close["GLD"] = close["BIL"]
    data = replace(market, close=close, raw_close=close.copy())
    original = sources(data)
    frames, _ = gold.build_from_sources(data, original, reference(data, original), policy)
    for frame in frames.values():
        assert frame.dropna(how="all")["GLD"].eq(0).all()


def test_only_original_months_after_full252_return_warmup_emit_targets(market, policy):
    original = sources(market)
    frames, _ = gold.build_from_sources(market, original, reference(market, original), policy)
    for frame in frames.values():
        assert frame.iloc[:252].isna().all().all()
        active = frame.dropna(how="all")
        assert all(is_month_end(day) for day in active.index)
        assert active.index.equals(original[gold.CANDIDATES[0]["source_id"]].iloc[252:].dropna(how="all").index)


def test_future_prices_do_not_change_prior_gold_targets_or_source_dates(market, policy):
    original = sources(market)
    before, _ = gold.build_from_sources(market, original, reference(market, original), policy)
    boundary = pd.Timestamp("2017-01-31")
    close = market.close.copy()
    later = close.index > boundary
    close.loc[later, "GLD"] *= np.linspace(1, 2, later.sum())
    changed = replace(market, close=close, raw_close=close.copy())
    after, _ = gold.build_from_sources(changed, original, reference(changed, original), policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_funded_and_bt_accounts_preserve_original_seed_and_costs(market, policy, cost, delay):
    original = sources(market)
    frames, _ = gold.build_from_sources(market, original, reference(market, original), policy)
    for frame in frames.values():
        targets = seed_window(frame, "2016-10-06")
        own = simulate(market, targets, "2016-10-06", "2017-06-30",
                       initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        separate = independent_equity(market, targets, "2016-10-06", "2017-06-30",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
        np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0


@pytest.mark.parametrize("problem", ["missing", "negative", "increase", "mismatched_budget", "future_warmup"])
def test_bad_prior_gold_reference_fails_without_cash_or_fraction_defaults(market, policy, problem):
    original = sources(market)
    ref = reference(market, original)
    day = ref["relative_gold"].dropna().index[0]
    if problem == "missing":
        ref.loc[day, "relative_gold"] = np.nan
    elif problem == "negative":
        ref.loc[day, "relative_gold"] = -0.1
    elif problem == "increase":
        ref.loc[day, "relative_gold"] = 1.0
    elif problem == "mismatched_budget":
        ref.loc[day, "original_gold"] += 0.1
    else:
        early = ref.iloc[:252]["original_gold"].dropna().index[0]
        ref.loc[early, "relative_gold"] = 0.0
    with pytest.raises(QuantError):
        gold.build_from_sources(market, original, ref, policy)


@pytest.mark.parametrize("field,value", [("gold_relative_sessions", [126]), ("gold_relative_benchmark", "risk_free"),
                                        ("decision_clock", "daily"), ("cash_reserve", 0.0),
                                        ("new_economic_factor_definitions", 1), ("new_configurations", 10)])
def test_no_retuned_horizons_vote_weights_clocks_or_factor_counts(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        gold.validate(policy)


def test_finite_generator_refuses_missing_frozen_relative_source_graph():
    spec = {"id": gold.CANDIDATES[0]["id"], "configuration": gold.CANDIDATES[0],
            "frozen_files": {"src/us_quant/breadth_gold_relative.py": "0" * 64,
                             "config/breadth-gold-relative.json": "0" * 64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)


def test_input_adapter_reuses_frozen_old_generator_gold_without_rewriting_equity(market, policy, monkeypatch):
    original = sources(market)
    raw = original[gold.CANDIDATES[0]["source_id"]]
    expected = reference(market, original)
    prior_frame = raw.copy()
    prior_frame["GLD"] = expected["relative_gold"]
    calls = []
    monkeypatch.setattr(gold, "safe_file", lambda *_args: ROOT)
    monkeypatch.setattr(gold, "source_targets", lambda *_args: original)
    monkeypatch.setattr(gold.prior, "monthly_source", lambda *_args: raw)

    def frozen_generator(data, baseline, prior_policy):
        assert data is market
        pd.testing.assert_frame_equal(baseline, raw)
        assert prior_policy == read_json(ROOT / "config/semiconductor-trend-transfer.json")
        calls.append(True)
        return {"semiconductor_equity_gold_relative_trends": prior_frame}, []

    monkeypatch.setattr(gold.prior, "build_from_baseline", frozen_generator)
    actual_sources, actual_gold = gold.inputs(market, policy)
    assert calls == [True]
    assert actual_sources is original
    pd.testing.assert_frame_equal(actual_gold, expected)
