from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import skew_factor_guard as guard
from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/skew-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def inputs(data, value=140.0):
    index = original(data).dropna(how="all").index
    result = pd.DataFrame({"SKEW":value, "VIX":20.0, "source_available":True, "source_reason":None}, index=index)
    result.loc[list(guard.PAUSED), "SKEW"] = np.nan
    result.loc[list(guard.PAUSED), "source_available"] = False
    result.loc[list(guard.PAUSED), "source_reason"] = "actual_SKEW_quote_missing_no_safe_default"
    return result


def test_equity_only_and_all_risk_scopes_preserve_exact_other_source_budgets(market, policy):
    source = original(market)
    frames = guard.build_from_inputs(market, inputs(market), policy)
    equity, portfolio = (frames[row["id"]] for row in guard.CANDIDATES)
    active = equity.dropna(how="all").index
    np.testing.assert_allclose(equity.loc[active, list(guard.EQUITY)], source.loc[active, list(guard.EQUITY)]/2)
    pd.testing.assert_series_equal(equity.loc[active, "GLD"], source.loc[active, "GLD"])
    np.testing.assert_allclose(portfolio.loc[active, source.columns.drop("BIL")],
                               source.loc[active, source.columns.drop("BIL")]/2)
    np.testing.assert_allclose(portfolio.loc[active, "BIL"], 0.49, atol=1e-12)
    for frame in frames.values():
        assert len(frame.dropna(how="all")) == 130
        assert frame.loc[list(guard.PAUSED)].isna().all().all()
        np.testing.assert_allclose(frame.loc[active].sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_reference_equality_triggers_without_implying_physical_probability(market, policy):
    outputs = guard.build_from_inputs(market, inputs(market, value=130.0), policy)
    active = outputs[guard.CANDIDATES[1]["id"]].dropna(how="all")
    np.testing.assert_allclose(active["BIL"], 0.49, atol=1e-12)


def test_lower_native_level_preserves_original_targets_except_exact_quote_pause(market, policy):
    expected = original(market)
    expected.loc[list(guard.PAUSED)] = np.nan
    for frame in guard.build_from_inputs(market, inputs(market, value=129.99), policy).values():
        pd.testing.assert_frame_equal(frame, expected)


@pytest.mark.parametrize("problem", ["missing_month", "unknown_nan", "changed_pause", "filled_known",
                                   "missing_reason", "wrong_available", "infinite", "wrong_columns"])
def test_unknown_missingness_cannot_enter_standby_or_a_safe_default(market, policy, problem):
    rows = inputs(market)
    day = rows.index[0]
    if problem == "missing_month":
        rows = rows.iloc[1:]
    elif problem == "unknown_nan":
        rows.loc[day, "SKEW"] = np.nan
        rows.loc[day, "source_available"] = False
    elif problem == "changed_pause":
        rows.loc[day, "source_available"] = False
    elif problem == "filled_known":
        rows.loc[guard.PAUSED[0], "SKEW"] = 100.0
        rows.loc[guard.PAUSED[0], "source_available"] = True
        rows.loc[guard.PAUSED[0], "source_reason"] = None
    elif problem == "missing_reason":
        rows.loc[guard.PAUSED[0], "source_reason"] = None
    elif problem == "wrong_available":
        rows["source_available"] = rows["source_available"].astype(int)
    elif problem == "infinite":
        rows.loc[day, "SKEW"] = np.inf
    else:
        rows = rows.rename(columns={"SKEW":"VIX3M"})
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, rows, policy)


def test_future_prices_and_tail_values_cannot_change_earlier_targets(market, policy):
    rows = inputs(market)
    before = guard.build_from_inputs(market, rows, policy)
    boundary = pd.Timestamp("2021-12-31")
    changed_rows = rows.copy()
    late = (rows.index > boundary) & rows["source_available"]
    changed_rows.loc[late, "SKEW"] = 100.0
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > boundary
    closing.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    after = guard.build_from_inputs(changed, changed_rows, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_pause_holds_positions_and_full_returns_in_funded_and_bt(market, policy, cost, delay):
    desired = guard.build_from_inputs(market, inputs(market), policy)[guard.CANDIDATES[1]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    assert own.frame.loc["2024-11-29", "gross_exposure"] > 0
    assert seeded.loc["2024-11-29"].isna().all()
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [("skew_reference_level", 140.0), ("risk_off_scale", 0.75),
                                        ("known_paused_months", []), ("strict_source_readiness_relabelled", True),
                                        ("new_economic_factor_definitions", 2)])
def test_reference_scopes_known_pause_and_source_integrity_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_finite_generator_requires_the_entire_original_source_and_tail_metadata_graph():
    spec = {"id":guard.CANDIDATES[0]["id"], "configuration":guard.CANDIDATES[0],
            "frozen_files":{"src/us_quant/skew_factor_guard.py":"0"*64, "config/skew-factor-guard.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
