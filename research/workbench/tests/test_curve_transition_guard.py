from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import curve_transition_guard as guard
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
    return read_json(ROOT / "config/curve-transition-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def comparisons(data, dollar=0.1, level=-0.2, change=0.1):
    rows = pd.DataFrame({"dollar_change":dollar}, index=original(data).dropna(how="all").index)
    for series in ("T10Y3M", "T10Y2Y"):
        rows[f"{series}_level"], rows[f"{series}_change"] = level, change
    return rows


def test_inverted_rising_curve_and_positive_dollar_halves_all_nonbil_budgets(market, policy):
    source = original(market)
    for frame in guard.build_from_inputs(market, comparisons(market), policy).values():
        active = frame.dropna(how="all")
        assets = frame.columns.drop("BIL")
        assert len(active) == 131
        assert frame.isna().equals(source.isna())
        np.testing.assert_allclose(active[assets], source.loc[active.index, assets] / 2)
        np.testing.assert_allclose(active["BIL"], source.loc[active.index, "BIL"] + 0.49, atol=1e-12)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert active[[*guard.FACTORS, "SOXX", "GLD"]].gt(0).all().all()


@pytest.mark.parametrize("dollar,level,change", [
    (0.0, -0.2, 0.1), (-0.1, -0.2, 0.1),
    (0.1, 0.0, 0.1), (0.1, 0.2, 0.1),
    (0.1, -0.2, 0.0), (0.1, -0.2, -0.1),
])
def test_every_nontrigger_or_equality_preserves_exact_original_allocation(market, policy, dollar, level, change):
    source = original(market)
    for frame in guard.build_from_inputs(market, comparisons(market, dollar, level, change), policy).values():
        pd.testing.assert_frame_equal(frame, source)


def test_maturity_views_are_distinct_fixed_comparisons_not_interchangeable(market, policy):
    rows = comparisons(market)
    rows["T10Y2Y_level"] = 0.1
    frames = guard.build_from_inputs(market, rows, policy)
    pd.testing.assert_frame_equal(frames[guard.CANDIDATES[1]["id"]], original(market))
    assert not frames[guard.CANDIDATES[0]["id"]].equals(original(market))


@pytest.mark.parametrize("problem", ["missing_month", "nan", "inf", "string", "duplicate", "reordered", "missing_column"])
def test_unknown_missingness_or_changed_source_shape_cannot_enter_standby(market, policy, problem):
    rows = comparisons(market)
    if problem == "missing_month":
        rows = rows.iloc[1:]
    elif problem in ("nan", "inf"):
        rows.iloc[0, 1] = np.nan if problem == "nan" else np.inf
    elif problem == "string":
        rows["T10Y3M_level"] = "missing"
    elif problem == "duplicate":
        rows = pd.concat([rows.iloc[:1], rows])
    elif problem == "reordered":
        rows = rows.iloc[::-1]
    else:
        rows = rows.drop(columns="T10Y2Y_change")
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, rows, policy)


def test_future_prices_and_macro_values_cannot_change_previous_targets(market, policy):
    rows = comparisons(market)
    before = guard.build_from_inputs(market, rows, policy)
    boundary = pd.Timestamp("2021-12-31")
    later = rows.index > boundary
    changed_rows = rows.copy()
    changed_rows.loc[later, "T10Y3M_level"] = 0.5
    closing, opening = market.close.copy(), market.open.copy()
    late_prices = closing.index > boundary
    scale = np.linspace(1.0, 1.4, late_prices.sum())
    closing.loc[late_prices, "SOXX"] *= scale
    opening.loc[late_prices, "SOXX"] *= scale
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    after = guard.build_from_inputs(changed, changed_rows, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_monthly_restore_and_full_calendar_preserved_in_funded_and_bt(market, policy, cost, delay):
    rows = comparisons(market)
    rows.iloc[::2, 0] = -0.1
    desired = guard.build_from_inputs(market, rows, policy)[guard.CANDIDATES[0]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert desired.isna().equals(original(market).isna())


@pytest.mark.parametrize("field,value", [
    ("risk_change_sessions", 21), ("risk_off_investment_scale", 0.75),
    ("cash_reserve", 0.0), ("new_economic_factor_definitions", 2),
    ("source_candidate", "semiconductor_dollar_activity_diffusion_confirmation"),
    ("candidates", [{"id":"new", "curve_series":"T10Y2Y"}]),
])
def test_frozen_rules_source_and_budget_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_finite_generator_requires_the_full_existing_macro_and_source_graph():
    spec = {"id":guard.CANDIDATES[0]["id"], "configuration":guard.CANDIDATES[0],
            "frozen_files":{"src/us_quant/curve_transition_guard.py":"0"*64,
                            "config/curve-transition-guard.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
