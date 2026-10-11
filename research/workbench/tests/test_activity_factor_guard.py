from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import activity_factor_guard as guard
from us_quant import activity_inputs as inputs
from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/activity-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def comparisons(data, dollar=0.1, magnitude=-0.8, diffusion=-0.4, change=-0.1):
    rows = pd.DataFrame({"dollar_change":dollar}, index=original(data).dropna(how="all").index)
    for series, level in (("CFNAIMA3", magnitude), ("CFNAIDIFF", diffusion)):
        rows[f"{series}_level"], rows[f"{series}_change"] = level, change
        rows[f"{series}_available"] = True
        rows.loc[list(guard.PAUSED), [f"{series}_level", f"{series}_change"]] = np.nan
        rows.loc[list(guard.PAUSED), f"{series}_available"] = False
    return rows


def vintages():
    index = sessions("2015-08-10", "2026-10-05")
    _, dates = inputs.cohort(index)
    months = pd.date_range("2014-01-01", "2026-08-01", freq="MS")
    stale = {row["vintage_cutoff"]:pd.Timestamp(row["observation_month"]) for row in guard.KNOWN_STALE}
    result = {}
    for series in inputs.SERIES:
        result[series] = {}
        for cutoff in dates:
            if cutoff in stale:
                actual = months[months <= stale[cutoff]]
            else:
                actual = months[months + pd.offsets.MonthEnd(0) <= pd.Timestamp(cutoff)]
            result[series][cutoff] = pd.Series(0.1, index=actual, name=series)
    return index, result


def test_only_exact_known_stale_queries_receive_explicit_unavailable_markers():
    index, snapshots = vintages()
    selected = guard.select_queries(index, snapshots)
    assert len(selected) == 450
    assert selected["source_available"].sum() == 440
    blocked = selected.loc[~selected["source_available"]]
    assert len(blocked) == 10 and blocked["index_value"].isna().all()
    assert blocked["source_reason"].notna().all()
    assert selected.loc[selected["source_available"], "index_value"].eq(0.1).all()


@pytest.mark.parametrize("problem", ["unknown_stale", "missing_archive", "future", "changed_known_gap"])
def test_unknown_source_gaps_or_changed_known_dates_never_enter_standby(problem):
    index, snapshots = vintages()
    series = "CFNAIMA3"
    if problem == "missing_archive":
        snapshots[series].pop(next(iter(snapshots[series])))
    elif problem == "changed_known_gap":
        row = guard.KNOWN_STALE[0]
        snapshots[series][row["vintage_cutoff"]] = pd.Series([0.1], index=pd.to_datetime(["2022-09-01"]))
    else:
        cutoff = next(iter(snapshots[series]))
        month = "2014-01-01" if problem == "unknown_stale" else "2026-08-01"
        snapshots[series][cutoff] = pd.Series([0.1], index=pd.to_datetime([month]))
    with pytest.raises(QuantError):
        guard.select_queries(index, snapshots)


def test_low_deteriorating_activity_and_positive_dollar_halves_only_known_active_months(market, policy):
    source = original(market)
    outputs = guard.build_from_inputs(market, comparisons(market), policy)
    for frame in outputs.values():
        assert frame.loc[list(guard.PAUSED)].isna().all().all()
        assert len(frame.dropna(how="all")) == 126
        assets = frame.columns.drop("BIL")
        active = frame.dropna(how="all").index
        np.testing.assert_allclose(frame.loc[active, assets], source.loc[active, assets] / 2)
        np.testing.assert_allclose(frame.loc[active].sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert frame.loc[active, [*guard.FACTORS, "SOXX", "GLD"]].gt(0).all().all()


@pytest.mark.parametrize("dollar,magnitude,diffusion,change", [
    (0.0, -0.8, -0.4, -0.1), (-0.1, -0.8, -0.4, -0.1),
    (0.1, -0.70, -0.35, -0.1), (0.1, -0.8, -0.4, 0.0),
    (0.1, -0.8, -0.4, 0.1), (0.1, 0.1, 0.1, -0.1),
])
def test_equality_nonrising_dollar_or_nonweakening_activity_preserves_original_targets(market, policy, dollar, magnitude, diffusion, change):
    expected = original(market)
    expected.loc[list(guard.PAUSED)] = np.nan
    for frame in guard.build_from_inputs(market, comparisons(market, dollar, magnitude, diffusion, change), policy).values():
        pd.testing.assert_frame_equal(frame, expected)


def test_composite_magnitude_and_diffusion_are_separate_fixed_confirmations_not_interchangeable(market, policy):
    outputs = guard.build_from_inputs(market, comparisons(market, magnitude=-0.5, diffusion=-0.4), policy)
    expected = original(market)
    expected.loc[list(guard.PAUSED)] = np.nan
    pd.testing.assert_frame_equal(outputs[guard.CANDIDATES[0]["id"]], expected)
    assert not outputs[guard.CANDIDATES[1]["id"]].equals(expected)


@pytest.mark.parametrize("problem", ["missing_month", "unknown_nan", "inf", "flag", "missing_known_pause"])
def test_target_missingness_must_be_exact_and_cannot_default_to_a_safe_state(market, policy, problem):
    rows = comparisons(market)
    day = rows.index[0]
    if problem == "missing_month":
        rows = rows.iloc[1:]
    elif problem == "unknown_nan":
        rows.loc[day, ["CFNAIMA3_level", "CFNAIMA3_change"]] = np.nan
        rows.loc[day, "CFNAIMA3_available"] = False
    elif problem == "inf":
        rows.loc[day, "CFNAIMA3_level"] = np.inf
    elif problem == "flag":
        rows.loc[day, "CFNAIMA3_available"] = False
    else:
        rows.loc[guard.PAUSED[0], ["CFNAIMA3_level", "CFNAIMA3_change"]] = -0.8, -0.1
        rows.loc[guard.PAUSED[0], "CFNAIMA3_available"] = True
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, rows, policy)


def test_future_prices_and_activity_cannot_change_earlier_targets(market, policy):
    rows = comparisons(market)
    before = guard.build_from_inputs(market, rows, policy)
    boundary = pd.Timestamp("2021-12-31")
    changed_rows = rows.copy()
    late_active = (changed_rows.index > boundary) & changed_rows["CFNAIMA3_available"]
    changed_rows.loc[late_active, "CFNAIMA3_level"] = 0.5
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > boundary
    multiplier = np.linspace(1.0, 1.5, later.sum())
    closing.loc[later, "SOXX"] *= multiplier
    opening.loc[later, "SOXX"] *= multiplier
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    after = guard.build_from_inputs(changed, changed_rows, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_pauses_keep_actual_holdings_and_every_daily_account_session_in_funded_and_bt(market, policy, cost, delay):
    desired = guard.build_from_inputs(market, comparisons(market), policy)[guard.CANDIDATES[0]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    independent = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].ge(0).all()
    for day in pd.to_datetime(guard.PAUSED):
        assert seeded.loc[day].isna().all()
        assert own.frame.loc[day, "gross_exposure"] > 0


@pytest.mark.parametrize("field,value", [("risk_change_sessions", 21), ("risk_off_investment_scale", 0.75),
                                        ("known_paused_months", []), ("known_stale_queries", []),
                                        ("original_source_full_readiness_relabelled", True),
                                        ("new_economic_factor_definitions", 2)])
def test_frozen_confirmation_windows_scale_and_known_missingness_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_finite_generator_requires_full_original_source_and_macro_helper_graph():
    spec = {"id":guard.CANDIDATES[0]["id"], "configuration":guard.CANDIDATES[0],
            "frozen_files":{"src/us_quant/activity_factor_guard.py":"0"*64,
                            "config/activity-factor-guard.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
