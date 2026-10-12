from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import housing_factor_guard as guard
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
    return read_json(ROOT / "config/housing-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def comparisons(data, permit_change=-100.0, starts_change=-100.0):
    frame = pd.DataFrame(index=original(data).dropna(how="all").index)
    for series, change in (("PERMIT", permit_change), ("HOUST", starts_change)):
        frame[f"{series}_level"] = 1500.0
        frame[f"{series}_reference_level"] = 1500.0-change
        frame[f"{series}_change"] = change
        frame[f"{series}_available"] = True
        frame[f"{series}_source_reason"] = None
        frame.loc[list(guard.PAUSED), [f"{series}_{suffix}" for suffix in ("level", "reference_level", "change")]] = np.nan
        frame.loc[list(guard.PAUSED), f"{series}_available"] = False
        frame.loc[list(guard.PAUSED), f"{series}_source_reason"] = guard.PAUSE_REASON
    return frame


@pytest.mark.parametrize("permit_change,starts_change", [
    (-100.0, -100.0), (-100.0, 0.0), (-100.0, 100.0),
    (0.0, -100.0), (0.0, 0.0), (0.0, 100.0),
    (100.0, -100.0), (100.0, 0.0), (100.0, 100.0),
])
def test_pipeline_stage_truth_table_strict_declines_and_unmodified_original_budget(market, policy, permit_change, starts_change):
    baseline = original(market)
    outputs = guard.build_from_inputs(market, comparisons(market, permit_change, starts_change), policy)
    for position, candidate in enumerate(guard.CANDIDATES):
        frame = outputs[candidate["id"]]
        risk = permit_change < 0 and (position == 0 or starts_change < 0)
        active = frame.dropna(how="all").index
        assert len(active) == 124
        assert frame.loc[list(guard.PAUSED)].isna().all().all()
        if risk:
            np.testing.assert_allclose(frame.loc[active, baseline.columns.drop("BIL")],
                                       baseline.loc[active, baseline.columns.drop("BIL")]/2)
            np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)
        else:
            pd.testing.assert_frame_equal(frame.loc[active], baseline.loc[active])
        np.testing.assert_allclose(frame.loc[active].sum(axis=1), 0.98, rtol=0, atol=1e-12)


@pytest.mark.parametrize("problem", [
    "missing_month", "unknown_nan", "filled_known", "wrong_available", "missing_reason", "infinite",
    "negative_units", "string_units", "complex_units", "false_change", "wrong_columns",
])
def test_unknown_missingness_or_wrong_units_cannot_enter_standby_or_safe_default(market, policy, problem):
    rows = comparisons(market)
    day = rows.index[0]
    if problem == "missing_month":
        rows = rows.iloc[1:]
    elif problem == "unknown_nan":
        rows.loc[day, "PERMIT_level"] = np.nan
        rows.loc[day, "PERMIT_change"] = np.nan
        rows.loc[day, "PERMIT_available"] = False
        rows.loc[day, "PERMIT_source_reason"] = guard.PAUSE_REASON
    elif problem == "filled_known":
        pause = guard.PAUSED[0]
        rows.loc[pause, ["PERMIT_level", "PERMIT_reference_level", "PERMIT_change"]] = [1500.0, 1600.0, -100.0]
        rows.loc[pause, "PERMIT_available"] = True
        rows.loc[pause, "PERMIT_source_reason"] = None
    elif problem == "wrong_available":
        rows["PERMIT_available"] = rows["PERMIT_available"].astype(int)
    elif problem == "missing_reason":
        rows.loc[guard.PAUSED[0], "PERMIT_source_reason"] = None
    elif problem == "infinite":
        rows.loc[day, "PERMIT_level"] = np.inf
    elif problem == "negative_units":
        rows.loc[day, "PERMIT_level"] = -1.0
    elif problem == "string_units":
        rows["PERMIT_level"] = rows["PERMIT_level"].astype(str)
    elif problem == "complex_units":
        rows["PERMIT_level"] = rows["PERMIT_level"].astype(complex)
    elif problem == "false_change":
        rows.loc[day, "PERMIT_change"] = 100.0
    else:
        rows = rows.rename(columns={"PERMIT_level": "PERMIT1_level"})
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, rows, policy)


def snapshots(index):
    _, dates = guard.inputs.cohort(index)
    months = pd.date_range("2014-01-01", "2026-08-01", freq="MS")
    stale = {row["vintage_cutoff"]: pd.Timestamp(row["observation_month"]) for row in guard.KNOWN_STALE}
    return {
        name: {
            cutoff: pd.Series(1500.0, index=months[
                (months+pd.offsets.MonthEnd(0) <= pd.Timestamp(cutoff))
                & (months <= stale.get(cutoff, pd.Timestamp("2026-08-01")))
            ])
            for cutoff in dates
        }
        for name in guard.inputs.SERIES
    }


def test_exact14known_stale_queries_remain_nan_and_not_a_zero_or_neutral_signal():
    index = sessions("2015-08-10", "2026-10-05")
    selected = guard.select_queries(index, snapshots(index))
    assert len(selected) == 450
    missing = selected.loc[~selected["source_available"]]
    assert len(missing) == 14
    assert missing["thousands_of_units_saar"].isna().all()
    assert missing["source_reason"].eq(guard.PAUSE_REASON).all()
    for series in guard.inputs.SERIES:
        assert missing.loc[missing["series"].eq(series), list(guard.KNOWN_STALE[0])].to_dict("records") == list(guard.KNOWN_STALE)


@pytest.mark.parametrize("problem", ["unknown_stale", "known_filled", "negative", "missing_snapshot", "future"])
def test_changed_historical_snapshot_cannot_be_folded_into_known_pause(problem):
    index = sessions("2015-08-10", "2026-10-05")
    values = snapshots(index)
    cutoff = next(iter(values["PERMIT"]))
    if problem == "unknown_stale":
        observed = values["PERMIT"][cutoff]
        values["PERMIT"][cutoff] = observed.iloc[:-3]
    elif problem == "known_filled":
        cutoff = guard.KNOWN_STALE[0]["vintage_cutoff"]
        values["PERMIT"][cutoff].loc[pd.Timestamp("2025-10-01")] = 1500.0
    elif problem == "negative":
        values["PERMIT"][cutoff].iloc[-1] = -1.0
    elif problem == "future":
        values["PERMIT"][cutoff].loc[pd.Timestamp("2026-08-01")] = 1500.0
    else:
        values["PERMIT"].pop(cutoff)
    with pytest.raises(QuantError):
        guard.select_queries(index, values)


def test_future_housing_readings_and_prices_cannot_change_earlier_targets(market, policy):
    rows = comparisons(market)
    before = guard.build_from_inputs(market, rows, policy)
    boundary = pd.Timestamp("2021-12-31")
    changed_rows = rows.copy()
    later_rows = (rows.index > boundary) & rows["PERMIT_available"]
    for series in guard.inputs.SERIES:
        changed_rows.loc[later_rows, f"{series}_level"] = 1700.0
        changed_rows.loc[later_rows, f"{series}_change"] = (
            changed_rows.loc[later_rows, f"{series}_level"]-changed_rows.loc[later_rows, f"{series}_reference_level"]
        )
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > boundary
    closing.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    after = guard.build_from_inputs(replace(market, close=closing, open=opening, raw_close=closing.copy()), changed_rows, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_all_seven_pauses_hold_positions_and_every_return_in_funded_and_bt(market, policy, cost, delay):
    desired = guard.build_from_inputs(market, comparisons(market), policy)[guard.CANDIDATES[1]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    for pause in guard.PAUSED:
        assert seeded.loc[pause].isna().all()
        due = own.frame.index[own.frame.index.get_loc(pause)+delay]
        assert own.frame.loc[due, "orders"] == 0 and own.frame.loc[due, "cost"] == 0
        assert own.frame.loc[due, "gross_exposure"] > 0
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [
    ("risk_change_sessions", 126), ("risk_off_scale", 0.75), ("known_paused_months", []),
    ("known_stale_queries", []), ("strict_source_readiness_relabelled", True),
    ("new_economic_factor_definitions", 2),
])
def test_decline_clock_stage_scale_and_known_pause_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_finite_generator_requires_complete_original_source_and_housing_metadata_graph():
    spec = {"id": guard.CANDIDATES[0]["id"], "configuration": guard.CANDIDATES[0],
            "frozen_files": {"src/us_quant/housing_factor_guard.py": "0"*64, "config/housing-factor-guard.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
