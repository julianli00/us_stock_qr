from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import implied_risk_execution as parent
from us_quant import profit_factor_guard as guard
from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT/"config/profit-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT/"config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def comparisons(data, dollar=0.01, profit_change=-100.0, share_change=-0.01):
    frame = pd.DataFrame({"dollar_change": dollar}, index=original(data).dropna(how="all").index)
    for name, level, change in (("CPATAX", 2500.0, profit_change), ("profit_share", 0.10, share_change)):
        frame[f"{name}_level"] = level
        frame[f"{name}_reference_level"] = level-change
        frame[f"{name}_change"] = frame[f"{name}_level"]-frame[f"{name}_reference_level"]
    return frame


def test_latest_common_quarter_uses_older_gdp_period_not_mismatched_latest_values():
    query = pd.Timestamp("2026-05-29")
    profit = pd.Series([2000.0, 2100.0], index=pd.to_datetime(["2025-07-01", "2025-10-01"]))
    output = pd.Series([30000.0, 31000.0, 32000.0],
                       index=pd.to_datetime(["2025-07-01", "2025-10-01", "2026-01-01"]))
    value, gdp, audit = guard.select_common(profit, output, query, "2026-05-27")
    assert value == 2100 and gdp == 31000
    assert audit["observation_quarter"] == "2025-10-01"
    assert audit["completed_period_end"] == "2025-12-31"
    assert audit["independently_latest_output_quarter"] == "2026-01-01"
    assert audit["macro_proxy_claimed_company_margin"] is False


@pytest.mark.parametrize("problem", ["no_common", "stale_common", "future", "incomplete", "negative_gdp", "wrong_cutoff"])
def test_missing_stale_or_wrongly_timed_join_is_fatal_not_filled_or_safe(problem):
    query = pd.Timestamp("2026-05-29")
    cutoff = "2026-05-27"
    profit = pd.Series([2100.0], index=pd.to_datetime(["2025-10-01"]))
    output = pd.Series([31000.0], index=pd.to_datetime(["2025-10-01"]))
    if problem == "no_common":
        output.index = pd.to_datetime(["2026-01-01"])
    elif problem == "stale_common":
        profit = pd.Series([2100.0, 2200.0], index=pd.to_datetime(["2025-01-01", "2025-10-01"]))
        output = pd.Series([31000.0, 32000.0], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))
    elif problem == "future":
        profit.index = pd.to_datetime(["2026-07-01"])
    elif problem == "incomplete":
        profit.index = pd.to_datetime(["2026-04-01"])
    elif problem == "negative_gdp":
        output.iloc[0] = -1
    else:
        cutoff = "2026-05-28"
    with pytest.raises(QuantError):
        guard.select_common(profit, output, query, cutoff)


def test_later_quarter_values_do_not_replace_the_actual_older_snapshot_values():
    query = pd.Timestamp("2026-05-29")
    profit = pd.Series([2100.0], index=pd.to_datetime(["2025-10-01"]))
    output = pd.Series([31000.0, 32000.0], index=pd.to_datetime(["2025-10-01", "2026-01-01"]))
    before = guard.select_common(profit, output, query, "2026-05-27")
    output.loc["2026-01-01"] = 99999
    after = guard.select_common(profit, output, query, "2026-05-27")
    assert before == after


@pytest.mark.parametrize("dollar,profit_change,share_change", [
    (0.01, -100.0, -0.01), (0.01, -100.0, 0.0), (0.01, 100.0, -0.01),
    (0.01, 0.0, 0.0), (0.0, -100.0, -0.01), (-0.01, -100.0, -0.01),
])
def test_strict_dollar_and_profit_measure_truth_table_preserves_original_allocation(market, policy, dollar, profit_change, share_change):
    source = original(market)
    rows = comparisons(market, dollar, profit_change, share_change)
    outputs = guard.build_from_inputs(market, source, rows, policy)
    for candidate, change in zip(guard.CANDIDATES, (profit_change, share_change), strict=True):
        frame = outputs[candidate["id"]]
        active = source.dropna(how="all").index
        assert frame.isna().equals(source.isna())
        if dollar > 0 and change < 0:
            np.testing.assert_allclose(frame.loc[active, source.columns.drop("BIL")],
                                       source.loc[active, source.columns.drop("BIL")]/2)
            np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)
        else:
            pd.testing.assert_frame_equal(frame, source)


@pytest.mark.parametrize("problem", ["missing_month", "nonfinite", "wrong_columns", "false_change"])
def test_unknown_monthly_comparisons_cannot_default_or_remove_target_rows(market, policy, problem):
    source, rows = original(market), comparisons(market)
    if problem == "missing_month":
        rows = rows.iloc[1:]
    elif problem == "nonfinite":
        rows.iloc[0, 0] = np.nan
    elif problem == "wrong_columns":
        rows = rows.rename(columns={"profit_share_level": "wrong_level"})
    else:
        rows.iloc[0, rows.columns.get_loc("CPATAX_change")] = 99
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, source, rows, policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_profit_adapter_reuses_exact_parent_band_and_synthetic_funded_bt_path(market, policy, cost, delay):
    desired = guard.build_from_inputs(market, original(market), comparisons(market), policy)[guard.CANDIDATES[0]["id"]]
    result, issued, decisions = guard.run(market, desired, guard.CANDIDATES[0], policy,
                                        "2021-10-06", "2026-10-05", cost, delay)
    old, old_issued, old_decisions = parent.run(
        market, desired, parent.CANDIDATES[0], read_json(ROOT/"config/implied-risk-execution.json"),
        "2021-10-06", "2026-10-05", cost, delay,
    )
    pd.testing.assert_frame_equal(result.frame, old.frame)
    pd.testing.assert_frame_equal(issued, old_issued)
    assert decisions == old_decisions
    own = simulate(market, issued, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(result.frame) == 1254 and own.frame["cash"].ge(0).all()
    np.testing.assert_allclose(result.frame["equity"], separate["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [
    ("profit_query_join", "independently_latest_quarters"), ("maximum_completed_quarter_age_days", 279),
    ("risk_change_sessions", 126), ("risk_off_scale", 0.75), ("actual_weight_trade_band", 0.05),
    ("new_economic_factor_definitions", 2),
])
def test_common_period_age_sign_window_scale_band_and_factor_scope_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_unique_generator_requires_complete_quarterly_allocation_execution_graph():
    candidate = guard.CANDIDATES[0]
    spec = {"id": candidate["id"], "configuration": candidate,
            "frozen_files": {"src/us_quant/profit_factor_guard.py": "0"*64, "config/profit-factor-guard.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
