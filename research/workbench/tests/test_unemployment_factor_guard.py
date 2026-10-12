from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import implied_risk_execution as parent
from us_quant import unemployment_factor_guard as guard
from us_quant import unemployment_inputs
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/unemployment-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2026-10-05",
        ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"),
    )


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def comparisons(data, change=0.2, dollar=0.01):
    frame = pd.DataFrame(index=original(data).dropna(how="all").index)
    frame["unemployment_level"] = 4.2
    frame["unemployment_reference_level"] = 4.2 - change
    frame["unemployment_change"] = frame["unemployment_level"] - frame["unemployment_reference_level"]
    frame["dollar_change"] = dollar
    return frame


def test_latest_only_contract_does_not_claim_to_repair_a_missing_rolling_history():
    values = pd.Series(4.2, index=pd.date_range("2025-01-01", "2026-07-01", freq="MS"))
    values = values.drop(pd.Timestamp("2025-10-01"))
    query, cutoff = pd.Timestamp("2026-08-31"), "2026-08-27"
    value, audit = guard.select_latest(values, query, cutoff)
    assert value == 4.2 and audit["completed_period_age_calendar_days"] == 31
    assert audit["consumed_history_scope"] == "latest_actual_completed_month_only"
    assert audit["rolling_history_source_readiness_repaired"] is False
    with pytest.raises(QuantError, match="15consecutive"):
        unemployment_inputs.select(values, query, cutoff)


@pytest.mark.parametrize("problem", ["stale", "future", "wrong_cutoff", "missing", "range", "bool", "complex"])
def test_latest_observation_unknown_or_invalid_data_is_fatal_not_fallback(problem):
    query, cutoff = pd.Timestamp("2026-08-31"), "2026-08-27"
    values = pd.Series([4.2], index=pd.to_datetime(["2026-07-01"]))
    if problem == "stale":
        values.index = pd.to_datetime(["2026-04-01"])
    elif problem == "future":
        values.index = pd.to_datetime(["2026-08-01"])
    elif problem == "wrong_cutoff":
        cutoff = "2026-08-28"
    elif problem == "missing":
        values[:] = np.nan
    elif problem == "range":
        values.iloc[0] = 101
    else:
        values = values.astype(bool if problem == "bool" else complex)
    with pytest.raises(QuantError):
        guard.select_latest(values, query, cutoff)


@pytest.mark.parametrize("change,dollar", [
    (0.2, 0.01), (0.2, 0.0), (0.2, -0.01), (0.0, 0.01), (-0.2, 0.01),
])
def test_fixed_strict_unemployment_and_optional_dollar_truth_table(market, policy, change, dollar):
    source = original(market)
    output = guard.build_from_inputs(market, source, comparisons(market, change, dollar), policy)
    active = source.dropna(how="all").index
    for candidate in guard.CANDIDATES:
        frame = output[candidate["id"]]
        assert frame.isna().equals(source.isna()) and len(frame.dropna(how="all")) == 131
        if change > 0 and (not candidate["dollar_confirmation"] or dollar > 0):
            np.testing.assert_allclose(frame.loc[active, source.columns.drop("BIL")],
                                       source.loc[active, source.columns.drop("BIL")] / 2)
            np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)
        else:
            pd.testing.assert_frame_equal(source, frame)


@pytest.mark.parametrize("problem", ["missing_month", "nonfinite", "false_change", "false_columns", "out_of_range"])
def test_all131causal_comparisons_and_actual_levels_are_required(market, policy, problem):
    source, values = original(market), comparisons(market)
    if problem == "missing_month":
        values = values.iloc[1:]
    elif problem == "nonfinite":
        values.iloc[0, 0] = np.nan
    elif problem == "false_change":
        values.iloc[0, 2] = 99
    elif problem == "false_columns":
        values = values.rename(columns={"unemployment_level": "Sahm_gap"})
    else:
        values.iloc[0, 0] = -1
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, source, values, policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_exact_existing_observed_band_and_independent_bt_account_are_preserved(market, policy, cost, delay):
    desired = guard.build_from_inputs(market, original(market), comparisons(market), policy)[guard.CANDIDATES[0]["id"]]
    result, issued, decisions = guard.run(market, desired, guard.CANDIDATES[0], policy,
                                        "2021-10-06", "2026-10-05", cost, delay)
    old, old_issued, old_decisions = parent.run(
        market, desired, parent.CANDIDATES[0], read_json(ROOT / "config/implied-risk-execution.json"),
        "2021-10-06", "2026-10-05", cost, delay,
    )
    pd.testing.assert_frame_equal(result.frame, old.frame)
    pd.testing.assert_frame_equal(issued, old_issued)
    assert decisions == old_decisions and len(result.frame) == 1254
    independent = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                    capital=10000, cost_bps=cost, commission=1, delay=delay)
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [
    ("risk_change_sessions", 126), ("risk_off_scale", 0.75), ("actual_weight_trade_band", 0.05),
    ("required_unemployment_observations_per_query", "repaired_Sahm15month_history"),
    ("source_id", "semiconductor_option_proxy_dollar_observed_band025"),
    ("new_economic_factor_definitions", 2),
])
def test_source_rule_horizon_scale_band_and_authority_are_not_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_unique_generator_requires_complete_labor_allocation_execution_source_graph():
    spec = {
        "id": guard.CANDIDATES[0]["id"], "configuration": guard.CANDIDATES[0],
        "frozen_files": {"src/us_quant/unemployment_factor_guard.py": "0" * 64,
                         "config/unemployment-factor-guard.json": "0" * 64},
    }
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
