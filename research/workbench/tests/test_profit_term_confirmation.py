from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import profit_factor_guard as parent
from us_quant import profit_term_confirmation as guard
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT/"config/profit-term-confirmation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def source(data):
    return sector_targets(data, read_json(ROOT/"config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def inputs(data, dollar=0.01, profit_change=-100.0, share_change=-0.01, vix=25.0, far=20.0):
    index = source(data).dropna(how="all").index
    comparisons = pd.DataFrame({"dollar_change": dollar}, index=index)
    for name, level, change in (("CPATAX", 2500.0, profit_change), ("profit_share", 0.10, share_change)):
        comparisons[f"{name}_level"] = level
        comparisons[f"{name}_reference_level"] = level-change
        comparisons[f"{name}_change"] = comparisons[f"{name}_level"]-comparisons[f"{name}_reference_level"]
    return comparisons, pd.DataFrame({"VIX": vix, "VIX3M": far}, index=index)


@pytest.mark.parametrize("dollar,profit_change,share_change,vix,far", [
    (0.01, -100.0, -0.01, 25.0, 20.0),
    (0.01, 100.0, 0.01, 25.0, 20.0),
    (0.01, 100.0, 0.01, 20.0, 20.0),
    (0.01, -100.0, 0.01, 19.0, 20.0),
    (0.01, 100.0, -0.01, 19.0, 20.0),
    (0.01, 0.0, 0.0, 19.0, 20.0),
    (0.0, -100.0, -0.01, 25.0, 20.0),
    (-0.01, -100.0, -0.01, 25.0, 20.0),
])
def test_fixed_dollar_and_profit_or_pressure_truth_table_scales_once(market, policy, dollar, profit_change, share_change, vix, far):
    original = source(market)
    comparisons, terms = inputs(market, dollar, profit_change, share_change, vix, far)
    outputs = guard.build_from_inputs(market, original, comparisons, terms, policy)
    for candidate, change in zip(guard.CANDIDATES, (profit_change, share_change), strict=True):
        frame = outputs[candidate["id"]]
        active = original.dropna(how="all").index
        if dollar > 0 and (change < 0 or vix >= far):
            np.testing.assert_allclose(frame.loc[active, original.columns.drop("BIL")],
                                       original.loc[active, original.columns.drop("BIL")]/2)
            np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)
        else:
            pd.testing.assert_frame_equal(frame, original)
        assert frame.isna().equals(original.isna())
        assert len(frame.dropna(how="all")) == 131


@pytest.mark.parametrize("problem", ["missing_month", "wrong_columns", "nonfinite", "zero", "bool", "complex"])
def test_unverified_option_input_cannot_default_to_safe_or_remove_a_month(market, policy, problem):
    original = source(market)
    comparisons, terms = inputs(market)
    if problem == "missing_month":
        terms = terms.iloc[1:]
    elif problem == "wrong_columns":
        terms = terms.rename(columns={"VIX3M": "VIX9D"})
    elif problem == "nonfinite":
        terms.iloc[0, 0] = np.inf
    elif problem == "zero":
        terms.iloc[0, 0] = 0.0
    elif problem == "bool":
        terms["VIX"] = terms["VIX"].astype(bool)
    else:
        terms["VIX"] = terms["VIX"].astype(complex)
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, original, comparisons, terms, policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_exact_inherited_holding_band_queue_and_bt_account_contract(market, policy, cost, delay):
    original = source(market)
    comparisons, terms = inputs(market)
    desired = guard.build_from_inputs(market, original, comparisons, terms, policy)[guard.CANDIDATES[0]["id"]]
    result, issued, decisions = guard.run(market, desired, guard.CANDIDATES[0], policy,
                                        "2021-10-06", "2026-10-05", cost, delay)
    old, old_issued, old_decisions = parent.run(
        market, desired, parent.CANDIDATES[0], read_json(ROOT/"config/profit-factor-guard.json"),
        "2021-10-06", "2026-10-05", cost, delay,
    )
    pd.testing.assert_frame_equal(result.frame, old.frame)
    pd.testing.assert_frame_equal(issued, old_issued)
    assert decisions == old_decisions
    separate = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(result.frame) == 1254 and result.frame["cash"].ge(0).all()
    np.testing.assert_allclose(result.frame["equity"], separate["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [
    ("option_ratio_risk_threshold", 1.1), ("risk_off_scale", 0.75),
    ("confirmation_schedule", "daily"), ("actual_weight_trade_band", 0.05),
    ("information_rule", "positive_dollar_and_both_profit_decline_and_option_pressure"),
    ("new_economic_factor_definitions", 2),
])
def test_no_option_ratio_scale_frequency_logic_or_band_grid_is_allowed(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_complete_earnings_option_allocation_execution_graph_is_required():
    candidate = guard.CANDIDATES[0]
    spec = {"id": candidate["id"], "configuration": candidate,
            "frozen_files": {"src/us_quant/profit_term_confirmation.py": "0"*64,
                             "config/profit-term-confirmation.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
