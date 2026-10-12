from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import implied_risk_execution as adapter
from us_quant import observed_weight_execution as prior
from us_quant.bt_audit import independent_equity
from us_quant.calendar import previous_session
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT/"config/implied-risk-execution.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *adapter.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def desired(data):
    return sector_targets(data, read_json(ROOT/"config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def test_adapter_reuses_exact_pure_band_and_original_cost_capital_law(policy):
    original = adapter.execution_policy(policy)
    band, _ = prior.pure_band(original)
    assert band["weight_band"] == policy["actual_weight_trade_band"] == 0.025
    assert band["qld_cap"] is None and band["volatility_target"] is None
    assert original["capital_usd"] == 10000 and original["commission_per_order"] == 1


@pytest.mark.parametrize("candidate", list(adapter.CANDIDATES))
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_exact_parent_execution_reuse_and_funded_bt_accounts_on_synthetic_contract_data(market, policy, candidate, cost, delay):
    original = desired(market)
    own, issued, decisions = adapter.run(market, original, candidate, policy, "2021-10-06", "2026-10-05", cost, delay)
    old, old_issued, old_decisions = prior.run(
        market, original, prior.CANDIDATES[0], adapter.execution_policy(policy),
        "2021-10-06", "2026-10-05", cost, delay,
    )
    pd.testing.assert_frame_equal(own.frame, old.frame)
    pd.testing.assert_frame_equal(issued, old_issued)
    assert decisions == old_decisions
    separate = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    funded = simulate(market, issued, "2021-10-06", "2026-10-05",
                      initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    pd.testing.assert_frame_equal(own.frame, funded.frame)
    assert decisions[0]["initial_capital"] and decisions[0]["reason"] == "initial_capital"
    anchor = previous_session(pd.Timestamp("2021-10-06"))
    pd.testing.assert_series_equal(issued.loc[anchor], original.loc[:anchor].dropna(how="all").iloc[-1], check_names=False)
    for row in decisions[1:]:
        day = pd.Timestamp(row["signal_session"])
        if row["reason"] == "hold":
            assert issued.loc[day].isna().all()
        else:
            pd.testing.assert_series_equal(issued.loc[day], original.loc[day])


@pytest.mark.parametrize("field,value", [
    ("actual_weight_trade_band", 0.05), ("commission_per_order", 0.0), ("capital_usd", 20000),
    ("cash_reserve", 0.0), ("new_economic_factor_definitions", 2),
])
def test_no_band_cost_budget_or_factor_parameter_grid_is_allowed(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        adapter.validate(policy)


def test_only_the_two_frozen_source_execution_configurations_are_allowed(market, policy):
    with pytest.raises(QuantError):
        adapter.run(market, desired(market), {"id": "different_source"}, policy,
                    "2021-10-06", "2026-10-05", 20, 2)


def test_complete_allocation_and_execution_dependency_graph_is_required():
    candidate = adapter.CANDIDATES[0]
    spec = {"id": candidate["id"], "configuration": candidate,
            "frozen_files": {"src/us_quant/implied_risk_execution.py": "0"*64,
                             "config/implied-risk-execution.json": "0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
