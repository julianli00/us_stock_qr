from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import rebalance as original_rebalance, simulate as original_simulate
from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import engine_reference, rebalance, simulate
from us_quant.config import QuantError
from us_quant.research_program import accounting_simulator
from us_quant.strategy import buy_and_hold_signals


def test_rounding_only_change_does_not_create_a_real_fixed_commission_trade():
    assets = np.array([9268.842267867001])
    cash = 189.1600462829999
    target = np.array([0.98])
    requested = target * (assets.sum() + cash) - assets
    assert 0 < abs(requested[0]) < 1e-6
    assert original_rebalance(assets, cash, target, 20, 1)[2] > 1
    actual, remaining, cost, turnover, orders = rebalance(assets, cash, target, 20, 1)
    np.testing.assert_array_equal(actual, assets)
    assert actual is not assets
    assert remaining == cash
    assert cost == turnover == orders == 0


@pytest.mark.parametrize("cost", [0.0, 5.0, 20.0])
@pytest.mark.parametrize("commission", [0.0, 1.0])
def test_real_trades_keep_the_original_cash_budget_and_fees(cost, commission):
    assets = np.array([3000.0, 4000.0])
    target = np.array([0.35, 0.63])
    original = original_rebalance(assets, 3000, target, cost, commission)
    corrected = rebalance(assets, 3000, target, cost, commission)
    for left, right in zip(original, corrected, strict=True):
        np.testing.assert_array_equal(left, right)
    assert corrected[-1] == 2
    assert corrected[1] >= 0


@pytest.mark.parametrize("target", [np.array([np.nan]), np.array([-0.1]), np.array([1.1])])
def test_invalid_inputs_cannot_be_treated_as_no_order(target):
    with pytest.raises(QuantError):
        rebalance(np.array([9800.0]), 200, target, 20, 1)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_ordinary_accounts_and_independent_bt_remain_identical(market_factory, cost, delay):
    data = market_factory()
    signals = buy_and_hold_signals(data.close, "SPY", "2020-01-02")
    args = ("2020-01-02", "2020-07-31")
    original = original_simulate(
        data, signals, *args, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    corrected = simulate(
        data, signals, *args, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    pd.testing.assert_frame_equal(original.frame, corrected.frame)
    pd.testing.assert_frame_equal(original.weights, corrected.weights)
    independent = independent_equity(
        data, signals, *args, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(corrected.frame["equity"], independent["equity"], rtol=0, atol=1e-8)


def test_bundle_selects_exact_accounting_version_and_preserves_legacy_replay():
    assert accounting_simulator({}, legacy_allowed=True) is original_simulate
    assert accounting_simulator({"accounting_engine": engine_reference()}) is simulate
    with pytest.raises(QuantError, match="accounting engine"):
        accounting_simulator({})
    with pytest.raises(QuantError, match="accounting engine"):
        accounting_simulator(
            {"accounting_engine": {**engine_reference(), "source_sha256": "0" * 64}}
        )
