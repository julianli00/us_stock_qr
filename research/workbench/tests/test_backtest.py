from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import rebalance, simulate
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData


def gap_market():
    index = sessions("2020-01-30", "2020-02-04")
    opening = pd.DataFrame({"SPY": [100.0, 100.0, 200.0, 210.0]}, index=index)
    close = pd.DataFrame({"SPY": [100.0, 100.0, 210.0, 210.0]}, index=index)
    return MarketData(
        open=opening,
        close=close,
        raw_close=close.copy(),
        volume=pd.DataFrame(1000.0, index=index, columns=["SPY"]),
        risk_free=pd.Series(0.0, index=index),
    )


def test_next_open_does_not_capture_gap_before_purchase():
    data = gap_market()
    # A 100% overnight move is intentional in this execution-timing unit test.
    data = replace(data, close=data.close / 2 + 50, open=data.open / 2 + 50)
    signal = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signal.loc["2020-01-31"] = 1.0
    result = simulate(
        data, signal, "2020-01-30", "2020-02-04", initial_capital=1000, cost_bps=0, commission=0
    )
    assert result.frame.loc["2020-01-31", "equity"] == 1000
    assert result.frame.loc["2020-02-03", "return"] == pytest.approx(155 / 150 - 1)
    assert result.frame.iloc[-1]["equity"] == pytest.approx(1000 * 155 / 150)


def test_delay_stress_uses_a_later_open(market_factory):
    data = market_factory("2020-01-30", "2020-02-05", ("SPY",))
    signal = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signal.loc["2020-01-31"] = 1.0
    result = simulate(data, signal, "2020-01-30", "2020-02-05", delay=2)
    assert result.frame.loc["2020-02-03", "gross_exposure"] == 0
    assert result.frame.loc["2020-02-04", "gross_exposure"] == pytest.approx(1)


def test_rebalance_self_finances_costs_and_never_borrows():
    values, cash, cost, turnover, orders = rebalance(np.array([0.0]), 1000, np.array([1.0]), 10, 1)
    expected = 999 / 1.001
    assert values[0] == pytest.approx(expected)
    assert cash == pytest.approx(0, abs=1e-8)
    assert cost == pytest.approx(1000 - expected)
    assert turnover == pytest.approx(expected / 1000)
    assert orders == 1


def test_both_sides_of_rotation_are_charged():
    values, cash, cost, turnover, orders = rebalance(
        np.array([1000.0, 0.0]), 0.0, np.array([0.0, 1.0]), 10, 1
    )
    assert values[0] == 0
    assert orders == 2
    assert 1.99 < turnover < 2
    assert cost > 3.99
    assert values.sum() + cash + cost == pytest.approx(1000)


def test_randomized_accounting_conservation():
    rng = np.random.default_rng(2)
    for _ in range(100):
        old = rng.uniform(0, 10000, 9)
        cash = float(rng.uniform(0, 1000))
        target = rng.dirichlet(np.ones(10))[:9]
        new, remainder, costs, _, _ = rebalance(old, cash, target, 20, 1)
        assert (new >= 0).all()
        assert remainder >= 0
        assert new.sum() + remainder + costs == pytest.approx(old.sum() + cash, abs=1e-7)


@pytest.mark.parametrize("problem", ["same_close", "leverage", "short", "partial_nan", "bad_price"])
def test_unsafe_or_ambiguous_simulations_rejected(market_factory, problem):
    data = market_factory()
    signal = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signal.iloc[1] = [0.5, 0.5]
    delay = 1
    if problem == "same_close":
        delay = 0
    elif problem == "leverage":
        signal.iloc[1] = [1.0, 1.0]
    elif problem == "short":
        signal.iloc[1] = [-0.1, 0.5]
    elif problem == "partial_nan":
        signal.iloc[1, 0] = np.nan
    else:
        opening = data.open.copy()
        opening.iloc[2, 0] = -1
        data = replace(data, open=opening)
    with pytest.raises(QuantError):
        simulate(data, signal, "2020-01-02", "2021-12-31", delay=delay)
