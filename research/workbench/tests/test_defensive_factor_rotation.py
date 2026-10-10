from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.defensive_factor_rotation import build_targets, target, validate_policy
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/defensive-factor-rotation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *FACTORS, "TLT"))


def test_no_positive_defense_uses_real_bills_without_dividing_by_bill_volatility(market, policy):
    history = market.close.copy()
    for symbol in ("GLD", "TLT", "IEF"):
        history[symbol] = np.linspace(100, 90, len(history))
    history["BIL"] = 100
    weights = target(history, policy["candidates"][0], policy)
    assert weights["BIL"] == pytest.approx(0.98 * 0.70)
    np.testing.assert_allclose(weights.loc[list(FACTORS)], 0.98 * 0.30 / 4)
    assert weights[["GLD", "TLT", "IEF"]].sum() == 0


def test_highest_defense_and_diversified_defense_are_distinct_fixed_rules(market, policy):
    history = market.close.copy()
    t = np.arange(len(history))
    for symbol, drift in (("GLD", 0.0005), ("TLT", 0.0003), ("IEF", 0.0001)):
        history[symbol] = 100 * np.exp(drift * t + 0.001 * np.sin(t))
    history["BIL"] = 100
    winner = target(history, policy["candidates"][0], policy)
    diversified = target(history, policy["candidates"][1], policy)
    assert winner["GLD"] > 0 and winner[["TLT", "IEF"]].sum() == 0
    assert (diversified[["GLD", "TLT", "IEF"]] > 0).all()
    for weights in (winner, diversified):
        assert weights.sum() == pytest.approx(0.98)
        eq = weights.loc[list(FACTORS)]
        np.testing.assert_allclose(eq / eq.sum(), 0.25)
        assert 0.98 * 0.30 - 1e-12 <= eq.sum() <= 0.98 * 0.70 + 1e-12


def test_future_defensive_returns_cannot_change_past_targets(market, policy):
    original = build_targets(market, policy)
    cut = pd.Timestamp("2017-01-31")
    close = market.close.copy()
    after = close.index > cut
    close.loc[after, "TLT"] *= np.linspace(1, 1.4, after.sum())
    changed = replace(market, close=close, raw_close=close.copy(), open=close * 0.999)
    updated = build_targets(changed, policy)
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:cut], updated[name].loc[:cut])


def test_no_post_result_defense_parameter_changes(policy):
    policy["momentum_sessions"] = 63
    with pytest.raises(QuantError):
        validate_policy(policy)
