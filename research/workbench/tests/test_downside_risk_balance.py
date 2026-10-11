from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.downside_risk_balance import build_targets, risk_budget, risk_estimate, validate_policy
from us_quant.dual_horizon import seed_window
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/downside-risk-balance.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS))


def test_semideviation_counts_all63observations_and_does_not_penalize_upside():
    values = np.array([-0.03, -0.02, 0.05] * 21)
    expected = float(np.sqrt((21 * 0.03**2 + 21 * 0.02**2) / 63))
    assert risk_estimate(values, "semideviation") == pytest.approx(expected)
    greater_upside = values.copy()
    greater_upside[greater_upside > 0] = 0.5
    assert risk_estimate(greater_upside, "semideviation") == pytest.approx(expected)


def test_expected_loss_includes_exact5percent_fractional_probability_mass():
    values = np.r_[[-0.10, -0.08, -0.06, -0.04], np.full(59, 0.01)]
    expected = (0.10 + 0.08 + 0.06 + 0.15 * 0.04) / 3.15
    assert risk_estimate(values, "expected_loss") == pytest.approx(expected, abs=1e-15)
    assert risk_estimate(values[::-1], "expected_loss") == pytest.approx(expected, abs=1e-15)
    assert risk_estimate(np.ones(63) * 0.01, "expected_loss") == 0


def test_zero_observed_losses_use_declared_bounds_not_fictitious_risk_free_claim():
    assert risk_budget(0.0, 0.01) == 0.70
    assert risk_budget(0.01, 0.0) == 0.30
    with pytest.raises(QuantError, match="zero or invalid"):
        risk_budget(0.0, 0.0)
    with pytest.raises(QuantError):
        risk_estimate(np.zeros(62), "semideviation")
    with pytest.raises(QuantError):
        risk_estimate(np.r_[np.zeros(62), np.nan], "expected_loss")
    with pytest.raises(QuantError):
        risk_estimate(np.zeros(63), "expected_loss", probability=0.10)


def test_original_core_composition_and_funding_are_not_changed(market, policy):
    for frame in build_targets(market, policy).values():
        active = frame.dropna(how="all")
        equity = active.loc[:, [*FACTORS, "QQQ"]].sum(axis=1)
        np.testing.assert_allclose(active["QQQ"], equity / 2)
        for symbol in FACTORS:
            np.testing.assert_allclose(active[symbol], equity / 8)
        assert (equity >= 0.98 * 0.30 - 1e-12).all()
        assert (equity <= 0.98 * 0.70 + 1e-12).all()
        assert active[["BIL", "SPY", "IEF", "TLT"]].eq(0).all().all()
        assert (active >= 0).all().all()
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert active.index[0] < pd.Timestamp("2016-10-05")


def test_future_prices_or_risk_free_rates_cannot_change_earlier_budgets(market, policy):
    before = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    close, opening, rates = market.close.copy(), market.open.copy(), market.risk_free.copy()
    later = close.index > cutoff
    close.loc[later, ["GLD", "QQQ"]] *= np.linspace(1, 1.4, later.sum())[:, None]
    opening.loc[later, ["GLD", "QQQ"]] *= np.linspace(1, 1.4, later.sum())[:, None]
    rates.loc[later] += 0.0001
    after = build_targets(replace(market, close=close, raw_close=close.copy(), open=opening, risk_free=rates), policy)
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], after[name].loc[:cutoff])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_source_replay_and_independent_funded_accounts(market, policy, cost, delay):
    source = ROOT / "src/us_quant/downside_risk_balance.py"
    config = ROOT / "config/downside-risk-balance.json"
    candidate = policy["candidates"][0]
    spec = {
        "id": candidate["id"], "configuration": candidate,
        "frozen_files": {
            source.relative_to(ROOT).as_posix(): file_digest(source),
            config.relative_to(ROOT).as_posix(): file_digest(config),
        },
    }
    start, end = "2016-10-06", "2017-06-30"
    target = registered_targets(spec, market, start, end, cost, delay, ROOT, {})
    pd.testing.assert_frame_equal(target, seed_window(build_targets(market, policy)[candidate["id"]], start))
    own = simulate(
        market, target, start, end, initial_capital=10000,
        cost_bps=cost, commission=1, delay=delay,
    )
    independent = independent_equity(
        market, target, start, end, capital=10000,
        cost_bps=cost, commission=1, delay=delay,
    )
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].min() >= 0 and own.frame["cost"].sum() > 0


@pytest.mark.parametrize("key,value", [
    ("risk_sessions", 126), ("tail_probability", 0.10), ("growth_share_of_equity", 0.75),
])
def test_downside_definitions_or_core_cannot_be_post_outcome_retuned(policy, key, value):
    policy[key] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
