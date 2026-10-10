from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import next_session
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol
from us_quant.portfolio_protection import (
    independent_check,
    protection_metrics,
    register,
    risk_budget,
    simulate_protection,
    validate_policy,
    verify_registration,
)
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/portfolio-protection.json")


@pytest.fixture
def comparison():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2020-01-02", "2020-06-30", ("SPY", "QQQ", "BIL", "TQQQ", "UPRO"))


def simulated(data, policy, *, end="2020-06-30", stress=False, candidate_index=1):
    return simulate_protection(
        data,
        policy["candidates"][candidate_index],
        policy,
        "2020-01-03",
        end,
        delay=policy["stress_delay_sessions"] if stress else policy["base_delay_sessions"],
        cost_bps=policy["stress_cost_bps_per_side"] if stress else policy["cost_bps_per_side"],
        commission=policy["commission_per_order"],
    )


def test_floor_formula_uses_only_observed_equity_and_peak():
    weight, floor = risk_budget(10000, 10000, 6, 0.90, 0.02)
    assert weight == pytest.approx(0.6) and floor == 9000
    weight, floor = risk_budget(9500, 10000, 6, 0.90, 0.02)
    assert weight == pytest.approx(3000 / 9500) and floor == 9000
    assert risk_budget(9000, 10000, 6, 0.90, 0.02)[0] == 0
    assert risk_budget(8800, 10000, 6, 0.90, 0.02)[0] == 0
    assert risk_budget(12000, 12000, 6, 0.90, 0.02)[0] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "args",
    [
        (0, 10000, 6, 0.9, 0.02),
        (10000, 9000, 6, 0.9, 0.02),
        (10000, 10000, float("nan"), 0.9, 0.02),
        (10000, 10000, -1, 0.9, 0.02),
        (10000, 10000, 6, 1.1, 0.02),
        (10000, 10000, 6, 0.9, -0.1),
    ],
)
def test_invalid_protection_inputs_do_not_create_fallback_orders(args):
    with pytest.raises(QuantError):
        risk_budget(*args)


def test_initial_known_capital_allocation_has_no_prior_return_or_lookahead(market, policy):
    run = simulated(market, policy)
    first = run.decisions[0]
    assert first["signal_session"] == "2020-01-02"
    assert first["execution_session"] == "2020-01-03"
    assert first["observed_equity"] == 10000 and first["observed_high_water"] == 10000
    assert first["desired_risk_weight"] == pytest.approx(0.6)
    assert run.issued_signals.loc["2020-01-02", "TQQQ"] == pytest.approx(0.6)
    assert run.issued_signals.loc["2020-01-02", "BIL"] == pytest.approx(0.38)
    assert run.result.frame.iloc[0]["orders"] == 2
    assert (run.result.weights >= 0).all().all()
    assert (run.result.frame["cash"] >= 0).all()


def test_extra_delay_leaves_first_session_uninvested_and_keeps_pending_signals(market, policy):
    stressed = simulated(market, policy, stress=True)
    assert stressed.decisions[0]["execution_session"] == "2020-01-06"
    first = stressed.result.frame.iloc[0]
    assert first["equity"] == 10000 and first["gross_exposure"] == 0
    assert first["orders"] == 0
    assert stressed.result.frame.loc["2020-01-06", "orders"] == 2
    for decision in stressed.decisions:
        day = pd.Timestamp(decision["signal_session"])
        assert pd.Timestamp(decision["execution_session"]) == next_session(next_session(day))


def test_floor_ratchets_with_past_peak_and_does_not_reset_after_loss(market, policy):
    result = simulated(market, policy).result.frame
    expected = np.maximum.accumulate(np.r_[10000, result["equity"]])[1:]
    np.testing.assert_allclose(result["high_water"], expected)
    np.testing.assert_allclose(result["floor"], expected * 0.9)
    assert (result["floor"].diff().dropna() >= 0).all()
    assert not protection_metrics(result)["floor_is_a_guarantee"]


@pytest.mark.parametrize("stress", [False, True])
def test_independent_engine_and_high_water_feedback_agree(market, policy, stress):
    candidate = policy["candidates"][1]
    own = simulated(market, policy, stress=stress)
    independent, audit = independent_check(
        market,
        own,
        candidate,
        policy,
        "2020-01-03",
        "2020-06-30",
        policy["stress_delay_sessions"] if stress else policy["base_delay_sessions"],
        policy["stress_cost_bps_per_side"] if stress else policy["cost_bps_per_side"],
    )
    assert audit["independent_high_water_feedback_passed"]
    assert audit["max_equity_difference_usd"] < 1e-6
    np.testing.assert_allclose(own.result.frame["equity"], independent["equity"], atol=1e-6)


def test_future_price_changes_do_not_change_earlier_floor_or_decisions(market, policy):
    cutoff = pd.Timestamp("2020-04-30")
    original = simulated(market, policy)
    prefix = replace(
        market,
        open=market.open.loc[:cutoff],
        close=market.close.loc[:cutoff],
        raw_close=market.raw_close.loc[:cutoff],
        volume=market.volume.loc[:cutoff],
        risk_free=market.risk_free.loc[:cutoff],
    )
    shorter = simulated(prefix, policy, end=str(cutoff.date()))
    changed_open, changed_close = market.open.copy(), market.close.copy()
    future = changed_close.index > cutoff
    multiplier = np.linspace(1.01, 1.3, future.sum())
    changed_open.loc[future, "TQQQ"] *= multiplier
    changed_close.loc[future, "TQQQ"] *= multiplier
    changed = simulated(
        replace(market, open=changed_open, close=changed_close, raw_close=changed_close.copy()),
        policy,
    )
    pd.testing.assert_frame_equal(original.result.frame.loc[:cutoff], shorter.result.frame)
    pd.testing.assert_frame_equal(
        original.result.frame.loc[:cutoff], changed.result.frame.loc[:cutoff]
    )
    pd.testing.assert_frame_equal(original.issued_signals.loc[:cutoff], shorter.issued_signals)
    assert [
        row for row in original.decisions if row["signal_session"] <= str(cutoff.date())
    ] == shorter.decisions


def test_gap_can_breach_the_floor_and_is_not_erased_by_next_close_exit(market, policy):
    opening = market.open * 0 + 100
    closing = market.close * 0 + 100
    opening.loc["2020-01-06":, "TQQQ"] = 50
    closing.loc["2020-01-06":, "TQQQ"] = 50
    crashed = replace(market, open=opening, close=closing, raw_close=closing.copy())
    run = simulated(crashed, policy, end="2020-01-10")
    day = run.result.frame.loc["2020-01-06"]
    assert day["opening_equity_before_orders"] < day["floor"]
    assert day["closing_floor_breach"]
    assert day["desired_risk_weight"] == 0
    assert run.issued_signals.loc["2020-01-06", "TQQQ"] == 0
    assert run.result.frame.loc["2020-01-07", "risky_asset_weight"] == 0
    assert protection_metrics(run.result.frame)["max_drawdown"] > 0.15
    assert any(row["reason"] == "floor_risk_exit" for row in run.decisions)


def test_constant_prices_do_not_create_daily_technical_fee_orders(market, policy):
    price = market.close * 0 + 100
    constant = replace(market, open=price.copy(), close=price.copy(), raw_close=price.copy())
    run = simulated(constant, policy)
    assert run.result.frame["orders"].sum() == 2
    assert (run.result.frame["cost"] > 0).sum() == 1
    assert run.result.frame["equity"].iloc[-1] < 10000


def test_changed_rules_or_authority_invalidate_preregistration(policy, comparison, tmp_path):
    validate_policy(policy, comparison)
    for changed in (
        {**policy, "broker_order_authority": True},
        {**policy, "floor_fraction_of_high_water": 0.8},
        {**policy, "prior_disclosed_trials": 0},
        {**policy, "rebalance_band": 0.10},
    ):
        with pytest.raises(QuantError):
            validate_policy(changed, comparison)
    changed = deepcopy(policy)
    changed["candidates"][0]["cushion_multiplier"] = 10
    with pytest.raises(QuantError):
        validate_policy(changed, comparison)
    path = tmp_path / "registration.json"
    record = register(policy, comparison, {"market": "fixed-source"}, path)
    assert record["global_trials_after_round"] == 52 and not record["order_authority"]
    verify_registration(policy, comparison, {"market": "fixed-source"}, path)
    with pytest.raises(QuantError, match="replace"):
        register(policy, comparison, {"market": "fixed-source"}, path)
    with pytest.raises(QuantError, match="data evidence"):
        verify_registration(policy, comparison, {"market": "changed"}, path)
