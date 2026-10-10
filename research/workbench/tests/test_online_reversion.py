from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol
from us_quant.online_reversion import (
    SimulationFundingError,
    actionable_schedule,
    audited_scenario,
    build_online_signals,
    register,
    simplex_projection,
    update_weights,
    validate,
    verify_registration,
)
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/online-reversion.json")


@pytest.fixture
def comparison():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


def test_projection_matches_independent_constrained_optimizer():
    rng = np.random.default_rng(12)
    for _ in range(12):
        value = rng.normal(0, 3, 5)
        actual = simplex_projection(value)
        independent = minimize(
            lambda weights, target=value: 0.5 * np.sum((weights - target) ** 2),
            np.full(5, 0.2),
            jac=lambda weights, target=value: weights - target,
            bounds=[(0, 1)] * 5,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(5)},
            method="SLSQP",
            options={"ftol": 1e-12},
        )
        assert independent.success
        np.testing.assert_allclose(actual, independent.x, atol=1e-7)
        np.testing.assert_allclose(simplex_projection(value + 1e6), actual, atol=1e-9)
        assert (actual >= 0).all() and actual.sum() == pytest.approx(1)


def test_two_asset_update_rules_have_expected_reversion_direction():
    weights = np.array([0.5, 0.5])
    # Moving-average forecast expects the recently cheap asset to recover.
    np.testing.assert_allclose(update_weights(weights, np.array([1.05, 0.95]), "olmar", 10), [1, 0])
    # Passive-aggressive rule moves away from today's relative winner.
    np.testing.assert_allclose(update_weights(weights, np.array([1.05, 0.95]), "pamr", 0.5), [0, 1])
    np.testing.assert_allclose(update_weights(weights, np.ones(2), "pamr", 0.5), weights)
    np.testing.assert_allclose(update_weights(weights, np.ones(2), "olmar", 10), weights)


def test_olmar_prediction_matches_original_inverse_relative_price_sum():
    rng = np.random.default_rng(73)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (30, 4)), axis=0)
    relatives = close[1:] / close[:-1]
    direct = close[-5:].mean(axis=0) / close[-1]
    reference = np.ones(4)
    product = np.ones(4)
    for i in range(4):
        product *= relatives[-i - 1]
        reference += 1 / product
    reference /= 5
    np.testing.assert_allclose(direct, reference, atol=1e-14)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    centered = reference - reference.mean()
    multiplier = max(0, 10 - weights @ reference) / (centered @ centered)
    expected = simplex_projection(weights + multiplier * centered)
    np.testing.assert_allclose(update_weights(weights, direct, "olmar", 10), expected, atol=1e-10)


def test_online_updates_are_sequential_prefix_causal_and_scale_invariant(
    policy, comparison, market_factory
):
    data = market_factory("2016-01-04", "2016-12-30", comparison.symbols)
    result = build_online_signals(data, policy)
    cutoff = pd.Timestamp("2016-06-30")
    shorter = replace(
        data,
        open=data.open.loc[:cutoff],
        close=data.close.loc[:cutoff],
        raw_close=data.raw_close.loc[:cutoff],
        volume=data.volume.loc[:cutoff],
        risk_free=data.risk_free.loc[:cutoff],
    )
    prefix = build_online_signals(shorter, policy)
    altered = data.close.copy()
    later = altered.index > cutoff
    altered.loc[later, "SPY"] *= np.linspace(1.01, 1.3, later.sum())
    future = build_online_signals(replace(data, close=altered), policy)
    scale = pd.Series(np.arange(1, len(data.close.columns) + 1), index=data.close.columns)
    scaled = build_online_signals(
        replace(
            data,
            open=data.open * scale,
            close=data.close * scale,
            raw_close=data.raw_close * scale,
        ),
        policy,
    )
    for candidate in policy["candidates"]:
        key = candidate["id"]
        pd.testing.assert_frame_equal(result[key].loc[:cutoff], prefix[key])
        pd.testing.assert_frame_equal(result[key].loc[:cutoff], future[key].loc[:cutoff])
        np.testing.assert_allclose(result[key], scaled[key], rtol=1e-9, atol=1e-9, equal_nan=True)
        assert result[key].iloc[:5].isna().all().all()
        known = result[key].dropna()
        assert len(known) == len(data.close) - 5
        assert (known >= 0).all().all()
        np.testing.assert_allclose(known.sum(axis=1), 0.98, atol=1e-12)


@pytest.mark.parametrize("value", [[], [1, float("nan")], [float("inf"), 0], [[1, 2], [3, 4]]])
def test_invalid_projection_input_is_not_silently_repaired(value):
    with pytest.raises(QuantError):
        simplex_projection(np.asarray(value))


def test_weight_update_rejects_negative_leveraged_or_invalid_vectors():
    for weights, ratio, method, threshold in (
        ([-0.2, 1.2], [1, 1], "olmar", 10),
        ([0.9, 0.9], [1, 1], "olmar", 10),
        ([0.5, 0.5], [-1, 1], "olmar", 10),
        ([0.5, 0.5], [1, 1], "unknown", 10),
        ([0.5, 0.5], [1, 1], "pamr", -1),
        ([0.5, 0.5], [1, 1], "olmar", float("nan")),
    ):
        with pytest.raises(QuantError):
            update_weights(np.array(weights), np.array(ratio), method, threshold)


def test_fixed_policy_no_parameter_search_or_trade_authority(policy, comparison):
    validate(policy, comparison)
    for changed in (
        {**policy, "parameter_search": True},
        {**policy, "broker_order_authority": True},
        {**policy, "prior_disclosed_trials": 0},
        {**policy, "cash_reserve": 0},
    ):
        with pytest.raises(QuantError):
            validate(changed, comparison)
    changed = deepcopy(policy)
    changed["candidates"][0]["epsilon"] = 100
    with pytest.raises(QuantError):
        validate(changed, comparison)


def test_registration_is_immutable_and_keeps_total_trial_count(policy, comparison, tmp_path):
    path = tmp_path / "registration.json"
    sources = {"prices": "fixed", "license": "Apache2"}
    receipt = register(policy, comparison, sources, path)
    assert receipt["prior_trials"] == 46 and receipt["new_trials"] == 2
    assert receipt["global_trials_after_round"] == 48 and not receipt["order_authority"]
    verify_registration(policy, comparison, sources, path)
    with pytest.raises(QuantError, match="overwrite"):
        register(policy, comparison, sources, path)
    with pytest.raises(QuantError, match="source"):
        verify_registration(policy, comparison, {"prices": "changed"}, path)


def test_ported_source_distributes_required_upstream_notices():
    root = files("us_quant") / "notices"
    assert "Apache License" in (root / "OLPS-LICENSE.txt").read_text()
    assert "Version 2.0" in (root / "OLPS-LICENSE.txt").read_text()
    assert "Bin Li" in (root / "OLPS-NOTICE.txt").read_text()


@pytest.mark.parametrize("stress", [False, True])
def test_zero_dollar_rebalances_cannot_trigger_repeated_fixed_commissions(
    comparison, market_factory, stress
):
    from us_quant.bt_audit import independent_equity
    from us_quant.dual_horizon import run_window

    data = market_factory("2021-01-04", "2021-03-31", ("SPY", "QQQ"))
    price = data.close * 0 + 100
    data = replace(data, open=price.copy(), close=price.copy(), raw_close=price.copy())
    signals = pd.DataFrame({"SPY": 0.98, "QQQ": 0.0}, index=price.index)
    original = signals.copy()
    start, end = str(price.index[0].date()), str(price.index[-1].date())
    scheduled, skipped = actionable_schedule(data, signals, start, end, comparison, stress=stress)
    result = run_window(data, scheduled, start, end, comparison, stress=stress)
    delay = 1 + (comparison.stress_additional_delay_sessions if stress else 0)
    independent = independent_equity(
        data,
        scheduled,
        start,
        end,
        capital=comparison.capital_usd,
        cost_bps=comparison.stress_cost_bps_per_side if stress else comparison.cost_bps_per_side,
        commission=comparison.commission_per_order,
        delay=delay,
    )
    pd.testing.assert_frame_equal(signals, original)
    assert skipped == len(price) - delay - 1
    assert result.frame["orders"].sum() == 1
    assert (result.frame["cost"] > 0).sum() == 1
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], atol=1e-7)


def test_actionable_schedule_does_not_hide_real_weight_changes(comparison, market_factory):
    from us_quant.bt_audit import independent_equity
    from us_quant.dual_horizon import run_window

    data = market_factory("2021-01-04", "2021-03-31", ("SPY", "QQQ"))
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signals.iloc[0] = [0.98, 0]
    signals.iloc[10] = [0, 0.98]
    signals.iloc[20] = [0.49, 0.49]
    start, end = str(data.close.index[0].date()), str(data.close.index[-1].date())
    scheduled, skipped = actionable_schedule(data, signals, start, end, comparison)
    assert skipped == 0
    pd.testing.assert_frame_equal(signals, scheduled)
    result = run_window(data, scheduled, start, end, comparison)
    independent = independent_equity(
        data,
        scheduled,
        start,
        end,
        capital=comparison.capital_usd,
        cost_bps=comparison.cost_bps_per_side,
        commission=comparison.commission_per_order,
        delay=1,
    )
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], atol=1e-7)


def test_unfundable_costs_stop_with_explicit_partial_evidence_not_fake_full_returns(
    comparison, market_factory
):
    data = market_factory("2021-01-04", "2021-02-26", ("SPY", "QQQ"))
    fixed_price = data.close * 0 + 100
    data = replace(
        data, open=fixed_price.copy(), close=fixed_price.copy(), raw_close=fixed_price.copy()
    )
    signal = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signal.iloc[0] = [0.98, 0]
    signal.iloc[1] = [0, 0.98]
    expensive = replace(comparison, commission_per_order=6000.0)
    start, end = str(data.close.index[0].date()), str(data.close.index[-1].date())
    with pytest.raises(SimulationFundingError) as failure:
        actionable_schedule(data, signal, start, end, expensive)
    assert failure.value.day == data.close.index[2]
    assert 0 < failure.value.equity < 6000
    result, independent, evidence = audited_scenario(data, signal, start, end, expensive)
    assert evidence["completed_requested_window"] is False
    assert evidence["requested_end"] == end
    assert evidence["audited_end"] == str(data.close.index[1].date())
    assert evidence["stopped"]["no_capital_injection_or_zero_fee_fallback"]
    assert len(result.frame) == 2
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], atol=1e-7)


def test_minimum_cost_counts_new_positions_as_well_as_liquidated_positions(
    comparison, market_factory
):
    data = market_factory("2021-01-04", "2021-02-26", ("SPY", "QQQ", "EFA", "VGK"))
    price = data.close * 0 + 100
    data = replace(data, open=price.copy(), close=price.copy(), raw_close=price.copy())
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signals.iloc[0] = [0.98, 0, 0, 0]
    signals.iloc[1] = [0, 0.30, 0.30, 0.38]
    tiny = replace(comparison, capital_usd=4.0, cost_bps_per_side=0.0)
    start, end = str(data.close.index[0].date()), str(data.close.index[-1].date())
    own, external, audit = audited_scenario(data, signals, start, end, tiny)
    assert audit["completed_requested_window"] is False
    assert audit["stopped"]["session"] == str(data.close.index[2].date())
    assert audit["stopped"]["equity_at_rejected_rebalance"] == pytest.approx(3.0)
    assert own.frame["orders"].sum() == 1
    np.testing.assert_allclose(own.frame["equity"], external["equity"], atol=1e-9)
