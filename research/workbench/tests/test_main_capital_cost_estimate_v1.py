from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pandas as pd
import pytest

from us_quant import main_capital_cost_estimate_v1 as model
from us_quant import strategy_expert_pool_v2 as original
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation


def series(a, b):
    return pd.Series([a, b], index=["A", "B"], dtype=float)


def observations(position=125, *, end=240, future_change=False):
    calendar = sessions("2016-01-04", "2016-12-30")[:end]
    day = calendar[position]
    elapsed = np.arange(len(calendar), dtype=float)
    result = []
    for i in range(8):
        returns = pd.Series(
            0.0002 + 0.00003 * i +
            0.005 * np.sin(elapsed / (3.0 + i) + i) +
            0.002 * np.cos(elapsed / 11), index=calendar,
        )
        if future_change:
            returns.loc[returns.index > day] = -0.8
        known = returns.loc[:day].tail(126)
        result.append(
            ExpertObservation(
                ExpertIdentity(f"synthetic_{i}", f"mechanism_{i}", "2026-11-04T00:00:00Z"),
                day,
                series(0.98 if i % 2 else 0.0, 0.0 if i % 2 else 0.98),
                day,
                known,
                pd.Series(0.00005, index=known.index),
                len(known) == 126,
            )
        )
    return result


def test_target_proxy_is_the_exact_original_expression():
    reference, target = series(0.49, 0.49), series(0.98, 0)
    estimate = model.TargetProxyCost(reference).estimate(target)
    change = abs(target - reference).to_numpy()
    expected = float(change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000)
    assert estimate.cost_fraction == expected
    assert estimate.cost_usd == expected * 10000
    assert estimate.model_tickets == 2
    assert estimate.model == "original_target_proxy"


def test_nominal_opinion_noop_is_not_a_physical_holdings_noop():
    target = series(0.98, 0)
    assert model.TargetProxyCost(target).estimate(target).cost_usd == 0
    actual = model.CloseMarkedCashCost(series(4900, 4900), 200).estimate(target)
    post_cost = (10000 - 2) / (1 + 0.98 * 5 / 10000)
    assert actual.cost_usd == pytest.approx(10000 - post_cost, abs=1e-9)
    assert actual.turnover_fraction == pytest.approx(0.98 * post_cost / 10000)
    assert actual.model_tickets == 2
    assert actual.reason == "observed_weight_gap"


def test_cash_seed_uses_one_fixed_ticket_and_post_cost_funding():
    estimate = model.CloseMarkedCashCost(
        series(0, 0), 10000, initial_capital=True
    ).estimate(series(0.98, 0))
    expected_post_cost = (10000 - 1) / (1 + 0.98 * 5 / 10000)
    assert estimate.cost_usd == pytest.approx(10000 - expected_post_cost, abs=1e-9)
    assert estimate.model_tickets == 1
    assert estimate.reason == "initial_capital"


def test_initialization_cannot_relabel_an_existing_funded_book():
    with pytest.raises(QuantError, match="initialization seed starts all cash"):
        model.CloseMarkedCashCost(series(4900, 4900), 200, initial_capital=True)


def test_fixed_commission_depends_on_actual_capital_not_nominal_10000():
    target = series(0.98, 0)
    small = model.CloseMarkedCashCost(series(4900, 4900), 200).estimate(target)
    large = model.CloseMarkedCashCost(series(49000, 49000), 2000).estimate(target)
    assert small.capital_usd == 10000 and large.capital_usd == 100000
    assert small.model_tickets == large.model_tickets == 2
    assert large.cost_fraction < small.cost_fraction
    assert large.cost_usd < 10 * small.cost_usd


@pytest.mark.parametrize(
    ("gap", "issues"), [(0, False), (0.0249, False), (0.025, True), (0.0251, True)]
)
def test_original_actual_holdings_band_is_not_retuned(gap, issues):
    estimate = model.CloseMarkedCashCost(
        series((0.49 - gap) * 10000, (0.49 + gap) * 10000), 200
    ).estimate(series(0.49, 0.49))
    assert (estimate.cost_usd > 0) is issues
    if not issues:
        assert estimate.model_tickets == 0
        assert estimate.turnover_fraction == 0
        assert estimate.reason == "hold_inside_original_actual_weight_band"


def test_snapshot_reference_and_estimates_are_not_mutable_caller_state():
    held, reference, target = series(4900, 4900), series(0.49, 0.49), series(0.98, 0)
    close, proxy = model.CloseMarkedCashCost(held, 200), model.TargetProxyCost(reference)
    before_close, before_proxy = close.estimate(target), proxy.estimate(target)
    held.iloc[:] = 0
    reference.iloc[:] = 0
    assert close.estimate(target) == before_close
    assert proxy.estimate(target) == before_proxy
    with pytest.raises(FrozenInstanceError):
        before_close.cost_usd = 0
    pd.testing.assert_series_equal(target, series(0.98, 0))


@pytest.mark.parametrize("bad", [series(-1, 100), series(np.nan, 100), series(np.inf, 100)])
def test_invalid_holdings_are_rejected_not_filled(bad):
    with pytest.raises(QuantError):
        model.CloseMarkedCashCost(bad, 100)


@pytest.mark.parametrize("cash", [-1, np.nan, np.inf, "200", True])
def test_invalid_cash_is_not_a_default(cash):
    with pytest.raises(QuantError):
        model.CloseMarkedCashCost(series(1, 1), cash)


def test_no_capital_or_unfundable_commission_is_not_a_zero_cost_success():
    with pytest.raises(QuantError, match="positive finite capital"):
        model.CloseMarkedCashCost(series(0, 0), 0)
    with pytest.raises(QuantError, match="cannot fund the complete"):
        model.CloseMarkedCashCost(
            series(0, 0), 0.5, initial_capital=True
        ).estimate(series(0.98, 0))


@pytest.mark.parametrize(
    "bad",
    [
        series(-0.1, 1.08),
        series(0.49, np.nan),
        series(0.49, np.inf),
        series(0.5, 0.5),
        series(0.49, 0.49).iloc[::-1],
        pd.Series([0.49, 0.49], index=["A", "A"]),
        pd.Series(["0.49", "0.49"], index=["A", "B"]),
        pd.Series([True, False], index=["A", "B"]),
    ],
)
def test_bad_target_is_not_reindexed_normalized_or_converted(bad):
    for estimator in (
        model.TargetProxyCost(series(0.49, 0.49)),
        model.CloseMarkedCashCost(series(4900, 4900), 200),
    ):
        with pytest.raises(QuantError):
            estimator.estimate(bad)


def test_invalid_cost_result_cannot_cross_the_interface():
    with pytest.raises(QuantError):
        model.CostEstimate("bad", 10000, -1, -0.0001, 0, 0, "invalid")
    with pytest.raises(QuantError):
        model.CostEstimate("bad", 10000, 1, 0, 0, 1, "inconsistent")
    with pytest.raises(QuantError):
        model.CostEstimate("bad", 10000, 10000, 1, 1, 1, "exhausts_capital")
    with pytest.raises(QuantError):
        model.CostEstimate("bad", "10000", 1, 0.0001, 1, 1, "invalid_capital_type")


def test_proxy_adapter_reproduces_every_original_pair_and_selected_weight_exactly():
    observed = observations()
    previous = np.full(8, 1 / 8)
    old_weights, old_assessment = original.choose_pair(observed, previous)
    proxy = model.TargetProxyCost(original.combine(observed, previous))
    weights, assessment = model.choose_pair_with_cost(observed, previous, proxy)
    np.testing.assert_array_equal(weights, old_weights)
    assert assessment["selected_pair"] == old_assessment["selected_pair"]
    assert len(assessment["pairs"]) == 28
    for old, new in zip(old_assessment["pairs"], assessment["pairs"], strict=True):
        assert new["pair"] == old["pair"]
        assert new["expert_weights"] == old["expert_weights"]
        for key in (
            "train_net_risk_score", "validation_net_risk_score",
            "selection_score", "selection_score_key",
        ):
            assert new[key] == old[key]
        assert new["cost_estimate"]["cost_fraction"] == (
            old["estimated_incremental_switch_cost_fraction"]
        )
        assert new["cost_estimation_is_actual_future_execution"] is False


def test_close_mark_adapter_reprices_same_complete_catalogue_without_mutating_sources():
    observed = observations()
    original_target = observed[0].target.copy()
    original_returns = observed[0].net_returns.copy()
    previous = np.full(8, 1 / 8)
    old_weights, old = original.choose_pair(observed, previous)
    snapshot = model.CloseMarkedCashCost(series(1000, 8800), 200)
    weights, new = model.choose_pair_with_cost(observed, previous, snapshot)
    assert len(new["pairs"]) == len(old["pairs"]) == 28
    excess = np.column_stack([o.net_returns - o.risk_free for o in observed])
    for before, after in zip(old["pairs"], new["pairs"], strict=True):
        assert before["expert_weights"] == after["expert_weights"]
        assert before["training_marginal_variance_contributions"] == (
            after["training_marginal_variance_contributions"]
        )
        allocation = np.asarray(after["expert_weights"])
        cost = snapshot.estimate(original.combine(observed, allocation))
        assert after["cost_estimate"] == asdict(cost)
        for sample, key in (
            (excess[:84], "train_net_risk_score"),
            (excess[84:], "validation_net_risk_score"),
        ):
            returns = sample @ allocation
            expected = float(
                np.sqrt(252) * (np.mean(returns) - cost.cost_fraction / 21) /
                np.std(returns, ddof=1)
            )
            assert after[key] == expected
    expected = sorted(new["pairs"], key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
    np.testing.assert_array_equal(weights, expected["expert_weights"])
    pd.testing.assert_series_equal(observed[0].target, original_target)
    pd.testing.assert_series_equal(observed[0].net_returns, original_returns)
    np.testing.assert_array_equal(previous, np.full(8, 1 / 8))
    np.testing.assert_array_equal(old_weights, original.choose_pair(observed, previous)[0])


@pytest.mark.parametrize("position", [125, 150, 175, 200, 220])
def test_synthetic_prefix_and_future_changes_preserve_all_pair_scores(position):
    previous = np.full(8, 1 / 8)
    estimator = model.CloseMarkedCashCost(series(1000, 8800), 200)
    full = model.choose_pair_with_cost(observations(position), previous, estimator)
    prefix = model.choose_pair_with_cost(
        observations(position, end=position + 1), previous, estimator
    )
    changed = model.choose_pair_with_cost(
        observations(position, future_change=True), previous, estimator
    )
    np.testing.assert_array_equal(full[0], prefix[0])
    np.testing.assert_array_equal(full[0], changed[0])
    assert full[1] == prefix[1] == changed[1]


def test_unmatured_is_explicit_equal_weights_without_cost_based_success():
    class Unavailable:
        def estimate(self, target):
            raise AssertionError("An unmatured assessment must not invent a cost-based score.")

    weights, assessment = model.choose_pair_with_cost(
        observations(124), np.full(8, 1 / 8), Unavailable()
    )
    np.testing.assert_array_equal(weights, np.full(8, 1 / 8))
    assert assessment["scores_available"] is False
    assert assessment["safety_or_success_claimed"] is False
    assert assessment["selected_pair"] is None


def test_same_mechanism_exclusion_and_zero_variance_remain_explicit():
    observed = observations()
    observed[1] = replace(
        observed[1],
        identity=replace(
            observed[1].identity, mechanism_group=observed[0].identity.mechanism_group
        ),
    )
    estimator = model.CloseMarkedCashCost(series(4900, 4900), 200)
    _, assessment = model.choose_pair_with_cost(observed, np.full(8, 1 / 8), estimator)
    excluded = [row for row in assessment["pairs"] if "unscorable_reason" in row]
    assert len(excluded) == 1
    assert excluded[0]["unscorable_reason"] == "same_mechanism_group"
    constant = [replace(o, net_returns=o.risk_free.copy()) for o in observed]
    weights, unscored = model.choose_pair_with_cost(constant, np.full(8, 1 / 8), estimator)
    np.testing.assert_array_equal(weights, np.full(8, 1 / 8))
    assert unscored["scores_available"] is False
    assert unscored["safety_or_success_claimed"] is False


def test_missing_or_future_expert_opinion_is_not_made_scorable_by_cost_estimation():
    observed = observations()
    for bad in (
        replace(observed[0], target=None),
        replace(observed[0], target_known_at=observed[0].as_of + pd.Timedelta(days=1)),
    ):
        with pytest.raises(QuantError):
            model.choose_pair_with_cost(
                [bad, *observed[1:]],
                np.full(8, 1 / 8),
                model.CloseMarkedCashCost(series(4900, 4900), 200),
            )


def test_module_cannot_execute_an_account_or_read_future_prices():
    assert not hasattr(model, "simulate")
    assert not hasattr(model.CloseMarkedCashCost, "execute")
    assert not hasattr(model.CloseMarkedCashCost, "observe_prices")
