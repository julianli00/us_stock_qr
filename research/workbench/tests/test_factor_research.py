from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol
from us_quant.factor_research import (
    build_signals,
    covariance_weights,
    factor_scores,
    fingerprint,
    goal_gates,
    growth_gold_weights,
    information_discreteness,
    inverse_variance_scale,
    macro_weights,
    rebalance_due,
    snapshot_descriptor,
    validate_policy,
    verify_registration,
)
from us_quant.storage import digest_json, file_digest, read_json, utc_now, write_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/factor-research.json")


@pytest.fixture
def comparison():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


@pytest.fixture
def allocation_policy():
    return read_json(Path(__file__).parents[1] / "config/allocation-research.json")


@pytest.fixture
def implementation_policy():
    return read_json(Path(__file__).parents[1] / "config/implementation-research.json")


@pytest.fixture
def factor_market(market_factory, policy, comparison):
    return market_factory(
        start="2010-01-04",
        end="2014-03-14",
        symbols=tuple(comparison.symbols) + tuple(policy["supplement_symbols"]),
    )


def test_round_preserves_all_configurations_and_both_goal_contracts(policy):
    validate_policy(policy)
    assert len(policy["candidates"]) == 8
    assert policy["prior_disclosed_configurations"] == 52
    assert policy["retained_secondary_goals"]["max_drawdown_at_most"] == 0.15


@pytest.mark.parametrize(
    "change",
    [
        {"cash_reserve": 0},
        {"capital_usd": 1000000},
        {"base_delay_sessions": 0},
        {"stress_delay_sessions": 1},
        {"cost_bps_per_side": 0},
        {"commission_per_order": float("nan")},
        {"prior_disclosed_configurations": 0},
        {"primary_goal": {"net_excess_sharpe_strictly_above": 0.8}},
    ],
)
def test_no_silent_goal_cost_or_authority_relaxation(policy, change):
    policy.update(change)
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_information_discreteness_distinguishes_continuous_and_jump_returns():
    returns = pd.DataFrame({"continuous": [0.01] * 10, "jump": [0.0] * 9 + [1.01**10 - 1]})
    values = information_discreteness(returns)
    assert values["continuous"] == -1
    assert values["jump"] == pytest.approx(-0.1)


def test_variance_management_is_not_inverse_volatility():
    assert inverse_variance_scale(0.4**2, 0.2) == pytest.approx(0.25)
    assert inverse_variance_scale(0.1**2, 0.2) == 1
    assert inverse_variance_scale(0.0, 0.2) == 1
    with pytest.raises(QuantError):
        inverse_variance_scale(float("nan"), 0.2)


def test_covariance_allocation_has_known_closed_form_answers():
    covariance = np.diag([0.04, 0.01])
    minimum = covariance_weights(covariance, "minimum_variance", 0.8)
    diversified = covariance_weights(covariance, "maximum_diversification", 0.8)
    np.testing.assert_allclose(minimum, [0.2, 0.8], atol=1e-6)
    np.testing.assert_allclose(diversified, [1 / 3, 2 / 3], atol=1e-6)
    with pytest.raises(QuantError):
        covariance_weights(np.array([[1.0, 2.0], [2.0, 1.0]]), "minimum_variance", 0.8)


def test_second_round_does_not_hide_first_round_or_change_primary_goal(policy, allocation_policy):
    validate_policy(allocation_policy)
    assert allocation_policy["primary_goal"] == policy["primary_goal"]
    assert allocation_policy["prior_disclosed_configurations"] == 60
    assert len(allocation_policy["candidates"]) == 6
    assert (
        "not an independent holdout" in (allocation_policy["previous_round"]["development_status"])
    )


def test_macro_allocation_uses_identical_selection_for_controls(allocation_policy, factor_market):
    history = factor_market.close.copy()
    length = len(history)
    for symbol, growth in (("QLD", 0.8), ("GLD", 0.4), ("TLT", -0.1), ("IEF", -0.2), ("DBC", -0.3)):
        history[symbol] = 100 * np.exp(np.linspace(0, growth, length))
    for candidate in allocation_policy["candidates"][:3]:
        weights = macro_weights(history, candidate, allocation_policy)
        assert set(weights[weights > 0].index) == {"QLD", "GLD"}
        assert weights.sum() == pytest.approx(0.98)
        assert weights.max() <= 0.98 * 0.8 + 1e-10


def test_allocation_round_signals_are_causal_and_preserve_cash(allocation_policy, factor_market):
    original = build_signals(factor_market, allocation_policy)
    boundary = pd.Timestamp("2013-09-30")
    changed_close = factor_market.close.copy()
    later = changed_close.index > boundary
    changed_close.loc[later, "QLD"] *= np.exp(np.linspace(0, -0.3, later.sum()))
    changed = replace(
        factor_market,
        close=changed_close,
        raw_close=changed_close.copy(),
        open=changed_close * 0.999,
    )
    observed = build_signals(changed, allocation_policy)
    for identifier, frame in original.items():
        defined = frame.dropna(how="all")
        assert np.allclose(defined.sum(axis=1), 0.98)
        assert (defined >= 0).all().all()
        assert all(is_month_end(day) for day in defined.index)
        pd.testing.assert_frame_equal(frame.loc[:boundary], observed[identifier].loc[:boundary])


def test_quarterly_and_target_band_rules_have_no_date_specific_exceptions():
    weight = pd.Series({"QLD": 0.3, "GLD": 0.68, "BIL": 0.0})
    candidate = {"rebalance": "quarterly"}
    assert not rebalance_due(pd.Timestamp("2020-02-28"), weight, None, candidate)
    assert rebalance_due(pd.Timestamp("2020-03-31"), weight, None, candidate)
    candidate = {"target_change_band": 0.05}
    assert rebalance_due(pd.Timestamp("2020-02-28"), weight, None, candidate)
    assert not rebalance_due(pd.Timestamp("2020-02-28"), weight, weight, candidate)
    prior = pd.Series({"QLD": 0.35, "GLD": 0.63, "BIL": 0.0})
    assert rebalance_due(pd.Timestamp("2020-02-28"), weight, prior, candidate)


def test_implementation_round_combines_targets_not_compounded_returns(
    implementation_policy, factor_market
):
    validate_policy(implementation_policy)
    assert implementation_policy["prior_disclosed_configurations"] == 66
    signals = build_signals(factor_market, implementation_policy)
    quarterly = signals["growth_gold_quarterly_risk"].dropna(how="all")
    assert all(day.month % 3 == 0 for day in quarterly.index)
    fixed = signals["growth_gold_equal_notional_control"].dropna(how="all")
    assert np.allclose(fixed["QLD"], 0.98 / 3)
    assert np.allclose(fixed["GLD"], 0.98 * 2 / 3)
    ensemble = signals["equal_growth_gold_min_variance_ensemble"].dropna(how="all")
    day = ensemble.index[-1]
    history = factor_market.close.loc[:day]
    expected = (
        macro_weights(
            history, {"family": "macro", "allocation": "minimum_variance"}, implementation_policy
        )
        / 2
    )
    expected.loc[["QLD", "GLD"]] += growth_gold_weights(history, implementation_policy) / 2
    expected["BIL"] = 0.98 - expected.sum()
    np.testing.assert_allclose(ensemble.loc[day], expected, atol=1e-12)
    for frame in signals.values():
        defined = frame.dropna(how="all")
        assert np.allclose(defined.sum(axis=1), 0.98)
        assert (defined >= 0).all().all()


def test_embedded_leverage_cannot_be_relabelled_as_unleveraged(implementation_policy):
    implementation_policy["candidates"][0]["embedded_leverage"] = False
    with pytest.raises(QuantError, match="leverage"):
        validate_policy(implementation_policy)


def test_implementation_round_is_causal_including_hysteresis(implementation_policy, factor_market):
    original = build_signals(factor_market, implementation_policy)
    boundary = pd.Timestamp("2013-09-30")
    close = factor_market.close.copy()
    later = close.index > boundary
    close.loc[later, "GLD"] *= np.exp(np.linspace(0, 0.4, later.sum()))
    changed = replace(factor_market, close=close, raw_close=close.copy(), open=close * 0.999)
    observed = build_signals(changed, implementation_policy)
    for identifier in original:
        pd.testing.assert_frame_equal(
            original[identifier].loc[:boundary], observed[identifier].loc[:boundary]
        )
    issued = original["growth_gold_target_change_band"].dropna(how="all")
    if len(issued) > 1:
        assert (issued.diff().abs().max(axis=1).iloc[1:] >= 0.05 - 1e-12).all()


def test_residual_factor_removes_exact_market_loading(policy, factor_market):
    close = factor_market.close.copy()
    returns = close.pct_change(fill_method=None).fillna(0)
    rf = factor_market.risk_free
    dependent = 0.0001 + 1.3 * (returns["SPY"] - rf) + rf
    close["XLK"] = 100 * (1 + dependent).cumprod()
    scores = factor_scores(close, rf, policy)
    assert abs(scores.loc["XLK", "residual"]) < 1e-8


def test_residual_warmup_and_alignment_are_required(policy, factor_market):
    with pytest.raises(QuantError, match="warmup"):
        factor_scores(factor_market.close.iloc[:100], factor_market.risk_free.iloc[:100], policy)
    with pytest.raises(QuantError, match="aligned"):
        factor_scores(factor_market.close, factor_market.risk_free.iloc[1:], policy)


def test_signals_are_month_end_cash_funded_and_do_not_use_future_data(policy, factor_market):
    original = build_signals(factor_market, policy)
    boundary = pd.Timestamp("2013-09-30")
    changed_close = factor_market.close.copy()
    later = changed_close.index > boundary
    changed_close.loc[later] *= np.exp(np.linspace(0, 0.3, later.sum()))[:, None]
    changed = replace(
        factor_market,
        close=changed_close,
        raw_close=changed_close.copy(),
        open=changed_close * 0.999,
    )
    changed_signals = build_signals(changed, policy)
    for identifier, frame in original.items():
        defined = frame.dropna(how="all")
        assert all(is_month_end(day) for day in defined.index)
        assert defined.index[-1] == pd.Timestamp("2014-02-28")
        assert np.allclose(defined.sum(axis=1), 0.98)
        assert (defined >= 0).all().all()
        pd.testing.assert_frame_equal(
            frame.loc[:boundary], changed_signals[identifier].loc[:boundary]
        )
    components = policy["candidates"][3]["components"]
    expected = sum(original[name] for name in components) / len(components)
    pd.testing.assert_frame_equal(original["sector_fixed_factor_ensemble"], expected)


def sample_metrics(cagr=0.15, sharpe=1.1):
    return {
        "start": "2016-10-06",
        "end": "2026-10-05",
        "sessions": 2512,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": 0.2,
    }


def test_latest_goal_is_separate_from_old_return_and_drawdown_targets(policy):
    result = goal_gates(sample_metrics(), sample_metrics(cagr=0.12), policy)
    assert all(result.values())
    assert sample_metrics()["cagr"] < policy["retained_secondary_goals"]["net_cagr_strictly_above"]


@pytest.mark.parametrize(
    "cagr,sharpe,expected",
    [
        (0.15, 1.0, False),
        (0.12, 1.2, False),
        (0.15, None, False),
        (0.15, 1.00001, True),
    ],
)
def test_goal_thresholds_are_strict(policy, cagr, sharpe, expected):
    result = goal_gates(sample_metrics(cagr, sharpe), sample_metrics(cagr=0.12), policy)
    assert all(result.values()) is expected


def test_mismatched_benchmark_window_is_not_an_outperformance_result(policy):
    benchmark = sample_metrics()
    benchmark["start"] = "2017-01-03"
    with pytest.raises(QuantError, match="same benchmark"):
        goal_gates(sample_metrics(), benchmark, policy)


def test_registration_binds_code_policy_and_comparison(policy, comparison):
    record = {
        "registered_at": utc_now(),
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "implementation_sha256": fingerprint(),
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "windows": comparison.windows(),
        "total_disclosed_after_round": 60,
        "order_authority": False,
        "history_previously_exposed": True,
    }
    verify_registration(policy, comparison, record)
    changed = deepcopy(policy)
    changed["parameters"]["variance_sessions"] = 63
    with pytest.raises(QuantError, match="changed"):
        verify_registration(changed, comparison, record)
    record["implementation_sha256"] = "0" * 64
    with pytest.raises(QuantError, match="changed"):
        verify_registration(policy, comparison, record)


def test_snapshot_requires_original_hashes_and_retrieval_time(tmp_path, comparison):
    (tmp_path / "raw").mkdir()
    (tmp_path / "SPY.csv").write_text("observed fixture")
    (tmp_path / "raw/SPY.json").write_text("{}")
    manifest = {
        "data_start": comparison.data_start,
        "data_end": comparison.as_of,
        "retrieved_at": "2026-10-06T05:16:59+00:00",
        "files": {name: file_digest(tmp_path / name) for name in ("SPY.csv", "raw/SPY.json")},
    }
    write_json(tmp_path / "manifest.json", manifest)
    descriptor = snapshot_descriptor(tmp_path, {"SPY"}, comparison)
    assert descriptor["retrieved_at"] == manifest["retrieved_at"]
    (tmp_path / "SPY.csv").write_text("revised fixture")
    with pytest.raises(QuantError, match="revised"):
        snapshot_descriptor(tmp_path, {"SPY"}, comparison)
