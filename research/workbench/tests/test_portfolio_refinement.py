from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import simulate
from us_quant.calendar import is_month_end, next_session
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.portfolio_refinement import (
    CONTROL,
    audit_run,
    constrained_target,
    predicted_volatility,
    promotion_assessment,
    rebalance_reason,
    register,
    simulate_refinement,
    validate_policy,
    verify_registration,
)
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/portfolio-refinement.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2019-01-02", "2020-06-30", ("SPY", "QQQ", "QLD", "GLD", "BIL"))


def baseline_signals(market):
    signals = pd.DataFrame(np.nan, index=market.close.index, columns=market.close.columns)
    for day in signals.index:
        if is_month_end(day):
            signals.loc[day] = 0.0
            signals.loc[day, ["QLD", "GLD"]] = [0.5, 0.48]
    return signals


def fixture_target():
    names = ["QLD", "GLD", "BIL"]
    target = pd.Series([0.6, 0.38, 0.0], index=names)
    covariance = pd.DataFrame(np.diag([0.16, 0.04, 0.0001]), index=names, columns=names)
    return target, covariance


@pytest.mark.parametrize(
    "change",
    [
        {"capital_usd": 100000},
        {"prior_disclosed_configurations": 0},
        {"new_configurations": 5},
        {"cash_reserve": 0},
        {"commission_per_order": 0},
        {"covariance_sessions": 126},
    ],
)
def test_declared_constraints_and_history_count_cannot_change(policy, change):
    policy.update(change)
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_no_trading_or_persistent_service_authority(policy):
    validate_policy(policy)
    for key in (
        "order_authority",
        "automatic_retuning",
        "old_forward_ledger_modified",
        "persistent_automation_started",
    ):
        changed = deepcopy(policy)
        changed["methodology"][key] = True
        with pytest.raises(QuantError, match="authority"):
            validate_policy(changed)


def test_research_candidate_names_cannot_escape_output_directory(policy):
    policy["candidates"][0]["id"] = "../outside"
    with pytest.raises(QuantError, match="candidate IDs"):
        validate_policy(policy)


def test_constraints_project_the_whole_portfolio_into_bil(policy):
    target, covariance = fixture_target()
    candidate = policy["candidates"][2]
    projected = constrained_target(target, covariance, candidate, 0.98)
    assert projected["QLD"] <= 0.30
    assert projected["GLD"] <= target["GLD"]
    assert projected.sum() == pytest.approx(0.98)
    assert predicted_volatility(projected, covariance) == pytest.approx(0.12, abs=1e-10)
    assert projected["BIL"] > 0.3
    assert (projected >= 0).all()
    unconstrained = constrained_target(target, covariance, CONTROL, 0.98)
    pd.testing.assert_series_equal(unconstrained, target)


def test_covariance_includes_defensive_asset_and_never_levers_up(policy):
    target, covariance = fixture_target()
    low_risk = target.copy()
    low_risk.loc[["QLD", "GLD", "BIL"]] = [0.01, 0.01, 0.96]
    projected = constrained_target(low_risk, covariance, policy["candidates"][1], 0.98)
    pd.testing.assert_series_equal(projected, low_risk)
    covariance.loc["BIL", "BIL"] = 1.0
    with pytest.raises(QuantError, match="defensive ETF"):
        constrained_target(target, covariance, policy["candidates"][1], 0.98)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -0.1])
def test_invalid_caps_do_not_produce_fallback_targets(value):
    target, covariance = fixture_target()
    candidate = {**CONTROL, "qld_cap": value}
    with pytest.raises(QuantError):
        constrained_target(target, covariance, candidate, 0.98)


def test_observed_weight_band_is_not_a_last_target_band(policy):
    _, covariance = fixture_target()
    desired = pd.Series({"QLD": 0.30, "GLD": 0.68, "BIL": 0.0})
    current = pd.Series({"QLD": 0.34, "GLD": 0.64, "BIL": 0.0})
    candidate = policy["candidates"][3]
    assert rebalance_reason(current, desired, covariance, candidate) == "observed_weight_gap"
    assert rebalance_reason(desired, desired, covariance, candidate) == "hold"
    current.loc[["QLD", "GLD"]] = [0.325, 0.655]
    assert rebalance_reason(current, desired, covariance, candidate) == "observed_weight_gap"
    current.loc[["QLD", "GLD"]] = [0.324, 0.656]
    assert rebalance_reason(current, desired, covariance, candidate) == "hold"


def test_risk_constraints_override_small_trade_band(policy):
    target, covariance = fixture_target()
    candidate = policy["candidates"][-1]
    desired = constrained_target(target, covariance, candidate, 0.98)
    current = desired.copy()
    current["QLD"] += 0.005
    current["BIL"] -= 0.005
    assert abs(current - desired).max() < 0.025
    assert rebalance_reason(current, desired, covariance, candidate) == "risk_limit_override"


@pytest.mark.parametrize("delay,cost", [(1, 5.0), (2, 20.0), (2, 50.0)])
def test_unchanged_control_reproduces_the_original_engine(market, policy, delay, cost):
    signals = baseline_signals(market)
    run = simulate_refinement(
        market,
        signals,
        CONTROL,
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=cost,
        delay=delay,
    )
    prior = simulate(
        market,
        seed_window(signals, "2020-01-02"),
        "2020-01-02",
        "2020-06-30",
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    pd.testing.assert_frame_equal(run.result.frame, prior.frame)
    pd.testing.assert_frame_equal(run.result.weights, prior.weights)
    if delay == 2:
        assert run.result.frame.iloc[0]["orders"] == 0
        assert run.result.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize("candidate_index", [2, 3, 5])
def test_issued_decisions_replay_independently(market, policy, candidate_index):
    signals, candidate = baseline_signals(market), policy["candidates"][candidate_index]
    run = simulate_refinement(
        market,
        signals,
        candidate,
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=20,
        delay=2,
    )
    independent, audit = audit_run(
        market,
        signals,
        run,
        candidate,
        policy,
        "2020-01-02",
        "2020-06-30",
        20,
        2,
    )
    assert audit["max_equity_difference_usd"] < 1e-6
    assert audit["position_feedback_replay_passed"]
    assert (run.result.frame["cash"] >= 0).all()
    np.testing.assert_allclose(independent["equity"], run.result.frame["equity"], atol=1e-6)
    for decision in run.decisions:
        day = pd.Timestamp(decision["signal_session"])
        assert pd.Timestamp(decision["execution_session"]) == next_session(next_session(day))


def test_constant_prices_do_not_generate_band_fee_churn(market, policy):
    flat = market.close * 0 + 100
    data = replace(market, open=flat.copy(), close=flat.copy(), raw_close=flat.copy())
    run = simulate_refinement(
        data,
        baseline_signals(data),
        policy["candidates"][3],
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=5,
        delay=1,
    )
    assert run.result.frame["orders"].sum() == 2
    assert sum(row["reason"] == "hold" for row in run.decisions) == 6


def test_monthly_target_limit_does_not_erase_between_decision_drift(market, policy):
    prices = market.close * 0 + 100
    prices.loc["2020-01-08":, "QLD"] = 125
    data = replace(market, open=prices.copy(), close=prices.copy(), raw_close=prices.copy())
    run = simulate_refinement(
        data,
        baseline_signals(data),
        policy["candidates"][0],
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=5,
        delay=1,
    )
    assert (run.issued["QLD"].dropna() <= 0.30).all()
    assert run.result.weights.loc["2020-01-08", "QLD"] > 0.30


def test_future_prices_and_opens_cannot_affect_earlier_risk_or_band_decisions(market, policy):
    cutoff = pd.Timestamp("2020-03-31")
    signals = baseline_signals(market)
    candidate = policy["candidates"][-1]
    original = simulate_refinement(
        market,
        signals,
        candidate,
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=5,
        delay=1,
    )
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    factor = np.linspace(1.05, 1.3, later.sum())
    close.loc[later, "QLD"] *= factor
    opening.loc[later, "QLD"] *= factor
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    other = simulate_refinement(
        changed,
        signals,
        candidate,
        policy,
        "2020-01-02",
        "2020-06-30",
        cost_bps=5,
        delay=1,
    )
    pd.testing.assert_frame_equal(original.issued.loc[:cutoff], other.issued.loc[:cutoff])
    pd.testing.assert_frame_equal(
        original.result.frame.loc[:cutoff], other.result.frame.loc[:cutoff]
    )
    assert [row for row in original.decisions if row["signal_session"] <= str(cutoff.date())] == [
        row for row in other.decisions if row["signal_session"] <= str(cutoff.date())
    ]
    prefix = replace(
        market,
        open=market.open.loc[:cutoff],
        close=market.close.loc[:cutoff],
        raw_close=market.raw_close.loc[:cutoff],
        volume=market.volume.loc[:cutoff],
        risk_free=market.risk_free.loc[:cutoff],
    )
    shortened = simulate_refinement(
        prefix,
        signals.loc[:cutoff],
        candidate,
        policy,
        "2020-01-02",
        str(cutoff.date()),
        cost_bps=5,
        delay=1,
    )
    pd.testing.assert_frame_equal(original.result.frame.loc[:cutoff], shortened.result.frame)


def fixture_results():
    return {
        window: {
            scenario: {
                "primary_pass": True,
                "metrics": {"max_drawdown": 0.20, "annualized_one_way_turnover": 1.0},
            }
            for scenario in ("base", "stress", "higher_cost")
        }
        for window in ("10y", "5y")
    }


def test_promotion_requires_real_improvement_not_a_better_single_metric(policy):
    baseline = fixture_results()
    candidate = deepcopy(baseline)
    assert not promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]
    for window in ("10y", "5y"):
        candidate[window]["higher_cost"]["metrics"]["annualized_one_way_turnover"] = 0.90
    assert promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]
    candidate["5y"]["stress"]["primary_pass"] = False
    assert not promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]
    candidate["5y"]["stress"]["primary_pass"] = True
    candidate["5y"]["base"]["metrics"]["max_drawdown"] = 0.211
    assert not promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]


def test_drawdown_goal_is_checked_exactly_not_rounded(policy):
    baseline = fixture_results()
    candidate = deepcopy(baseline)
    for window in ("10y", "5y"):
        for scenario in ("base", "stress"):
            candidate[window][scenario]["metrics"]["max_drawdown"] = 0.15
    assert promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]
    candidate["5y"]["stress"]["metrics"]["max_drawdown"] = 0.150001
    assert not promotion_assessment(candidate, baseline, policy)["research_improvement_qualified"]


def test_registration_preserves_trials_and_rejects_changed_code(policy, tmp_path, monkeypatch):
    import us_quant.portfolio_refinement as module

    base = read_json(Path(__file__).parents[1] / "config/implementation-research.json")
    monkeypatch.setattr(module, "baseline_context", lambda *args: (base, {}, None))
    output = tmp_path / "registration.json"
    record = register(policy, tmp_path, output)
    assert record["total_disclosed_configurations"] == 78
    assert len(record["candidate_ids"]) == 6
    assert record["order_authority"] is False
    verify_registration(policy, record)
    with pytest.raises(QuantError, match="overwrite"):
        register(policy, tmp_path, output)
    monkeypatch.setattr(module, "fingerprint", lambda: "changed")
    with pytest.raises(QuantError, match="changed"):
        verify_registration(policy, record)
