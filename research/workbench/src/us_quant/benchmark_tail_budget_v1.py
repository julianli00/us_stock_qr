from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from us_quant import joint_tail_construction_v1 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = prior.ROOT
POLICY = ROOT / "config/benchmark-tail-budget-v1.json"
CORE, SLEEVES, UNIVERSE, FACTOR_IDS = prior.CORE, prior.SLEEVES, prior.UNIVERSE, prior.FACTOR_IDS
LOOKBACK, ALPHA = prior.LOOKBACK, prior.ALPHA
SCORE_SLACK, SOLVER_TOL = 1e-10, prior.SOLVER_TOL
FEAS_TOL, ROUND_TOL = prior.FEAS_TOL, prior.ROUND_TOL
PRIOR_HEAD = "ccb22d6dfcfa4f6c7d24fec3d6f2052eb80c06df9c041ad7d581bd48227a8f84"
CANDIDATES = (
    {"id": "benchmark_tail_budget_growth_v1", "construction": "growth_first_benchmark_tail_budget"},
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_benchmark_tail_budget_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 187,
        "prior_new_lane_registered_candidates": 9,
        "prior_new_lane_completed_configurations": 9,
        "prior_journal_sequence": 106,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "lookback_sessions": LOOKBACK,
        "tail_probability": ALPHA,
        "sleeves": list(SLEEVES),
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "minimum_economic_factor_families": 3,
        "used_economic_factor_families": 6,
        "new_economic_factor_definitions": 0,
        "core_at_least_SOXX": True,
        "cash_reserve": 0.02,
        "weight_band": 0.025,
        "opportunity_optimum_slack": SCORE_SLACK,
        "solver_tolerance": SOLVER_TOL,
        "certificate_tolerance": prior.CERT_TOL,
        "weight_roundoff_tolerance": ROUND_TOL,
        "decision_feasibility_tolerance": FEAS_TOL,
        "risk_budget_multiplier": 1.0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("The fixed benchmark-tail-budget priority, inputs or safeguards changed.")


def risk_anchor(scenarios: np.ndarray, benchmark_returns: np.ndarray) -> dict:
    if (
        scenarios.shape != (LOOKBACK, 6)
        or benchmark_returns.shape != (LOOKBACK, 2)
        or not np.isfinite(scenarios).all()
        or not np.isfinite(benchmark_returns).all()
    ):
        raise QuantError("All actual sleeve and benchmark return scenarios are required.")
    losses = {
        "SPY": prior.tail_loss(-benchmark_returns[:, 0]),
        "QQQ": prior.tail_loss(-benchmark_returns[:, 1]),
        "BIL": prior.tail_loss(-scenarios[:, -1]),
    }
    return {
        "observed_tail_losses": losses,
        "risk_budget": max(losses["BIL"], min(losses["SPY"], losses["QQQ"])),
        "budget_is_drawdown_or_future_risk_guarantee": False,
    }


def linear_problem(scenarios: np.ndarray, scores: np.ndarray, budget: float) -> tuple:
    risk, a, b, eq, rhs, bounds = prior.linear_problem(scenarios, scores, 0.0)
    n = len(scenarios)
    selected = [*range(n), n + 1]
    return risk, np.vstack([a[selected], risk]), np.r_[b[selected], budget], eq, rhs, bounds


def allocate(
    scenarios: np.ndarray,
    scores: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[np.ndarray, dict]:
    scenarios = np.asarray(scenarios, dtype=float)
    scores = np.asarray(scores, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    anchor = risk_anchor(scenarios, benchmark_returns)
    if scores.shape != (6,) or not np.isfinite(scores).all() or scores[-1] != 0:
        raise QuantError("Finite cash-relative scores, including exact zero BIL, are required.")
    maximum = prior.maximum_opportunity(scores)
    standby = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    common = {**anchor, "analytical_unconstrained_opportunity_maximum": maximum}
    if maximum <= 0:
        return standby, {
            **common,
            "reason": "no_positive_asset_opportunity",
            "solver_called": False,
        }
    risk, a, b, eq, rhs, bounds = linear_problem(scenarios, scores, anchor["risk_budget"])
    reward = np.r_[-scores, np.zeros(LOOKBACK + 1)]
    options = {"primal_feasibility_tolerance": SOLVER_TOL, "dual_feasibility_tolerance": SOLVER_TOL}
    first = linprog(
        reward, A_ub=a, b_ub=b, A_eq=eq, b_eq=rhs, bounds=bounds, method="highs-ds", options=options
    )
    first_check = prior.certificate(first, reward, a, b, eq, rhs, bounds)
    opportunity = -first_check["objective"]
    primary = {
        **common,
        "solver_called": True,
        "primary_opportunity": opportunity,
        "primary_certificate": first_check,
        "primary_solution": prior.solution_record(first),
    }
    if opportunity < -FEAS_TOL:
        raise QuantError("The certified optimum is worse than its feasible actual BIL standby.")
    if opportunity <= 0:
        return standby, {**primary, "reason": "no_positive_risk_feasible_opportunity"}
    second_a = np.vstack([a, reward])
    second_b = np.r_[b, -opportunity + SCORE_SLACK]
    second = linprog(
        risk,
        A_ub=second_a,
        b_ub=second_b,
        A_eq=eq,
        b_eq=rhs,
        bounds=bounds,
        method="highs-ds",
        options=options,
    )
    second_check = prior.certificate(second, risk, second_a, second_b, eq, rhs, bounds)
    raw = second.x[:6]
    if (raw < -ROUND_TOL).any() or abs(float(raw.sum()) - 1) > FEAS_TOL:
        raise QuantError("The risk-budget optimum is not an unleveraged simplex portfolio.")
    weights = raw.copy()
    weights[np.abs(weights) <= ROUND_TOL] = 0
    weights /= weights.sum()
    exact_risk = prior.tail_loss(-scenarios @ weights)
    score = float(scores @ weights)
    if (
        (weights < 0).any()
        or weights[1] > weights[0] + FEAS_TOL
        or exact_risk > anchor["risk_budget"] + FEAS_TOL
        or score < opportunity - FEAS_TOL
        or abs(exact_risk - second_check["objective"]) > FEAS_TOL
    ):
        raise QuantError("Exact risk, opportunity or core constraints reject numerical cleanup.")
    return weights, {
        **primary,
        "reason": "benchmark_tail_budget_optimum",
        "achieved_observed_opportunity": score,
        "empirical_joint_tail_loss": exact_risk,
        "maximum_weight_roundoff_change": float(np.max(np.abs(weights - raw))),
        "secondary_certificate": second_check,
        "secondary_solution": prior.solution_record(second),
    }


def verify_plan(
    weights: np.ndarray,
    record: dict,
    scenarios: np.ndarray,
    scores: np.ndarray,
    benchmark_returns: np.ndarray,
) -> None:
    anchor = risk_anchor(scenarios, benchmark_returns)
    if (
        weights.shape != (6,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
        or weights[1] > weights[0] + FEAS_TOL
        or not np.isclose(record["risk_budget"], anchor["risk_budget"], rtol=0, atol=1e-12)
        or any(
            not np.isclose(record["observed_tail_losses"][key], value, rtol=0, atol=1e-12)
            for key, value in anchor["observed_tail_losses"].items()
        )
    ):
        raise QuantError("Independent price scenarios reject the retained risk budget or weights.")
    maximum = prior.maximum_opportunity(scores)
    standby = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if maximum <= 0:
        if (
            record["reason"] != "no_positive_asset_opportunity"
            or record["solver_called"] is not False
            or not np.array_equal(weights, standby)
        ):
            raise QuantError("Economic standby differs from the fixed nonpositive-score rule.")
        return
    if record["solver_called"] is not True:
        raise QuantError("Risk-feasible opportunity requires a real optimization certificate.")
    risk, a, b, eq, rhs, bounds = linear_problem(scenarios, scores, anchor["risk_budget"])
    reward = np.r_[-scores, np.zeros(LOOKBACK + 1)]
    first = prior.certificate(
        prior.recorded_solution(record["primary_solution"]),
        reward,
        a,
        b,
        eq,
        rhs,
        bounds,
    )
    opportunity = -first["objective"]
    if opportunity <= 0:
        if record["reason"] != "no_positive_risk_feasible_opportunity" or not np.array_equal(
            weights, standby
        ):
            raise QuantError("Risk-feasible standby has inconsistent recorded evidence.")
        return
    if record["reason"] != "benchmark_tail_budget_optimum":
        raise QuantError("A positive certified opportunity cannot be replaced by a cash default.")
    second = prior.recorded_solution(record["secondary_solution"])
    second_check = prior.certificate(
        second,
        risk,
        np.vstack([a, reward]),
        np.r_[b, -opportunity + SCORE_SLACK],
        eq,
        rhs,
        bounds,
    )
    exact_risk = prior.tail_loss(-scenarios @ weights)
    if (
        exact_risk > anchor["risk_budget"] + FEAS_TOL
        or float(scores @ weights) < opportunity - FEAS_TOL
        or np.max(np.abs(weights - second.x[:6])) > FEAS_TOL
        or abs(exact_risk - second_check["objective"]) > FEAS_TOL
        or abs(exact_risk - record["empirical_joint_tail_loss"]) > FEAS_TOL
    ):
        raise QuantError("Independent tail/score checks reject the retained optimized allocation.")


def observed_problem(close: pd.DataFrame, day: pd.Timestamp) -> tuple:
    scenarios, scores, _ = prior.observed_problem(close, day)
    past = prior.price_window(close, day)
    benchmarks = past.loc[:, ["SPY", "QQQ"]].pct_change(fill_method=None).iloc[1:].to_numpy()
    return scenarios, scores, benchmarks


def independent_problem(close: pd.DataFrame, day: pd.Timestamp) -> tuple:
    scenarios, scores, _ = prior.independent_problem(close, day)
    past = prior.price_window(close, day)
    benchmarks = np.array(
        [
            [float(past.iloc[i][name] / past.iloc[i - 1][name] - 1) for name in ("SPY", "QQQ")]
            for i in range(1, len(past))
        ]
    )
    return scenarios, scores, benchmarks


def build_targets(data: MarketData, policy: dict) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if tuple(data.close.columns) != UNIVERSE or not data.close.index.equals(
        sessions(data.close.index[0], data.close.index[-1])
    ):
        raise QuantError("Use the full admitted fourteen-asset calendar.")
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    records = []
    for i, day in enumerate(data.close.index):
        if i < LOOKBACK or not is_month_end(day):
            continue
        scenarios, scores, benchmarks = observed_problem(data.close, day)
        weights, record = allocate(scenarios, scores, benchmarks)
        targets.loc[day] = prior.scatter(weights)
        records.append(
            {
                "decision_session": str(day.date()),
                "first_observation": str(data.close.index[i - LOOKBACK].date()),
                "last_observation": str(day.date()),
                "return_observations": LOOKBACK,
                "sleeve_weights": dict(zip(SLEEVES, map(float, weights), strict=True)),
                "observed_scores": dict(zip(SLEEVES, map(float, scores), strict=True)),
                **record,
            }
        )
    return {CANDIDATES[0]["id"]: targets.ffill()}, records
