from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.ssd_factor_catalog_v1 import ADDED, EXISTING, ROOT

POLICY = ROOT / "config/joint-tail-construction-v1.json"
CORE = ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
SLEEVES = ("factor_core", "SOXX", "GLD", "IEF", "TLT", "BIL")
UNIVERSE = (*EXISTING, *ADDED)
FACTOR_IDS = (
    "price_momentum",
    "value_exposure",
    "quality_exposure",
    "low_volatility_exposure",
    "size_exposure_ijr_v1",
    "net_buyback_exposure_pkw_v1",
)
CANDIDATES = (
    {
        "id": "benchmark_opportunity_joint_tail_v1",
        "construction": "joint_ES_with_benchmark_opportunity",
    },
)
PRIOR_HEAD = "c2b8002e31187e374c8fd11c095f1df7c27d80647ca3229b98f8e35818b863e9"
LOOKBACK = 252
ALPHA = 0.05
RISK_SLACK = 1e-10
SOLVER_TOL = 1e-9
CERT_TOL = 1e-7
ROUND_TOL = 1e-10
FEAS_TOL = 1e-8


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_joint_tail_construction_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 186,
        "prior_new_lane_registered_candidates": 8,
        "prior_new_lane_completed_configurations": 8,
        "prior_journal_sequence": 94,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "catalog_publication_commit": "83514dd1186ff4fe612c5078a7787509919f80cb",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
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
        "risk_optimum_slack": RISK_SLACK,
        "solver_tolerance": SOLVER_TOL,
        "certificate_tolerance": CERT_TOL,
        "weight_roundoff_tolerance": ROUND_TOL,
        "decision_feasibility_tolerance": FEAS_TOL,
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
        raise QuantError("The fixed joint-tail construction, safeguards or source scope changed.")


def tail_loss(losses: np.ndarray, alpha: float = ALPHA) -> float:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < alpha <= 1:
        raise QuantError("Expected shortfall needs finite loss observations and a valid tail mass.")
    mass = alpha * len(values)
    ordered = np.sort(values)[::-1]
    count = int(np.floor(mass))
    fraction = mass - count
    total = float(ordered[:count].sum())
    if fraction:
        total += fraction * float(ordered[count])
    return total / mass


def maximum_opportunity(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise QuantError("All six observed sleeve opportunity scores are required.")
    return float(max(values[0], (values[0] + values[1]) / 2, *values[2:]))


def certificate(
    result,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eq: np.ndarray,
    rhs: np.ndarray,
    bounds: list[tuple],
) -> dict:
    if not result.success or result.status != 0 or not np.isfinite(result.x).all():
        raise QuantError(f"Joint-tail solver failed explicitly: {result.message}")
    x = result.x
    yi, ye = result.ineqlin.marginals, result.eqlin.marginals
    lo, hi = result.lower.marginals, result.upper.marginals
    lower = np.array([-np.inf if item[0] is None else item[0] for item in bounds])
    upper = np.array([np.inf if item[1] is None else item[1] for item in bounds])
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    if (
        x.shape != c.shape
        or yi.shape != b.shape
        or ye.shape != rhs.shape
        or lo.shape != c.shape
        or hi.shape != c.shape
        or not all(np.isfinite(value).all() for value in (yi, ye, lo, hi))
    ):
        raise QuantError("The primal/dual certificate has malformed dimensions or values.")
    dual = float(
        b @ yi
        + rhs @ ye
        + lower[finite_lower] @ lo[finite_lower]
        + upper[finite_upper] @ hi[finite_upper]
    )
    stationary = c - a.T @ yi - eq.T @ ye - lo - hi
    gaps = {
        "inequality_violation": float(max(0, np.max(a @ x - b))),
        "equality_violation": float(np.max(np.abs(eq @ x - rhs))),
        "lower_bound_violation": float(max(0, np.max(lower[finite_lower] - x[finite_lower]))),
        "upper_bound_violation": float(max(0, np.max(x[finite_upper] - upper[finite_upper]))),
        "dual_sign_violation": float(max(0, np.max(yi), np.max(-lo), np.max(hi))),
        "stationarity_error": float(np.max(np.abs(stationary))),
        "inequality_complementarity": float(np.max(np.abs(yi * (b - a @ x)))),
        "lower_complementarity": float(
            np.max(np.abs(lo[finite_lower] * (x[finite_lower] - lower[finite_lower])))
        ),
        "upper_complementarity": float(
            np.max(np.abs(hi[finite_upper] * (upper[finite_upper] - x[finite_upper])))
        ),
        "primal_dual_gap": abs(float(c @ x) - dual),
        "unbounded_multiplier_violation": float(
            max(np.max(np.abs(lo[~finite_lower])), np.max(np.abs(hi[~finite_upper])))
        ),
    }
    if not all(np.isfinite(value) and value <= CERT_TOL for value in gaps.values()):
        raise QuantError(f"Independent joint-tail optimality certificate failed: {gaps}")
    return {"objective": float(c @ x), "dual_objective": dual, **gaps}


def solution_record(result) -> dict:
    return {
        "primal": result.x.tolist(),
        "inequality_dual": result.ineqlin.marginals.tolist(),
        "equality_dual": result.eqlin.marginals.tolist(),
        "lower_dual": result.lower.marginals.tolist(),
        "upper_dual": result.upper.marginals.tolist(),
    }


def recorded_solution(value: dict):
    return SimpleNamespace(
        success=True,
        status=0,
        message="retained primal/dual solution",
        x=np.asarray(value["primal"], dtype=float),
        ineqlin=SimpleNamespace(marginals=np.asarray(value["inequality_dual"], dtype=float)),
        eqlin=SimpleNamespace(marginals=np.asarray(value["equality_dual"], dtype=float)),
        lower=SimpleNamespace(marginals=np.asarray(value["lower_dual"], dtype=float)),
        upper=SimpleNamespace(marginals=np.asarray(value["upper_dual"], dtype=float)),
    )


def linear_problem(returns: np.ndarray, scores: np.ndarray, hurdle: float) -> tuple:
    n = len(returns)
    width = 7 + n
    risk = np.r_[np.zeros(6), 1.0, np.full(n, 1 / (ALPHA * n))]
    a = np.zeros((n + 2, width))
    a[:n, :6] = -returns
    a[:n, 6] = -1
    a[:n, 7:] = -np.eye(n)
    a[n, :6] = -scores
    a[n + 1, 0:2] = [-1, 1]
    b = np.r_[np.zeros(n), -hurdle, 0.0]
    eq = np.zeros((1, width))
    eq[0, :6] = 1
    return risk, a, b, eq, np.ones(1), [(0, 1)] * 6 + [(None, None)] + [(0, None)] * n


def allocate(scenarios: np.ndarray, scores: np.ndarray, hurdle: float) -> tuple[np.ndarray, dict]:
    returns = np.asarray(scenarios, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (
        returns.shape != (LOOKBACK, len(SLEEVES))
        or not np.isfinite(returns).all()
        or scores.shape != (6,)
        or not np.isfinite(scores).all()
        or not np.isfinite(hurdle)
        or scores[-1] != 0
    ):
        raise QuantError("The complete fixed joint scenarios and opportunity inputs are required.")
    maximum = maximum_opportunity(scores)
    standby = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if hurdle <= 0:
        return standby, {
            "reason": "nonpositive_benchmark_opportunity",
            "benchmark_hurdle": float(hurdle),
            "maximum_feasible_opportunity": maximum,
            "solver_called": False,
        }
    if hurdle > maximum:
        return standby, {
            "reason": "observed_benchmark_opportunity_unavailable",
            "benchmark_hurdle": float(hurdle),
            "maximum_feasible_opportunity": maximum,
            "solver_called": False,
        }
    risk, a, b, eq, rhs, bounds = linear_problem(returns, scores, hurdle)
    options = {"primal_feasibility_tolerance": SOLVER_TOL, "dual_feasibility_tolerance": SOLVER_TOL}
    first = linprog(
        risk, A_ub=a, b_ub=b, A_eq=eq, b_eq=rhs, bounds=bounds, method="highs-ds", options=options
    )
    primary = certificate(first, risk, a, b, eq, rhs, bounds)
    second_a = np.vstack([a, risk])
    second_b = np.r_[b, first.fun + RISK_SLACK]
    reward = np.r_[-scores, np.zeros(1 + len(returns))]
    second = linprog(
        reward,
        A_ub=second_a,
        b_ub=second_b,
        A_eq=eq,
        b_eq=rhs,
        bounds=bounds,
        method="highs-ds",
        options=options,
    )
    secondary = certificate(second, reward, second_a, second_b, eq, rhs, bounds)
    raw = second.x[:6]
    if (raw < -ROUND_TOL).any() or abs(float(raw.sum()) - 1) > FEAS_TOL:
        raise QuantError("The optimum is not an unleveraged simplex portfolio.")
    weights = raw.copy()
    weights[np.abs(weights) <= ROUND_TOL] = 0
    weights /= weights.sum()
    exact_risk = tail_loss(-returns @ weights)
    score = float(scores @ weights)
    if (
        (weights < 0).any()
        or weights[1] > weights[0] + FEAS_TOL
        or score < hurdle - FEAS_TOL
        or exact_risk > first.fun + FEAS_TOL
        or abs(tail_loss(-returns @ first.x[:6]) - first.fun) > FEAS_TOL
    ):
        raise QuantError("Exact tail/score validation rejects the numerical optimum.")
    return weights, {
        "reason": "joint_tail_optimum",
        "benchmark_hurdle": float(hurdle),
        "maximum_feasible_opportunity": maximum,
        "achieved_observed_opportunity": score,
        "empirical_joint_tail_loss": exact_risk,
        "solver_called": True,
        "maximum_weight_roundoff_change": float(np.max(np.abs(weights - raw))),
        "primary_certificate": primary,
        "secondary_certificate": secondary,
        "primary_solution": solution_record(first),
        "secondary_solution": solution_record(second),
    }


def verify_plan(
    weights: np.ndarray,
    record: dict,
    scenarios: np.ndarray,
    scores: np.ndarray,
    hurdle: float,
) -> None:
    maximum = maximum_opportunity(scores)
    expected_reason = (
        "nonpositive_benchmark_opportunity"
        if hurdle <= 0
        else "observed_benchmark_opportunity_unavailable"
        if hurdle > maximum
        else "joint_tail_optimum"
    )
    if record["reason"] != expected_reason or not np.allclose(
        [record["benchmark_hurdle"], record["maximum_feasible_opportunity"]],
        [hurdle, maximum],
        rtol=0,
        atol=1e-12,
    ):
        raise QuantError("A retained decision does not match the independently observed problem.")
    if expected_reason != "joint_tail_optimum":
        if record["solver_called"] is not False or not np.array_equal(
            weights, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ):
            raise QuantError("An explicit economic standby is not a solver fallback.")
        return
    if record["solver_called"] is not True:
        raise QuantError("An optimized plan needs actual primal/dual evidence.")
    risk, a, b, eq, rhs, bounds = linear_problem(scenarios, scores, hurdle)
    first = recorded_solution(record["primary_solution"])
    checked = certificate(first, risk, a, b, eq, rhs, bounds)
    second = recorded_solution(record["secondary_solution"])
    certificate(
        second,
        np.r_[-scores, np.zeros(1 + len(scenarios))],
        np.vstack([a, risk]),
        np.r_[b, checked["objective"] + RISK_SLACK],
        eq,
        rhs,
        bounds,
    )
    exact = tail_loss(-scenarios @ weights)
    if (
        weights.shape != (6,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
        or weights[1] > weights[0] + FEAS_TOL
        or float(scores @ weights) < hurdle - FEAS_TOL
        or exact > checked["objective"] + FEAS_TOL
        or np.max(np.abs(weights - second.x[:6])) > FEAS_TOL
        or abs(exact - record["empirical_joint_tail_loss"]) > FEAS_TOL
    ):
        raise QuantError("Independent tail/score/simplex checks reject the retained plan.")


def price_window(close: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if (
        tuple(close.columns) != UNIVERSE
        or not close.index.is_unique
        or not close.index.is_monotonic_increasing
        or day not in close.index
    ):
        raise QuantError("Require a complete ordered price schema and actual decision date.")
    location = close.index.get_loc(day)
    if location < LOOKBACK:
        raise QuantError("A joint-tail decision requires 253 actual price observations.")
    past = close.iloc[location - LOOKBACK : location + 1]
    if not np.isfinite(past).all().all() or (past <= 0).any().any():
        raise QuantError("Missing or invalid historical prices cannot become an opportunity score.")
    return past


def observed_problem(
    close: pd.DataFrame, day: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, float]:
    past = price_window(close, day)
    daily = past.pct_change(fill_method=None).iloc[1:]
    fund_score = past.iloc[-1] / past.iloc[0] - 1
    excess = fund_score - fund_score["BIL"]
    scenario = np.column_stack(
        [
            daily.loc[:, list(CORE)].mean(axis=1),
            *(daily[name] for name in SLEEVES[1:]),
        ]
    )
    scores = np.array([excess.loc[list(CORE)].mean(), *(excess[name] for name in SLEEVES[1:])])
    return scenario, scores, float(max(excess["SPY"], excess["QQQ"]))


def independent_problem(
    close: pd.DataFrame, day: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, float]:
    past = price_window(close, day)
    rows = []
    for position in range(1, len(past)):
        changes = {
            name: float(past.iloc[position][name] / past.iloc[position - 1][name] - 1)
            for name in UNIVERSE
        }
        rows.append(
            [sum(changes[name] for name in CORE) / 6, *(changes[name] for name in SLEEVES[1:])]
        )
    changes = {name: float(past.iloc[-1][name] / past.iloc[0][name] - 1) for name in UNIVERSE}
    scores = np.array(
        [
            sum(changes[name] - changes["BIL"] for name in CORE) / 6,
            *(changes[name] - changes["BIL"] for name in SLEEVES[1:]),
        ]
    )
    hurdle = max(changes["SPY"] - changes["BIL"], changes["QQQ"] - changes["BIL"])
    return np.asarray(rows), scores, hurdle


def scatter(weights: np.ndarray) -> pd.Series:
    result = pd.Series(0.0, index=UNIVERSE)
    for name in CORE:
        result[name] = 0.98 * weights[0] / 6
    for name, value in zip(SLEEVES[1:], weights[1:], strict=True):
        result[name] = 0.98 * value
    if (result < 0).any() or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12):
        raise QuantError("A sleeve optimum did not produce a funded actual-ETF target.")
    return result


def build_targets(data: MarketData, policy: dict) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    close = data.close
    if (
        tuple(close.columns) != UNIVERSE
        or not close.index.is_unique
        or not close.index.equals(sessions(close.index[0], close.index[-1]))
    ):
        raise QuantError("Use the complete admitted fourteen-asset calendar.")
    target = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for position, day in enumerate(close.index):
        if position < LOOKBACK or not is_month_end(day):
            continue
        matrix, score, hurdle = observed_problem(close, day)
        weights, record = allocate(matrix, score, hurdle)
        target.loc[day] = scatter(weights)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_observation": str(close.index[position - LOOKBACK].date()),
                "last_observation": str(day.date()),
                "return_observations": LOOKBACK,
                "sleeve_weights": {
                    name: float(value) for name, value in zip(SLEEVES, weights, strict=True)
                },
                "observed_scores": {
                    name: float(value) for name, value in zip(SLEEVES, score, strict=True)
                },
                **record,
            }
        )
    target = target.ffill()
    return {CANDIDATES[0]["id"]: target}, audits
