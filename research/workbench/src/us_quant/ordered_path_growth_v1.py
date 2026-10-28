from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from us_quant import daily_variance_disagreement_v1 as existing
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/ordered-path-growth-v1.json"
CORE, UNIVERSE, FACTOR_IDS = existing.CORE, existing.UNIVERSE, existing.FACTOR_IDS
SLEEVES = ("stock_basket", "GLD", "BIL")
BASKET = {"SOXX": 0.5, **dict.fromkeys(CORE, 0.125)}
LOOKBACK, CASH, BUDGET, STOCK_MIN, DRAWDOWN = 252, 0.02, 0.98, 0.147, 0.15
SOLVER_TOL, CERT_TOL, ROUND_TOL, FEAS_TOL, SLACK = 1e-9, 1e-7, 1e-10, 1e-8, 1e-10
PRIOR_HEAD = "3f0956c227caff3189d042cb002a72e19f7faf32e8234eacaeeb1fa8ff42a4ec"
CANDIDATES = (
    {
        "id": "ordered_path_growth_budget_v1",
        "construction": "terminal_wealth_with_all_ordered_peak_trough_constraints",
    },
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_ordered_path_growth_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 195,
        "prior_journal_sequence": 202,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "lookback_return_sessions": LOOKBACK,
        "training_price_observations": LOOKBACK + 1,
        "training_maximum_drawdown": DRAWDOWN,
        "cash_reserve": CASH,
        "invested_target_budget": BUDGET,
        "stock_target_minimum": STOCK_MIN,
        "stock_basket": BASKET,
        "sleeves": list(SLEEVES),
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "actual_weight_trade_band": 0.025,
        "solver": "highs-ds",
        "solver_tolerance": SOLVER_TOL,
        "certificate_tolerance": CERT_TOL,
        "weight_roundoff_tolerance": ROUND_TOL,
        "decision_feasibility_tolerance": FEAS_TOL,
        "primary_optimum_slack": SLACK,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        key not in policy or policy[key] != value or type(policy[key]) is not type(value)
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
        raise QuantError(
            "The fixed ordered-path construction or its qualification boundary changed."
        )


def problem(paths):
    p = np.asarray(paths, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or len(p) < 2
        or not np.isfinite(p).all()
        or (p <= 0).any()
        or not np.array_equal(p[0], np.ones(3))
    ):
        raise QuantError("Require complete positive normalized paths including initial wealth.")
    peaks, troughs = np.triu_indices(len(p), k=1)
    a = (1 - DRAWDOWN) * p[peaks] - p[troughs]
    b = np.full(len(peaks), DRAWDOWN * CASH)
    return (
        -p[-1],
        a,
        b,
        np.ones((1, 3)),
        np.array([BUDGET]),
        [(STOCK_MIN, BUDGET), (0.0, BUDGET), (0.0, BUDGET)],
    )


def certificate(x, yi, ye, lo, hi, c, a, b, eq, rhs, bounds):
    arrays = [np.asarray(value, dtype=float) for value in (x, yi, ye, lo, hi)]
    x, yi, ye, lo, hi = arrays
    if [v.shape for v in arrays] != [(3,), b.shape, rhs.shape, (3,), (3,)] or not all(
        np.isfinite(v).all() for v in arrays
    ):
        raise QuantError("Malformed bounded-LP primal/dual certificate.")
    lower, upper = np.asarray(bounds, dtype=float).T
    dual = float(b @ yi + rhs @ ye + lower @ lo + upper @ hi)
    checks = {
        "inequality_violation": float(max(0, np.max(a @ x - b))),
        "equality_violation": float(np.max(abs(eq @ x - rhs))),
        "lower_violation": float(max(0, np.max(lower - x))),
        "upper_violation": float(max(0, np.max(x - upper))),
        "dual_sign_violation": float(max(0, np.max(yi), np.max(-lo), np.max(hi))),
        "stationarity_error": float(np.max(abs(c - a.T @ yi - eq.T @ ye - lo - hi))),
        "inequality_complementarity": float(np.max(abs(yi * (b - a @ x)))),
        "lower_complementarity": float(np.max(abs(lo * (x - lower)))),
        "upper_complementarity": float(np.max(abs(hi * (upper - x)))),
        "primal_dual_gap": abs(float(c @ x) - dual),
    }
    if not all(np.isfinite(v) and v <= CERT_TOL for v in checks.values()):
        raise QuantError(f"Ordered-path optimality certificate failed: {checks}")
    return {"objective": float(c @ x), "dual_objective": dual, **checks}


def record_solution(result, c, a, b, eq, rhs, bounds):
    if not result.success or result.status != 0:
        raise QuantError(f"Ordered-path LP failed without fallback: {result.message}")
    checked = certificate(
        result.x,
        result.ineqlin.marginals,
        result.eqlin.marginals,
        result.lower.marginals,
        result.upper.marginals,
        c,
        a,
        b,
        eq,
        rhs,
        bounds,
    )
    dual = result.ineqlin.marginals
    record = {
        "primal": result.x.tolist(),
        "inequality_count": len(dual),
        "nonzero_inequality_dual": [
            {"row": int(i), "value": float(dual[i])} for i in np.flatnonzero(dual)
        ],
        "equality_dual": result.eqlin.marginals.tolist(),
        "lower_dual": result.lower.marginals.tolist(),
        "upper_dual": result.upper.marginals.tolist(),
    }
    return record, checked


def verify_solution(record, c, a, b, eq, rhs, bounds):
    if record["inequality_count"] != len(b):
        raise QuantError("Sparse dual certificate has a different constraint count.")
    yi = np.zeros(len(b))
    indices = [row["row"] for row in record["nonzero_inequality_dual"]]
    if indices != sorted(set(indices)) or any(
        type(i) is not int or not 0 <= i < len(b) for i in indices
    ):
        raise QuantError("Sparse dual indices are malformed or duplicated.")
    for row in record["nonzero_inequality_dual"]:
        if not np.isfinite(row["value"]) or row["value"] == 0:
            raise QuantError("Sparse dual values must retain every actual nonzero multiplier.")
        yi[row["row"]] = row["value"]
    return certificate(
        record["primal"],
        yi,
        record["equality_dual"],
        record["lower_dual"],
        record["upper_dual"],
        c,
        a,
        b,
        eq,
        rhs,
        bounds,
    )


def drawdown(paths, weights):
    wealth = CASH + np.asarray(paths) @ np.asarray(weights)
    if not np.isfinite(wealth).all() or (wealth <= 0).any():
        raise QuantError("Training wealth must remain positive without synthetic cash interest.")
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1 - wealth / peak)), float(wealth[-1])


def allocate(paths):
    c, a, b, eq, rhs, bounds = problem(paths)
    options = {"primal_feasibility_tolerance": SOLVER_TOL, "dual_feasibility_tolerance": SOLVER_TOL}
    first = linprog(
        c, A_ub=a, b_ub=b, A_eq=eq, b_eq=rhs, bounds=bounds, method="highs-ds", options=options
    )
    first_record, first_check = record_solution(first, c, a, b, eq, rhs, bounds)
    second_a, second_b = np.vstack([a, c]), np.r_[b, first_check["objective"] + SLACK]
    cash_preference = np.array([0.0, 0.0, -1.0])
    second = linprog(
        cash_preference,
        A_ub=second_a,
        b_ub=second_b,
        A_eq=eq,
        b_eq=rhs,
        bounds=bounds,
        method="highs-ds",
        options=options,
    )
    second_record, second_check = record_solution(
        second, cash_preference, second_a, second_b, eq, rhs, bounds
    )
    raw = second.x
    if (raw < -ROUND_TOL).any() or abs(float(raw.sum()) - BUDGET) > FEAS_TOL:
        raise QuantError("The optimum does not retain the nonnegative funded budget.")
    weights = raw.copy()
    weights[np.abs(weights) <= ROUND_TOL] = 0
    weights *= BUDGET / weights.sum()
    risk, terminal = drawdown(paths, weights)
    if (
        (weights < 0).any()
        or weights[0] < STOCK_MIN - FEAS_TOL
        or risk > DRAWDOWN + FEAS_TOL
        or float(c @ weights) > first_check["objective"] + FEAS_TOL
        or -weights[-1] > second_check["objective"] + FEAS_TOL
    ):
        raise QuantError("Numerical cleanup fails risk, growth, family or funding checks.")
    return weights, {
        "primary_solution": first_record,
        "primary_certificate": first_check,
        "secondary_solution": second_record,
        "secondary_certificate": second_check,
        "ordered_peak_trough_constraints": len(b),
        "training_gross_maximum_drawdown": risk,
        "training_gross_terminal_wealth": terminal,
        "maximum_roundoff_weight_change": float(np.max(abs(weights - raw))),
        "training_statistics_are_formal_strategy_results": False,
        "future_or_net_drawdown_guaranteed": False,
    }


def verify_plan(weights, record, paths):
    c, a, b, eq, rhs, bounds = problem(paths)
    first = verify_solution(record["primary_solution"], c, a, b, eq, rhs, bounds)
    second_a, second_b = np.vstack([a, c]), np.r_[b, first["objective"] + SLACK]
    second = verify_solution(
        record["secondary_solution"],
        np.array([0.0, 0.0, -1.0]),
        second_a,
        second_b,
        eq,
        rhs,
        bounds,
    )
    weights = np.asarray(weights, dtype=float)
    risk, terminal = drawdown(paths, weights)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), BUDGET, rtol=0, atol=1e-12)
        or weights[0] < STOCK_MIN - FEAS_TOL
        or risk > DRAWDOWN + FEAS_TOL
        or float(c @ weights) > first["objective"] + FEAS_TOL
        or -weights[-1] > second["objective"] + FEAS_TOL
        or abs(risk - record["training_gross_maximum_drawdown"]) > FEAS_TOL
        or abs(terminal - record["training_gross_terminal_wealth"]) > FEAS_TOL
        or record["ordered_peak_trough_constraints"] != len(b)
    ):
        raise QuantError("Independent ordered-path allocation verification failed.")


def price_window(close, day):
    position = close.index.get_loc(day)
    if not isinstance(position, int) or position < LOOKBACK:
        raise QuantError("A path decision needs253 actual prices,not a shortened window.")
    past = close.iloc[position - LOOKBACK : position + 1]
    if not np.isfinite(past).all().all() or (past <= 0).any().any():
        raise QuantError("Invalid training prices cannot become a safe allocation.")
    return past


def observed_paths(close, day):
    past = price_window(close, day)
    relative = past / past.iloc[0]
    stock = sum(weight * relative[name] for name, weight in BASKET.items())
    return np.column_stack([stock, relative["GLD"], relative["BIL"]])


def independent_paths(close, day):
    past = price_window(close, day)
    units = {name: weight / float(past.iloc[0][name]) for name, weight in BASKET.items()}
    return np.array(
        [
            [
                1.0 + sum(units[name] * float(row[name] - past.iloc[0][name]) for name in BASKET),
                1.0 + float((row["GLD"] - past.iloc[0]["GLD"]) / past.iloc[0]["GLD"]),
                1.0 + float((row["BIL"] - past.iloc[0]["BIL"]) / past.iloc[0]["BIL"]),
            ]
            for _, row in past.iterrows()
        ]
    )


def scatter(weights):
    result = pd.Series(0.0, index=UNIVERSE)
    for name, fraction in BASKET.items():
        result[name] = float(weights[0] * fraction)
    result["GLD"], result["BIL"] = float(weights[1]), float(weights[2])
    if (
        not np.isfinite(result).all()
        or (result < 0).any()
        or not np.isclose(result.sum(), BUDGET, rtol=0, atol=1e-12)
        or not (result.loc[list(CORE)] > 0).all()
    ):
        raise QuantError("Scatter changed budget or genuine factor participation.")
    return result


def build_targets(data: MarketData, policy: dict):
    validate(policy)
    data.validate()
    close = data.close
    if tuple(close.columns) != UNIVERSE or not close.index.equals(
        sessions(close.index[0], close.index[-1])
    ):
        raise QuantError("Use the complete original admitted twelve-fund calendar.")
    target = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for position, day in enumerate(close.index):
        if position < LOOKBACK or not is_month_end(day):
            continue
        paths = observed_paths(close, day)
        weights, record = allocate(paths)
        verify_plan(weights, record, independent_paths(close, day))
        target.loc[day] = scatter(weights)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_observation": str(close.index[position - LOOKBACK].date()),
                "last_observation": str(day.date()),
                "return_observations": LOOKBACK,
                "sleeve_weights_of_total_capital": dict(
                    zip(SLEEVES, weights.tolist(), strict=True)
                ),
                **record,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits
