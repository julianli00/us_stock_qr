from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.ordered_path_growth_v1 import (
    BASKET,
    BUDGET,
    CASH,
    CERT_TOL,
    CORE,
    DRAWDOWN,
    FACTOR_IDS,
    FEAS_TOL,
    LOOKBACK,
    ROUND_TOL,
    SLACK,
    SLEEVES,
    STOCK_MIN,
    UNIVERSE,
    drawdown,
    independent_paths,
    observed_paths,
    problem,
    record_solution,
    scatter,
    verify_plan,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/ordered-path-growth-v2.json"
SOLVER_TOL = 1e-10
PRIOR_HEAD = "aefd8adaafd8df9efaa6a61b41784bbbfdecfc3d6949f3c5f7a8bf7259af6a0b"
PREVIOUS_FAILURE = {
    "path": "evidence/ordered_path_growth_20261028_preparation_failure_v1.json",
    "sha256": "7cb58f891b759493d2cb0d9266e43de41f26fda3aa8a8c40a4117b0ec773a634",
}
CANDIDATES = (
    {
        "id": "ordered_path_growth_budget_precision_v2",
        "construction": "terminal_wealth_with_all_ordered_peak_trough_constraints",
    },
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_ordered_path_growth_20261028_v2",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 195,
        "prior_journal_sequence": 204,
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
        "revision_of_candidate": "ordered_path_growth_budget_v1",
        "revision_kind": "solver_precision_only",
        "new_independent_economic_hypotheses": 0,
        "preserved_preparation_failure": PREVIOUS_FAILURE,
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
