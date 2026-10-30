from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from us_quant import joint_tail_construction_v1 as lp
from us_quant import strategy_expert_pool_v4 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-tail-risk-budget-v1.json"
PRIOR_HEAD = "1b68a018352ef4c975265ce8f2dbe011fdbfd904ebd3b06fb8025a625555e75a"
EXPERT_IDS, MECHANISMS = source.EXPERT_IDS, source.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_robust_tail_budget_v1",
        "weighting": "robust_net_opportunity_under_joint_tail_budget",
    },
)
BENCHMARKS = ("SPY", "QQQ")
combine = source.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_tail_risk_budget_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 220,
        "prior_new_registered_candidates": 46,
        "prior_new_completed_strategy_evaluations": 42,
        "prior_new_technical_failures": 4,
        "prior_journal_sequence": 504,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "joint_selection_is_independent_validation": False,
        "tail_probability": 0.05,
        "solver_tolerance": lp.SOLVER_TOL,
        "certificate_tolerance": lp.CERT_TOL,
        "decision_feasibility_tolerance": lp.FEAS_TOL,
        "weight_roundoff_tolerance": lp.ROUND_TOL,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "switch_cost_amortization_sessions": 21,
        "changed_asset_numeric_tolerance": 1e-12,
        "switch_proxy_reference": "last_nominal_meta_target_not_actual_holdings",
        "infeasibility_requires_phase_one_certificate": True,
        "funding_standby_symbol": "BIL",
        "funding_standby_is_strategy_expert": False,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()
    ):
        raise QuantError("Freeze one joint-tail expert policy and all original funding gates.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A tail-risk assessment cannot claim qualification or trading authority.")


def linear_problem(excess, benchmarks, targets, previous):
    if (
        excess.shape != (126, 6)
        or benchmarks.shape != (126, 2)
        or targets.shape[1] != 6
        or previous.shape != (targets.shape[0],)
        or not all(np.isfinite(v).all() for v in (excess, benchmarks, targets, previous))
    ):
        raise QuantError("Use exact mature expert, benchmark and underlying-target coordinates.")
    width = 135 + len(previous)
    cost = np.zeros(width)
    cost[6] = -1
    cost[135:] = 5 / 10000 / 21
    rows, values, periods, cap_rows = [], [], {}, []
    for label, start, stop, theta, first_hinge in (
        ("older_84", 0, 84, 7, 8),
        ("recent_42", 84, 126, 92, 93),
    ):
        sample, b = excess[start:stop], benchmarks[start:stop]
        losses = {name: lp.tail_loss(-b[:, i], 0.05) for i, name in enumerate(BENCHMARKS)}
        budget = min(losses.values())
        for i, scenario in enumerate(sample):
            row = np.zeros(width)
            row[:6], row[theta], row[first_hinge + i] = -scenario, -1, -1
            rows.append(row)
            values.append(0.0)
        risk = np.zeros(width)
        risk[theta], risk[first_hinge : first_hinge + len(sample)] = 1, 1 / (0.05 * len(sample))
        cap_rows.append(len(rows))
        rows.append(risk)
        values.append(budget)
        mean = sample.mean(axis=0)
        reward = np.zeros(width)
        reward[:6], reward[6] = -mean, 1
        rows.append(reward)
        values.append(0.0)
        periods[label] = {
            "observations": len(sample),
            "NET_RF_excess_means": mean.tolist(),
            "benchmark_RF_tail_losses": losses,
            "risk_budget": budget,
        }
    for i, target in enumerate(targets):
        for sign in (1, -1):
            row = np.zeros(width)
            row[:6], row[135 + i] = sign * target, -1
            rows.append(row)
            values.append(sign * previous[i])
    eq = np.zeros((1, width))
    eq[0, :6] = 1
    bounds = (
        [(0, 1)] * 6
        + [(None, None)] * 2
        + [(0, None)] * 84
        + [(None, None)]
        + [(0, None)] * (42 + len(previous))
    )
    return cost, np.array(rows), np.array(values), eq, np.ones(1), bounds, periods, cap_rows


def phase_one(problem):
    _, a, b, eq, rhs, bounds, _, cap_rows = problem
    extended = np.column_stack([a, np.zeros(len(a))])
    extended[cap_rows, -1] = -1
    objective = np.r_[np.zeros(a.shape[1]), 1.0]
    return (
        objective,
        extended,
        b,
        np.column_stack([eq, np.zeros(len(eq))]),
        rhs,
        [*bounds, (0, None)],
    )


def solve(problem):
    c, a, b, eq, rhs, bounds, periods, _ = problem
    options = {
        "primal_feasibility_tolerance": lp.SOLVER_TOL,
        "dual_feasibility_tolerance": lp.SOLVER_TOL,
    }
    result = linprog(
        c, A_ub=a, b_ub=b, A_eq=eq, b_eq=rhs, bounds=bounds, method="highs-ds", options=options
    )
    if result.status == 2:
        pc, pa, pb, peq, prhs, pbound = phase_one(problem)
        feasible = linprog(
            pc,
            A_ub=pa,
            b_ub=pb,
            A_eq=peq,
            b_eq=prhs,
            bounds=pbound,
            method="highs-ds",
            options=options,
        )
        proof = lp.certificate(feasible, pc, pa, pb, peq, prhs, pbound)
        if proof["objective"] <= lp.FEAS_TOL or proof["dual_objective"] <= lp.FEAS_TOL:
            raise QuantError("Reported infeasibility has no positive independent PhaseI gap.")
        return np.zeros(6), {
            "mode": "funding_standby",
            "reason": "certified_observed_joint_tail_budget_infeasible",
            "primary_solver_status": 2,
            "phase_one_certificate": proof,
            "phase_one_solution": lp.solution_record(feasible),
            "periods": periods,
            "allocation_risk_feasibility_or_safety_claimed": False,
        }
    proof = lp.certificate(result, c, a, b, eq, rhs, bounds)
    raw = result.x[:6]
    if (raw < -lp.ROUND_TOL).any() or abs(float(raw.sum()) - 1) > lp.FEAS_TOL:
        raise QuantError("The expert LP solution is not a funded nonnegative simplex.")
    weights = raw.copy()
    weights[abs(weights) <= lp.ROUND_TOL] = 0
    weights /= weights.sum()
    if (weights < 0).any() or abs(float(weights.sum()) - 1) > 1e-12:
        raise QuantError("Numerical cleanup cannot introduce leverage or unfunded expert capital.")
    return weights, {
        "mode": "expert_mix",
        "reason": "robust_NET_opportunity_under_joint_tail_budget",
        "primary_solver_status": 0,
        "primary_certificate": proof,
        "primary_solution": lp.solution_record(result),
        "maximum_weight_roundoff_change": float(abs(weights - raw).max()),
        "periods": periods,
        "allocation_risk_feasibility_or_safety_claimed": False,
    }


def tail_contributions(scenarios, weights):
    losses = np.array(
        [-math.fsum(float(weights[i]) * float(row[i]) for i in range(6)) for row in scenarios]
    )
    order = np.argsort(-losses, kind="stable")
    mass = 0.05 * len(losses)
    full = int(np.floor(mass))
    marginal = -scenarios[order[:full]].sum(axis=0)
    if mass > full:
        marginal -= (mass - full) * scenarios[order[full]]
    marginal /= mass
    return marginal, weights * marginal


def choose_allocation(
    observed: list[ExpertObservation],
    previous_target: pd.Series | None,
    benchmark_returns: pd.DataFrame,
):
    if (
        len(observed) != 6
        or tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != observed[0].as_of
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > o.as_of
            or not o.net_returns.index.equals(o.risk_free.index)
            or not np.isfinite(o.net_returns).all()
            or not np.isfinite(o.risk_free).all()
            or (o.net_returns <= -1).any()
            or len(o.net_returns) > 126
            or o.score_mature != (len(o.net_returns) == 126)
            for o in observed
        )
    ):
        raise QuantError("Preserve six complete known expert policies and original NET history.")
    index, rf = observed[0].net_returns.index, observed[0].risk_free
    if (
        any(not o.net_returns.index.equals(index) or not o.risk_free.equals(rf) for o in observed)
        or len(index)
        and (index[-1] != observed[0].as_of or not index.equals(sessions(index[0], index[-1])))
    ):
        raise QuantError("Missing or different expert/RF calendars cannot become a neutral score.")
    equal = np.full(6, 1 / 6)
    current_equal = combine(observed, equal)
    if "BIL" not in current_equal.index:
        raise QuantError("The funding asset must belong to the declared admitted universe.")
    if previous_target is None:
        previous_target = current_equal
    if (
        not previous_target.index.equals(current_equal.index)
        or not np.isfinite(previous_target).all()
        or (previous_target < 0).any()
        or abs(float(previous_target.sum()) - 0.98) > 1e-12
    ):
        raise QuantError("The cost reference is a declared nominal target, not invented holdings.")
    if not observed[0].score_mature:
        return (
            current_equal,
            equal,
            {
                "mode": "unscored_equal",
                "reason": "unmatured_initial_history",
                "risk_problem_mature": False,
                "available_observations": len(index),
                "allocation_risk_feasibility_or_safety_claimed": False,
            },
        )
    if (
        not isinstance(benchmark_returns, pd.DataFrame)
        or not isinstance(benchmark_returns.index, pd.DatetimeIndex)
        or not benchmark_returns.index.is_unique
        or not benchmark_returns.columns.is_unique
        or not set(BENCHMARKS).issubset(benchmark_returns.columns)
        or not index.isin(benchmark_returns.index).all()
    ):
        raise QuantError("Use original NET SPY/QQQ on every exact matured source date.")
    b = benchmark_returns.loc[index, list(BENCHMARKS)].to_numpy()
    if not np.isfinite(b).all() or (b <= -1).any():
        raise QuantError("Missing benchmark observations cannot be filled or ignored.")
    raw = np.column_stack([o.net_returns.to_numpy() for o in observed])
    excess = raw - rf.to_numpy()[:, None]
    benchmark_excess = b - rf.to_numpy()[:, None]
    coordinates = np.column_stack([o.target.to_numpy() for o in observed])
    problem = linear_problem(excess, benchmark_excess, coordinates, previous_target.to_numpy())
    weights, allocation = solve(problem)
    if allocation["mode"] == "funding_standby":
        target = current_equal * 0
        target.loc["BIL"] = 0.98
    else:
        target = combine(observed, weights)
    delta = abs(target - previous_target).to_numpy()
    proportional = float(delta.sum() * 5 / 10000)
    fixed = int((delta > 1e-12).sum()) / 10000
    if allocation["mode"] == "expert_mix":
        realized_means = []
        for label, bounds in (("older_84", slice(0, 84)), ("recent_42", slice(84, 126))):
            sample = excess[bounds]
            tail = lp.tail_loss(-sample @ weights, 0.05)
            period = allocation["periods"][label]
            marginal, contributions = tail_contributions(sample, weights)
            mean = float((sample @ weights).mean())
            if (
                tail > period["risk_budget"] + lp.FEAS_TOL
                or abs(float(contributions.sum()) - tail) > 1e-12
            ):
                raise QuantError("Exact post-cleanup joint tail risk rejects the LP solution.")
            period.update(
                {
                    "observed_tail_loss": tail,
                    "achieved_RF_excess_mean": mean,
                    "tail_marginal_losses": marginal.tolist(),
                    "signed_tail_risk_contributions": contributions.tolist(),
                }
            )
            realized_means.append(mean)
        achieved = min(realized_means) - proportional / 21
        if achieved < -allocation["primary_certificate"]["objective"] - lp.FEAS_TOL:
            raise QuantError("Exact means and nominal switch costs reject numerical cleanup.")
        allocation["achieved_robust_RF_opportunity_after_proportional_proxy"] = achieved
    allocation.update(
        {
            "risk_problem_mature": True,
            "risk_sample_start": str(index[0].date()),
            "risk_sample_end": str(index[-1].date()),
            "estimated_incremental_proportional_cost_fraction": proportional,
            "estimated_incremental_fixed_ticket_cost_fraction": fixed,
            "estimated_incremental_changed_assets": int((delta > 1e-12).sum()),
            "cost_proxy_is_actual_main_account_cost": False,
            "joint_selection_is_independent_validation": False,
            "funding_standby_is_strategy_expert": False,
        }
    )
    return target, weights, allocation


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the genuine capital anchor and all original exchange sessions.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar,
    universe,
    policy,
    benchmark_returns,
    before_decision=None,
):
    dates = decision_sessions(calendar, policy)
    if len(experts) != 6 or len(universe) != len(set(universe)) or "BIL" not in universe:
        raise QuantError(
            "Use six admitted complete experts and the funded fourteen-asset universe."
        )
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits, previous = [], None
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if (
            any(o.target is None or tuple(o.target.index) != universe for o in observed)
            or position > 0
            and not len(observed[0].net_returns)
        ):
            raise QuantError("Missing complete recommendations cannot be substituted or dropped.")
        target, weights, allocation = choose_allocation(observed, previous, benchmark_returns)
        targets.loc[day] = target
        audits.append(
            {
                "decision_session": str(day.date()),
                "decision_role": "initial_capital"
                if position == 0
                else "completed_exchange_month_end",
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
                "preserved_archive_identity_introduced_at": [
                    o.identity.introduced_at for o in observed
                ],
                "expert_shadow_last_observation": [
                    str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                    for o in observed
                ],
                "previous_nominal_meta_target": previous.tolist() if previous is not None else None,
                "expert_weights": weights.tolist(),
                "allocation": allocation,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
        )
        previous = target.copy()
    return targets, audits
