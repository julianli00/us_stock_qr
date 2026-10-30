from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as pool
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_full_covariance_risk_budget_v1 import sample_covariance
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-hierarchical-risk-v1.json"
PRIOR_HEAD = "b28c8b9b8c469d62d6d82837d27698270c2026479473c67ea4a623135ed44629"
EXPERT_IDS, MECHANISMS = pool.EXPERT_IDS, pool.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_hierarchical_risk_v1",
        "weighting": "hierarchical_single_linkage_recursive_risk",
    },
)
combine = pool.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_hierarchical_risk_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 221,
        "prior_new_registered_candidates": 47,
        "prior_new_completed_strategy_evaluations": 43,
        "prior_new_technical_failures": 4,
        "prior_journal_sequence": 516,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "sample_covariance_ddof": 1,
        "all_observations_used_for_risk_fit": True,
        "risk_fit_is_heldout_validation": False,
        "linkage": "canonical_single",
        "distance_key_decimal_places": 12,
        "correlation_roundoff_tolerance": 1e-12,
        "allocation": "seriated_balanced_bisection_inverse_child_IVP_variance",
        "zero_child_variance": "one_zero_child_receives_parent_both_zero_split_equally",
        "independent_weight_tolerance": 1e-12,
        "independent_target_tolerance": 1e-12,
        "independent_covariance_tolerance": 1e-12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
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
        key not in policy or type(policy[key]) is not type(value) or policy[key] != value
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze the independent hierarchical rule and original funding contract.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "source_expert_introduction_backdated",
            "risk_clusters_are_independent_factor_families",
            "source_shadow_wealth_or_floor_transferred",
            "return_score_or_rank_used",
            "switch_cost_proxy_used_for_allocation",
            "covariance_inverse_or_global_optimizer_used",
            "covariance_regularization_or_variance_floor",
            "prior_failed_covariance_solver_invoked",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Risk clusters are neither new factors nor a qualification guarantee.")


def cluster_order(covariance: np.ndarray) -> tuple[tuple[int, ...], list[dict], np.ndarray]:
    if (
        covariance.shape != (6, 6)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        or (np.diag(covariance) <= 0).any()
        or float(np.linalg.eigvalsh(covariance / np.trace(covariance)).min()) < -1e-12
    ):
        raise QuantError("A hierarchy needs finite PSD covariance and positive individual risk.")
    deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(deviation, deviation)
    if float(abs(correlation).max()) > 1 + 1e-12:
        raise QuantError("Invalid correlations cannot be repaired by distance clipping.")
    distance = np.sqrt((1 - np.clip(correlation, -1, 1)) / 2)
    np.fill_diagonal(distance, 0)
    orders = {(i,): (i,) for i in range(6)}
    merges = []
    while len(orders) > 1:
        active = sorted(orders)
        pairs = [
            (
                round(min(float(distance[i, j]) for i in left for j in right), 12),
                left,
                right,
            )
            for position, left in enumerate(active)
            for right in active[position + 1 :]
        ]
        key, left, right = min(pairs)
        combined = tuple(sorted((*left, *right)))
        order = orders.pop(left) + orders.pop(right)
        orders[combined] = order
        merges.append(
            {
                "left_members": list(left),
                "right_members": list(right),
                "distance_key": key,
                "merged_members": list(combined),
                "leaf_order": list(order),
            }
        )
    return next(iter(orders.values())), merges, distance


def hierarchical_weights(values: np.ndarray) -> tuple[np.ndarray, dict]:
    covariance = sample_covariance(values)
    order, merges, distance = cluster_order(covariance)
    centered = values - values[0]
    weights = np.zeros(6)
    splits = []

    def variance(members):
        inverse = 1 / np.diag(covariance)[list(members)]
        inverse /= inverse.sum()
        observed = centered[:, list(members)] @ inverse
        observed -= observed.mean()
        result = float(observed @ observed / (len(observed) - 1))
        if not math.isfinite(result) or result < 0:
            raise QuantError("Observed cluster risk must be finite and nonnegative.")
        return result

    def allocate(members, capital):
        if len(members) == 1:
            weights[members[0]] = capital
            return
        midpoint = len(members) // 2
        left, right = members[:midpoint], members[midpoint:]
        left_risk, right_risk = variance(left), variance(right)
        fraction = 0.5 if left_risk == right_risk == 0 else right_risk / (left_risk + right_risk)
        splits.append(
            {
                "leaf_order": list(members),
                "left": list(left),
                "right": list(right),
                "parent_capital_fraction": float(capital),
                "left_IVP_variance": left_risk,
                "right_IVP_variance": right_risk,
                "left_capital_fraction_of_parent": float(fraction),
                "both_child_risks_zero": left_risk == right_risk == 0,
                "exactly_one_child_risk_zero": (left_risk == 0) != (right_risk == 0),
            }
        )
        allocate(left, capital * fraction)
        allocate(right, capital * (1 - fraction))

    allocate(order, 1.0)
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
    ):
        raise QuantError("Hierarchical capital must reconcile without borrowing or shorting.")
    return weights, {
        "covariance": covariance.tolist(),
        "risk_distance": distance.tolist(),
        "canonical_merges": merges,
        "seriated_expert_coordinates": list(order),
        "recursive_capital_splits": splits,
        "forecast_annualized_RF_excess_volatility": float(
            np.std(centered @ weights, ddof=1) * math.sqrt(252)
        ),
        "variance_floor_or_regularization_applied": False,
        "global_optimizer_or_previous_failed_solver_used": False,
        "risk_clusters_are_independent_factor_families": False,
    }


def choose_allocation(
    observed: list[ExpertObservation], previous: np.ndarray
) -> tuple[np.ndarray, dict]:
    if (
        len(observed) != 6
        or tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or previous.shape != (6,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(float(previous.sum()) - 1) > 1e-12
        or any(
            o.as_of != observed[0].as_of
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > o.as_of
            or not o.net_returns.index.equals(o.risk_free.index)
            or not np.isfinite(o.net_returns).all()
            or not np.isfinite(o.risk_free).all()
            or (o.net_returns <= -1).any()
            or o.score_mature != (len(o.net_returns) == 126)
            or len(o.net_returns) > 126
            or len(o.net_returns)
            and (
                o.net_returns.index[-1] != o.as_of
                or not o.net_returns.index.equals(
                    sessions(o.net_returns.index[0], o.net_returns.index[-1])
                )
            )
            for o in observed
        )
    ):
        raise QuantError("Keep complete expert identities, matured clocks, RF and known targets.")
    equal = np.full(6, 1 / 6)
    if any(not o.score_mature for o in observed):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "risk_estimate_available": False,
            "available_shadow_observations": [len(o.net_returns) for o in observed],
            "safety_or_success_claimed": False,
        }
    dates, rf = observed[0].net_returns.index, observed[0].risk_free
    if any(not o.net_returns.index.equals(dates) or not o.risk_free.equals(rf) for o in observed):
        raise QuantError("The hierarchy needs exactly common mature source dates and RF.")
    excess = np.column_stack([o.net_returns.to_numpy() - rf.to_numpy() for o in observed])
    covariance = sample_covariance(excess)
    common = {
        "risk_sample_start": str(dates[0].date()),
        "risk_sample_end": str(dates[-1].date()),
        "safety_or_success_claimed": False,
    }
    if (np.diag(covariance) <= 0).any():
        return equal, {
            **common,
            "reason": "unscored_nonpositive_individual_variance_equal_weights",
            "risk_estimate_available": False,
            "covariance": covariance.tolist(),
        }
    weights, allocation = hierarchical_weights(excess)
    change = abs(combine(observed, weights) - combine(observed, previous)).to_numpy()
    return weights, {
        **common,
        **allocation,
        "reason": "full_126_NET_RF_hierarchical_risk_allocation",
        "risk_estimate_available": True,
        "observations_used_for_fit": 126,
        "risk_fit_is_heldout_validation": False,
        "return_score_or_rank_used": False,
        "nominal_target_change_L1": float(change.sum()),
        "nominal_target_change_is_actual_turnover": False,
        "cost_proxy_used_for_allocation": False,
    }


def decision_sessions(calendar: pd.DatetimeIndex, policy: dict) -> pd.DatetimeIndex:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the real initial anchor and every original source session.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
) -> tuple[pd.DataFrame, list[dict]]:
    dates = decision_sessions(calendar, policy)
    if len(experts) != 6 or len(universe) != len(set(universe)):
        raise QuantError("Use the unchanged six complete experts and unique asset coordinates.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits = []
    previous = np.full(6, 1 / 6)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if any(o.target is None or tuple(o.target.index) != universe for o in observed):
            raise QuantError("An unavailable expert cannot disappear from the fixed risk pool.")
        weights, allocation = choose_allocation(observed, previous)
        targets.loc[day] = combine(observed, weights)
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
                "previous_expert_weights": previous.tolist(),
                "expert_weights": weights.tolist(),
                "allocation": allocation,
                "counterfactual_historical_expert_availability": True,
                "source_shadow_wealth_or_protection_floor_transferred": False,
                "independent_forward_validation": False,
            }
        )
        previous = weights
    return targets, audits
