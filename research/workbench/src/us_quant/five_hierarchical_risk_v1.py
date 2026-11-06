from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import conditional_volatility_quintet_v1 as source
from us_quant import expert_hierarchical_risk_v1 as original
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.median_equity_budget_v1 import checked_opinions
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/five-hierarchical-risk-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
ID = "expert_pool_five_hierarchical_risk_v1"
CANDIDATES = ({"id": ID, "weighting": "fixed_five_single_linkage_recursive_risk"},)
PRIOR_HEAD = "ccd9626b845cc244cf08264d7674c146e66b354a97284a8d7dc95549dce080e8"


def expected_policy():
    old, pool = read_json(original.POLICY), read_json(source.POLICY)
    inherited = (
        "protocol_id",
        "initial_decision_session",
        "lookback_sessions",
        "sample_covariance_ddof",
        "all_observations_used_for_risk_fit",
        "risk_fit_is_heldout_validation",
        "linkage",
        "distance_key_decimal_places",
        "correlation_roundoff_tolerance",
        "allocation",
        "zero_child_variance",
        "independent_weight_tolerance",
        "independent_target_tolerance",
        "independent_covariance_tolerance",
        "shadow_reference_path",
        "reference_capital_usd",
        "invested_target_budget",
        "cash_reserve",
        "actual_weight_trade_band",
        "minimum_economic_factor_families",
        "study_size_is_a_total_research_quota",
    )
    flags = (
        "risk_clusters_are_independent_factor_families",
        "return_score_or_rank_used",
        "switch_cost_proxy_used_for_allocation",
        "covariance_inverse_or_global_optimizer_used",
        "covariance_regularization_or_variance_floor",
        "prior_failed_covariance_solver_invoked",
        "independent_forward_validation",
        "investment_objective_verified",
        "order_authority",
    )
    return {
        **{k: old[k] for k in inherited},
        **{k: old["methodology"][k] for k in flags},
        "schema_version": 1,
        "study_id": "ssd_five_hierarchical_risk_20261106_v1",
        "authorization_id": "november-research-20261103-v1",
        "actual_native_event_seq": 184317,
        "prior_journal_sequence": 1381,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 294,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/five-hierarchical-risk-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "37057d0c62e3bf19860c256b0cfc5caa224af324355d55ec9f6af0d48a780988",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "median_equity_budget_promoted_or_retuned": False,
        "admitted_legacy_defensive_views_reused_unchanged": pool[
            "admitted_legacy_defensive_views_reused_unchanged"
        ],
        "all_source_native_RF_claimed_equal_without_view": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "global_pool_changed": False,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Only the already-fixed five-source hierarchical construction is declared."
        )


def sample_covariance(values):
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise QuantError("Use complete finite observations and at least two real coordinates.")
    centered = values - values[0]
    centered -= centered.mean(axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    if not np.isfinite(covariance).all():
        raise QuantError("Nonfinite covariance is not a successful risk estimate.")
    return covariance


def cluster_order(covariance):
    if (
        not isinstance(covariance, np.ndarray)
        or covariance.ndim != 2
        or covariance.dtype.kind != "f"
    ):
        raise QuantError("Require a complete numeric covariance matrix.")
    n = len(covariance)
    if (
        covariance.shape != (n, n)
        or n < 2
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        or (np.diag(covariance) <= 0).any()
        or float(np.linalg.eigvalsh(covariance / np.trace(covariance)).min()) < -1e-12
    ):
        raise QuantError("Require finite PSD covariance and positive individual risk.")
    deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(deviation, deviation)
    if float(abs(correlation).max()) > 1 + 1e-12:
        raise QuantError("Invalid correlation cannot be repaired by clipping.")
    distance = np.sqrt((1 - np.clip(correlation, -1, 1)) / 2)
    np.fill_diagonal(distance, 0)
    orders, merges = {(i,): (i,) for i in range(n)}, []
    while len(orders) > 1:
        active = sorted(orders)
        pairs = [
            (round(min(float(distance[i, j]) for i in left for j in right), 12), left, right)
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


def hierarchical_weights(values):
    covariance = sample_covariance(values)
    n = values.shape[1]
    order, merges, distance = cluster_order(covariance)
    centered, weights, splits = values - values[0], np.zeros(n), []

    def variance(members):
        inverse = 1 / np.diag(covariance)[list(members)]
        inverse /= inverse.sum()
        observed = centered[:, list(members)] @ inverse
        observed -= observed.mean()
        result = float(observed @ observed / (len(observed) - 1))
        if not math.isfinite(result) or result < 0:
            raise QuantError("Child risk must remain finite and nonnegative.")
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
        raise QuantError("Allocate one unit of nonnegative cash-funded capital.")
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


def combine(observed, weights):
    return pd.Series(
        weights @ np.stack([o.target.to_numpy() for o in observed]),
        index=ASSETS,
        name=observed[0].as_of,
    )


def choose_allocation(observed, previous):
    if len(observed) != 5 or not isinstance(previous, np.ndarray):
        raise QuantError("Require the fixed five complete experts and their prior allocation.")
    checked_opinions(observed, observed[0].as_of)
    if (
        previous.shape != (5,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(float(previous.sum()) - 1) > 1e-12
        or any(
            not o.net_returns.index.equals(o.risk_free.index)
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
        raise QuantError("Keep exact matured NET/RF clocks and a valid previous allocation.")
    equal = np.full(5, 0.2)
    if any(not o.score_mature for o in observed):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "risk_estimate_available": False,
            "available_shadow_observations": [len(o.net_returns) for o in observed],
            "safety_or_success_claimed": False,
        }
    dates, rf = observed[0].net_returns.index, observed[0].risk_free
    if any(not o.net_returns.index.equals(dates) or not o.risk_free.equals(rf) for o in observed):
        raise QuantError("All five risk histories require exactly common dates and assessment RF.")
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


def decision_sessions(calendar, policy):
    validate(policy)
    return source.decision_sessions(calendar, read_json(source.POLICY))


def snapshot(observed):
    return [
        {
            "id": o.identity.identifier,
            "mechanism_group": o.identity.mechanism_group,
            "introduced_at": o.identity.introduced_at,
            "as_of": str(o.as_of.date()),
            "target_known_at": str(o.target_known_at.date())
            if o.target_known_at is not None
            else None,
            "asset_order": list(o.target.index) if o.target is not None else None,
            "target_binary64_hex": [float(v).hex() for v in o.target]
            if o.target is not None
            else None,
            "NET_dates": o.net_returns.index.strftime("%Y-%m-%d").tolist(),
            "NET_binary64_hex": [float(v).hex() for v in o.net_returns],
            "RF_binary64_hex": [float(v).hex() for v in o.risk_free],
            "score_mature": o.score_mature,
        }
        for o in observed
    ]


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    dates = decision_sessions(calendar, policy)
    if len(experts) != 5 or universe != ASSETS:
        raise QuantError("Use exactly the fixed five complete experts and original14assets.")
    targets, audits = pd.DataFrame(np.nan, index=calendar, columns=universe), []
    previous = np.full(5, 0.2)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": position,
                    "as_of": str(day.date()),
                    "observations": snapshot(observed),
                    "previous_expert_weights_binary64_hex": [float(v).hex() for v in previous],
                }
            )
        weights, allocation = choose_allocation(observed, previous)
        target = combine(observed, weights)
        if (
            not np.isfinite(target).all()
            or (target < 0).any()
            or abs(float(target.sum()) - 0.98) > 1e-12
        ):
            raise QuantError("The hierarchy must produce one funded underlying-asset target.")
        targets.loc[day] = target
        audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": CLOCK,
            "previous_expert_weights": previous.tolist(),
            "expert_weights": weights.tolist(),
            "allocation": allocation,
            "source_score_mature": [o.score_mature for o in observed],
            "source_shadow_wealth_transferred": False,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "decision_index": position,
                    "audit": audit,
                    "asset_order": list(universe),
                    "combined_targets": {ID: target.tolist()},
                }
            )
        previous = weights
    return {ID: targets}, audits
