from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-full-covariance-risk-budget-v1.json"
PRIOR_HEAD = "6b17fa3f52d43633ad8ae342f2c4617a666e58a93b5154ed8ed86469801810ab"
EXPERT_IDS, MECHANISMS = source.EXPERT_IDS, source.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_full_covariance_erc_v1",
        "weighting": "full_covariance_equal_risk_budget",
    },
)
combine = source.combine
MAX_ITERATIONS = 10000
BUDGET_TOLERANCE = 1e-12


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_full_covariance_risk_budget_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 220,
        "prior_new_registered_candidates": 45,
        "prior_new_completed_strategy_evaluations": 42,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 502,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "sample_covariance_ddof": 1,
        "all_observations_used_for_risk_fit": True,
        "subperiod_diagnostic_sessions": [84, 42],
        "subperiod_diagnostics_are_heldout_validation": False,
        "solver_max_iterations": MAX_ITERATIONS,
        "solver_scaled_budget_tolerance": BUDGET_TOLERANCE,
        "covariance_integrity_tolerance": 1e-12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": 1e-12,
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
        raise QuantError("Freeze one full-covariance expert allocator and original funding gates.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A risk allocation is not qualification or authority to trade.")


def sample_covariance(values: np.ndarray) -> np.ndarray:
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] != 6
        or not np.isfinite(values).all()
    ):
        raise QuantError("Use six complete finite observation vectors without filling.")
    centered = values - values[0]
    centered -= centered.mean(axis=0)
    return centered.T @ centered / (len(centered) - 1)


def risk_budget(covariance: np.ndarray) -> tuple[np.ndarray, dict]:
    if (
        covariance.shape != (6, 6)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
        or (np.diag(covariance) <= 0).any()
    ):
        raise QuantError("Solve only finite symmetric covariance with positive individual risk.")
    matrix = covariance / float(np.trace(covariance))
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if minimum_eigenvalue < -1e-12:
        raise QuantError("A non-PSD covariance cannot be silently regularized.")
    x = np.ones(6)
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        for i in range(6):
            cross = float(matrix[i] @ x - matrix[i, i] * x[i])
            root = math.sqrt(cross * cross + 4 * float(matrix[i, i]) / 6)
            x[i] = (2 / 6) / (cross + root) if cross >= 0 else (root - cross) / (2 * matrix[i, i])
        residual = float(abs(x * (matrix @ x) - 1 / 6).max())
        if residual <= BUDGET_TOLERANCE:
            break
    else:
        raise QuantError("Full expert risk budgeting did not converge; no success fallback.")
    if not np.isfinite(x).all() or (x <= 0).any():
        raise QuantError("The positive risk solution is not finite and funded.")
    weights = x / x.sum()
    variance = float(weights @ covariance @ weights)
    if variance <= 0 or not math.isfinite(variance):
        raise QuantError("The solved expert covariance does not have positive finite total risk.")
    shares = weights * (covariance @ weights) / variance
    if (
        variance <= 0
        or not np.isfinite(shares).all()
        or float(abs(shares - 1 / 6).max()) > 1e-9
        or abs(float(weights.sum()) - 1) > 1e-12
    ):
        raise QuantError("The whole-pool equal marginal risk budgets do not reconcile.")
    return weights, {
        "solver": "positive_cyclic_coordinate_descent",
        "iterations": iteration,
        "unnormalized_solution": x.tolist(),
        "scaled_budget_equation_max_residual": residual,
        "scaled_covariance_minimum_eigenvalue": minimum_eigenvalue,
        "signed_variance_contribution_shares": shares.tolist(),
        "forecast_annualized_RF_excess_volatility": math.sqrt(252 * variance),
        "all_six_expert_weights_positive": True,
        "variance_floor_or_regularization_applied": False,
    }


def choose_allocation(
    observed: list[ExpertObservation],
    previous: np.ndarray,
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
        raise QuantError(
            "Preserve complete expert identities, aligned NET clocks and known targets."
        )
    equal = np.full(6, 1 / 6)
    if any(not o.score_mature for o in observed):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "risk_estimate_available": False,
            "available_shadow_observations": [len(o.net_returns) for o in observed],
            "safety_or_success_claimed": False,
        }
    index, rf = observed[0].net_returns.index, observed[0].risk_free
    if any(not o.net_returns.index.equals(index) or not o.risk_free.equals(rf) for o in observed):
        raise QuantError("All six risk coordinates need exactly the same mature dates and RF.")
    raw = np.column_stack([o.net_returns.to_numpy() for o in observed])
    excess = raw - rf.to_numpy()[:, None]
    covariance = sample_covariance(excess)
    if (np.diag(covariance) <= 0).any():
        return equal, {
            "reason": "unscored_nonpositive_individual_variance_equal_weights",
            "risk_estimate_available": False,
            "risk_sample_start": str(index[0].date()),
            "risk_sample_end": str(index[-1].date()),
            "covariance": covariance.tolist(),
            "safety_or_success_claimed": False,
        }
    weights, solved = risk_budget(covariance)
    change = abs(combine(observed, weights) - combine(observed, previous)).to_numpy()
    subperiods = {}
    for label, positions in (("older_84", slice(0, 84)), ("recent_42", slice(84, 126))):
        sample = excess[positions]
        cov = sample_covariance(sample)
        variance = float(weights @ cov @ weights)
        subperiods[label] = {
            "observations": len(sample),
            "start": str(index[positions][0].date()),
            "end": str(index[positions][-1].date()),
            "NET_shadow_mean_returns": raw[positions].mean(axis=0).tolist(),
            "RF_excess_shadow_mean_returns": sample.mean(axis=0).tolist(),
            "combined_RF_excess_mean_return": float((sample @ weights).mean()),
            "combined_annualized_RF_excess_volatility": math.sqrt(252 * variance),
            "signed_variance_contribution_shares": (weights * (cov @ weights) / variance).tolist()
            if variance > 0
            else None,
            "zero_combined_variance": variance == 0,
            "heldout_validation": False,
        }
    return weights, {
        "reason": "full_126_NET_RF_covariance_equal_risk_budget",
        "risk_estimate_available": True,
        "risk_sample_start": str(index[0].date()),
        "risk_sample_end": str(index[-1].date()),
        "observations_used_for_fit": 126,
        "covariance": covariance.tolist(),
        **solved,
        "subperiod_risk_stability": subperiods,
        "estimated_incremental_switch_cost_fraction": float(
            change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000
        ),
        "estimated_incremental_changed_assets": int((change > 1e-12).sum()),
        "cost_proxy_is_actual_main_account_cost": False,
        "return_score_or_rank_used": False,
        "safety_or_success_claimed": False,
    }


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the genuine initial anchor and every original source session.")
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
        raise QuantError("Use six complete strategy experts, not renamed holdings.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits, previous = [], np.full(6, 1 / 6)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if (
            any(o.target is None or tuple(o.target.index) != universe for o in observed)
            or position > 0
            and any(not len(o.net_returns) for o in observed)
        ):
            raise QuantError("Missing complete expert recommendations cannot be dropped or filled.")
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
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
        )
        previous = weights
    return targets, audits
