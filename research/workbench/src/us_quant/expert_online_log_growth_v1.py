from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as pool
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.expert_net_wealth_mixture_v1 import period_relative
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-online-log-growth-v1.json"
PRIOR_HEAD = "c7bd3ad1634d8632781f4f67529f36b5eee1e1de48143af5b95ddb8ac558f6fa"
EXPERT_IDS, MECHANISMS = pool.EXPERT_IDS, pool.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_online_log_growth_v1",
        "weighting": "second_order_monthly_NET_log_growth",
    },
)
combine = pool.combine
SIMPLEX_TOL = 1e-12
KKT_TOL = 1e-10


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_online_log_growth_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 222,
        "prior_new_registered_candidates": 48,
        "prior_new_completed_strategy_evaluations": 44,
        "prior_new_technical_failures": 4,
        "prior_journal_sequence": 528,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "minimum_common_source_observations": 126,
        "shadow_reference_path": "10y-base",
        "learning_clock": "completed_full_exchange_month_intervals",
        "training_memory": "expanding_all_completed_intervals",
        "initial_partial_source_month_assimilated": False,
        "learning_state_resets_for_five_year_account": False,
        "maximum_complete_learning_intervals": 119,
        "initial_curvature_identity_scale": 1.0,
        "step_size": 1.0,
        "simplex_roundoff_tolerance": SIMPLEX_TOL,
        "KKT_tolerance": KKT_TOL,
        "curvature_integrity_tolerance": 1e-10,
        "independent_weight_and_target_tolerance": 1e-12,
        "maximum_simplex_faces_per_update": 63,
        "reference_capital_usd": 10000,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        key not in policy or type(policy[key]) is not type(value) or policy[key] != value
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze one causal online law, the same six experts and original funding.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "learning_score_is_actual_main_net_wealth",
            "source_shadow_wealth_or_protection_floor_transferred",
            "universal_NET_wealth_identity_or_bound_claimed",
            "exact_uniform_policy_integral_repaired_or_approximated",
            "source_covariance_or_risk_rank_used",
            "switch_cost_proxy_affects_allocation",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A learning statistic is not funded wealth or a qualification theorem.")


@dataclass(frozen=True)
class AllocationLearningState:
    weights: np.ndarray
    curvature: np.ndarray
    completed_intervals: int

    @classmethod
    def initial(cls) -> AllocationLearningState:
        return cls(np.full(6, 1 / 6), np.eye(6), 0)

    def validate(self) -> None:
        if (
            self.weights.shape != (6,)
            or not np.isfinite(self.weights).all()
            or (self.weights < 0).any()
            or abs(float(self.weights.sum()) - 1) > SIMPLEX_TOL
            or self.curvature.shape != (6, 6)
            or not np.isfinite(self.curvature).all()
            or not np.allclose(self.curvature, self.curvature.T, rtol=0, atol=1e-10)
            or float(np.linalg.eigvalsh(self.curvature).min()) < 1 - 1e-10
            or type(self.completed_intervals) is not int
            or not 0 <= self.completed_intervals <= 119
        ):
            raise QuantError(
                "Require finite simplex learning weights and the positive identity prior."
            )


def certificate(matrix: np.ndarray, linear: np.ndarray, weights: np.ndarray) -> dict:
    if (
        matrix.shape != (6, 6)
        or linear.shape != (6,)
        or weights.shape != (6,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(linear).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > SIMPLEX_TOL
    ):
        raise QuantError("A projection certificate needs a finite funded simplex point.")
    gradient = matrix @ weights - linear
    positive = weights > SIMPLEX_TOL
    multiplier = -math.fsum(float(v) for v in gradient[positive]) / int(positive.sum())
    dual = gradient + multiplier
    stationarity = float(abs(dual[positive]).max(initial=0))
    dual_error = float(np.maximum(-dual[~positive], 0).max(initial=0))
    complementarity = float(abs(weights * dual).max())
    if max(stationarity, dual_error, complementarity) > KKT_TOL:
        raise QuantError("The simplex projection lacks a valid global quadratic KKT certificate.")
    return {
        "simplex_multiplier": multiplier,
        "nonnegative_coordinate_duals": dual.tolist(),
        "maximum_positive_stationarity_error": stationarity,
        "maximum_inactive_dual_violation": dual_error,
        "maximum_complementarity_error": complementarity,
    }


def project(matrix: np.ndarray, linear: np.ndarray) -> tuple[np.ndarray, dict]:
    if (
        matrix.shape != (6, 6)
        or linear.shape != (6,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(linear).all()
        or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10)
        or float(np.linalg.eigvalsh(matrix).min()) <= 0
    ):
        raise QuantError("Project only a finite symmetric positive-definite quadratic.")
    candidates = []
    for size in range(1, 7):
        for face in itertools.combinations(range(6), size):
            positions = list(face)
            block = matrix[np.ix_(positions, positions)]
            try:
                free = np.linalg.solve(block, linear[positions])
                direction = np.linalg.solve(block, np.ones(size))
            except np.linalg.LinAlgError as error:
                raise QuantError("A positive-definite face solve failed.") from error
            multiplier = (float(free.sum()) - 1) / float(direction.sum())
            raw = free - multiplier * direction
            if (
                not np.isfinite(raw).all()
                or (raw < -SIMPLEX_TOL).any()
                or abs(float(raw.sum()) - 1) > SIMPLEX_TOL
            ):
                continue
            weights = np.zeros(6)
            weights[positions] = np.maximum(raw, 0)
            weights /= weights.sum()
            objective = float(0.5 * weights @ matrix @ weights - linear @ weights)
            candidates.append((objective, face, weights))
    if not candidates:
        raise QuantError("No feasible simplex face exists; no neutral success fallback.")
    objective, face, weights = min(candidates, key=lambda row: (row[0], row[1]))
    return weights, {
        "enumerated_nonempty_faces": 63,
        "feasible_faces": len(candidates),
        "selected_face": list(face),
        "quadratic_objective": objective,
        **certificate(matrix, linear, weights),
    }


def advance(
    state: AllocationLearningState, relatives: np.ndarray
) -> tuple[AllocationLearningState, dict]:
    state.validate()
    if (
        relatives.shape != (6,)
        or not np.isfinite(relatives).all()
        or (relatives <= 0).any()
        or state.completed_intervals >= 119
    ):
        raise QuantError("Assimilate one genuinely complete positive NET source month.")
    score = float(state.weights @ relatives)
    if not math.isfinite(score) or score <= 0:
        raise QuantError("The prior learning score must remain finite and positive.")
    gradient = relatives / score
    curvature = state.curvature + np.outer(gradient, gradient)
    linear = curvature @ state.weights + gradient
    weights, proof = project(curvature, linear)
    result = AllocationLearningState(weights, curvature, state.completed_intervals + 1)
    result.validate()
    return result, {
        "learner_weights_before": state.weights.tolist(),
        "source_NET_period_relatives": relatives.tolist(),
        "pre_update_learning_score_relative": score,
        "gradient": gradient.tolist(),
        "curvature_before": state.curvature.tolist(),
        "curvature_after": curvature.tolist(),
        "learner_weights_after": weights.tolist(),
        "complete_learning_intervals": result.completed_intervals,
        "persistent_array_buffer_bytes": weights.nbytes + curvature.nbytes,
        "buffer_bytes_are_total_process_RSS": False,
        "learning_score_is_actual_main_net_wealth": False,
        **proof,
    }


def observations(experts: tuple[StrategyExpert, ...], day: pd.Timestamp) -> list[ExpertObservation]:
    result = [e.observe(day, 126) for e in experts]
    if (
        len(result) != 6
        or tuple(o.identity.identifier for o in result) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in result) != MECHANISMS
        or any(
            o.as_of != day
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > day
            or not o.net_returns.index.equals(o.risk_free.index)
            or not np.isfinite(o.net_returns).all()
            or not np.isfinite(o.risk_free).all()
            or (o.net_returns <= -1).any()
            or len(o.net_returns) > 126
            or o.score_mature != (len(o.net_returns) == 126)
            or len(o.net_returns)
            and (
                o.net_returns.index[-1] != day
                or not o.net_returns.index.equals(
                    sessions(o.net_returns.index[0], o.net_returns.index[-1])
                )
            )
            for o in result
        )
        or any(
            not o.net_returns.index.equals(result[0].net_returns.index)
            or not o.risk_free.equals(result[0].risk_free)
            for o in result
        )
    ):
        raise QuantError(
            "Keep all six causal recommendations and identical matured source dates/RF."
        )
    return result


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
    state = AllocationLearningState.initial()
    previous_day = None
    previous_main = np.full(6, 1 / 6)
    mature_started = False
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("Every complete expert must retain the admitted asset lattice.")
        update = None
        partial = False
        if previous_day is not None:
            if is_month_end(previous_day):
                if day.year * 12 + day.month - previous_day.year * 12 - previous_day.month != 1:
                    raise QuantError(
                        "A complete source month cannot skip a decision or be assimilated twice."
                    )
                expected = sessions(next_session(previous_day), day)
                if any(not expected.isin(o.net_returns.index).all() for o in observed):
                    raise QuantError("A missing matured month cannot be filled or dropped.")
                relatives = period_relative(observed, previous_day, day)
                state, update = advance(state, relatives)
                update.update(
                    source_period_previous_decision=str(previous_day.date()),
                    source_period_first_session=str(expected[0].date()),
                    source_period_end=str(day.date()),
                    source_period_sessions=len(expected),
                    score_recorded_before_assimilation=True,
                )
            else:
                if position != 1:
                    raise QuantError("Only the real initial partial source month may be excluded.")
                partial = True
        mature = all(o.score_mature for o in observed)
        if mature_started and not mature:
            raise QuantError(
                "Post-maturity source history cannot silently become an unscored fallback."
            )
        mature_started = mature_started or mature
        applied = state.weights.copy() if mature else np.full(6, 1 / 6)
        targets.loc[day] = combine(observed, applied)
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
                "source_observation_counts": [len(o.net_returns) for o in observed],
                "previous_applied_expert_weights": previous_main.tolist(),
                "applied_expert_weights": applied.tolist(),
                "learner_expert_weights": state.weights.tolist(),
                "complete_learning_intervals": state.completed_intervals,
                "source_assessment_mature": mature,
                "reason": "online_NET_log_growth_weights"
                if mature
                else "unscored_initial_equal_weights",
                "initial_partial_source_month_not_assimilated": partial,
                "online_update": update,
                "source_shadow_wealth_or_protection_floor_transferred": False,
                "learning_score_is_actual_main_net_wealth": False,
                "universal_NET_wealth_identity_or_bound_claimed": False,
                "counterfactual_historical_expert_availability": True,
                "independent_forward_validation": False,
            }
        )
        previous_day, previous_main = day, applied
    return targets, audits
