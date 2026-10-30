from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import expert_online_log_growth_v1 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-path-admissible-baskets-v1.json"
PRIOR_HEAD = "5ff52302a7b36693ba30b578613aa353fb0e65da18052a073ca4f7570968240e"
EXPERT_IDS, MECHANISMS = prior.EXPERT_IDS, prior.MECHANISMS
BENCHMARKS = ("SPY", "QQQ")
SEGMENTS = (("full_126", 0, 126), ("older_84", 0, 84), ("recent_42", 84, 126))
CANDIDATES = (
    {
        "id": "expert_pool_six_path_admissible_baskets_v1",
        "weighting": "ordered_NET_drawdown_admissible_equal_baskets",
    },
)
observations, combine = prior.observations, prior.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_path_baskets_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 223,
        "prior_new_registered_candidates": 49,
        "prior_new_completed_strategy_evaluations": 45,
        "prior_new_technical_failures": 4,
        "prior_journal_sequence": 540,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "enumerated_nonempty_subsets": 63,
        "subset_weighting": "equal_initial_capital_in_complete_expert_NET_reference_paths",
        "assessment_segments": [label for label, _, _ in SEGMENTS],
        "maximum_observed_source_drawdown": 0.15,
        "conservative_numeric_risk_margin": 1e-10,
        "selection_score_decimal_places": 10,
        "target_L1_decimal_places": 12,
        "score_positive_threshold_used": False,
        "independent_risk_and_score_tolerance": 1e-10,
        "independent_weight_and_target_tolerance": 1e-12,
        "funding_standby_symbol": "BIL",
        "funding_standby_is_strategy_expert": False,
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
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()
    ):
        raise QuantError("Freeze one chronological NET basket criterion and the original pool.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "assessment_paths_are_main_wealth_or_future_drawdown_guarantee",
            "source_fixed_fees_scale_exactly_to_main_expert_capital",
            "subset_cases_are_independent_trials_or_validation",
            "target_L1_is_actual_turnover_or_fee_forecast",
            "old_ordered_price_sleeve_LP_or_solver_invoked",
            "old_online_learning_rate_or_curvature_retuned",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Assessment admissibility is not main qualification or trading authority.")


def assess_baskets(
    returns: np.ndarray,
    benchmarks: np.ndarray,
    targets: np.ndarray,
    previous: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    if (
        returns.shape != (126, 6)
        or benchmarks.shape != (126, 2)
        or targets.ndim != 2
        or targets.shape[1] != 6
        or previous.shape != (targets.shape[0],)
        or not all(np.isfinite(a).all() for a in (returns, benchmarks, targets, previous))
        or (returns <= -1).any()
        or (benchmarks <= -1).any()
        or (targets < 0).any()
        or (previous < 0).any()
        or float(previous.sum()) > 0.98 + 1e-12
        or not np.allclose(targets.sum(axis=0), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Use exact mature NET samples and funded recommendation coordinates.")
    paths = {}
    benchmark_growth = {}
    for label, start, stop in SEGMENTS:
        relative = np.vstack([np.ones(6), np.cumprod(1 + returns[start:stop], axis=0)])
        if not np.isfinite(relative).all() or (relative <= 0).any():
            raise QuantError("A source reference path must stay finite and strictly positive.")
        paths[label] = relative
        if label != "full_126":
            gross = np.prod(1 + benchmarks[start:stop], axis=0)
            if not np.isfinite(gross).all() or (gross <= 0).any():
                raise QuantError("Keep the original positive NET benchmark reference paths.")
            benchmark_growth[label] = np.log(gross) * 252 / (stop - start)
    cards = []
    for mask in range(1, 64):
        chosen = tuple(i for i in range(6) if mask & (1 << i))
        weights = np.zeros(6)
        weights[list(chosen)] = 1 / len(chosen)
        risks, scores = {}, {}
        for label, start, stop in SEGMENTS:
            wealth = paths[label][:, list(chosen)].mean(axis=1)
            risks[label] = float((1 - wealth / np.maximum.accumulate(wealth)).max())
            if label != "full_126":
                growth = math.log(float(wealth[-1])) * 252 / (stop - start)
                for j, symbol in enumerate(BENCHMARKS):
                    scores[f"{label}:{symbol}"] = float(growth - benchmark_growth[label][j])
        score = min(scores.values())
        distance = float(abs(targets @ weights - previous).sum())
        cards.append(
            {
                "mask": mask,
                "selected_indices": list(chosen),
                "expert_count": len(chosen),
                "assessment_drawdowns": risks,
                "annualized_NET_log_growth_differences": scores,
                "minimum_active_growth": score,
                "rank_score": round(score, 10),
                "nominal_target_L1_distance": distance,
                "rank_target_L1": round(distance, 12),
                "assessment_admissible": all(r <= 0.15 - 1e-10 for r in risks.values()),
            }
        )
    admissible = [card for card in cards if card["assessment_admissible"]]
    selected = (
        min(
            admissible,
            key=lambda c: (
                -c["rank_score"],
                c["rank_target_L1"],
                -c["expert_count"],
                c["selected_indices"],
            ),
        )
        if admissible
        else None
    )
    weights = None
    if selected is not None:
        weights = np.zeros(6)
        weights[selected["selected_indices"]] = 1 / selected["expert_count"]
    return weights, {
        "enumerated_nonempty_subsets": 63,
        "assessment_admissible_subsets": len(admissible),
        "selected_mask": selected["mask"] if selected is not None else None,
        "selected_expert_indices": selected["selected_indices"] if selected is not None else [],
        "selected_minimum_active_growth": selected["minimum_active_growth"] if selected else None,
        "funding_standby": selected is None,
        "reason": "ordered_NET_drawdown_admissible_robust_active_growth"
        if selected is not None
        else "no_observed_path_admissible_subset",
        "all_subset_assessments": cards,
        "assessment_paths_are_actual_main_or_future_drawdown_guarantee": False,
        "subset_cases_are_independent_trials_or_validation": False,
        "target_L1_is_actual_turnover_or_fee_forecast": False,
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
    benchmarks: pd.DataFrame,
    policy: dict,
    before_decision=None,
) -> tuple[pd.DataFrame, list[dict]]:
    dates = decision_sessions(calendar, policy)
    if (
        len(experts) != 6
        or len(universe) != len(set(universe))
        or "BIL" not in universe
        or tuple(benchmarks.columns) != BENCHMARKS
        or benchmarks.empty
        or not benchmarks.index.equals(sessions(benchmarks.index[0], benchmarks.index[-1]))
    ):
        raise QuantError("Retain the six complete experts and original tradable benchmark clock.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    previous = np.zeros(len(universe))
    mature_started = False
    audits = []
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("Complete recommendations cannot change the admitted asset lattice.")
        benchmark_sample = benchmarks.loc[:day].tail(126)
        if not np.isfinite(benchmark_sample).all().all() or (benchmark_sample <= -1).any().any():
            raise QuantError("Invalid benchmark observations cannot become neutral scores.")
        mature = all(o.score_mature for o in observed) and len(benchmark_sample) == 126
        if mature_started and not mature:
            raise QuantError("Post-maturity missing history cannot silently become warmup.")
        mature_started = mature_started or mature
        if mature:
            if any(not o.net_returns.index.equals(benchmark_sample.index) for o in observed):
                raise QuantError(
                    "Every mature expert and benchmark assessment needs identical dates."
                )
            weights, assessment = assess_baskets(
                np.column_stack([o.net_returns.to_numpy() for o in observed]),
                benchmark_sample.to_numpy(),
                np.column_stack([o.target.to_numpy() for o in observed]),
                previous,
            )
        else:
            weights = np.full(6, 1 / 6)
            assessment = {
                "enumerated_nonempty_subsets": 0,
                "assessment_admissible_subsets": None,
                "selected_mask": None,
                "selected_expert_indices": list(range(6)),
                "selected_minimum_active_growth": None,
                "funding_standby": False,
                "reason": "unscored_initial_equal_weights",
                "all_subset_assessments": [],
                "safety_or_success_claimed": False,
            }
        if weights is None:
            desired = pd.Series(0.0, index=universe)
            desired["BIL"] = 0.98
        else:
            desired = combine(observed, weights)
        targets.loc[day] = desired
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
                "source_observation_counts": [len(o.net_returns) for o in observed],
                "source_observation_start": str(observed[0].net_returns.index[0].date())
                if len(observed[0].net_returns)
                else None,
                "source_observation_end": str(observed[0].net_returns.index[-1].date())
                if len(observed[0].net_returns)
                else None,
                "benchmark_observations": len(benchmark_sample),
                "source_assessment_mature": mature,
                "previous_nominal_meta_target": previous.tolist(),
                "expert_weights": weights.tolist() if weights is not None else None,
                "assessment": assessment,
                "source_shadow_wealth_or_protection_floor_transferred": False,
                "assessment_admissibility_is_final_qualification": False,
                "counterfactual_historical_expert_availability": True,
                "independent_forward_validation": False,
            }
        )
        previous = desired.to_numpy().copy()
    return targets, audits
