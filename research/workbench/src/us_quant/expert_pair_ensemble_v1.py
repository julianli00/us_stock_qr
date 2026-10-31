from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v5 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_pair_dominance_v1 import observation_record
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-pair-ensemble-v1.json"
PRIOR_HEAD = "2a3d72cac1a58f9de394a5e0975902f289951e1c2fd65e211c6d5cf48e3daadf"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
CANDIDATES = (
    {
        "id": "expert_pool_seven_uniform_pair_ensemble_v1",
        "aggregation": "uniform_scorable_pair_ERC",
    },
    {
        "id": "expert_pool_seven_rank_pair_ensemble_v1",
        "aggregation": "tie_averaged_ordinal_pair_ERC",
    },
)
SCHEMES = tuple(candidate["aggregation"] for candidate in CANDIDATES)
combine = source.combine


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_pair_allocation_ensemble_20261031_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 232,
        "prior_new_registered_candidates": 62,
        "prior_new_completed_strategy_evaluations": 54,
        "prior_new_technical_failures": 8,
        "prior_journal_sequence": 652,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "decision_information_clock": CLOCK,
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "selected_experts_field_describes_original_pair_size_not_ensemble_support": True,
        "ensemble_support": "every_causally_scorable_original_pair",
        "selection_score_decimal_places": 12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "switch_cost_amortization_sessions": 21,
        "changed_asset_numeric_tolerance": 1e-12,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze full pair support with unchanged assessment,clock and execution.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Pair aggregation is not independent experts or final qualification.")


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the real initial anchor and full original source calendar.")
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def select(observed: list[ExpertObservation], previous: np.ndarray, scheme: str):
    if (
        scheme not in SCHEMES
        or len(observed) != 7
        or tuple(item.identity.identifier for item in observed) != EXPERT_IDS
        or tuple(item.identity.mechanism_group for item in observed) != MECHANISMS
    ):
        raise QuantError("Use one pre-frozen positive full-support pair aggregation law.")
    instant, assessment = source.choose_pair(observed, previous)
    evidence = {
        "aggregation": scheme,
        "pair_assessments": assessment,
        "pair_masses": [],
        "overlapping_pairs_are_independent_experts_or_capital": False,
        "confidence_interval_or_significance_claimed": False,
    }
    if not assessment["scores_available"]:
        evidence.update(
            reason=assessment["reason"],
            scorable_pair_count=0,
            aggregation_point_total=None,
            chosen_expert_weights=instant.tolist(),
        )
        return instant, evidence
    scored = sorted(
        (pair for pair in assessment["pairs"] if "unscorable_reason" not in pair),
        key=lambda pair: (-pair["selection_score_key"], pair["pair"]),
    )
    m = len(scored)
    if not m:
        raise QuantError("A successful original pair assessment must contain scorable opinions.")
    points = np.ones(m)
    if scheme == SCHEMES[1]:
        start = 0
        while start < m:
            end = start + 1
            while end < m and (
                scored[end]["selection_score_key"] == scored[start]["selection_score_key"]
            ):
                end += 1
            points[start:end] = m - (start + end - 1) / 2
            start = end
    total = float(points.sum())
    masses = points / total
    weights = masses @ np.asarray([pair["expert_weights"] for pair in scored])
    if (
        weights.shape != (len(observed),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
        or not np.isfinite(masses).all()
        or (masses <= 0).any()
        or abs(float(masses.sum()) - 1) > 1e-12
        or total != (m if scheme == SCHEMES[0] else m * (m + 1) / 2)
    ):
        raise QuantError("Each scorable pair needs positive mass and one funded aggregate vector.")
    evidence.update(
        reason="full_support_pair_ERC_aggregation",
        scorable_pair_count=m,
        aggregation_point_total=total,
        pair_masses=[
            {
                "pair": pair["pair"],
                "selection_score_key": pair["selection_score_key"],
                "aggregation_points": float(point),
                "pair_mass": float(mass),
            }
            for pair, point, mass in zip(scored, points, masses, strict=True)
        ],
        chosen_expert_weights=weights.tolist(),
    )
    return weights, evidence


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    retain=None,
):
    dates = decision_sessions(calendar, policy)
    if len(experts) != 7 or len(set(universe)) != len(universe):
        raise QuantError("Use the fixed complete seven experts and unique asset coordinates.")
    targets, audits = {}, {}
    for candidate in CANDIDATES:
        name, scheme = candidate["id"], candidate["aggregation"]
        targets[name] = pd.DataFrame(np.nan, index=calendar, columns=universe)
        audits[name] = []
        previous = np.full(7, 1 / 7)
        for position, day in enumerate(dates):
            if before_decision is not None:
                before_decision()
            observed = [expert.observe(day, 126) for expert in experts]
            if retain is not None:
                retain(
                    {
                        "record": "decision_input",
                        "candidate_id": name,
                        "decision_index": position,
                        "as_of": str(day.date()),
                        "information_clock": CLOCK,
                        "previous_weights": previous.tolist(),
                        "previous_weights_binary64_hex": [float(x).hex() for x in previous],
                        "observations": [observation_record(item) for item in observed],
                        "binary64_values_truncated_to_fixed_decimals": False,
                    }
                )
            if (
                tuple(item.identity.identifier for item in observed) != EXPERT_IDS
                or tuple(item.identity.mechanism_group for item in observed) != MECHANISMS
                or any(item.target is None for item in observed)
                or any(tuple(item.target.index) != universe for item in observed)
                or position > 0
                and any(not len(item.net_returns) for item in observed)
                or position < 7
                and all(item.score_mature for item in observed)
                or position >= 7
                and any(not item.score_mature for item in observed)
            ):
                raise QuantError("Missing original expert observations cannot become defaults.")
            weights, evidence = select(observed, previous, scheme)
            targets[name].loc[day] = combine(observed, weights)
            audit = {
                "decision_session": str(day.date()),
                "decision_index": position,
                "decision_role": "initial_capital"
                if position == 0
                else "completed_exchange_month_end",
                "decision_information_clock": CLOCK,
                "expert_target_known_at": [str(item.target_known_at.date()) for item in observed],
                "preserved_archive_identity_introduced_at": [
                    item.identity.introduced_at for item in observed
                ],
                "expert_shadow_last_observation": [
                    str(item.net_returns.index[-1].date()) if len(item.net_returns) else None
                    for item in observed
                ],
                "previous_expert_weights": previous.tolist(),
                "expert_weights": weights.tolist(),
                "pair_ensemble": evidence,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
            audits[name].append(audit)
            if retain is not None:
                retain(
                    {
                        "record": "decision_resolved",
                        "candidate_id": name,
                        "decision_index": position,
                        "audit": audit,
                        "asset_order": list(universe),
                        "underlying_target": targets[name].loc[day].tolist(),
                    }
                )
            previous = weights
    return targets, audits
