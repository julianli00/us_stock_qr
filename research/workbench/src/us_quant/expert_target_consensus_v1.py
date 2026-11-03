from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v5 as source
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-target-consensus-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
CANDIDATES = (
    {"id": "expert_pool_target_medoid_consensus_v1", "weighting": "exact_target_L1_medoid"},
)
SOURCE_POLICY = {
    "path": "config/strategy-expert-pool-seven-v1.json",
    "sha256": "f27b1ac96253faef7ce57520682dac83ef426272057dfa2c277c61fcc319416f",
}


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_target_consensus_20261103_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "authorization_id": "november-research-20261103-v1",
        "prior_journal_sequence": 675,
        "prior_journal_head_sha256": (
            "30b7ead58cc386de0d25db3426150c78af535d0d7d53e17cd99d55ef2ca314db"
        ),
        "prior_new_registered_candidates": 64,
        "prior_new_completed_strategy_evaluations": 56,
        "prior_new_technical_failures": 8,
        "source_pool_policy": SOURCE_POLICY,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "decision_information_clock": CLOCK,
        "initial_decision_session": "2016-10-05",
        "source_observation_lookback_sessions": 126,
        "aggregation": "exact_binary64_L1_medoid_of_current_complete_recommendations",
        "include_residual_idle_cash_coordinate": True,
        "equal_vote_per_mechanism_representative": True,
        "tie_rule": "equal_capital_over_all_exact_minimizers",
        "return_ranking_or_score_maturity_drives_allocation": False,
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
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze the complete seven-expert exact target-consensus law.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Recommendation agreement is not qualification or trading authority.")


def _checked_targets(observed: list[ExpertObservation]) -> np.ndarray:
    if (
        len(observed) != len(EXPERT_IDS)
        or tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            not isinstance(o.target, pd.Series)
            or o.target.empty
            or not o.target.index.is_unique
            or o.target.dtype.kind not in "fi"
            or not np.isfinite(o.target).all()
            or (o.target < 0).any()
            or abs(float(o.target.sum()) - 0.98) > 1e-12
            for o in observed
        )
    ):
        raise QuantError("Consensus requires all seven valid complete expert recommendations.")
    if (
        any(not o.target.index.equals(observed[0].target.index) for o in observed)
        or any(o.as_of != observed[0].as_of for o in observed)
        or any(o.target_known_at is None or o.target_known_at > o.as_of for o in observed)
        or any(o.score_mature != (len(o.net_returns) == 126) for o in observed)
        or any(not o.net_returns.index.equals(observed[0].net_returns.index) for o in observed)
        or any(not o.risk_free.index.equals(o.net_returns.index) for o in observed)
        or any(not o.risk_free.equals(observed[0].risk_free) for o in observed)
        or any(len(o.net_returns) and o.net_returns.index[-1] != o.as_of for o in observed)
        or (
            len(observed[0].net_returns)
            and not observed[0].net_returns.index.equals(
                sessions(
                    observed[0].net_returns.index[0],
                    observed[0].net_returns.index[-1],
                )
            )
        )
        or any(
            not np.isfinite(o.net_returns).all()
            or (o.net_returns <= -1).any()
            or not np.isfinite(o.risk_free).all()
            for o in observed
        )
    ):
        raise QuantError("Keep the original complete source coordinates, clocks, NET and exact RF.")
    return np.stack([o.target.to_numpy() for o in observed])


def select(observed: list[ExpertObservation]) -> tuple[np.ndarray, dict]:
    matrix = _checked_targets(observed)
    rows = [[Fraction.from_float(float(value)) for value in row] for row in matrix]
    for row in rows:
        row.append(1 - sum(row))
    totals = [
        sum(
            sum(abs(left - right) for left, right in zip(row, other, strict=True)) for other in rows
        )
        for row in rows
    ]
    winners = [index for index, value in enumerate(totals) if value == min(totals)]
    weights = np.zeros(len(rows))
    weights[winners] = 1 / len(winners)
    return weights, {
        "reason": "exact_current_recommendation_L1_medoid",
        "objective_exact_fractions": [
            {"numerator": str(value.numerator), "denominator": str(value.denominator)}
            for value in totals
        ],
        "selected_expert_ids": [EXPERT_IDS[index] for index in winners],
        "equal_weight_over_all_exact_minimizers": True,
        "residual_idle_cash_coordinate_included": True,
        "return_scores_used": False,
        "source_score_mature": [o.score_mature for o in observed],
        "source_NET_or_RF_values_ranked_or_filled": False,
        "target_distance_is_actual_main_turnover_or_risk_forecast": False,
        "main_drawdown_guarantee": False,
    }


def independent_target(observed: list[ExpertObservation]) -> tuple[pd.Series, list[str]]:
    matrix = _checked_targets(observed)
    ratios = [[float(value).as_integer_ratio() for value in row] for row in matrix]
    denominator = max(bottom for row in ratios for _, bottom in row)
    integers = [[top * (denominator // bottom) for top, bottom in row] for row in ratios]
    for row in integers:
        row.append(denominator - sum(row))
    totals = [
        sum(
            abs(integers[i][column] - integers[j][column])
            for j in range(len(integers))
            for column in range(matrix.shape[1] + 1)
        )
        for i in range(len(integers))
    ]
    best = [index for index in range(len(integers)) if totals[index] == min(totals)]
    target = pd.Series(
        [
            float(Fraction(sum(integers[i][column] for i in best), denominator * len(best)))
            for column in range(matrix.shape[1])
        ],
        index=observed[0].target.index,
    )
    return target, [EXPERT_IDS[index] for index in best]


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision: Callable[[], None] | None = None,
    retain: Callable[[dict], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    validate(policy)
    if (
        file_digest(source.POLICY) != SOURCE_POLICY["sha256"]
        or len(experts) != len(EXPERT_IDS)
        or len(universe) != len(set(universe))
    ):
        raise QuantError("Preserve the exact source pool, full calendar and asset order.")
    dates = source.decision_sessions(calendar, read_json(source.POLICY))
    identifier = CANDIDATES[0]["id"]
    target = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits = []
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [expert.observe(day, 126) for expert in experts]
        inputs = {
            "record": "decision_input",
            "decision_index": position,
            "as_of": str(day.date()),
            "information_clock": CLOCK,
            "observations": [
                {
                    "id": o.identity.identifier,
                    "mechanism_group": o.identity.mechanism_group,
                    "introduced_at": o.identity.introduced_at,
                    "target_known_at": (
                        str(o.target_known_at.date()) if o.target_known_at is not None else None
                    ),
                    "asset_order": list(o.target.index) if o.target is not None else None,
                    "target": o.target.tolist() if o.target is not None else None,
                    "target_binary64_hex": (
                        [float(value).hex() for value in o.target] if o.target is not None else None
                    ),
                    "source_NET_observations": len(o.net_returns),
                    "source_score_mature": o.score_mature,
                }
                for o in observed
            ],
            "unused_source_NET_RF_values_are_not_inputs_to_the_allocation_operator": True,
        }
        if retain is not None:
            retain(inputs)
        weights, evidence = select(observed)
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("A complete consensus recommendation cannot silently remap assets.")
        combined = source.combine(observed, weights)
        target.loc[day] = combined
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if position == 0 else "completed_exchange_month_end",
            "decision_information_clock": CLOCK,
            "expert_target_known_at": [
                record["target_known_at"] for record in inputs["observations"]
            ],
            "actual_expert_introduction": [
                record["introduced_at"] for record in inputs["observations"]
            ],
            "expert_weights": weights.tolist(),
            "consensus": evidence,
            "counterfactual_historical_expert_availability": True,
            "shadow_wealth_added_to_main_capital": False,
            "source_feedback_floor_transferred_to_main_account": False,
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
                    "combined_target": combined.tolist(),
                }
            )
    return {identifier: target}, {identifier: audits}
