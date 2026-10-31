from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v5 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-pair-dominance-v1.json"
PRIOR_HEAD = "c433aaa787972866e919c5452307f1b3552749777ac2c3bf7f1bf84e73318293"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
CANDIDATES = (
    {
        "id": "expert_pool_seven_pareto_pair_commitment_v1",
        "switch_gate": "strict_train_and_validation_dominance",
    },
    {
        "id": "expert_pool_seven_envelope_pair_commitment_v1",
        "switch_gate": "challenger_lower_above_incumbent_upper",
    },
)
GATES = tuple(candidate["switch_gate"] for candidate in CANDIDATES)
combine = source.combine


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_pair_assessment_dominance_20261031_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 230,
        "prior_new_registered_candidates": 60,
        "prior_new_completed_strategy_evaluations": 52,
        "prior_new_technical_failures": 8,
        "prior_journal_sequence": 629,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "decision_information_clock": CLOCK,
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
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
        raise QuantError(
            "Freeze strict empirical gates with unchanged source,assessment and execution."
        )
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Empirical dominance is not statistical confidence or trading authority.")


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
        raise QuantError("Keep the real initial anchor and complete original source calendar.")
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def select(
    observed: list[ExpertObservation],
    previous: np.ndarray,
    incumbent: tuple[str, str] | None,
    gate: str,
):
    if gate not in GATES or (
        incumbent is not None
        and (
            len(incumbent) != 2
            or tuple(sorted(incumbent)) != incumbent
            or len(set(incumbent)) != 2
            or not set(incumbent).issubset(EXPERT_IDS)
        )
    ):
        raise QuantError("Use one fixed gate and a genuine admitted incumbent pair.")
    instant, assessment = source.choose_pair(observed, previous)
    evidence = {
        "gate": gate,
        "incumbent_pair_before": list(incumbent) if incumbent else None,
        "pair_assessments": assessment,
        "qualified_challengers": [],
        "dominance_comparisons": [],
        "confidence_interval_or_significance_claimed": False,
    }
    if not assessment["scores_available"]:
        evidence.update(
            reason=assessment["reason"],
            incumbent_pair_after=None,
            chosen_expert_weights=instant.tolist(),
        )
        return instant, None, evidence
    scored = {
        tuple(pair["pair"]): pair for pair in assessment["pairs"] if "unscorable_reason" not in pair
    }
    best = tuple(assessment["selected_pair"])
    if incumbent is None or incumbent not in scored:
        reason = (
            "no_incumbent_select_original_best"
            if incumbent is None
            else "incumbent_unscorable_select_original_best"
        )
        chosen = best
    else:
        current = scored[incumbent]
        current_train = round(current["train_net_risk_score"], 12)
        current_validation = round(current["validation_net_risk_score"], 12)
        current_upper = round(
            max(current["train_net_risk_score"], current["validation_net_risk_score"]), 12
        )
        eligible = []
        for pair in assessment["pairs"]:
            names = tuple(pair["pair"])
            if "unscorable_reason" in pair:
                evidence["dominance_comparisons"].append(
                    {
                        "pair": list(names),
                        "unscorable_reason": pair["unscorable_reason"],
                        "qualifies": None,
                    }
                )
                continue
            train, validation = (
                round(pair["train_net_risk_score"], 12),
                round(pair["validation_net_risk_score"], 12),
            )
            lower = pair["selection_score_key"]
            qualifies = (
                train > current_train and validation > current_validation
                if gate == GATES[0]
                else lower > current_upper
            )
            evidence["dominance_comparisons"].append(
                {
                    "pair": list(names),
                    "challenger_train_key": train,
                    "challenger_validation_key": validation,
                    "challenger_lower_key": lower,
                    "incumbent_train_key": current_train,
                    "incumbent_validation_key": current_validation,
                    "incumbent_upper_key": current_upper,
                    "qualifies": qualifies,
                }
            )
            if qualifies:
                eligible.append(names)
        eligible.sort(key=lambda names: (-scored[names]["selection_score_key"], names))
        evidence["qualified_challengers"] = [list(names) for names in eligible]
        chosen = eligible[0] if eligible else incumbent
        reason = "dominant_challenger_selected" if eligible else "no_dominance_retain_incumbent"
    weights = np.asarray(scored[chosen]["expert_weights"])
    evidence.update(
        reason=reason,
        incumbent_pair_after=list(chosen),
        chosen_expert_weights=weights.tolist(),
    )
    return weights, chosen, evidence


def observation_record(item):
    return {
        "id": item.identity.identifier,
        "mechanism_group": item.identity.mechanism_group,
        "introduced_at": item.identity.introduced_at,
        "as_of": str(item.as_of.date()),
        "target_known_at": str(item.target_known_at.date())
        if item.target_known_at is not None
        else None,
        "target": item.target.tolist() if item.target is not None else None,
        "asset_order": list(item.target.index) if item.target is not None else None,
        "return_dates": item.net_returns.index.strftime("%Y-%m-%d").tolist(),
        "NET_returns": item.net_returns.tolist(),
        "RF": item.risk_free.tolist(),
        "score_mature": item.score_mature,
    }


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
        raise QuantError("Use the fixed seven complete experts and unique asset coordinates.")
    targets, audits = {}, {}
    for candidate in CANDIDATES:
        name, gate = candidate["id"], candidate["switch_gate"]
        targets[name] = pd.DataFrame(np.nan, index=calendar, columns=universe)
        audits[name] = []
        previous, incumbent = np.full(7, 1 / 7), None
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
                        "previous_weights_binary64_hex": [float(value).hex() for value in previous],
                        "incumbent_pair": list(incumbent) if incumbent else None,
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
                raise QuantError("Missing post-start expert observations cannot become defaults.")
            weights, chosen, evidence = select(observed, previous, incumbent, gate)
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
                "switch_evidence": evidence,
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
            previous, incumbent = weights, chosen
    return targets, audits
