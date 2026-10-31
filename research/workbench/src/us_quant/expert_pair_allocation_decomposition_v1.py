from __future__ import annotations

import math

from us_quant import strategy_expert_pool_v5 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_pair_menu_extension_v1 import _pairs, _stats, _weights

EXPERT_IDS, CLOCK = source.EXPERT_IDS, source.CLOCK
UNIFORM = "expert_pool_seven_uniform_pair_ensemble_v1"
ORDINAL = "expert_pool_seven_rank_pair_ensemble_v1"
SCHEMES = ("uniform_scorable_pair_ERC", "tie_averaged_ordinal_pair_ERC")
NEUTRAL = [1 / 7] * 7


def _masses(evidence, pairs, scheme):
    scored = {names for names, row in pairs.items() if "unscorable_reason" not in row}
    records = evidence["pair_masses"]
    result = {}
    if (
        not scored
        or evidence["reason"] != "full_support_pair_ERC_aggregation"
        or evidence["scorable_pair_count"] != len(scored)
        or len(records) != len(scored)
        or evidence["aggregation_point_total"]
        != (len(scored) if scheme == SCHEMES[0] else len(scored) * (len(scored) + 1) / 2)
    ):
        raise QuantError("Preserve every recorded scorable pair mass and declared source scheme.")
    for row in records:
        names, mass, points = tuple(row["pair"]), row["pair_mass"], row["aggregation_points"]
        if (
            names not in scored
            or names in result
            or type(mass) not in (int, float)
            or type(points) not in (int, float)
            or not math.isfinite(mass)
            or not math.isfinite(points)
            or mass <= 0
            or points <= 0
            or row["selection_score_key"] != pairs[names]["selection_score_key"]
            or abs(mass - points / evidence["aggregation_point_total"]) > 1e-12
            or scheme == SCHEMES[0]
            and (points != 1 or abs(mass - 1 / len(scored)) > 1e-12)
        ):
            raise QuantError("Recorded pair mass is not finite positive original allocation data.")
        result[names] = float(mass)
    if set(result) != scored or abs(math.fsum(result.values()) - 1) > 1e-12:
        raise QuantError("Keep a complete unit-mass recorded allocation menu.")
    return result


def _allocation_summary(rows):
    concentration = [math.fsum(value * value for value in row) for row in rows]
    effective = [1 / value for value in concentration]
    return {
        "observations": len(rows),
        "per_expert": [
            {
                "expert_id": name,
                "recorded_assessment_fraction": _stats([row[i] for row in rows]),
                "strict_above_neutral_records": sum(row[i] > NEUTRAL[i] for row in rows),
                "strict_below_neutral_records": sum(row[i] < NEUTRAL[i] for row in rows),
                "exact_neutral_records": sum(row[i] == NEUTRAL[i] for row in rows),
            }
            for i, name in enumerate(EXPERT_IDS)
        ],
        "recorded_allocation_HHI": _stats(concentration),
        "effective_allocation_number": _stats(effective),
        "effective_number_is_independent_factor_or_expert_count": False,
        "neutral_comparisons_are_strict_binary64_not_significance": True,
    }


def compare(uniform, ordinal, retain=None):
    calendar = sessions("2016-10-05", "2026-10-05")
    expected = [
        "2016-10-05",
        *[str(day.date()) for day in calendar if day > calendar[0] and is_month_end(day)],
    ]
    if (
        len(uniform) != 121
        or len(ordinal) != 121
        or [row["decision_session"] for row in uniform] != expected
        or [row["decision_session"] for row in ordinal] != expected
    ):
        raise QuantError("Compare all original121initial/month-end source allocations.")
    previous_u, previous_o = NEUTRAL, NEUTRAL
    all_u, all_o, mature_u, mature_o = [], [], [], []
    mass_terms, risk_terms, differences, l1 = [], [], [], []
    risk_weight, risk_correlation, risk_variance = [], [], []
    reconstruction = identity = 0.0
    unscored = mature = matched_pairs = 0
    for position, (u, o) in enumerate(zip(uniform, ordinal, strict=True)):
        if retain is not None:
            retain(
                {
                    "record": "matched_source_input",
                    "decision_index": position,
                    "uniform": u,
                    "ordinal": o,
                    "new_diagnostic_arithmetic_not_source_replay": True,
                }
            )
        day = u["decision_session"]
        current_u = _weights(u["expert_weights"], 7).tolist()
        current_o = _weights(o["expert_weights"], 7).tolist()
        if (
            u["decision_index"] != position
            or o["decision_index"] != position
            or u["previous_expert_weights"] != previous_u
            or o["previous_expert_weights"] != previous_o
            or any(
                row["decision_information_clock"] != CLOCK
                or row["decision_role"]
                != ("initial_capital" if position == 0 else "completed_exchange_month_end")
                for row in (u, o)
            )
            or u["expert_target_known_at"] != o["expert_target_known_at"]
            or u["preserved_archive_identity_introduced_at"]
            != o["preserved_archive_identity_introduced_at"]
            or u["expert_shadow_last_observation"] != o["expert_shadow_last_observation"]
            or len(u["expert_target_known_at"]) != 7
            or any(value > day for value in u["expert_target_known_at"])
            or any(
                row[key] is not value
                for row in (u, o)
                for key, value in (
                    ("counterfactual_historical_expert_availability", True),
                    ("shadow_wealth_added_to_main_capital", False),
                    ("source_feedback_floor_transferred_to_main_account", False),
                    ("independent_forward_validation", False),
                )
            )
        ):
            raise QuantError("Keep source state chains,information clocks and capital separation.")
        eu, eo = u["pair_ensemble"], o["pair_ensemble"]
        for evidence, scheme in zip((eu, eo), SCHEMES, strict=True):
            if (
                evidence["aggregation"] != scheme
                or evidence["confidence_interval_or_significance_claimed"] is not False
                or evidence["overlapping_pairs_are_independent_experts_or_capital"] is not False
                or evidence["chosen_expert_weights"]
                != (current_u if scheme == SCHEMES[0] else current_o)
            ):
                raise QuantError(
                    "Recorded allocations are not confidence or new independent capital."
                )
        au, ao = eu["pair_assessments"], eo["pair_assessments"]
        if position < 7:
            if any(
                assessment["scores_available"] is not False
                or assessment["pairs"] != []
                or assessment["selected_pair"] is not None
                or assessment["reason"] != "unscored_initial_equal_weights"
                or assessment["safety_or_success_claimed"] is not False
                for assessment in (au, ao)
            ) or any(
                evidence["pair_masses"] != []
                or evidence["scorable_pair_count"] != 0
                or evidence["aggregation_point_total"] is not None
                for evidence in (eu, eo)
            ):
                raise QuantError("Unmatured records cannot acquire successful pair statistics.")
            if current_u != NEUTRAL or current_o != NEUTRAL:
                raise QuantError("Preserve the actual original neutral initial source states.")
            mass, risk, delta = [0.0] * 7, [0.0] * 7, [0.0] * 7
            pair_rows = []
            unscored += 1
        else:
            clocks = ("train_start", "train_end", "validation_start", "validation_end")
            if (
                au["scores_available"] is not True
                or ao["scores_available"] is not True
                or au["safety_or_success_claimed"] is not False
                or ao["safety_or_success_claimed"] is not False
                or any(au[key] != ao[key] for key in clocks)
                or not (
                    au["train_start"]
                    <= au["train_end"]
                    < au["validation_start"]
                    <= au["validation_end"]
                    <= day
                )
            ):
                raise QuantError("Keep every recorded mature matched past-only risk clock.")
            pu, po = _pairs(au, EXPERT_IDS), _pairs(ao, EXPERT_IDS)
            qu, qo = _masses(eu, pu, SCHEMES[0]), _masses(eo, po, SCHEMES[1])
            pair_rows = []
            for names in pu:
                bu, bo = pu[names], po[names]
                if bu.get("unscorable_reason") != bo.get("unscorable_reason"):
                    raise QuantError(
                        "Preserve matched original unscorable reasons,not rescue them."
                    )
                pair = {"pair": list(names), "unscorable_reason": bu.get("unscorable_reason")}
                if "unscorable_reason" not in bu:
                    weight = max(
                        abs(x - y)
                        for x, y in zip(bu["expert_weights"], bo["expert_weights"], strict=True)
                    )
                    correlation = abs(bu["training_correlation"] - bo["training_correlation"])
                    variance = max(
                        abs(x - y)
                        for x, y in zip(
                            bu["training_marginal_variance_contributions"],
                            bo["training_marginal_variance_contributions"],
                            strict=True,
                        )
                    )
                    if weight > 1e-10 or correlation > 1e-10 or variance > 1e-12:
                        raise QuantError(
                            "Recorded pair risk identity exceeds original fixed limits."
                        )
                    risk_weight.append(weight)
                    risk_correlation.append(correlation)
                    risk_variance.append(variance)
                    pair.update(
                        uniform_mass=qu[names],
                        ordinal_mass=qo[names],
                        uniform_member_weights=bu["expert_weights"],
                        ordinal_member_weights=bo["expert_weights"],
                        maximum_member_weight_difference=weight,
                        training_correlation_difference=correlation,
                        marginal_variance_difference=variance,
                    )
                pair_rows.append(pair)
            if set(qu) != set(qo):
                raise QuantError("Match the same original scorable allocation support.")
            for weights, pairs, masses in (
                (current_u, pu, qu),
                (current_o, po, qo),
            ):
                for i in range(7):
                    total = math.fsum(
                        value * float(pairs[names]["expert_weights"][i])
                        for names, value in masses.items()
                    )
                    reconstruction = max(reconstruction, abs(total - weights[i]))
            mass = [
                math.fsum(
                    (qo[names] - qu[names]) * float(pu[names]["expert_weights"][i]) for names in qu
                )
                for i in range(7)
            ]
            risk = [
                math.fsum(
                    qo[names]
                    * (
                        float(po[names]["expert_weights"][i])
                        - float(pu[names]["expert_weights"][i])
                    )
                    for names in qu
                )
                for i in range(7)
            ]
            delta = [a - b for a, b in zip(current_o, current_u, strict=True)]
            identity = max(
                identity,
                max(
                    abs(value - math.fsum((a, b)))
                    for value, a, b in zip(delta, mass, risk, strict=True)
                ),
            )
            matched_pairs += len(pair_rows)
            mature += 1
            mature_u.append(current_u)
            mature_o.append(current_o)
            mass_terms.append(mass)
            risk_terms.append(risk)
            differences.append(delta)
            l1.append(math.fsum(abs(value) for value in delta))
        all_u.append(current_u)
        all_o.append(current_o)
        row = {
            "decision_session": day,
            "decision_index": position,
            "mature_recorded_assessment": position >= 7,
            "uniform_recorded_expert_weights": current_u,
            "ordinal_recorded_expert_weights": current_o,
            "ordinal_minus_uniform": delta,
            "uniform_anchored_pair_mass_term": mass,
            "ordinal_mass_pair_risk_difference_term": risk,
            "all_matched_pair_records": pair_rows,
            "new_diagnostic_arithmetic_not_source_memory": True,
            "main_account_PNL_or_causal_effect": False,
        }
        if retain is not None:
            retain(
                {
                    "record": "resolved_allocation_decomposition",
                    "decision_index": position,
                    "decomposition": row,
                }
            )
        previous_u, previous_o = current_u, current_o
    if unscored != 7 or mature != 114 or matched_pairs != 2394:
        raise QuantError("Retain all initial,mature and original pair comparisons,not a subset.")
    if reconstruction > 1e-10 or identity > 1e-10:
        raise QuantError(
            "The fixed stored-allocation diagnostic identity could not be established."
        )
    return {
        "matched_decisions": 121,
        "unscored_initial_decisions": unscored,
        "mature_recorded_allocations": mature,
        "all_original_named_pair_comparisons": matched_pairs,
        "maximum_source_allocation_sum_residual": reconstruction,
        "maximum_uniform_anchored_decomposition_residual": identity,
        "paired_recorded_risk_weight_differences": _stats(risk_weight),
        "paired_recorded_training_correlation_differences": _stats(risk_correlation),
        "paired_recorded_marginal_variance_differences": _stats(risk_variance),
        "all_source_allocations": {
            "uniform": _allocation_summary(all_u),
            "ordinal": _allocation_summary(all_o),
        },
        "mature_source_allocations": {
            "uniform": _allocation_summary(mature_u),
            "ordinal": _allocation_summary(mature_o),
        },
        "mature_ordinal_minus_uniform_L1": _stats(l1),
        "mature_per_expert_decomposition": [
            {
                "expert_id": name,
                "recorded_fraction_difference": _stats([row[i] for row in differences]),
                "uniform_anchored_mass_term": _stats([row[i] for row in mass_terms]),
                "recorded_risk_difference_term": _stats([row[i] for row in risk_terms]),
            }
            for i, name in enumerate(EXPERT_IDS)
        ],
        "neutral_initial_states_are_successful_risk_assessments": False,
        "HHI_effective_number_is_independent_factor_count": False,
        "decomposition_is_uniform_anchored_not_unique_causal_attribution": True,
        "new_diagnostic_arithmetic_not_original_source_memory": True,
        "original_scores_risks_ranks_targets_or_accounts_regenerated": False,
        "main_account_PNL_fees_CAGR_or_drawdown_counterfactual_computed": False,
        "new_financial_configuration_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
