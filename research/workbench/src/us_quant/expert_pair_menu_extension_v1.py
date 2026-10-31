from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v5 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

SIX, SEVEN = source.EXPERT_IDS[:6], source.EXPERT_IDS
VVIX, CLOCK = SEVEN[-1], source.CLOCK


def _weights(values, n):
    result = np.asarray(values, dtype=float)
    if (
        result.shape != (n,)
        or not np.isfinite(result).all()
        or (result < 0).any()
        or abs(float(result.sum()) - 1) > 1e-12
    ):
        raise QuantError("Stored assessment weights need their complete funded coordinate lattice.")
    return result


def _pairs(selection, ids):
    expected = {tuple(sorted(names)) for names in itertools.combinations(ids, 2)}
    result = {}
    for pair in selection["pairs"]:
        names = tuple(pair["pair"])
        if names not in expected or names in result:
            raise QuantError("A stored menu needs every unique named pair,not a selected subset.")
        if "unscorable_reason" in pair:
            if not isinstance(pair["unscorable_reason"], str) or not pair["unscorable_reason"]:
                raise QuantError("An unscorable assessment needs its actual recorded reason.")
        else:
            weights = _weights(pair["expert_weights"], len(ids))
            contributions = np.asarray(
                pair["training_marginal_variance_contributions"], dtype=float
            )
            if (
                contributions.shape != weights.shape
                or not np.isfinite(contributions).all()
                or any(
                    weights[i] != 0 or contributions[i] != 0
                    for i, name in enumerate(ids)
                    if name not in names
                )
                or not np.isfinite(
                    [
                        pair[key]
                        for key in (
                            "train_net_risk_score",
                            "validation_net_risk_score",
                            "selection_score",
                            "selection_score_key",
                            "training_correlation",
                            "estimated_incremental_switch_cost_fraction",
                        )
                    ]
                ).all()
                or pair["estimated_incremental_switch_cost_fraction"] < 0
                or type(pair["estimated_incremental_changed_assets"]) is not int
                or pair["estimated_incremental_changed_assets"] < 0
                or pair["selection_score"]
                != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
                or pair["selection_score_key"] != round(pair["selection_score"], 12)
                or pair["cost_proxy_is_actual_main_account_cost"] is not False
            ):
                raise QuantError(
                    "Recorded pair risk,net keys or switch proxies cannot be invented."
                )
        result[names] = pair
    if set(result) != expected:
        raise QuantError("A stored menu needs all original shared and added pairs.")
    return result


def _rank(pairs):
    return sorted(
        (names for names, row in pairs.items() if "unscorable_reason" not in row),
        key=lambda names: (-pairs[names]["selection_score_key"], names),
    )


def _stats(values):
    if not values:
        return {"observations": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "observations": len(values),
        "minimum": min(values),
        "mean": math.fsum(values) / len(values),
        "maximum": max(values),
    }


def compare(six, seven, retain=None):
    calendar = sessions("2016-10-05", "2026-10-05")
    expected = [
        "2016-10-05",
        *[str(day.date()) for day in calendar if day > calendar[0] and is_month_end(day)],
    ]
    if (
        len(six) != 121
        or len(seven) != 121
        or [row["decision_session"] for row in six] != expected
        or [row["decision_session"] for row in seven] != expected
    ):
        raise QuantError(
            "Match every actual original initial/month-end decision before comparison."
        )
    counts = {
        "added_VVIX_pair_selected": 0,
        "same_original_shared_pair_selected": 0,
        "different_original_shared_pair_selected": 0,
    }
    shared_cost, shared_train, shared_validation, state_l1 = [], [], [], []
    risk_weight, risk_correlation, risk_contribution = [], [], []
    full_ranks, shared_ranks = [], []
    before_previous, after_previous = np.full(6, 1 / 6), np.full(7, 1 / 7)
    before_pair = after_pair = None
    six_switches = seven_switches = shared_restricted_changes = previous_VVIX_positive = 0
    mature = unscored = shared_records = added_records = 0
    for position, (before, after) in enumerate(zip(six, seven, strict=True)):
        if retain is not None:
            retain(
                {
                    "record": "matched_source_input",
                    "decision_index": position,
                    "six": before,
                    "seven": after,
                    "original_sources_replayed": False,
                }
            )
        day = before["decision_session"]
        old_prev = _weights(before["previous_rolling_expert_weights"], 6)
        new_prev = _weights(after["previous_rolling_expert_weights"], 7)
        old_current = _weights(before["rolling_expert_weights"], 6)
        new_current = _weights(after["rolling_expert_weights"], 7)
        if (
            not np.array_equal(old_prev, before_previous)
            or not np.array_equal(new_prev, after_previous)
            or after["decision_information_clock"] != CLOCK
            or before["decision_role"] != after["decision_role"]
            or before["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or len(before["expert_target_known_at"]) != 6
            or len(after["expert_target_known_at"]) != 7
            or before["expert_target_known_at"] != after["expert_target_known_at"][:6]
            or before["preserved_archive_identity_introduced_at"]
            != after["preserved_archive_identity_introduced_at"][:6]
            or before["expert_shadow_last_observation"]
            != after["expert_shadow_last_observation"][:6]
            or any(
                pd.Timestamp(value) > pd.Timestamp(day) for value in after["expert_target_known_at"]
            )
            or any(
                row[key] is not value
                for row in (before, after)
                for key, value in (
                    ("counterfactual_historical_expert_availability", True),
                    ("shadow_wealth_added_to_main_capital", False),
                    ("source_feedback_floor_transferred_to_main_account", False),
                    ("independent_forward_validation", False),
                )
            )
        ):
            raise QuantError("Recorded clocks,previous state or capital separation changed.")
        before_previous, after_previous = old_current, new_current
        difference = math.fsum(
            abs(float(x - y)) for x, y in zip(np.r_[old_prev, 0.0], new_prev, strict=True)
        )
        state_l1.append(difference)
        row = {
            "decision_session": day,
            "decision_index": position,
            "previous_six_weights_in_seven_coordinates": [*old_prev.tolist(), 0.0],
            "previous_seven_weights": new_prev.tolist(),
            "previous_assessment_weight_L1_difference": difference,
            "previous_VVIX_weight": float(new_prev[6]),
            "diagnostic_restricted_ranking_is_original_six_replay": False,
        }
        old_selection, new_selection = before["selection"], after["selection"]
        if position < 7:
            if (
                any(
                    selection["scores_available"] is not False
                    or selection["pairs"] != []
                    or selection["selected_pair"] is not None
                    or selection["reason"] != "unscored_initial_equal_weights"
                    or selection["safety_or_success_claimed"] is not False
                    for selection in (old_selection, new_selection)
                )
                or not np.array_equal(old_current, np.full(6, 1 / 6))
                or not np.array_equal(new_current, np.full(7, 1 / 7))
            ):
                raise QuantError("Seven unmatured initial states cannot acquire successful ranks.")
            unscored += 1
            row.update(
                scored=False, choice_classification="unscored_initial", restricted_shared_best=None
            )
        else:
            if any(
                selection["scores_available"] is not True
                for selection in (old_selection, new_selection)
            ):
                raise QuantError("Preserve each actual mature recorded choice without a default.")
            keys = ("train_start", "train_end", "validation_start", "validation_end")
            if any(old_selection[key] != new_selection[key] for key in keys) or not (
                old_selection["train_start"]
                <= old_selection["train_end"]
                < old_selection["validation_start"]
                <= old_selection["validation_end"]
                <= day
            ):
                raise QuantError("Mature pair assessments must retain identical past-only clocks.")
            old_pairs, new_pairs = _pairs(old_selection, SIX), _pairs(new_selection, SEVEN)
            original_rank, expanded_rank = _rank(old_pairs), _rank(new_pairs)
            restricted = {names: new_pairs[names] for names in old_pairs}
            restricted_rank = _rank(restricted)
            if not original_rank or not expanded_rank or not restricted_rank:
                raise QuantError(
                    "A recorded mature choice cannot be reconstructed from an empty rank."
                )
            old_best, new_best, restricted_best = (
                original_rank[0],
                expanded_rank[0],
                restricted_rank[0],
            )
            if (
                tuple(old_selection["selected_pair"]) != old_best
                or tuple(new_selection["selected_pair"]) != new_best
                or not np.array_equal(
                    old_current, np.asarray(old_pairs[old_best]["expert_weights"])
                )
                or not np.array_equal(
                    new_current, np.asarray(new_pairs[new_best]["expert_weights"])
                )
            ):
                raise QuantError(
                    "The stored winner must follow its actual recorded score-key tie order."
                )
            paired = []
            for names, old_record in old_pairs.items():
                new_record = new_pairs[names]
                old_reason, new_reason = (
                    old_record.get("unscorable_reason"),
                    new_record.get("unscorable_reason"),
                )
                if old_reason != new_reason:
                    raise QuantError(
                        "Unscorable shared-pair states cannot be silently reclassified."
                    )
                item = {"pair": list(names), "unscorable_reason": old_reason}
                if old_reason is None:
                    weights = max(
                        abs(np.r_[old_record["expert_weights"], 0.0] - new_record["expert_weights"])
                    )
                    contributions = max(
                        abs(
                            np.r_[old_record["training_marginal_variance_contributions"], 0.0]
                            - new_record["training_marginal_variance_contributions"]
                        )
                    )
                    correlation = abs(
                        old_record["training_correlation"] - new_record["training_correlation"]
                    )
                    if weights > 1e-10 or correlation > 1e-10 or contributions > 1e-12:
                        raise QuantError(
                            "Shared risk/weight evidence exceeds the original fixed limits."
                        )
                    risk_weight.append(float(weights))
                    risk_correlation.append(correlation)
                    risk_contribution.append(float(contributions))
                    cost = (
                        new_record["estimated_incremental_switch_cost_fraction"]
                        - old_record["estimated_incremental_switch_cost_fraction"]
                    )
                    train = new_record["train_net_risk_score"] - old_record["train_net_risk_score"]
                    validation = (
                        new_record["validation_net_risk_score"]
                        - old_record["validation_net_risk_score"]
                    )
                    shared_cost.append(cost)
                    shared_train.append(train)
                    shared_validation.append(validation)
                    item.update(
                        {
                            "maximum_weight_difference": float(weights),
                            "training_correlation_difference": correlation,
                            "maximum_marginal_variance_difference": float(contributions),
                            "seven_minus_six_switch_proxy_fraction": cost,
                            "seven_minus_six_proxy_changed_assets": new_record[
                                "estimated_incremental_changed_assets"
                            ]
                            - old_record["estimated_incremental_changed_assets"],
                            "seven_minus_six_train_net_score": train,
                            "seven_minus_six_validation_net_score": validation,
                            "seven_minus_six_selection_score_key": new_record["selection_score_key"]
                            - old_record["selection_score_key"],
                        }
                    )
                paired.append(item)
            classification = (
                "added_VVIX_pair_selected"
                if VVIX in new_best
                else (
                    "same_original_shared_pair_selected"
                    if old_best == new_best
                    else "different_original_shared_pair_selected"
                )
            )
            counts[classification] += 1
            shared_restricted_changes += restricted_best != old_best
            previous_VVIX_positive += new_prev[6] > 0
            six_switches += before_pair is not None and old_best != before_pair
            seven_switches += after_pair is not None and new_best != after_pair
            before_pair, after_pair = old_best, new_best
            full_rank, shared_rank = (
                expanded_rank.index(old_best) + 1,
                restricted_rank.index(old_best) + 1,
            )
            full_ranks.append(full_rank)
            shared_ranks.append(shared_rank)
            added = [new_pairs[names] for names in new_pairs if VVIX in names]
            shared_records += len(paired)
            added_records += len(added)
            mature += 1
            row.update(
                {
                    "scored": True,
                    "original_six_best": list(old_best),
                    "actual_seven_best": list(new_best),
                    "diagnostic_restricted_shared_best": list(restricted_best),
                    "choice_classification": classification,
                    "old_winner_rank_in_full_seven_menu": full_rank,
                    "old_winner_rank_in_restricted_shared_menu": shared_rank,
                    "restricted_shared_best_differs_from_original": restricted_best != old_best,
                    "all_shared_pair_comparisons": paired,
                    "all_added_VVIX_pair_records": added,
                    "recorded_score_or_covariance_recomputed": False,
                }
            )
        if retain is not None:
            retain(
                {
                    "record": "resolved_menu_comparison",
                    "decision_index": position,
                    "comparison": row,
                }
            )
    if unscored != 7 or mature != 114 or shared_records != 1710 or added_records != 684:
        raise QuantError(
            "Retain exactly all initial states,mature choices,shared pairs and added pairs."
        )
    return {
        "matched_decisions": 121,
        "unscored_initial_decisions": unscored,
        "mature_recorded_choices": mature,
        "all_shared_pair_record_comparisons": shared_records,
        "all_added_VVIX_pair_records": added_records,
        **counts,
        "restricted_shared_best_differs_from_original_records": shared_restricted_changes,
        "mature_previous_state_has_positive_VVIX_weight_records": int(previous_VVIX_positive),
        "recorded_six_adjacent_pair_switches": six_switches,
        "recorded_seven_adjacent_pair_switches": seven_switches,
        "all_previous_assessment_weight_L1_differences": _stats(state_l1),
        "old_winner_rank_in_full_seven_menu": _stats(full_ranks),
        "old_winner_rank_in_diagnostic_restricted_shared_menu": _stats(shared_ranks),
        "shared_weight_absolute_differences": _stats(risk_weight),
        "shared_training_correlation_absolute_differences": _stats(risk_correlation),
        "shared_marginal_variance_absolute_differences": _stats(risk_contribution),
        "all_shared_seven_minus_six_switch_proxy_fractions": _stats(shared_cost),
        "all_shared_seven_minus_six_train_net_scores": _stats(shared_train),
        "all_shared_seven_minus_six_validation_net_scores": _stats(shared_validation),
        "restricted_ranking_is_a_new_diagnostic_of_recorded_seven_scores": True,
        "restricted_ranking_replays_original_six_strategy_or_previous_state": False,
        "original_scoring_targets_or_accounts_recomputed": False,
        "pair_switches_or_weight_differences_are_actual_orders_or_fees": False,
        "isolated_causal_cost_or_source_effect_claimed": False,
        "new_financial_configuration_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
