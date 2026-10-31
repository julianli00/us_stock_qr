import copy
import itertools

import numpy as np
import pytest

from us_quant import expert_pair_menu_extension_v1 as rules
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError


def saved_decisions():
    calendar = sessions("2016-10-05", "2026-10-05")
    dates = [calendar[0], *[day for day in calendar if day > calendar[0] and is_month_end(day)]]
    before, after = [], []
    previous_six, previous_seven = np.full(6, 1 / 6).tolist(), np.full(7, 1 / 7).tolist()
    old_names = [tuple(sorted(pair)) for pair in itertools.combinations(rules.SIX, 2)]
    added_names = [tuple(sorted((name, rules.VVIX))) for name in rules.SIX]

    def pair(names, ids, score):
        weights = [0.0] * len(ids)
        contribution = [0.0] * len(ids)
        weights[ids.index(names[0])], weights[ids.index(names[1])] = 0.4, 0.6
        contribution[ids.index(names[0])] = contribution[ids.index(names[1])] = 0.001
        return {
            "pair": list(names),
            "expert_weights": weights,
            "training_marginal_variance_contributions": contribution,
            "training_correlation": 0.25,
            "train_net_risk_score": score + 0.1,
            "validation_net_risk_score": score,
            "selection_score": score,
            "selection_score_key": round(score, 12),
            "estimated_incremental_switch_cost_fraction": 0.001,
            "estimated_incremental_changed_assets": 3,
            "cost_proxy_is_actual_main_account_cost": False,
        }

    for position, day in enumerate(dates):
        records = []
        for ids, previous in ((rules.SIX, previous_six), (rules.SEVEN, previous_seven)):
            n = len(ids)
            if position < 7:
                selection = {
                    "pairs": [],
                    "selected_pair": None,
                    "scores_available": False,
                    "reason": "unscored_initial_equal_weights",
                    "safety_or_success_claimed": False,
                }
                current = np.full(n, 1 / n).tolist()
            else:
                pairs = [
                    pair(names, ids, 1.3 - index * 0.01) for index, names in enumerate(old_names)
                ]
                if n == 7:
                    pairs.extend(
                        pair(names, ids, 0.5 - index * 0.01)
                        for index, names in enumerate(added_names)
                    )
                    if position == 7:
                        pairs[15] = pair(added_names[0], ids, 2.0)
                    elif position == 8:
                        pairs[1] = pair(old_names[1], ids, 1.5)
                        pairs[1]["estimated_incremental_switch_cost_fraction"] = 0.002
                best = sorted(pairs, key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
                known = sessions("2016-10-06", day)[-126:]
                selection = {
                    "pairs": pairs,
                    "selected_pair": best["pair"],
                    "scores_available": True,
                    "reason": "recorded_rolling_pair",
                    "safety_or_success_claimed": False,
                    "train_start": str(known[0].date()),
                    "train_end": str(known[83].date()),
                    "validation_start": str(known[84].date()),
                    "validation_end": str(known[-1].date()),
                }
                current = best["expert_weights"].copy()
            record = {
                "decision_session": str(day.date()),
                "decision_role": "initial_capital"
                if position == 0
                else "completed_exchange_month_end",
                "expert_target_known_at": [str(previous_session(day).date())] * n,
                "preserved_archive_identity_introduced_at": ["2026-10-28T00:00:00+00:00"] * n,
                "expert_shadow_last_observation": [str(day.date()) if position else None] * n,
                "previous_rolling_expert_weights": previous.copy(),
                "rolling_expert_weights": current,
                "selection": selection,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
            if n == 7:
                record["decision_information_clock"] = rules.CLOCK
            records.append(record)
        before.append(records[0])
        after.append(records[1])
        previous_six, previous_seven = (
            records[0]["rolling_expert_weights"],
            records[1]["rolling_expert_weights"],
        )
    return before, after


def test_direct_and_shared_choice_categories_are_distinct():
    records = []
    summary = rules.compare(*saved_decisions(), retain=records.append)
    assert summary["matched_decisions"] == 121 and summary["mature_recorded_choices"] == 114
    assert summary["unscored_initial_decisions"] == 7
    assert summary["all_shared_pair_record_comparisons"] == 1710
    assert summary["all_added_VVIX_pair_records"] == 684
    assert summary["added_VVIX_pair_selected"] == 1
    assert summary["different_original_shared_pair_selected"] == 1
    assert summary["same_original_shared_pair_selected"] == 112
    assert summary["recorded_six_adjacent_pair_switches"] == 0
    assert summary["recorded_seven_adjacent_pair_switches"] == 2
    assert len(records) == 242


def test_restricted_recorded_ranking_is_not_original_six_replay():
    summary = rules.compare(*saved_decisions())
    assert summary["restricted_shared_best_differs_from_original_records"] == 1
    assert summary["mature_previous_state_has_positive_VVIX_weight_records"] == 2
    assert summary["restricted_ranking_is_a_new_diagnostic_of_recorded_seven_scores"] is True
    assert summary["restricted_ranking_replays_original_six_strategy_or_previous_state"] is False
    assert summary["isolated_causal_cost_or_source_effect_claimed"] is False


def test_unmatured_states_do_not_receive_successful_ranks():
    before, after = saved_decisions()
    after[0]["selection"]["scores_available"] = True
    with pytest.raises(QuantError, match="unmatured"):
        rules.compare(before, after)


@pytest.mark.parametrize(
    "kind",
    ["missing_date", "missing_shared", "missing_added", "wrong_clock", "changed_train_clock"],
)
def test_all_shared_and_added_pairs_and_clocks_are_required(kind):
    before, after = saved_decisions()
    if kind == "missing_date":
        after.pop()
    elif kind == "missing_shared":
        after[7]["selection"]["pairs"].pop(0)
    elif kind == "missing_added":
        after[7]["selection"]["pairs"].pop()
    elif kind == "wrong_clock":
        after[7]["decision_information_clock"] = "16_00_ET"
    else:
        after[7]["selection"]["train_start"] = "2010-01-04"
    with pytest.raises(QuantError):
        rules.compare(before, after)


@pytest.mark.parametrize(
    "field", ["expert_weights", "training_correlation", "training_marginal_variance_contributions"]
)
def test_original_shared_risk_limits_cannot_be_waived(field):
    before, after = saved_decisions()
    record = after[7]["selection"]["pairs"][0]
    if field == "expert_weights":
        indices = [i for i, value in enumerate(record[field]) if value]
        record[field][indices[0]] += 1e-4
        record[field][indices[1]] -= 1e-4
    elif field == "training_correlation":
        record[field] += 2e-10
    else:
        record[field][0] += 2e-12
    with pytest.raises(QuantError):
        rules.compare(before, after)


def test_failure_preserves_matched_inputs_without_rescue():
    before, after = saved_decisions()
    after[7]["selection"]["pairs"][0]["training_correlation"] += 1e-5
    records = []
    with pytest.raises(QuantError, match="original fixed limits"):
        rules.compare(before, after, retain=records.append)
    assert len(records) == 15 and records[-1]["record"] == "matched_source_input"
    assert records[-1]["decision_index"] == 7
    assert records[-1]["seven"]["selection"]["pairs"][0]["training_correlation"] == 0.25001


def test_shared_unscorable_states_cannot_be_reclassified():
    before, after = saved_decisions()
    after[7]["selection"]["pairs"][1] = {
        "pair": copy.deepcopy(after[7]["selection"]["pairs"][1]["pair"]),
        "unscorable_reason": "zero_training_variance",
    }
    with pytest.raises(QuantError, match="reclassified"):
        rules.compare(before, after)


def test_previous_recorded_state_is_not_replaced():
    before, after = saved_decisions()
    after[8]["previous_rolling_expert_weights"] = np.full(7, 1 / 7).tolist()
    with pytest.raises(QuantError, match="previous state"):
        rules.compare(before, after)
