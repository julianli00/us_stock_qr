import copy
import itertools
import math

import pytest

from us_quant import expert_pair_allocation_decomposition_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError


def source_records():
    calendar = sessions("2016-10-05", "2026-10-05")
    days = [calendar[0], *[day for day in calendar if day > calendar[0] and is_month_end(day)]]
    result = [[], []]
    previous = [rules.NEUTRAL, rules.NEUTRAL]
    for position, day in enumerate(days):
        known = calendar[calendar <= day][-126:]
        pairs = []
        if position >= 7:
            for rank, (a, b) in enumerate(itertools.combinations(range(7), 2)):
                weights = [0.0] * 7
                weights[a], weights[b] = (b + 1) / (a + b + 2), (a + 1) / (a + b + 2)
                contributions = [0.0] * 7
                contributions[a] = contributions[b] = 0.001
                key = round(2 - rank / 100, 12)
                pairs.append(
                    {
                        "pair": sorted((rules.EXPERT_IDS[a], rules.EXPERT_IDS[b])),
                        "expert_weights": weights,
                        "training_marginal_variance_contributions": contributions,
                        "training_correlation": 0.0,
                        "train_net_risk_score": key + 0.25,
                        "validation_net_risk_score": key,
                        "selection_score": key,
                        "selection_score_key": key,
                        "estimated_incremental_switch_cost_fraction": 0.001,
                        "estimated_incremental_changed_assets": 2,
                        "cost_proxy_is_actual_main_account_cost": False,
                    }
                )
        for mode, scheme in enumerate(rules.SCHEMES):
            if position < 7:
                weights = list(rules.NEUTRAL)
                assessment = {
                    "scores_available": False,
                    "pairs": [],
                    "selected_pair": None,
                    "reason": "unscored_initial_equal_weights",
                    "safety_or_success_claimed": False,
                }
                evidence = {
                    "reason": "unscored_initial_equal_weights",
                    "pair_masses": [],
                    "scorable_pair_count": 0,
                    "aggregation_point_total": None,
                }
            else:
                m = len(pairs)
                points = [1 if mode == 0 else m - i for i in range(m)]
                masses = [p / sum(points) for p in points]
                weights = [
                    math.fsum(
                        mass * pair["expert_weights"][i]
                        for pair, mass in zip(pairs, masses, strict=True)
                    )
                    for i in range(7)
                ]
                assessment = {
                    "scores_available": True,
                    "pairs": copy.deepcopy(pairs),
                    "selected_pair": pairs[0]["pair"],
                    "reason": "rolling_trained_pair_validated_on_mature_past",
                    "safety_or_success_claimed": False,
                    "train_start": str(known[0].date()),
                    "train_end": str(known[83].date()),
                    "validation_start": str(known[84].date()),
                    "validation_end": str(known[-1].date()),
                }
                evidence = {
                    "reason": "full_support_pair_ERC_aggregation",
                    "scorable_pair_count": m,
                    "aggregation_point_total": sum(points),
                    "pair_masses": [
                        {
                            "pair": pair["pair"],
                            "pair_mass": mass,
                            "aggregation_points": point,
                            "selection_score_key": pair["selection_score_key"],
                        }
                        for pair, mass, point in zip(pairs, masses, points, strict=True)
                    ],
                }
            evidence.update(
                aggregation=scheme,
                pair_assessments=assessment,
                chosen_expert_weights=weights,
                overlapping_pairs_are_independent_experts_or_capital=False,
                confidence_interval_or_significance_claimed=False,
            )
            result[mode].append(
                {
                    "decision_session": str(day.date()),
                    "decision_index": position,
                    "decision_information_clock": rules.CLOCK,
                    "decision_role": "initial_capital"
                    if position == 0
                    else "completed_exchange_month_end",
                    "previous_expert_weights": previous[mode],
                    "expert_weights": weights,
                    "expert_target_known_at": [str(day.date())] * 7,
                    "preserved_archive_identity_introduced_at": ["2026-10-28T00:00:00Z"] * 7,
                    "expert_shadow_last_observation": [str(day.date()) if position else None] * 7,
                    "pair_ensemble": evidence,
                    "counterfactual_historical_expert_availability": True,
                    "shadow_wealth_added_to_main_capital": False,
                    "source_feedback_floor_transferred_to_main_account": False,
                    "independent_forward_validation": False,
                }
            )
            previous[mode] = weights
    return result


@pytest.fixture(scope="module")
def recorded():
    return source_records()


def test_complete_uniform_anchored_identity_and_concentration(recorded):
    rows = []
    checked = rules.compare(*recorded, retain=rows.append)
    assert len(rows) == 242
    assert checked["all_original_named_pair_comparisons"] == 2394
    assert checked["paired_recorded_risk_weight_differences"]["maximum"] == 0
    assert checked["maximum_uniform_anchored_decomposition_residual"] < 1e-12
    assert checked["mature_ordinal_minus_uniform_L1"]["minimum"] > 0
    assert checked["all_source_allocations"]["uniform"]["observations"] == 121
    assert checked["mature_source_allocations"]["ordinal"]["observations"] == 114
    assert checked["main_account_PNL_fees_CAGR_or_drawdown_counterfactual_computed"] is False
    assert checked["decomposition_is_uniform_anchored_not_unique_causal_attribution"] is True


def test_neutral_initial_states_are_not_mature_risk_results(recorded):
    checked = rules.compare(*recorded)
    assert checked["unscored_initial_decisions"] == 7
    assert checked["neutral_initial_states_are_successful_risk_assessments"] is False
    assert checked["HHI_effective_number_is_independent_factor_count"] is False
    for row in checked["all_source_allocations"]["uniform"]["per_expert"]:
        assert row["exact_neutral_records"] >= 7


@pytest.mark.parametrize(
    "field",
    ["expert_weights", "training_correlation", "training_marginal_variance_contributions"],
)
def test_recorded_pair_risk_limits_cannot_be_waived(recorded, field):
    changed = copy.deepcopy(recorded)
    pair = changed[1][7]["pair_ensemble"]["pair_assessments"]["pairs"][0]
    if field == "expert_weights":
        ids = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
        pair[field][ids[0]] += 0.001
        pair[field][ids[1]] -= 0.001
    elif field == "training_correlation":
        pair[field] += 0.001
    else:
        pair[field][0] += 0.001
    with pytest.raises(QuantError, match="risk identity"):
        rules.compare(*changed)


@pytest.mark.parametrize(
    "field",
    [
        "missing_pair",
        "missing_mass",
        "nan_mass",
        "nonunit_mass",
        "previous_state",
        "future_clock",
        "successful_unscored",
        "source_wealth_transfer",
        "chosen_weight",
    ],
)
def test_recorded_mass_state_clock_or_capital_tampering_is_rejected(recorded, field):
    changed = copy.deepcopy(recorded)
    row = changed[1][7]
    evidence = row["pair_ensemble"]
    if field == "missing_pair":
        evidence["pair_assessments"]["pairs"].pop()
    elif field == "missing_mass":
        evidence["pair_masses"].pop()
    elif field in ("nan_mass", "nonunit_mass"):
        evidence["pair_masses"][0]["pair_mass"] = math.nan if field == "nan_mass" else 0.8
    elif field == "previous_state":
        row["previous_expert_weights"] = [1.0, 0, 0, 0, 0, 0, 0]
    elif field == "future_clock":
        row["expert_target_known_at"][0] = "2030-01-01"
    elif field == "successful_unscored":
        changed[1][0]["pair_ensemble"]["pair_assessments"]["scores_available"] = True
    elif field == "source_wealth_transfer":
        row["shadow_wealth_added_to_main_capital"] = True
    elif field == "chosen_weight":
        evidence["chosen_expert_weights"] = [1.0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(QuantError):
        rules.compare(*changed)


def test_nonzero_recorded_risk_difference_term_is_not_silently_dropped(recorded):
    changed = copy.deepcopy(recorded)
    row = changed[1][-1]
    evidence = row["pair_ensemble"]
    pair = evidence["pair_assessments"]["pairs"][0]
    a, b = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
    pair["expert_weights"][a] += 1e-11
    pair["expert_weights"][b] -= 1e-11
    masses = {tuple(r["pair"]): r["pair_mass"] for r in evidence["pair_masses"]}
    weights = [
        math.fsum(
            masses[tuple(p["pair"])] * p["expert_weights"][i]
            for p in evidence["pair_assessments"]["pairs"]
        )
        for i in range(7)
    ]
    row["expert_weights"] = evidence["chosen_expert_weights"] = weights
    checked = rules.compare(*changed)
    assert checked["paired_recorded_risk_weight_differences"]["maximum"] > 0
    terms = checked["mature_per_expert_decomposition"]
    assert max(r["recorded_risk_difference_term"]["maximum"] for r in terms) > 0
    assert checked["maximum_uniform_anchored_decomposition_residual"] < 1e-10


def test_failure_preserves_matched_input_before_decomposition(recorded):
    changed = copy.deepcopy(recorded)
    changed[0][0]["previous_expert_weights"] = [1.0, 0, 0, 0, 0, 0, 0]
    rows = []
    with pytest.raises(QuantError, match="state chains"):
        rules.compare(*changed, retain=rows.append)
    assert len(rows) == 1 and rows[0]["record"] == "matched_source_input"
