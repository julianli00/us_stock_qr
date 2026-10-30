from copy import deepcopy
from fractions import Fraction

import pandas as pd
import pytest

from us_quant import expert_contest_credit_diagnosis_v1 as rules
from us_quant.calendar import next_session, sessions
from us_quant.config import QuantError


def audit(day, state, context=0, mature=True, update=None, partial=False):
    result = {
        "decision_session": day,
        "credit_state_after": state,
        "complete_learning_months": sum(map(sum, state)) // 60,
        "current_context": context,
        "source_assessment_mature": mature,
        "source_observation_counts": [126 if mature else 10] * 6,
        "initial_partial_source_month_not_assimilated": partial,
        "monthly_credit_update": update,
        "allocations": {},
    }
    for index, name in enumerate(rules.IDS):
        credit = [sum(row[i] for row in state) for i in range(6)] if index == 0 else state[context]
        result["allocations"][name] = {
            "counted_complete_months": sum(credit) // 60,
            "categorical_forecast_is_prior_only": sum(credit) == 0,
            "posterior_rational_fractions": [
                rules.rational_record(Fraction(value + 60, sum(credit) + 360)) for value in credit
            ],
        }
    return result


def pair(tied=False, mature=True, context=0):
    state = [[60, 0, 0, 0, 0, 0], [0, 60, 0, 0, 0, 0], [0] * 6, [0] * 6]
    previous = audit("2017-04-28", state, context=context, mature=mature)
    period = sessions(next_session(pd.Timestamp("2017-04-28")), pd.Timestamp("2017-05-31"))
    units = [30, 30, 0, 0, 0, 0] if tied else [60, 0, 0, 0, 0, 0]
    keys = [2, 2, 1, 1, 1, 1] if tied else [2, 1, 1, 1, 1, 1]
    update = {
        "winner_relative_keys": keys,
        "winner_indices": [0, 1] if tied else [0],
        "credit_units": units,
        "previous_context": context,
        "first_source_session": str(period[0].date()),
        "last_source_session": str(period[-1].date()),
        "source_sessions": len(period),
    }
    new_state = deepcopy(state)
    new_state[context] = [x + y for x, y in zip(state[context], units, strict=True)]
    current = audit("2017-05-31", new_state, update=update)
    inputs = {
        "stage": "input",
        "session": current["decision_session"],
        "previous_decision": previous["decision_session"],
        "previous_context": context,
        "credit_state_before": deepcopy(state),
        "complete_months_before": previous["complete_learning_months"],
    }
    return previous, inputs, current


def test_exact_prequential_loss_uses_prior_not_assimilated_posterior():
    previous, inputs, current = pair()
    before = deepcopy((previous, inputs, current))
    result = rules.assess_transition(previous, inputs, current)
    assert result["forecast_assessments"][rules.IDS[0]]["expected_brier"] == {
        "numerator": "11",
        "denominator": "16",
    }
    assert result["forecast_assessments"][rules.IDS[1]]["expected_brier"] == {
        "numerator": "30",
        "denominator": "49",
    }
    assert result["contextual_minus_unconditional_loss"] == {
        "numerator": "-59",
        "denominator": "784",
    }
    assert (previous, inputs, current) == before


def test_tied_credit_uses_expected_multiclass_loss_not_zero_distribution_loss():
    result = rules.assess_transition(*pair(tied=True))
    assert result["forecast_assessments"][rules.IDS[1]]["expected_brier"] == {
        "numerator": "37",
        "denominator": "49",
    }
    assert sum(result["outcome_credit_units"]) == 60


def test_unmatured_main_assessment_does_not_erase_recorded_credit_forecast():
    result = rules.assess_transition(*pair(mature=False))
    assert result["forecast_main_assessment_mature"] is False
    assert result["forecast_assessments"][rules.IDS[0]]["completed_month_support"] == 2
    assert result["forecast_assessments"][rules.IDS[0]]["expected_brier"] is not None


def test_unseen_context_retains_declared_prior_and_actual_support():
    result = rules.assess_transition(*pair(context=3))
    value = result["forecast_assessments"][rules.IDS[1]]
    assert value["prior_only"] is True and value["completed_month_support"] == 0
    assert value["unit_prior_mass"] == {"numerator": "1", "denominator": "1"}
    assert value["expected_brier"] == {"numerator": "5", "denominator": "6"}


def test_declared_initial_partial_is_explicitly_unscored():
    state = [[0] * 6 for _ in range(4)]
    previous = audit("2016-10-05", state, mature=False)
    current = audit("2016-10-31", state, mature=False, partial=True)
    inputs = {
        "stage": "input",
        "session": "2016-10-31",
        "previous_decision": "2016-10-05",
        "previous_context": 0,
        "credit_state_before": state,
        "complete_months_before": 0,
    }
    result = rules.assess_transition(previous, inputs, current)
    assert result["status"] == "excluded_declared_initial_partial"
    assert "forecast_assessments" not in result


@pytest.mark.parametrize(
    "mutation",
    [
        "lookahead_state",
        "wrong_context",
        "wrong_interval",
        "wrong_credit",
        "rounded_posterior",
        "wrong_support",
        "maturity_loss",
        "missing_input",
    ],
)
def test_credit_clock_count_posterior_and_maturity_tampering_fails(mutation):
    previous, inputs, current = pair()
    if mutation == "lookahead_state":
        inputs["credit_state_before"] = deepcopy(current["credit_state_after"])
    elif mutation == "wrong_context":
        current["monthly_credit_update"]["previous_context"] = 2
    elif mutation == "wrong_interval":
        current["monthly_credit_update"]["first_source_session"] = "2017-05-02"
    elif mutation == "wrong_credit":
        current["monthly_credit_update"]["credit_units"] = [30, 30, 0, 0, 0, 0]
    elif mutation == "rounded_posterior":
        previous["allocations"][rules.IDS[0]]["posterior_rational_fractions"][0] = 0.25
    elif mutation == "wrong_support":
        previous["allocations"][rules.IDS[1]]["counted_complete_months"] = 5
    elif mutation == "maturity_loss":
        current["source_assessment_mature"] = False
        current["source_observation_counts"] = [20] * 6
    else:
        inputs.pop("previous_decision")
    with pytest.raises(QuantError):
        rules.assess_transition(previous, inputs, current)


def test_following_context_and_applied_risk_weights_do_not_leak_into_current_score():
    previous, inputs, current = pair()
    baseline = rules.assess_transition(previous, inputs, current)
    changed = audit(
        "2017-05-31",
        current["credit_state_after"],
        context=3,
        update=current["monthly_credit_update"],
    )
    for value in (previous, changed):
        for record in value["allocations"].values():
            record["applied_expert_weights"] = ["not a probability forecast"]
    assert rules.assess_transition(previous, inputs, changed) == baseline


def test_exact_aggregation_and_explicit_empty_context():
    rows = [rules.assess_transition(*pair()), rules.assess_transition(*pair(tied=True))]
    result = rules.summarize(rows)
    assert result["pairs"] == 2
    assert result["paired_loss"] == {
        "contextual_minus_unconditional_mean": {"numerator": "-3", "denominator": "784"},
        "contextual_minus_unconditional_mean_display": float(Fraction(-3, 784)),
        "contextual_lower_loss": 1,
        "equal_loss": 0,
        "contextual_higher_loss": 1,
    }
    assert rules.summarize([]) == {
        "pairs": 0,
        "expected_brier": None,
        "paired_loss": None,
        "no_pairs": True,
    }
