from __future__ import annotations

from datetime import date
from fractions import Fraction

import pandas as pd

from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.expert_contextual_contest_v1 import rational_record

IDS = ("expert_pool_six_contest_frequency_v1", "expert_pool_six_contextual_contest_v1")
CREDIT = 60


def _day(value):
    if not isinstance(value, str):
        raise QuantError("A saved credit assessment needs an explicit ISO decision date.")
    try:
        day = date.fromisoformat(value)
    except ValueError as exc:
        raise QuantError("A saved credit assessment has an invalid decision date.") from exc
    if day.isoformat() != value:
        raise QuantError("Use the original canonical decision date.")
    return pd.Timestamp(day)


def _state(value, months):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 6 for row in value)
        or any(type(x) is not int or x < 0 for row in value for x in row)
        or type(months) is not int
        or not 0 <= months <= 119
        or sum(map(sum, value)) != CREDIT * months
        or any(sum(row) % CREDIT for row in value)
    ):
        raise QuantError("Preserve the exact four-context completed-month credit ledger.")
    return value


def _ratio(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise QuantError("A stored posterior needs its exact numerator and denominator.")
    if any(not isinstance(value[k], str) for k in value):
        raise QuantError("Do not round the saved rational posterior into binary64.")
    try:
        numerator, denominator = int(value["numerator"]), int(value["denominator"])
    except ValueError as exc:
        raise QuantError("The saved posterior contains a noninteger rational operand.") from exc
    if numerator <= 0 or denominator <= 0:
        raise QuantError("The declared unit prior requires positive posterior operands.")
    return Fraction(numerator, denominator)


def snapshot(audit: dict):
    if not isinstance(audit, dict):
        raise QuantError("Require a saved whole resolved contest decision.")
    _day(audit.get("decision_session"))
    months = audit.get("complete_learning_months")
    state = _state(audit.get("credit_state_after"), months)
    context, mature = audit.get("current_context"), audit.get("source_assessment_mature")
    counts = audit.get("source_observation_counts")
    if (
        type(context) is not int
        or not 0 <= context < 4
        or type(mature) is not bool
        or not isinstance(counts, list)
        or len(counts) != 6
        or any(type(x) is not int or not 0 <= x <= 126 for x in counts)
        or mature != all(x == 126 for x in counts)
        or not isinstance(audit.get("allocations"), dict)
        or set(audit["allocations"]) != set(IDS)
    ):
        raise QuantError("Keep the original context,maturity and two saved assessment identities.")
    result = {}
    for index, name in enumerate(IDS):
        record = audit["allocations"][name]
        credit = [sum(row[i] for row in state) for i in range(6)] if index == 0 else state[context]
        support = sum(credit) // CREDIT
        denominator = sum(credit) + 6 * CREDIT
        expected = tuple(Fraction(x + CREDIT, denominator) for x in credit)
        if not isinstance(record, dict):
            raise QuantError("A saved allocation has no credit assessment.")
        values = record.get("posterior_rational_fractions")
        if (
            not isinstance(values, list)
            or len(values) != 6
            or tuple(_ratio(value) for value in values) != expected
            or type(record.get("counted_complete_months")) is not int
            or record.get("counted_complete_months") != support
            or record.get("categorical_forecast_is_prior_only") is not (support == 0)
        ):
            raise QuantError("The stored posterior must equal its then-known integer counts.")
        result[name] = {
            "fractions": expected,
            "support": support,
            "prior_mass": Fraction(6, support + 6),
            "prior_only": support == 0,
        }
    return result


def assess_transition(previous: dict, current_input: dict, current: dict):
    forecasts = snapshot(previous)
    snapshot(current)
    before, after = _day(previous["decision_session"]), _day(current["decision_session"])
    if (
        not isinstance(current_input, dict)
        or current_input.get("stage") != "input"
        or current_input.get("session") != current["decision_session"]
        or current_input.get("previous_decision") != previous["decision_session"]
        or type(current_input.get("previous_context")) is not int
        or current_input.get("previous_context") != previous["current_context"]
        or current_input.get("credit_state_before") != previous["credit_state_after"]
        or type(current_input.get("complete_months_before")) is not int
        or current_input.get("complete_months_before") != previous["complete_learning_months"]
        or after <= before
        or not is_month_end(after)
        or previous["source_assessment_mature"]
        and not current["source_assessment_mature"]
    ):
        raise QuantError("Assess the previous decision before assimilating the next saved outcome.")
    if not is_month_end(before):
        if (
            str(before.date()) != "2016-10-05"
            or str(after.date()) != "2016-10-31"
            or current.get("initial_partial_source_month_not_assimilated") is not True
            or current.get("monthly_credit_update") is not None
            or current["credit_state_after"] != previous["credit_state_after"]
            or current["complete_learning_months"] != previous["complete_learning_months"]
        ):
            raise QuantError("Exclude only the declared initial partial source month.")
        return {
            "status": "excluded_declared_initial_partial",
            "forecast_session": previous["decision_session"],
            "outcome_as_of": current["decision_session"],
        }
    update = current.get("monthly_credit_update")
    if (
        after.year * 12 + after.month - before.year * 12 - before.month != 1
        or current.get("initial_partial_source_month_not_assimilated") is not False
        or not isinstance(update, dict)
        or type(update.get("previous_context")) is not int
        or update.get("previous_context") != previous["current_context"]
        or current["complete_learning_months"] != previous["complete_learning_months"] + 1
    ):
        raise QuantError("A credit outcome needs exactly one following complete exchange month.")
    period = sessions(next_session(before), after)
    if (
        update.get("first_source_session") != str(period[0].date())
        or update.get("last_source_session") != str(period[-1].date())
        or update.get("source_sessions") != len(period)
    ):
        raise QuantError("Do not skip,backdate or extend the recorded credit outcome interval.")
    keys, winners, units = (
        update.get("winner_relative_keys"),
        update.get("winner_indices"),
        update.get("credit_units"),
    )
    if (
        not isinstance(keys, list)
        or len(keys) != 6
        or any(type(x) is not int for x in keys)
        or not isinstance(winners, list)
        or any(type(x) is not int for x in winners)
        or winners != [i for i, x in enumerate(keys) if x == max(keys)]
        or not isinstance(units, list)
        or len(units) != 6
        or any(type(x) is not int or x < 0 for x in units)
        or sum(units) != CREDIT
        or units != [CREDIT // len(winners) if i in winners else 0 for i in range(6)]
    ):
        raise QuantError("Use the saved exact equal-split winning credit,not new source returns.")
    expected = [list(row) for row in previous["credit_state_after"]]
    context = previous["current_context"]
    expected[context] = [expected[context][i] + units[i] for i in range(6)]
    if expected != current["credit_state_after"]:
        raise QuantError("The next outcome must not already be present in the forecast state.")
    outcome = tuple(Fraction(x, CREDIT) for x in units)
    scores = {}
    details = {}
    for name in IDS:
        forecast = forecasts[name]
        p = forecast["fractions"]
        score = 1 - 2 * sum(a * b for a, b in zip(p, outcome, strict=True)) + sum(x * x for x in p)
        if not 0 <= score <= 2:
            raise QuantError("Expected multiclass Brier left its fixed unscaled range.")
        scores[name] = score
        details[name] = {
            "posterior_rational_fractions": [rational_record(x) for x in p],
            "completed_month_support": forecast["support"],
            "unit_prior_mass": rational_record(forecast["prior_mass"]),
            "prior_only": forecast["prior_only"],
            "expected_brier": rational_record(score),
            "expected_brier_display": float(score),
        }
    difference = scores[IDS[1]] - scores[IDS[0]]
    return {
        "status": "scored_complete_month",
        "forecast_session": previous["decision_session"],
        "outcome_as_of": current["decision_session"],
        "first_outcome_source_session": update["first_source_session"],
        "outcome_source_sessions": len(period),
        "forecast_context": context,
        "forecast_main_assessment_mature": previous["source_assessment_mature"],
        "outcome_credit_units": units,
        "forecast_assessments": details,
        "contextual_minus_unconditional_loss": rational_record(difference),
        "contextual_minus_unconditional_loss_display": float(difference),
    }


def _read_signed_ratio(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def summarize(rows: list[dict]):
    if not rows:
        return {"pairs": 0, "expected_brier": None, "paired_loss": None, "no_pairs": True}
    scores = {
        name: [
            _read_signed_ratio(row["forecast_assessments"][name]["expected_brier"]) for row in rows
        ]
        for name in IDS
    }
    differences = [_read_signed_ratio(row["contextual_minus_unconditional_loss"]) for row in rows]
    if differences != [b - a for a, b in zip(*scores.values(), strict=True)]:
        raise QuantError("Aggregate only the preserved paired expected losses.")
    assessment = {}
    for name in IDS:
        values = [row["forecast_assessments"][name] for row in rows]
        support = [value["completed_month_support"] for value in values]
        prior_mass = [_read_signed_ratio(value["unit_prior_mass"]) for value in values]
        mean = sum(scores[name], Fraction(0)) / len(rows)
        assessment[name] = {
            "loss_sum": rational_record(sum(scores[name], Fraction(0))),
            "loss_mean": rational_record(mean),
            "loss_mean_display": float(mean),
            "minimum_completed_month_support": min(support),
            "maximum_completed_month_support": max(support),
            "prior_only_forecasts": sum(value["prior_only"] for value in values),
            "minimum_unit_prior_mass": rational_record(min(prior_mass)),
            "maximum_unit_prior_mass": rational_record(max(prior_mass)),
        }
    difference = sum(differences, Fraction(0)) / len(rows)
    return {
        "pairs": len(rows),
        "expected_brier": assessment,
        "paired_loss": {
            "contextual_minus_unconditional_mean": rational_record(difference),
            "contextual_minus_unconditional_mean_display": float(difference),
            "contextual_lower_loss": sum(x < 0 for x in differences),
            "equal_loss": sum(x == 0 for x in differences),
            "contextual_higher_loss": sum(x > 0 for x in differences),
        },
        "no_pairs": False,
    }
