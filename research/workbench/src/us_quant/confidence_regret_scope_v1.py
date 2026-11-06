from __future__ import annotations

import math
from datetime import date
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import next_session, sessions
from us_quant.config import QuantError
from us_quant.protective_source_mass_audit_v1 import _hex

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/confidence-regret-scope-v1.json"
N, TOLERANCE = 6, Fraction.from_float(1e-12)
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_confidence_regret_scope_20261106_v1",
        "actual_native_event_seq": 194677,
        "prior_journal_sequence": 1490,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/confidence-regret-scope-20261106-v1/"
            "source-inputs-v1.json",
            "sha256": "789702be87bbb2702ce21253650cb71f04edb64e5370d1ec466c060786f55dad",
        },
        "source_decisions": 121,
        "mature_source_months": 119,
        "expert_coordinates": N,
        "source_controller_records": 242,
        "saved_main_books": 4,
        "original_main_return_rows": 7532,
        "complete_main_month_cases": 356,
        "expected_main_months": {
            "10y-base": 119,
            "10y-stress": 119,
            "5y-base": 59,
            "5y-stress": 59,
        },
        "expected_boundary_days_per_book": 21,
        "expected_input_resolution_records": 962,
        "confidence_groups": ["zero", "partial", "full"],
        "comparator_identity": "plain_excess_equals_confidence_excess_plus_omitted_excess",
        "arithmetic": "exact_Fractions_of_saved_decimal_and_binary64_operands",
        "stored_rounding_coherence_limit": 1e-12,
        "confidence_clock": "prior_month_origin_not_end_month_lookahead",
        "main_month_statistic": "exact_product_of_saved_binary64_NET_return_relatives",
        "initial_partial_and_terminal_boundary_returns_retained": True,
        "source_learner_or_target_or_account_or_prior_diagnosis_replayed": False,
        "new_financial_trials": 0,
        "new_rankings_targets_or_accounts": 0,
        "new_source_or_RF_or_event_Adapter": False,
        "confidence_is_probability_or_source_admission": False,
        "confidence_regret_is_main_wealth": False,
        "retrospective_association_is_causal_forecast_validation": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze saved comparator scope, original clocks and descriptive-only interpretation.",
    )


def decimal_fraction(value):
    require(type(value) is str, "Preserve the actual saved decimal operand,not a cast float.")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise QuantError("The saved decimal operand is malformed.") from error
    require(parsed.is_finite(), "A nonfinite stored source-game value cannot be repaired.")
    return Fraction(parsed)


def encode(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def decode(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def confidence_vector(values):
    row = np.asarray(values)
    require(
        row.shape == (N,)
        and row.dtype.kind in "fi"
        and np.isfinite(row).all()
        and (row >= 0).all()
        and (row <= 1).all()
        and np.array_equal(row[:5], np.ones(5)),
        "Retain five always-participating policies and the recorded bounded specialist rating.",
    )
    return [Fraction.from_float(float(v)) for v in row]


def inspect_game(before, after, previous):
    day, audit = before["decision_session"], after["audit"]
    require(
        type(day) is str
        and str(date.fromisoformat(day)) == day
        and before["phase"] == "before_decision"
        and after["phase"] == "after_decision"
        and after["decision_session"] == audit["decision_session"] == day
        and before["confidence_wealth_input"] == audit["confidence_wealth_input"]
        and audit["confidence_wealth_input"]["as_of"] == day
        and audit["confidence_wealth_input"]["main_wealth_or_future_shadow_used"] is False
        and (previous is None or day > previous["decision_session"]),
        "Inspect exact saved same-source decisions without changing information clocks.",
    )
    current = confidence_vector(audit["current_confidences"])
    if previous is not None:
        require(
            before["state"] == previous["learner_state"]
            and before["previous_confidences_hex"]
            == [float(v).hex() for v in previous["current_confidences"]],
            "Keep the actual previous saved learner state and confidence origin.",
        )
    update = audit["period_update"]
    if update is None:
        require(
            audit["complete_learning_intervals"] == before["state"]["complete_intervals"] == 0
            and before["cached_loss_labels_hex"] is None
            and not before["matured_period_dates"],
            "Unscored initial and partial-month observations cannot acquire a fabricated loss.",
        )
        return {
            "decision_session": day,
            "mature": False,
            "end_confidences": [float(v) for v in current],
        }
    require(previous is not None, "A complete source month needs its actual prior decision.")
    confidence = confidence_vector(update["confidence_held_from_prior_origin"])
    require(
        [float(v).hex() for v in confidence] == before["previous_confidences_hex"]
        and update["start_after"] == previous["decision_session"]
        and update["end"] == day
        and audit["complete_learning_intervals"] == before["state"]["complete_intervals"] + 1
        and update["state_after"] == audit["learner_state"],
        "End-month confidence cannot retroactively weight the completed source loss.",
    )
    labels = _hex(before["cached_loss_labels_hex"], N)
    require(
        ((0 <= labels) & (labels <= 1)).all()
        and [float(v).hex() for v in labels]
        == [float(v).hex() for v in update["input_loss_labels"]],
        "Use unchanged sealed binary64 source losses in their original coordinates.",
    )
    dates = sessions(next_session(pd.Timestamp(update["start_after"])), day)
    require(
        dates.strftime("%Y-%m-%d").tolist() == before["matured_period_dates"],
        "A missing, cropped or shifted source month cannot be filled.",
    )
    played = decimal_fraction(update["played_loss_decimal"])
    require(
        -TOLERANCE <= played <= 1 + TOLERANCE, "Keep the saved bounded source-game played loss."
    )
    plain = [played - Fraction.from_float(float(v)) for v in labels]
    rated = [c * value for c, value in zip(confidence, plain, strict=True)]
    omitted = [(1 - c) * value for c, value in zip(confidence, plain, strict=True)]
    require(
        all(p == r + o for p, r, o in zip(plain, rated, omitted, strict=True)),
        "The signed exact comparator decomposition must reconcile without residual.",
    )
    stored = [decimal_fraction(v) for v in update["confidence_excess_decimal"]]
    old_regret = [decimal_fraction(v) for v in before["state"]["confidence_regret_decimal"]]
    new_regret = [decimal_fraction(v) for v in audit["learner_state"]["confidence_regret_decimal"]]
    require(
        len(stored) == len(old_regret) == len(new_regret) == N,
        "Every original source-game coordinate must remain.",
    )
    errors = [abs(a - b) for a, b in zip(stored, rated, strict=True)] + [
        abs(new - old - increment)
        for new, old, increment in zip(new_regret, old_regret, stored, strict=True)
    ]
    require(max(errors) <= TOLERANCE, "Stored rounding coherence failed;do not alter old operands.")
    return {
        "decision_session": day,
        "start_after": update["start_after"],
        "mature": True,
        "source_return_dates": before["matured_period_dates"],
        "origin_confidences": [float(v) for v in confidence],
        "end_confidences": [float(v) for v in current],
        "plain_excess": [encode(v) for v in plain],
        "confidence_excess_exact": [encode(v) for v in rated],
        "omitted_excess": [encode(v) for v in omitted],
        "stored_confidence_regret_after_decimal": audit["learner_state"][
            "confidence_regret_decimal"
        ],
        "maximum_stored_rounding_error": float(max(errors)),
        "source_game_units_are_main_returns_or_USD": False,
    }


def group(value):
    require(
        math.isfinite(value) and 0 <= value <= 1, "Confidence group needs its exact bounded value."
    )
    return "zero" if value == 0 else ("full" if value == 1 else "partial")


def summarize_games(rows, identities):
    mature = [row for row in rows if row["mature"]]
    require(
        mature and len(identities) == len(set(identities)) == N,
        "Keep nonempty complete-source-game evidence with distinct original identities.",
    )
    sources = {}
    for i, identifier in enumerate(identities):
        plain = sum((decode(row["plain_excess"][i]) for row in mature), Fraction(0))
        rated = sum((decode(row["confidence_excess_exact"][i]) for row in mature), Fraction(0))
        omitted = sum((decode(row["omitted_excess"][i]) for row in mature), Fraction(0))
        final = decimal_fraction(mature[-1]["stored_confidence_regret_after_decimal"][i])
        require(
            plain == rated + omitted and abs(final - rated) <= TOLERANCE,
            "The whole-history signed comparator and saved cumulative regret must reconcile.",
        )
        sources[identifier] = {
            "complete_source_months": len(mature),
            "unweighted_source_game_regret": float(plain),
            "confidence_weighted_source_game_regret": float(rated),
            "omitted_participation_excess": float(omitted),
            "recorded_final_confidence_regret": float(final),
            "maximum_final_rounding_difference": float(abs(final - rated)),
            "sum_confidence_ratings": math.fsum(row["origin_confidences"][i] for row in mature),
            "exact_confidence_groups": {
                kind: sum(group(row["origin_confidences"][i]) == kind for row in mature)
                for kind in ("zero", "partial", "full")
            },
            "unweighted_excess_positive_months": sum(
                decode(row["plain_excess"][i]) > 0 for row in mature
            ),
            "unweighted_excess_negative_months": sum(
                decode(row["plain_excess"][i]) < 0 for row in mature
            ),
        }
    return {
        "source_decisions": len(rows),
        "complete_source_months": len(mature),
        "sources": sources,
        "comparator_identity_exact": True,
        "source_game_units_are_main_returns_or_USD": False,
        "confidence_sum_is_independent_sample_size": False,
        "new_learner_state_or_allocation_generated": False,
    }


def inspect_main_month(row):
    game = row["game_month"]
    dates = pd.DatetimeIndex(row["return_dates"])
    require(
        row["path_key"] in PATHS
        and game["mature"] is True
        and dates.equals(
            sessions(next_session(pd.Timestamp(game["start_after"])), game["decision_session"])
        )
        and row["return_dates"] == game["source_return_dates"],
        "Match only the complete original source month inside the saved main book.",
    )
    returns = _hex(row["NET_returns_hex"], len(dates))
    require(
        (returns > -1).all(), "A nonfinite or insolvent saved return cannot become a safe month."
    )
    relative = math.prod(1 + Fraction.from_float(float(v)) for v in returns)
    start = float(confidence_vector(game["origin_confidences"])[-1])
    end = float(confidence_vector(game["end_confidences"])[-1])
    return {
        "path_key": row["path_key"],
        "start_after": game["start_after"],
        "end": game["decision_session"],
        "formal_return_rows": len(dates),
        "origin_feedback_confidence": start,
        "end_feedback_confidence": end,
        "origin_confidence_group": group(start),
        "confidence_change": "increased"
        if end > start
        else ("decreased" if end < start else "same"),
        "main_compounded_relative": float(relative),
        "main_net_loss": relative < 1,
        "main_net_positive": relative > 1,
        "retrospective_association_only": True,
        "new_account_or_annual_metric_computed": False,
    }


def summarize_main(rows):
    losses = [row for row in rows if row["main_net_loss"]]
    return {
        "complete_main_months": len(rows),
        "complete_main_return_rows": sum(row["formal_return_rows"] for row in rows),
        "main_loss_months": len(losses),
        "main_loss_months_by_origin_confidence": {
            kind: sum(row["origin_confidence_group"] == kind for row in losses)
            for kind in ("zero", "partial", "full")
        },
        "main_loss_months_by_later_confidence_change": {
            kind: sum(row["confidence_change"] == kind for row in losses)
            for kind in ("increased", "decreased", "same")
        },
        "main_loss_months_zero_origin_then_positive_end_confidence": sum(
            row["origin_feedback_confidence"] == 0 and row["end_feedback_confidence"] > 0
            for row in losses
        ),
        "confidence_is_calibrated_loss_probability": False,
        "source_game_regret_is_main_drawdown_protection": False,
        "overlapping_windows_or_scenarios_are_independent": False,
    }
