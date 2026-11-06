import copy
from decimal import Decimal, localcontext
from fractions import Fraction

import pandas as pd
import pytest

from us_quant import confidence_regret_scope_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json

IDS = tuple(f"synthetic_complete_policy_{i}" for i in range(6))


def initial(day, confidence=0.25, previous=None):
    state = (
        {"complete_intervals": 0, "confidence_regret_decimal": ["0"] * 6}
        if previous is None
        else copy.deepcopy(previous["learner_state"])
    )
    wealth = {"as_of": day, "main_wealth_or_future_shadow_used": False}
    before = {
        "phase": "before_decision",
        "decision_session": day,
        "state": state,
        "confidence_wealth_input": wealth,
        "previous_confidences_hex": None
        if previous is None
        else [float(v).hex() for v in previous["current_confidences"]],
        "cached_loss_labels_hex": None,
        "matured_period_dates": [],
    }
    audit = {
        "decision_session": day,
        "confidence_wealth_input": wealth,
        "current_confidences": [1.0] * 5 + [confidence],
        "period_update": None,
        "complete_learning_intervals": 0,
        "learner_state": copy.deepcopy(state),
    }
    after = {"phase": "after_decision", "decision_session": day, "audit": audit}
    return before, after


def mature(previous, day="2016-11-30", current_confidence=0.75, played="0.5"):
    labels = [0.0, 0.25, 0.5, 0.75, 1.0, 0.4]
    confidence = previous["current_confidences"]
    before, after = initial(day, current_confidence, previous)
    with localcontext() as context:
        context.prec = 160
        excess = [
            Decimal.from_float(c) * (Decimal(played) - Decimal.from_float(v))
            for c, v in zip(confidence, labels, strict=True)
        ]
        state = {
            "complete_intervals": previous["learner_state"]["complete_intervals"] + 1,
            "confidence_regret_decimal": [
                str(Decimal(old) + value)
                for old, value in zip(
                    previous["learner_state"]["confidence_regret_decimal"], excess, strict=True
                )
            ],
        }
    before["cached_loss_labels_hex"] = [v.hex() for v in labels]
    before["matured_period_dates"] = (
        rules.sessions(rules.next_session(pd.Timestamp(previous["decision_session"])), day)
        .strftime("%Y-%m-%d")
        .tolist()
    )
    update = {
        "start_after": previous["decision_session"],
        "end": day,
        "input_loss_labels": labels,
        "confidence_held_from_prior_origin": confidence.copy(),
        "played_loss_decimal": played,
        "confidence_excess_decimal": [str(v) for v in excess],
        "state_after": copy.deepcopy(state),
    }
    after["audit"].update(
        period_update=update,
        complete_learning_intervals=state["complete_intervals"],
        learner_state=state,
    )
    return before, after


def fixture(confidence=0.25, played="0.5"):
    _, first = initial("2016-10-05", confidence)
    _, anchor = initial("2016-10-31", confidence, first["audit"])
    pair = mature(anchor["audit"], played=played)
    return pair, anchor["audit"]


def test_exact_signed_comparator_identity_and_prior_confidence():
    (before, after), previous = fixture()
    result = rules.inspect_game(before, after, previous)
    plain = rules.decode(result["plain_excess"][-1])
    rated = rules.decode(result["confidence_excess_exact"][-1])
    omitted = rules.decode(result["omitted_excess"][-1])
    assert plain == Fraction(1, 2) - Fraction.from_float(0.4)
    assert rated == plain / 4 and omitted == 3 * plain / 4 and plain == rated + omitted
    assert result["origin_confidences"][-1] == 0.25
    assert result["end_confidences"][-1] == 0.75
    assert result["maximum_stored_rounding_error"] == 0
    assert result["source_game_units_are_main_returns_or_USD"] is False


def test_negative_excess_is_not_absolute_or_clipped_to_zero():
    (before, after), previous = fixture(played="0.25")
    result = rules.inspect_game(before, after, previous)
    assert rules.decode(result["plain_excess"][-1]) < 0
    assert rules.decode(result["confidence_excess_exact"][-1]) < 0
    summary = rules.summarize_games([result], IDS)["sources"][IDS[-1]]
    assert (
        summary["unweighted_source_game_regret"]
        < summary["confidence_weighted_source_game_regret"]
        < 0
    )
    assert summary["omitted_participation_excess"] < 0


def test_zero_participation_preserves_unweighted_comparator_difference():
    (before, after), previous = fixture(confidence=0.0)
    result = rules.inspect_game(before, after, previous)
    assert rules.decode(result["confidence_excess_exact"][-1]) == 0
    assert rules.decode(result["plain_excess"][-1]) == rules.decode(result["omitted_excess"][-1])
    summary = rules.summarize_games([result], IDS)["sources"][IDS[-1]]
    assert summary["exact_confidence_groups"] == {"zero": 1, "partial": 0, "full": 0}
    assert summary["sum_confidence_ratings"] == 0


def test_saved_decimal_rounding_is_disclosed_without_repairing_old_state():
    (before, after), previous = fixture(played="0." + "3" * 160)
    frozen = copy.deepcopy((before, after, previous))
    result = rules.inspect_game(before, after, previous)
    assert 0 < result["maximum_stored_rounding_error"] <= 1e-12
    assert (before, after, previous) == frozen


def test_initial_and_partial_source_months_do_not_acquire_losses():
    first, after_first = initial("2016-10-05")
    a = rules.inspect_game(first, after_first, None)
    second, after_second = initial("2016-10-31", previous=after_first["audit"])
    b = rules.inspect_game(second, after_second, after_first["audit"])
    assert a["mature"] is False and b["mature"] is False
    assert "plain_excess" not in a and "plain_excess" not in b


@pytest.mark.parametrize(
    "kind",
    [
        "phase",
        "prior_confidence",
        "end_confidence",
        "label",
        "missing_day",
        "stored_excess",
        "cumulative",
        "state_clock",
        "wealth_clock",
    ],
)
def test_invalid_saved_operand_or_clock_stops_without_a_replacement(kind):
    (before, after), previous = fixture()
    if kind == "phase":
        before["phase"] = "after_decision"
    elif kind == "prior_confidence":
        after["audit"]["period_update"]["confidence_held_from_prior_origin"][-1] = 0.75
    elif kind == "end_confidence":
        after["audit"]["current_confidences"][-1] = float("nan")
    elif kind == "label":
        before["cached_loss_labels_hex"][0] = (0.01).hex()
    elif kind == "missing_day":
        before["matured_period_dates"] = before["matured_period_dates"][1:]
    elif kind == "stored_excess":
        after["audit"]["period_update"]["confidence_excess_decimal"][0] = "99"
    elif kind == "cumulative":
        after["audit"]["learner_state"]["confidence_regret_decimal"][0] = "99"
    elif kind == "state_clock":
        before["state"]["complete_intervals"] = 99
    else:
        before["confidence_wealth_input"]["as_of"] = "2016-12-30"
    with pytest.raises(QuantError):
        rules.inspect_game(before, after, previous)


def test_exact_cumulative_summary_and_unchanged_retained_prefix():
    (before, after), previous = fixture()
    first = rules.inspect_game(before, after, previous)
    second_pair = mature(after["audit"], "2016-12-30", current_confidence=1.0, played="0.25")
    second = rules.inspect_game(*second_pair, after["audit"])
    summary = rules.summarize_games([first, second], IDS)
    assert summary["complete_source_months"] == 2 and summary["comparator_identity_exact"] is True
    for identifier in IDS[:-1]:
        row = summary["sources"][identifier]
        assert row["unweighted_source_game_regret"] == row["confidence_weighted_source_game_regret"]
        assert row["omitted_participation_excess"] == 0
    old = digest_json(first)
    second_pair[1]["audit"]["period_update"]["played_loss_decimal"] = "999"
    assert digest_json(rules.inspect_game(before, after, previous)) == old


def month_case(returns=-0.001):
    (before, after), previous = fixture()
    game = rules.inspect_game(before, after, previous)
    return {
        "path_key": "10y-base",
        "game_month": game,
        "return_dates": game["source_return_dates"],
        "NET_returns_hex": [float(returns).hex()] * len(game["source_return_dates"]),
    }


def test_main_loss_is_exact_saved_NET_product_not_source_regret():
    value = month_case()
    result = rules.inspect_main_month(value)
    assert result["main_net_loss"] is True and result["main_net_positive"] is False
    assert result["main_compounded_relative"] < 1
    assert result["origin_feedback_confidence"] == 0.25
    assert result["confidence_change"] == "increased"
    summary = rules.summarize_main([result])
    assert summary["main_loss_months_by_origin_confidence"] == {"zero": 0, "partial": 1, "full": 0}
    assert summary["main_loss_months_by_later_confidence_change"]["increased"] == 1
    assert summary["source_game_regret_is_main_drawdown_protection"] is False


def test_zero_main_return_is_not_a_loss_or_positive_return():
    result = rules.inspect_main_month(month_case(0.0))
    assert result["main_compounded_relative"] == 1
    assert result["main_net_loss"] is result["main_net_positive"] is False


@pytest.mark.parametrize("kind", ["missing", "minus_one", "nan", "end_confidence", "path"])
def test_missing_or_changed_main_month_is_not_filled_or_retimed(kind):
    row = month_case()
    if kind == "missing":
        row["return_dates"] = row["return_dates"][1:]
    elif kind == "minus_one":
        row["NET_returns_hex"][0] = (-1.0).hex()
    elif kind == "nan":
        row["NET_returns_hex"][0] = float("nan").hex()
    elif kind == "end_confidence":
        row["game_month"]["end_confidences"][-1] = float("nan")
    else:
        row["path_key"] = "chosen_new_window"
    with pytest.raises(QuantError):
        rules.inspect_main_month(row)


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence_groups", ["low", "high"]),
        ("confidence_clock", "end_month_lookahead"),
        ("new_financial_trials", 1),
        ("source_learner_or_target_or_account_or_prior_diagnosis_replayed", True),
        ("retrospective_association_is_causal_forecast_validation", True),
        ("expected_input_resolution_records", 961),
    ],
)
def test_scope_and_noncausal_interpretation_cannot_be_changed(field, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[field] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
