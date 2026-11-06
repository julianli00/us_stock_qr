from copy import deepcopy

import numpy as np
import pytest

from us_quant import expert_opinion_clock_review_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def row(day="2020-01-30", known="2020-01-30"):
    target = np.zeros(14)
    target[0] = 0.98
    return {
        "expert_id": "synthetic",
        "mechanism_group": "complete_synthetic_policy",
        "introduced_at": "2026-11-06T00:00:00+00:00",
        "as_of": day,
        "target_known_at": known,
        "asset_order": list(rules.ASSETS),
        "target_hex": [float(v).hex() for v in target],
        "assessment_RF_matches_original_view": True,
        "source_window_exact_calendar": True,
        "source_return_count": 126,
        "source_score_mature": True,
    }


def test_initial_observation_is_not_backdated_refresh():
    actual = rules.inspect(row(known="2019-12-31"), None)
    assert actual["initial_observation"]
    assert actual["initial_observes_prior_dated_opinion"]
    assert not actual["new_same_day_known_row"]
    assert not actual["known_row_is_native_request_or_execution"]


def test_carried_and_repeated_current_rows_are_distinct():
    previous = row()
    carried = row(day="2020-01-31")
    assert rules.inspect(carried, previous)["carried_known_timestamp"]
    refreshed = row(day="2020-01-31", known="2020-01-31")
    actual = rules.inspect(refreshed, previous)
    assert actual["new_same_day_known_row"] and actual["new_same_day_row_same_numeric_target"]
    assert not actual["off_month_end_new_same_day_row"]


def test_non_month_end_changed_opinion_is_not_main_order():
    previous = row(day="2020-01-28", known="2020-01-28")
    current = row(day="2020-01-29", known="2020-01-29")
    current["target_hex"][0] = (0.97).hex()
    current["target_hex"][3] = (0.01).hex()
    actual = rules.inspect(current, previous)
    assert actual["off_month_end_changed_opinion"]
    assert not actual["new_main_target_or_rebalance_generated"]


def test_delayed_new_known_row_is_not_same_day_refresh():
    previous = row(day="2020-01-28", known="2020-01-27")
    current = row(day="2020-01-30", known="2020-01-29")
    actual = rules.inspect(current, previous)
    assert actual["newly_observed_prior_dated_row"]
    assert not actual["new_same_day_known_row"]


def test_signed_zero_equal_but_tiny_numeric_change_remains_different():
    previous = row()
    current = row(day="2020-01-31")
    current["target_hex"][3] = (-0.0).hex()
    assert not rules.inspect(current, previous)["numeric_target_changed"]
    current["target_known_at"] = "2020-01-31"
    current["target_hex"][3] = (1e-300).hex()
    assert rules.inspect(current, previous)["numeric_target_changed"]


@pytest.mark.parametrize(
    "kind",
    [
        "future",
        "backward",
        "same_clock_changed",
        "identity",
        "RF",
        "calendar",
        "maturity",
        "count",
        "asset",
        "negative",
    ],
)
def test_invalid_observation_never_fills_or_admits(kind):
    previous = row()
    current = row(day="2020-01-31")
    if kind == "future":
        current["target_known_at"] = "2020-02-03"
    elif kind == "backward":
        current["target_known_at"] = "2020-01-29"
    elif kind == "same_clock_changed":
        current["target_hex"][0], current["target_hex"][3] = (0.97).hex(), (0.01).hex()
    elif kind == "identity":
        current["expert_id"] = "foreign"
    elif kind == "RF":
        current["assessment_RF_matches_original_view"] = False
    elif kind == "calendar":
        current["source_window_exact_calendar"] = False
    elif kind == "maturity":
        current["source_score_mature"] = False
    elif kind == "count":
        current["source_return_count"] = True
    elif kind == "asset":
        current["asset_order"].reverse()
    else:
        current["target_hex"][1] = (-0.01).hex()
    with pytest.raises(QuantError):
        rules.inspect(current, previous)


def test_prefix_is_independent_of_future_source_change():
    rows = [row("2020-01-28", "2020-01-28"), row("2020-01-29", "2020-01-29")]
    expected = [rules.inspect(rows[0], None), rules.inspect(rows[1], rows[0])]
    changed = deepcopy(rows) + [row("2020-01-30", "2030-01-01")]
    assert [rules.inspect(changed[0], None), rules.inspect(changed[1], changed[0])] == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("known_opinion_rows_are_native_requests", True),
        ("source_assessment_RF_view_changed", True),
        ("new_rankings_targets_accounts_or_financial_trials", True),
    ],
)
def test_scope_and_interpretation_cannot_be_promoted(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
