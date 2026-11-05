from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant import saved_inclusion_execution_v1 as saved
from us_quant import ssd_signed_activity_execution_diagnosis_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def meta():
    ids = tuple(f"expert{i}" for i in range(9))
    assets = ["A", "B", "C", "BIL"]
    target, extra = [0.49, 0.49, 0, 0], [0, 0, 0, 0.98]
    observations = [
        {
            "id": name,
            "mechanism_group": name,
            "introduced_at": "2026-11-05T00:00:00Z",
            "as_of": "2020-10-30",
            "target_known_at": "2020-09-30",
            "target": target if i < 8 else extra,
            "asset_order": assets,
            "return_dates": ["2020-10-29"],
            "NET_returns": [0.001],
            "RF": [0.0001],
            "score_mature": True,
        }
        for i, name in enumerate(ids)
    ]
    before = {
        "record": "decision_input",
        "decision_index": 0,
        "as_of": "2020-10-30",
        "observations": observations,
    }

    def after(count, names, weights, rolling):
        return {
            "record": "decision_resolved",
            "decision_index": 0,
            "asset_order": assets,
            "audit": {
                "decision_session": "2020-10-30",
                "rolling_expert_weights": weights,
                "selection": {"selected_pair": names, "scores_available": True},
            },
            "combined_targets": {
                f"eq{count}": target
                if count == 8
                else ((8 * np.array(target) + extra) / 9).tolist(),
                f"roll{count}": rolling,
            },
        }

    eight = deepcopy(before)
    eight["observations"] = eight["observations"][:8]
    after8 = after(8, list(ids[:2]), [0.5, 0.5] + [0] * 6, target)
    after9 = after(
        9,
        [ids[1], ids[-1]],
        [0, 0.4] + [0] * 6 + [0.6],
        (0.4 * np.array(target) + 0.6 * np.array(extra)).tolist(),
    )
    return (
        (eight, after8),
        (deepcopy(before), after9),
        {"experts": ids, "old8": ("eq8", "roll8"), "new9": ("eq9", "roll9")},
    )


def books():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    old = pd.DataFrame({"orders": [2, 0, 1, 0], "cost": [2.2, 0, 1.1, 0]}, index=dates)
    new = pd.DataFrame({"orders": [2, 1, 0, 0], "cost": [2.4, 1.3, 0, 0]}, index=dates)

    def rows(reasons):
        return [
            {
                "signal_session": signal,
                "execution_session": execution,
                "initial_capital": i == 0,
                "reason": reason,
                "maximum_observed_target_gap": 0.01 if reason == "hold" else 0.98,
            }
            for i, (signal, execution, reason) in enumerate(
                zip(
                    ["2020-01-01", "2020-01-02", "2020-01-03"],
                    ["2020-01-02", "2020-01-03", "2020-01-06"],
                    reasons,
                    strict=True,
                )
            )
        ]

    return (
        old,
        new,
        rows(["initial_capital", "hold", "observed_weight_gap"]),
        rows(["initial_capital", "observed_weight_gap", "hold"]),
    )


def test_actual_pair_participation_is_not_initial_equal_weight_or_a_new_scorer():
    old, new, ids = meta()
    result = saved.inspect_meta(old, new, ids)
    assert result["new_selected_pair_contains_extra"]
    assert result["new_extra_statistical_weight"] == 0.6
    assert result["new_extra_opinion_is_actual_BIL_only"]
    assert result["same_original_eight_observations"]
    assert result["equal_inclusion_max_residual"] < 1e-12
    assert result["independent_fraction_equal_inclusion_max_residual"] < 1e-12


def test_unscored_uniform_extra_budget_is_not_selected_pair_success():
    old, new, ids = meta()
    for pair, count in ((old, 8), (new, 9)):
        pair[1]["audit"]["rolling_expert_weights"] = [1 / count] * count
        pair[1]["audit"]["selection"] = {"scores_available": False, "selected_pair": None}
    result = saved.inspect_meta(old, new, ids)
    assert not result["new_scored"] and not result["new_selected_pair_contains_extra"]
    assert result["new_extra_statistical_weight"] == 1 / 9


@pytest.mark.parametrize(
    "failure",
    ["future", "original8", "index", "asset", "pair", "weights", "budget", "identity", "string"],
)
def test_no_missing_source_identity_clock_or_allocation_repair(failure):
    old, new, ids = meta()
    if failure == "future":
        new[0]["observations"][-1]["target_known_at"] = "2020-11-30"
    elif failure == "original8":
        new[0]["observations"][0]["NET_returns"] = [0.1]
    elif failure == "index":
        old[1]["decision_index"] = new[1]["decision_index"] = 1
    elif failure == "asset":
        new[1]["asset_order"] = list(reversed(new[1]["asset_order"]))
    elif failure == "pair":
        new[1]["audit"]["selection"]["selected_pair"] = ["foreign", "expert8"]
    elif failure == "weights":
        new[1]["audit"]["rolling_expert_weights"] = [1 / 9] * 9
    elif failure == "budget":
        new[0]["observations"][-1]["target"] = [0, 0, 0, 1]
    elif failure == "identity":
        new[1]["combined_targets"]["eq9"] = [0, 0, 0, 0.98]
    else:
        new[0]["observations"][-1]["target"] = ["0", "0", "0", "0.98"]
    with pytest.raises(QuantError):
        saved.inspect_meta(old, new, ids)


def test_actual_fee_ticket_identity_retains_positive_and_negative_group_changes():
    rows, result = saved.compare_execution(*books())
    assert result["group_counts"] == {
        "both_issued": 1,
        "only_old_issued": 1,
        "only_new_issued": 1,
        "neither_issued": 0,
    }
    assert result["total_cost_difference_USD"] == pytest.approx(0.4)
    assert result["fixed_ticket_difference_USD"] == 0
    assert result["proportional_fee_difference_USD"] == pytest.approx(0.4)
    assert result["group_cost_difference_USD"]["only_old_issued"] == -1.1
    assert rows[0]["initial_capital"] is True


def test_issued_numeric_noop_and_outside_window_censor_are_not_dropped():
    old, new, left, right = books()
    right[2]["reason"] = "observed_weight_gap"
    late = {
        "signal_session": "2020-01-07",
        "execution_session": "2020-01-08",
        "initial_capital": False,
        "reason": "observed_weight_gap",
        "maximum_observed_target_gap": 0.1,
    }
    left.append(late.copy())
    right.append(late.copy())
    rows, result = saved.compare_execution(old, new, left, right)
    assert result["new_issued_numeric_noops"] == 1
    assert result["censored_cases"] == 1
    assert rows[-1]["status"] == "outside_original_window_censored"
    assert "new_cost_USD" not in rows[-1]


@pytest.mark.parametrize(
    "failure",
    [
        "hidden_cost",
        "hold_fill",
        "fractional_order",
        "low_fee",
        "missing_date",
        "duplicate_clock",
        "retime",
        "missing_decision",
        "nan",
        "negative",
        "commission",
    ],
)
def test_unmatched_missing_invalid_or_altered_saved_execution_is_explicit(failure):
    old, new, left, right = books()
    commission = 1.0
    if failure == "hidden_cost":
        new.iloc[-1] = [1, 1.1]
    elif failure == "hold_fill":
        new.iloc[2] = [1, 1.1]
    elif failure == "fractional_order":
        new = new.astype(float)
        new.iloc[0, 0] = 1.5
    elif failure == "low_fee":
        new.iloc[0, 1] = 1.9
    elif failure == "missing_date":
        new = new.iloc[1:]
    elif failure == "duplicate_clock":
        right[2]["execution_session"] = right[1]["execution_session"]
    elif failure == "retime":
        right[1]["execution_session"] = "2020-01-07"
    elif failure == "missing_decision":
        right.pop()
    elif failure == "nan":
        new.iloc[0, 1] = np.nan
    elif failure == "negative":
        new.iloc[0, 1] = -1
    else:
        commission = 0
    with pytest.raises(QuantError):
        saved.compare_execution(old, new, left, right, ticket_usd=commission)


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_financial_evaluations", 1),
        ("model_ticket_USD", 0.0),
        ("money_reconciliation_tolerance_USD", 0.01),
        ("cost_difference_is_causal_marginal_expert_effect", True),
    ],
)
def test_criteria_do_not_become_financial_rule_or_causal_claim(key, value):
    policy = read_json(study.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        study.validate(policy)
