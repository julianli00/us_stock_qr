import copy
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from us_quant import pair_score_persistence_diagnosis_v1 as rules
from us_quant.config import QuantError


def example():
    dates = rules.risk.future_calendar(pd.Timestamp("2017-04-28"))
    values = np.column_stack(
        (
            0.001 + 0.004 * np.sin(np.arange(len(dates))),
            0.0004 + 0.003 * np.cos(np.arange(len(dates))),
        )
    )
    rf = np.full(len(dates), 0.0001)
    pair = sorted(rules.IDS[:2])
    weights = [0.0] * 8
    for name in pair:
        weights[rules.IDS.index(name)] = 0.5
    case = {
        "decision_session": "2017-04-28",
        "pair": pair,
        "was_original_selected_pair": True,
        "required_future_start": str(dates[0].date()),
        "required_future_end": str(dates[-1].date()),
        "required_future_sessions": len(dates),
        "available_future_sessions": len(dates),
    }
    before = {
        "record": "new_diagnostic_input",
        "case": case,
        "stored_pair_record": {
            "pair": pair,
            "expert_weights": weights,
            "selection_score": 0.75,
            "selection_score_key": 0.75,
            "train_net_risk_score": 1.0,
            "validation_net_risk_score": 0.75,
            "estimated_incremental_switch_cost_fraction": 0.001,
            "cost_proxy_is_actual_main_account_cost": False,
        },
        "future_dates": dates.strftime("%Y-%m-%d").tolist(),
        "original_source_NET_binary64_hex": [[float(v).hex() for v in row] for row in values],
        "original_source_RF_binary64_hex": [float(v).hex() for v in rf],
    }
    resolved = {
        "record": "new_diagnostic_resolved",
        "result": {
            **case,
            "status": "complete_future_risk_evaluated",
            "future_portfolio_variance": float(np.var((values - rf[:, None]).mean(axis=1), ddof=1)),
        },
    }
    return before, resolved, values, rf


def test_fixed_future_mean_uses_original_proxy_and_stored_variance_not_new_covariance(monkeypatch):
    before, resolved, values, rf = example()

    def forbidden(*args, **kwargs):
        raise AssertionError("A closed modeling pipeline must not be replayed.")

    monkeypatch.setattr(rules.risk, "run", forbidden)
    monkeypatch.setattr(rules.risk, "risk_case", forbidden)
    monkeypatch.setattr(rules.saved.rules, "choose_pair", forbidden)
    row = rules.assess(before, resolved)
    expected = (
        math.sqrt(252)
        * (((values - rf[:, None]).mean()) - 0.001 / 21)
        / math.sqrt(resolved["result"]["future_portfolio_variance"])
    )
    assert row["future_reference_assessment"] == pytest.approx(expected)
    assert row["independent_normalized_assessment_difference"] < 1e-8
    assert not row["new_variance_or_covariance_estimate"]


def test_censored_full_month_retained_without_partial_outcome():
    before, resolved, _, _ = example()
    for key in (
        "future_dates",
        "original_source_NET_binary64_hex",
        "original_source_RF_binary64_hex",
    ):
        before[key] = before[key][:3]
    before["case"]["available_future_sessions"] = 3
    resolved["result"].update(before["case"])
    resolved["result"]["status"] = "future_month_incomplete_not_evaluated"
    row = rules.assess(before, resolved)
    assert row["status"] == "future_month_incomplete_not_assessed"
    assert row["future_reference_assessment"] is None


def test_exact_zero_future_risk_is_undefined_not_a_safe_score():
    before, resolved, _, _ = example()
    resolved["result"]["status"] = "complete_future_zero_portfolio_risk"
    resolved["result"]["future_portfolio_variance"] = 0.0
    row = rules.assess(before, resolved)
    assert row["future_reference_assessment"] is None
    assert row["status"] == "complete_future_zero_portfolio_risk"


@pytest.mark.parametrize(
    "kind", ["future_date", "nonfinite", "variance", "third_source", "score_key"]
)
def test_corrupt_source_clock_values_weights_or_old_score_are_not_repaired(kind):
    before, resolved, _, _ = example()
    if kind == "future_date":
        before["future_dates"][0] = "2017-04-28"
    elif kind == "nonfinite":
        before["original_source_NET_binary64_hex"][0][0] = "nan"
    elif kind == "variance":
        resolved["result"]["future_portfolio_variance"] = -1.0
    elif kind == "third_source":
        weights = before["stored_pair_record"]["expert_weights"]
        weights[2] = 0.1
        weights[0] -= 0.1
    else:
        before["stored_pair_record"]["selection_score_key"] = 0.5
    with pytest.raises(QuantError):
        rules.assess(before, resolved)


def monthly_rows():
    return [
        {
            "decision_session": "2017-04-28",
            "pair": sorted(pair),
            "was_original_selected_pair": i == 0,
            "past_score_key": float(i),
            "future_reference_assessment": float(i),
            "status": "complete_future_reference_assessed",
        }
        for i, pair in enumerate(itertools.combinations(rules.IDS, 2))
    ]


def test_full28_midrank_and_tau_keep_actual_selected_flag_without_future_chooser():
    rows = monthly_rows()
    result = rules.month(rows)
    assert result["evaluable_pairs"] == 28
    assert result["selected_future_midrank_percentile"] == 0.5 / 28
    assert result["tau"]["tau_b"] == 1
    assert rows[0]["was_original_selected_pair"]
    assert not rows[-1]["was_original_selected_pair"]


def test_exact_ties_get_half_rank_and_tau_null_without_denominator_floor():
    rows = monthly_rows()
    for row in rows:
        row["past_score_key"] = 1.0
        row["future_reference_assessment"] = 2.0
    result = rules.month(rows)
    assert result["selected_future_midrank_percentile"] == 0.5
    assert result["tau"]["tau_b"] is None
    assert result["tau"]["status"] == "undefined_tied_population"
    assert result["tau"]["both_ties"] == 378


def test_tau_b_handles_past_only_and_future_only_ties_exactly():
    result = rules.tau_b([1.0, 1.0, 2.0], [1.0, 2.0, 2.0])
    assert result["concordant"] == 1
    assert result["past_only_ties"] == 1
    assert result["future_only_ties"] == 1
    assert result["tau_b"] == 0.5


def test_selected_undefined_is_retained_not_dropped_as_a_success():
    rows = monthly_rows()
    rows[0]["status"] = "complete_future_zero_portfolio_risk"
    rows[0]["future_reference_assessment"] = None
    result = rules.month(rows)
    assert result["status"] == "selected_future_assessment_undefined"
    assert result["selected_future_midrank_percentile"] is None
    assert result["evaluable_pairs"] == 27


def test_missing_or_duplicate_pair_is_not_a_smaller_successful_month():
    rows = monthly_rows()
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(QuantError):
        rules.month(rows)


def test_truncated_source_input_resolved_stream_fails(tmp_path):
    before, _, _, _ = example()
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(before) + "\n")
    with pytest.raises(QuantError):
        list(rules.read_pairs(path))


def test_second_complete_or_partial_actual_diagnosis_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(rules.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(rules, "FOLDER", tmp_path)
    monkeypatch.setattr(rules, "RESULT", tmp_path / "result.json")
    (tmp_path / "attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="never repair and retry"):
        rules.run()


def test_original_unequal_weights_follow_saved_pair_column_order():
    before, resolved, values, rf = example()
    pair = before["case"]["pair"]
    before["stored_pair_record"]["expert_weights"] = [
        0.25 if name == pair[0] else (0.75 if name == pair[1] else 0.0) for name in rules.IDS
    ]
    row = rules.assess(before, resolved)
    expected_mean = float(np.mean((values - rf[:, None]) @ [0.25, 0.75]))
    assert row["future_source_NET_RF_mean"] == pytest.approx(expected_mean)


def test_public_report_contains_no_raw_pair_outcomes_targets_or_source_observations():
    result = {
        "completed_at": "2026-11-04T19:13:44Z",
        "intent": {"sha256": "synthetic"},
        "aggregates": rules.aggregate([rules.month(monthly_rows())]),
        "maximum_independent_scalar_error": 0.0,
        "financial_counts_unchanged": {},
    }
    public = rules.public_report(result)
    assert public["new_financial_trials"] == 0
    assert not public["investment_objective_verified"]
    text = json.dumps(public)
    assert all(
        word not in text
        for word in (
            "binary64_hex",
            "expert_weights",
            "NET_returns",
            "decision_session",
            "future_source_NET_RF_mean",
            'future_reference_assessment"',
        )
    )
