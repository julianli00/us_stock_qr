import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from us_quant import hierarchical_risk_transfer_v1 as rules
from us_quant import ssd_hierarchical_risk_transfer_v1 as study
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json


def data():
    dates = sessions("2016-10-06", "2026-10-05")
    rng = np.random.default_rng(184910)
    rf = pd.Series(0.00001, index=dates)
    net = pd.DataFrame(
        rng.normal(0.0003, 0.008, size=(len(dates), 5)) + rf.to_numpy()[:, None],
        index=dates,
        columns=rules.EXPERT_IDS,
    )
    mains = {}
    for key in rules.PATHS:
        index = sessions(*rules.WINDOWS[key])
        values = net.loc[index].mean(axis=1).to_numpy()
        mains[key] = pd.DataFrame({"return": values, "risk_free": rf.loc[index]}, index=index)
    return net, rf, mains


def forecast(day, variance=0.0001):
    return {
        "decision_session": day,
        "expert_weights": [0.2] * 5,
        "allocation": {
            "reason": "full_126_NET_RF_hierarchical_risk_allocation",
            "risk_estimate_available": True,
            "risk_sample_end": day,
            "forecast_annualized_RF_excess_volatility": math.sqrt(252 * variance),
            "covariance": (np.eye(5) * variance / 0.2).tolist(),
        },
    }


def test_exact_recorded_forecast_projection_without_refitting():
    value, weights, gap = rules.forecast_variance(forecast("2021-09-30"))
    assert value == pytest.approx(0.0001) and gap <= 1e-12
    assert np.array_equal(weights, np.full(5, 0.2))


def test_two_variance_channels_are_independently_checked_and_distinct():
    rng = np.random.default_rng(14)
    net = rng.normal(0, 0.01, size=(23, 5))
    rf = np.zeros(23)
    weights = np.full(5, 0.2)
    main = (net @ weights) * 2
    result = rules.assess(0.0001, weights, net, rf, main)
    assert result["actual_main_NET_RF_daily_variance"] == pytest.approx(
        4 * result["source_statistical_NET_RF_daily_variance"]
    )
    assert result["independent_main_variance_difference"] <= 1e-18
    assert result["label_gap_is_causal_fee_or_source_profit_attribution"] is False
    assert result["source_mixture_is_main_account_or_tradable_portfolio"] is False


def test_exact_zero_future_variance_is_not_a_risk_floor_or_safe_label():
    net = np.full((21, 5), 0.00001)
    result = rules.assess(0.0001, np.full(5, 0.2), net, np.full(21, 0.00001), np.full(21, 0.00001))
    assert result["source_statistical_NET_RF_daily_variance"] == 0
    assert result["actual_main_NET_RF_daily_variance"] == 0
    assert result["exact_constant_source_mixture"] and result["exact_constant_main_excess_returns"]
    assert not result["zero_variance_is_safe_success"]


def test_nonzero_exact_constant_scalar_series_has_exact_zero_variance():
    assert rules.variance([0.0001] * 23) == 0


def test_zero_forecast_keeps_undefined_ratios_and_positive_future_risk():
    net = np.column_stack([np.arange(21, dtype=float) / 1000] * 5)
    result = rules.assess(0, np.full(5, 0.2), net, np.zeros(21), net[:, 0])
    assert result["source_to_forecast_variance_ratio"] is None
    assert result["main_to_forecast_variance_ratio"] is None
    assert result["zero_forecast_risk_ratio_undefined"]
    assert result["source_underforecast"] and result["main_underforecast"]


def test_unscored_initial_origin_keeps_four_contexts_without_new_forecast():
    net, rf, mains = data()
    audit = {
        "decision_session": "2016-10-05",
        "expert_weights": [0.2] * 5,
        "allocation": {
            "risk_estimate_available": False,
            "reason": "unscored_initial_equal_weights",
            "safety_or_success_claimed": False,
        },
    }
    rows = rules.diagnose([audit], net, rf, mains)
    assert len(rows) == 4
    assert all(r["status"] == "original_unscored_no_forecast" and r["label"] is None for r in rows)


def test_full_month_not_arbitrary_21_row_slice_and_partial_five_year_seed():
    net, rf, mains = data()
    rows = rules.diagnose([forecast("2021-09-30")], net, rf, mains)
    expected = rules.future_calendar(pd.Timestamp("2021-09-30"))
    assert all(r["required_label_sessions"] == len(expected) for r in rows)
    assert all(r["status"] == "complete_paired_risk_label" for r in rows[:2])
    assert all(r["status"] == "future_month_incomplete_not_assessed" for r in rows[2:])
    assert all(r["initial_window_boundary_censored"] and r["label"] is None for r in rows[2:])
    assert all(0 < r["available_main_sessions"] < len(expected) for r in rows[2:])


def test_terminal_partial_month_is_not_filled_or_shortened():
    net, rf, mains = data()
    rows = rules.diagnose([forecast("2026-09-30")], net, rf, mains)
    assert all(r["status"] == "future_month_incomplete_not_assessed" for r in rows)
    assert all(r["terminal_window_boundary_censored"] and r["label"] is None for r in rows)
    assert all(r["available_source_sessions"] == r["available_main_sessions"] == 3 for r in rows)


def test_outside_five_year_context_is_not_a_zero_return_label():
    net, rf, mains = data()
    rows = rules.diagnose([forecast("2017-04-28")], net, rf, mains)
    assert all(r["status"] == "outside_main_path_window_not_assessed" for r in rows[2:])
    assert all(r["available_main_sessions"] == 0 and r["label"] is None for r in rows[2:])


def test_already_complete_past_labels_and_forecasts_are_prefix_stable():
    net, rf, mains = data()
    audits = [forecast(d) for d in ("2017-04-28", "2021-09-30", "2021-10-29", "2026-09-30")]
    full = rules.diagnose(audits, net, rf, mains)
    cut = pd.Timestamp("2021-10-29")
    early_audits = [a for a in audits if pd.Timestamp(a["decision_session"]) <= cut]
    early = rules.diagnose(
        early_audits,
        net.loc[:cut],
        rf.loc[:cut],
        {k: frame.loc[:cut] for k, frame in mains.items()},
    )
    lookup = {(r["decision_session"], r["path_key"]): r for r in early}
    for row in full:
        key = row["decision_session"], row["path_key"]
        if key in lookup:
            assert lookup[key]["predicted_daily_variance"] == row["predicted_daily_variance"]
        if row["status"] == "complete_paired_risk_label" and (
            pd.Timestamp(row["required_label_end"]) <= cut
        ):
            assert lookup[key] == row
    changed = net.copy()
    changed.loc[changed.index > cut] += 0.1
    later = rules.diagnose(audits, changed, rf, mains)
    for before, after in zip(full, later, strict=True):
        if before["status"] == "complete_paired_risk_label" and (
            pd.Timestamp(before["required_label_end"]) <= cut
        ):
            assert before == after


def test_input_and_resolved_trace_keeps_original_forecasts_and_binary64_values(tmp_path):
    net, rf, mains = data()
    audits = [forecast("2021-09-30"), forecast("2026-09-30")]
    tape = []
    rows = rules.diagnose(audits, net, rf, mains, retain=tape.append)
    assert len(tape) == 2 * len(rows)
    path = tmp_path / "inputs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in tape))
    assert study.verify_trace(path, rows, audits, net, rf, mains)[
        "actual_input_and_resolved_record_count"
    ] == len(tape)
    damaged = copy.deepcopy(tape)
    damaged[0]["source_RF_binary64_hex"][0] = float(0.1).hex()
    path.write_text("".join(json.dumps(r) + "\n" for r in damaged))
    with pytest.raises(QuantError):
        study.verify_trace(path, rows, audits, net, rf, mains)


def test_missing_calendar_or_different_rf_cannot_be_padded_or_normalized():
    net, rf, mains = data()
    with pytest.raises(QuantError):
        rules.diagnose([forecast("2021-09-30")], net.iloc[1:], rf.iloc[1:], mains)
    mains["5y-base"] = mains["5y-base"].copy()
    mains["5y-base"].iloc[0, 1] += 1e-10
    with pytest.raises(QuantError):
        rules.diagnose([forecast("2021-09-30")], net, rf, mains)


@pytest.mark.parametrize(
    "key,value",
    [
        ("label_horizon", "next21rows"),
        ("full_calendar_month_required_for_both_labels", False),
        ("initial_five_year_partial_month_kept_unlabelled", False),
        ("terminal_partial_month_kept_unlabelled", False),
        ("independent_variance_absolute_tolerance", 1e-8),
        ("label_gap_is_causal_fee_or_source_profit_attribution", True),
        ("new_covariance_fits_targets_accounts_or_financial_trials", 1),
        ("new_model_or_allocation_selected", True),
        ("new_source_requests_or_RF_views", 1),
    ],
)
def test_measurement_scope_and_financial_flags_cannot_be_waived(monkeypatch, key, value):
    policy = {**read_json(study.POLICY), key: value}
    monkeypatch.setattr(study, "read_json", lambda p: policy)
    with pytest.raises(QuantError):
        study.policy()


def test_summaries_remain_per_path_not_independent_pooled_trials():
    net, rf, mains = data()
    rows = rules.diagnose([forecast("2021-09-30"), forecast("2026-09-30")], net, rf, mains)
    summary = rules.summarize(rows)
    assert tuple(summary["path_summaries"]) == rules.PATHS
    assert not summary["path_contexts_are_independent_samples"]
    assert summary["new_model_choices_targets_accounts_or_financial_trials"] == 0
