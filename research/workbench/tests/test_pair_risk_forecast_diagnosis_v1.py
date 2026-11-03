from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from us_quant import pair_risk_forecast_diagnosis_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError


def forecast():
    return {
        "training_marginal_variance_contributions": [0.00005, 0.00005, *([0.0] * 6)],
        "training_correlation": -0.5,
    }


def arrays():
    t = np.arange(21)
    rf = np.full(21, 0.0001)
    net = (
        np.column_stack([0.001 + 0.01 * np.sin(t / 3), 0.002 + 0.02 * np.cos(t / 5)]) + rf[:, None]
    )
    return net, rf, np.array([0.5, 0.5])


def test_full_future_risk_variance_ratio_and_marginal_contributions_match_independent_scalar():
    net, rf, w = arrays()
    result = rules.risk_case(net, rf, w, forecast())
    assert result["status"] == "complete_future_risk_evaluated"
    expected = np.var((net - rf[:, None]) @ w, ddof=1)
    assert result["future_portfolio_variance"] == pytest.approx(expected)
    assert result["forecast_variance_ratio"] == pytest.approx(expected / 0.0001)
    assert sum(result["future_marginal_variance_contributions"]) == pytest.approx(expected)
    assert result["independent_scalar_variance_relative_difference"] <= 1e-10
    assert not result["statistical_source_mixture_is_tradable_main_account"]


def test_exact_constant_future_portfolio_has_no_safe_success_or_ratio_floor():
    net = np.full((21, 2), 0.0002)
    rf = np.full(21, 0.0001)
    result = rules.risk_case(net, rf, np.array([0.5, 0.5]), forecast())
    assert result["status"] == "complete_future_zero_portfolio_risk"
    assert result["forecast_variance_ratio"] is None
    assert not result["zero_risk_is_safe_success"]
    assert not result["variance_floor_or_source_normalization"]


def test_exact_constant_one_source_retains_undefined_correlation_but_valid_other_source_risk():
    net, rf, w = arrays()
    net[:, 0] = 0.0002
    result = rules.risk_case(net, rf, w, forecast())
    assert result["future_pair_correlation"] is None
    assert result["forecast_variance_ratio"] > 0
    assert result["exact_constant_future_source_series"] == [True, False]


def test_negative_recorded_correlation_can_flip_without_claiming_future_hedging():
    net, rf, w = arrays()
    net[:, 1] = 2 * net[:, 0]
    result = rules.risk_case(net, rf, w, forecast())
    assert result["negative_recorded_correlation_became_nonnegative"]
    assert not result["future_risk_is_bound_or_independent_forward_proof"]


def audit(day):
    pairs = []
    ids = rules.source.rules.EXPERT_IDS
    for i, j in itertools.combinations(range(8), 2):
        names = sorted([ids[i], ids[j]])
        weights = [0.0] * 8
        weights[i] = weights[j] = 0.5
        pairs.append(
            {"pair": names, "expert_weights": weights, "selection_score_key": 1.0, **forecast()}
        )
    return {
        "decision_session": day,
        "selection": {"scores_available": True, "selected_pair": pairs[0]["pair"], "pairs": pairs},
    }


def source_frames():
    dates = sessions("2026-08-03", "2026-10-05")
    values = 0.001 + 0.003 * np.sin(np.arange(len(dates)) / 3)
    matrix = np.column_stack(
        [values + 0.001 * np.cos(np.arange(len(dates)) / (i + 2)) for i in range(8)]
    )
    return pd.DataFrame(matrix, index=dates, columns=rules.source.rules.EXPERT_IDS), pd.Series(
        0.0001, index=dates
    )


def test_next_exchange_month_is_not_same_calendar_month_or_fixed_twenty_one_days():
    dates = rules.future_calendar(pd.Timestamp("2017-04-28"))
    assert dates[0] == pd.Timestamp("2017-05-01")
    assert dates[-1] == pd.Timestamp("2017-05-31")
    assert len(dates) == 22
    with pytest.raises(QuantError, match="genuine"):
        rules.future_calendar(pd.Timestamp("2017-04-18"))


def test_all_28_pairs_and_final_three_days_remain_unevaluated_without_full_month():
    net, rf = source_frames()
    records = []
    decisions, cases = rules.diagnose(
        [audit("2026-08-31"), audit("2026-09-30")], net, rf, retain=records.append
    )
    assert len(cases) == 56 and len(records) == 112
    assert decisions[0]["status"] == "future_month_complete"
    assert decisions[1]["available_future_sessions"] == 3
    incomplete = [r for r in cases if r["status"] == "future_month_incomplete_not_evaluated"]
    assert len(incomplete) == 28 and all(r["forecast_variance_ratio"] is None for r in incomplete)
    assert all(r["required_future_end"] == "2026-10-30" for r in incomplete)
    assert rules.aggregate(cases)["original_selected_pairs"]["pairs"] == 1


def test_actual_new_inputs_are_retained_before_a_failed_statistical_calculation(monkeypatch):
    net, rf = source_frames()
    records = []

    def failed(*args):
        raise QuantError("Synthetic diagnostic calculation failure")

    monkeypatch.setattr(rules, "risk_case", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.diagnose([audit("2026-08-31")], net, rf, retain=records.append)
    assert len(records) == 1 and records[0]["record"] == "new_diagnostic_input"
    assert records[0]["original_source_NET_binary64_hex"]


def test_missing_nonselected_pair_or_third_source_weight_cannot_be_hidden():
    net, rf = source_frames()
    row = audit("2026-08-31")
    row["selection"]["pairs"].pop()
    with pytest.raises(QuantError, match="all28"):
        rules.diagnose([row], net, rf)
    row = audit("2026-08-31")
    row["selection"]["pairs"][0]["expert_weights"] = [1 / 8] * 8
    with pytest.raises(QuantError, match="third"):
        rules.diagnose([row], net, rf)


def test_invalid_forecast_or_future_source_cannot_become_zero_risk_success():
    net, rf, w = arrays()
    old = forecast()
    old["training_marginal_variance_contributions"] = [0.0] * 8
    with pytest.raises(QuantError, match="positive"):
        rules.risk_case(net, rf, w, old)
    net[0, 0] = np.nan
    with pytest.raises(QuantError, match="exact complete"):
        rules.risk_case(net, rf, w, forecast())


def test_dependency_closure_contains_actual_diagnostic_and_frozen_source_controller():
    paths = {r["path"] for r in rules.dependencies()}
    assert set(rules.OWN).issubset(paths)
    assert "src/us_quant/ssd_eight_expert_meta_v1.py" in paths
