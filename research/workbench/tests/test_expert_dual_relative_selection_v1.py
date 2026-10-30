from __future__ import annotations

import runpy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_dual_relative_selection_v1 as rules
from us_quant import strategy_expert_pool_v4 as original
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic(changed_after=None):
    helper = runpy.run_path(str(rules.ROOT / "tests/test_strategy_expert_pool_v4.py"))
    experts, calendar, assets = helper["archived_experts"](changed_after=changed_after)
    index = calendar[calendar >= "2016-10-06"]
    phase = np.arange(len(index))
    benchmarks = pd.DataFrame(
        {
            "SPY": 0.0007 + 0.003 * np.sin(phase / 9),
            "QQQ": 0.0009 + 0.005 * np.cos(phase / 11),
        },
        index=index,
    )
    return experts, calendar, assets, benchmarks


def test_dual_objective_keeps_original_risk_weights_and_all_four_scores():
    experts, _, _, benchmarks = synthetic()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    previous = np.full(6, 1 / 6)
    _, old = original.choose_pair(observed, previous)
    weights, audit = rules.choose_pair(observed, previous, benchmarks)
    assert len(audit["pairs"]) == 15 and (weights > 0).sum() == 2
    old_pairs = {tuple(p["pair"]): p for p in old["pairs"]}
    for pair in audit["pairs"]:
        prior = old_pairs[tuple(pair["pair"])]
        for key in (
            "expert_weights",
            "training_correlation",
            "training_marginal_variance_contributions",
            "estimated_incremental_switch_cost_fraction",
            "estimated_incremental_changed_assets",
        ):
            assert pair[key] == prior[key]
        assert pair["training_risk_basis"] == "original_RF_excess_returns"
        scores = [
            pair["benchmark_scores"][name][part + "_net_active_risk_score"]
            for name in ("SPY", "QQQ")
            for part in ("train", "validation")
        ]
        assert pair["selection_score"] == min(scores)
        assert pair["selection_score_key"] == round(min(scores), 12)
        assert pair["cost_proxy_is_actual_main_account_cost"] is False
    assert audit["scoring_benchmark_last_observation"] == "2017-04-28"


def test_maturity_and_actual_initial_anchor_preserve_all_days():
    experts, calendar, assets, benchmarks = synthetic()
    full, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), benchmarks
    )
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["decision_role"] == "initial_capital"
    assert audits[0]["selection"]["scoring_benchmark_last_observation"] is None
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert (
        next(a["decision_session"] for a in audits if a["selection"]["scores_available"])
        == "2017-04-28"
    )
    assert full.ffill().loc["2016-10-05":].notna().all().all()
    for end in ("2016-10-05", "2017-04-28", "2017-10-18"):
        early, earlier = rules.build_targets(
            experts,
            calendar[calendar <= end],
            assets,
            read_json(rules.POLICY),
            benchmarks.loc[:end],
        )
        pd.testing.assert_frame_equal(early, full.loc[:end])
        assert earlier == [a for a in audits if a["decision_session"] <= end]


def test_future_benchmark_and_source_changes_do_not_change_past():
    experts, calendar, assets, benchmarks = synthetic()
    changed, _, _, future_benchmarks = synthetic(changed_after=pd.Timestamp("2017-04-28"))
    future_benchmarks.loc[future_benchmarks.index > "2017-04-28"] *= 5
    before, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), benchmarks
    )
    after, later = rules.build_targets(
        changed, calendar, assets, read_json(rules.POLICY), future_benchmarks
    )
    pd.testing.assert_frame_equal(before.loc[:"2017-04-28"], after.loc[:"2017-04-28"])
    assert [a for a in audits if a["decision_session"] <= "2017-04-28"] == [
        a for a in later if a["decision_session"] <= "2017-04-28"
    ]
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    future_benchmarks.loc[future_benchmarks.index > "2017-04-28"] = np.nan
    assert (
        rules.choose_pair(observed, np.full(6, 1 / 6), benchmarks)[1]
        == (rules.choose_pair(observed, np.full(6, 1 / 6), future_benchmarks)[1])
    )


@pytest.mark.parametrize("damage", ["date", "column", "NaN", "duplicate_column"])
def test_missing_or_invalid_benchmark_is_not_filled(damage):
    experts, _, _, benchmarks = synthetic()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    if damage == "date":
        benchmarks = benchmarks.drop(observed[0].net_returns.index[10])
    elif damage == "column":
        benchmarks = benchmarks.drop(columns="QQQ")
    elif damage == "NaN":
        benchmarks.loc["2017-04-28", "QQQ"] = np.nan
    else:
        benchmarks = pd.concat([benchmarks, benchmarks[["QQQ"]]], axis=1)
    with pytest.raises(QuantError):
        rules.choose_pair(observed, np.full(6, 1 / 6), benchmarks)


def test_exact_zero_tracking_risk_is_explicit_unscored_not_a_safe_score():
    experts, _, _, _ = synthetic()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    values = observed[0].net_returns.copy()
    target = observed[0].target.copy()
    same = [replace(o, net_returns=values, target=target) for o in observed]
    benchmarks = pd.DataFrame({"SPY": values, "QQQ": values})
    weights, audit = rules.choose_pair(same, np.full(6, 1 / 6), benchmarks)
    np.testing.assert_array_equal(weights, np.full(6, 1 / 6))
    assert audit["reason"] == "unscored_degenerate_pair_equal_weights"
    assert audit["scores_available"] is False and audit["safety_or_success_claimed"] is False
    assert all(
        p["unscorable_reason"] == "nonpositive_or_nonfinite_active_tracking_risk"
        for p in audit["pairs"]
    )


def test_negative_finite_active_scores_still_rank_without_new_cash_cutoff():
    experts, _, _, benchmarks = synthetic()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    benchmarks += 0.05
    weights, audit = rules.choose_pair(observed, np.full(6, 1 / 6), benchmarks)
    assert audit["scores_available"] is True and (weights > 0).sum() == 2
    assert max(p["selection_score"] for p in audit["pairs"]) < 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("actual_weight_trade_band", 0.01),
        ("switch_cost_bps_per_side", 0.0),
        ("initial_decision_session", "2016-09-30"),
    ],
)
def test_new_objective_cannot_rewrite_original_parameters(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="single active-score"):
        rules.validate(policy)
