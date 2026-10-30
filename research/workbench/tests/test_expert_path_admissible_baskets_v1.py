from __future__ import annotations

import copy
import runpy

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_path_admissible_baskets_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic():
    helper = runpy.run_path(str(rules.ROOT / "tests/test_expert_hierarchical_risk_v1.py"))
    return helper["synthetic"]()


def benchmarks(calendar):
    return pd.DataFrame({"SPY": 0.00015, "QQQ": 0.00025}, index=calendar)


def coordinates():
    targets = np.zeros((14, 6))
    targets[0] = 0.98
    return targets, targets[:, 0].copy()


def test_all63_equal_capital_baskets_are_assessed_as_paths():
    target, previous = coordinates()
    weights, audit = rules.assess_baskets(
        np.full((126, 6), 0.0004), np.zeros((126, 2)), target, previous
    )
    assert audit["enumerated_nonempty_subsets"] == 63
    assert audit["assessment_admissible_subsets"] == 63
    assert len({c["mask"] for c in audit["all_subset_assessments"]}) == 63
    assert audit["selected_mask"] == 63
    assert np.array_equal(weights, np.full(6, 1 / 6))
    assert all(
        set(c["assessment_drawdowns"]) == {"full_126", "older_84", "recent_42"}
        for c in audit["all_subset_assessments"]
    )


def test_chronological_order_changes_risk_not_terminal_growth():
    target, previous = coordinates()
    alternating = np.tile([0.025, -1 / 41], 63)
    grouped = np.r_[np.full(63, 0.025), np.full(63, -1 / 41)]
    assert abs(np.prod(1 + alternating) - np.prod(1 + grouped)) < 1e-12
    first, a = rules.assess_baskets(
        np.tile(alternating[:, None], (1, 6)), np.zeros((126, 2)), target, previous
    )
    second, b = rules.assess_baskets(
        np.tile(grouped[:, None], (1, 6)), np.zeros((126, 2)), target, previous
    )
    assert first is not None and second is None
    assert a["all_subset_assessments"][0]["assessment_drawdowns"]["full_126"] < 0.03
    assert b["all_subset_assessments"][0]["assessment_drawdowns"]["full_126"] > 0.7


def test_admissibility_is_not_average_individual_drawdown():
    target, previous = coordinates()
    returns = np.full((126, 6), -0.02)
    returns[:, :2] = 0
    returns[0, :2] = [0.30, -0.25]
    returns[1, :2] = [-0.25, 0.30]
    _, audit = rules.assess_baskets(returns, np.zeros((126, 2)), target, previous)
    cards = {c["mask"]: c for c in audit["all_subset_assessments"]}
    assert cards[1]["assessment_admissible"] is False
    assert cards[2]["assessment_admissible"] is False
    assert cards[3]["assessment_admissible"] is True
    assert cards[3]["assessment_drawdowns"]["full_126"] < 0.05


def test_full126_path_does_not_reset_its_peak_at_training_split():
    target, previous = coordinates()
    returns = np.zeros((126, 6))
    returns[[82, 85]] = -0.1
    weights, audit = rules.assess_baskets(returns, np.zeros((126, 2)), target, previous)
    risk = audit["all_subset_assessments"][0]["assessment_drawdowns"]
    assert risk["older_84"] < 0.11 and risk["recent_42"] < 0.11
    assert risk["full_126"] > 0.18 and weights is None


@pytest.mark.parametrize(
    "loss,admissible", [(0.15, False), (0.15 - 5e-11, False), (0.15 - 2e-10, True)]
)
def test_source_risk_boundary_is_conservative_not_a_relaxed_main_gate(loss, admissible):
    target, previous = coordinates()
    returns = np.zeros((126, 6))
    returns[10] = -loss
    weights, _ = rules.assess_baskets(returns, np.zeros((126, 2)), target, previous)
    assert (weights is not None) is admissible


def test_no_admissible_subset_is_explicit_funding_standby():
    target, previous = coordinates()
    weights, audit = rules.assess_baskets(
        np.full((126, 6), -0.02), np.zeros((126, 2)), target, previous
    )
    assert weights is None and audit["funding_standby"] is True
    assert audit["assessment_admissible_subsets"] == 0
    assert audit["selected_mask"] is None and audit["selected_minimum_active_growth"] is None
    assert audit["reason"] == "no_observed_path_admissible_subset"


def test_negative_finite_active_growth_is_not_a_cash_cutoff():
    target, previous = coordinates()
    weights, audit = rules.assess_baskets(
        np.zeros((126, 6)), np.full((126, 2), 0.001), target, previous
    )
    assert audit["selected_minimum_active_growth"] < 0
    assert weights is not None and audit["funding_standby"] is False
    assert audit["assessment_paths_are_actual_main_or_future_drawdown_guarantee"] is False


def test_ties_prefer_nominal_target_distance_not_an_invented_fee():
    target = np.zeros((14, 6))
    for i in range(6):
        target[i, i] = 0.98
    weights, audit = rules.assess_baskets(
        np.zeros((126, 6)), np.zeros((126, 2)), target, target[:, 0]
    )
    assert audit["selected_mask"] == 1
    assert np.array_equal(weights, np.array([1, 0, 0, 0, 0, 0]))
    assert audit["target_L1_is_actual_turnover_or_fee_forecast"] is False


@pytest.mark.parametrize("damage", ["shape", "NaN", "insolvent", "target", "previous"])
def test_invalid_assessment_cannot_be_funding_success(damage):
    target, previous = coordinates()
    returns, benchmark = np.zeros((126, 6)), np.zeros((126, 2))
    if damage == "shape":
        returns = returns[:-1]
    elif damage == "NaN":
        benchmark[0, 0] = np.nan
    elif damage == "insolvent":
        returns[0, 0] = -1
    elif damage == "target":
        target[0, 0] = 1.01
    else:
        previous[0] = -0.1
    with pytest.raises(QuantError):
        rules.assess_baskets(returns, benchmark, target, previous)


def test_warmup_and_actual_standby_are_different_states():
    experts, calendar, universe = synthetic()
    for expert in experts:
        expert._shadow["return"] = -0.02
    target, audits = rules.build_targets(
        experts, calendar, universe, benchmarks(calendar), read_json(rules.POLICY)
    )
    assert sum(not a["source_assessment_mature"] for a in audits) == 7
    assert all(not a["assessment"]["funding_standby"] for a in audits[:7])
    assert all(a["expert_weights"] is None for a in audits[7:])
    assert all(target.loc[a["decision_session"], "BIL"] == 0.98 for a in audits[7:])


def test_future_source_and_benchmark_changes_preserve_prior_targets():
    experts, calendar, universe = synthetic()
    policy, benchmark = read_json(rules.POLICY), benchmarks(calendar)
    cutoff = rules.decision_sessions(calendar, policy)[-2]
    prefix = calendar[calendar <= cutoff]
    before = rules.build_targets(experts, prefix, universe, benchmark, policy)
    changed = copy.deepcopy(experts)
    for expert in changed:
        expert._shadow.loc[expert._shadow.index > cutoff, "return"] += 0.2
    benchmark.loc[benchmark.index > cutoff] += 0.3
    after = rules.build_targets(changed, prefix, universe, benchmark, policy)
    assert before[0].equals(after[0]) and before[1] == after[1]
