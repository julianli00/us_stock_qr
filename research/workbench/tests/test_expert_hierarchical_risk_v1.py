from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_hierarchical_risk_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)


def synthetic():
    calendar = sessions("2016-09-01", "2017-12-29")
    universe = (*UNIVERSE, "IJR", "PKW")
    history = calendar[calendar >= pd.Timestamp("2016-10-06")]
    phase = np.arange(len(history), dtype=float)
    rf = pd.Series(0.00005, index=history, name="risk_free")
    experts = []
    for i, identifier in enumerate(rules.EXPERT_IDS):
        target = pd.DataFrame(np.nan, index=calendar, columns=universe)
        monthly = i % 2 == 1
        active = [
            d
            for d in calendar
            if d >= pd.Timestamp("2016-09-30" if monthly else "2016-10-05")
            and (not monthly or is_month_end(d))
        ]
        for day in active:
            target.loc[day] = 0.0
            share = 0.65 + 0.04 * np.sin(day.month / (i + 1))
            target.loc[day, universe[4 + i]] += share
            target.loc[day, "BIL"] += 0.98 - share
        returns = (
            rf
            + 0.0001
            + (i + 1) * 0.0015 * (np.sin(phase / (i + 2.7)) + 0.3 * np.cos(phase / (i + 3.1)))
        )
        equity = 10000 * (1 + returns).cumprod()
        shadow = pd.DataFrame(
            {
                "equity": equity,
                "return": returns,
                "risk_free": rf,
                "cash": equity * 0.02,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0,
            },
            index=history,
        )
        identity = ExpertIdentity(identifier, rules.MECHANISMS[i], "2026-10-30T00:00:00Z")
        adapter = ArchivedMonthlyExpert if monthly else ArchivedDenseExpert
        experts.append(adapter(identity, target, shadow, universe))
    return tuple(experts), calendar, universe


def orthogonal():
    return np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [-1, 1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1],
            [1, 1, 1, -1, -1, -1],
            [-1, 1, -1, -1, 1, -1],
            [1, -1, -1, -1, -1, 1],
            [-1, -1, 1, -1, 1, 1],
        ],
        dtype=float,
    )


def test_uncorrelated_hierarchy_is_inverse_variance_not_global_erc():
    values = orthogonal() * np.arange(1, 7)
    weights, proof = rules.hierarchical_weights(values)
    expected = 1 / np.arange(1, 7, dtype=float) ** 2
    expected /= expected.sum()
    assert np.max(abs(weights - expected)) < 1e-12
    assert len(proof["canonical_merges"]) == len(proof["recursive_capital_splits"]) == 5
    assert proof["variance_floor_or_regularization_applied"] is False
    assert proof["global_optimizer_or_previous_failed_solver_used"] is False
    scaled, _ = rules.hierarchical_weights(values * 17)
    assert np.max(abs(scaled - weights)) < 1e-12


def test_singular_covariance_and_deterministic_ties_need_no_solver():
    values = np.arange(126, dtype=float)[:, None] * np.ones((1, 6))
    weights, proof = rules.hierarchical_weights(values)
    again, repeated = rules.hierarchical_weights(values)
    assert np.array_equal(weights, again) and proof == repeated
    assert proof["canonical_merges"][0]["left_members"] == [0]
    assert proof["canonical_merges"][0]["right_members"] == [1]
    assert sorted(proof["seriated_expert_coordinates"]) == list(range(6))
    assert np.isfinite(weights).all() and (weights > 0).all() and abs(weights.sum() - 1) < 1e-12


def test_zero_cluster_risk_is_explicit_not_a_safety_claim(monkeypatch):
    base = orthogonal()
    values = np.column_stack(
        (
            base[:, 0] + base[:, 1],
            -base[:, 0] + base[:, 1],
            -base[:, 1],
            base[:, 2] + base[:, 3],
            -base[:, 2] + base[:, 3],
            -base[:, 3],
        )
    )
    monkeypatch.setattr(
        rules, "cluster_order", lambda covariance: (tuple(range(6)), [], np.zeros((6, 6)))
    )
    weights, proof = rules.hierarchical_weights(values)
    assert proof["recursive_capital_splits"][0]["both_child_risks_zero"] is True
    assert proof["recursive_capital_splits"][0]["left_capital_fraction_of_parent"] == 0.5
    assert np.isfinite(weights).all() and abs(weights.sum() - 1) < 1e-12
    assert proof["risk_clusters_are_independent_factor_families"] is False


def test_one_exact_zero_risk_child_receives_parent_capital():
    base = orthogonal()[:, 0]
    weights, proof = rules.hierarchical_weights(np.column_stack([base] * 5 + [-base]))
    zero = [row for row in proof["recursive_capital_splits"] if row["exactly_one_child_risk_zero"]]
    assert zero and zero[0]["right_IVP_variance"] == 0
    assert zero[0]["left_capital_fraction_of_parent"] == 0
    assert weights[3] == 0 and np.isfinite(weights).all()


@pytest.mark.parametrize("kind", ("shape", "asymmetric", "nonPSD", "nonfinite", "zero"))
def test_invalid_cluster_covariance_is_not_repaired(kind):
    covariance = np.eye(6)
    if kind == "shape":
        covariance = np.eye(5)
    elif kind == "asymmetric":
        covariance[0, 1] = 0.5
    elif kind == "nonPSD":
        covariance[0, 1] = covariance[1, 0] = 2
    elif kind == "nonfinite":
        covariance[0, 0] = np.nan
    else:
        covariance[0, 0] = 0
    with pytest.raises(QuantError):
        rules.cluster_order(covariance)


def test_missing_history_and_zero_individual_risk_are_unscored():
    experts, calendar, universe = synthetic()
    targets, audits = rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))
    assert targets.index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["allocation"]["reason"] == "unscored_initial_equal_weights"
    assert audits[0]["allocation"]["risk_estimate_available"] is False
    assert audits[0]["allocation"]["safety_or_success_claimed"] is False
    observed = [e.observe(calendar[-1], 126) for e in experts]
    observed[0] = replace(observed[0], net_returns=observed[0].risk_free.copy())
    weights, proof = rules.choose_allocation(observed, np.full(6, 1 / 6))
    assert np.array_equal(weights, np.full(6, 1 / 6))
    assert proof["reason"] == "unscored_nonpositive_individual_variance_equal_weights"
    assert proof["risk_estimate_available"] is False and proof["safety_or_success_claimed"] is False


def test_risk_fit_is_not_return_ranking_or_heldout_validation():
    experts, calendar, universe = synthetic()
    _, audits = rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))
    mature = [row for row in audits if row["allocation"]["risk_estimate_available"]]
    assert mature
    for row in mature:
        allocation = row["allocation"]
        assert allocation["observations_used_for_fit"] == 126
        assert allocation["risk_fit_is_heldout_validation"] is False
        assert allocation["return_score_or_rank_used"] is False
        assert allocation["cost_proxy_used_for_allocation"] is False
        assert allocation["nominal_target_change_is_actual_turnover"] is False
        assert allocation["safety_or_success_claimed"] is False


def test_future_source_changes_preserve_prior_allocations():
    experts, calendar, universe = synthetic()
    policy = read_json(rules.POLICY)
    cutoff = rules.decision_sessions(calendar, policy)[-2]
    before = rules.build_targets(experts, calendar[calendar <= cutoff], universe, policy)
    changed = copy.deepcopy(experts)
    for expert in changed:
        expert._shadow.loc[expert._shadow.index > cutoff, "return"] += 0.15
    after = rules.build_targets(changed, calendar[calendar <= cutoff], universe, policy)
    assert before[0].equals(after[0]) and before[1] == after[1]


@pytest.mark.parametrize("kind", ("identity", "future_target", "RF", "date", "nonfinite"))
def test_changed_or_missing_expert_state_is_rejected(kind):
    experts, calendar, _ = synthetic()
    observed = [e.observe(calendar[-1], 126) for e in experts]
    if kind == "identity":
        observed.reverse()
    elif kind == "future_target":
        observed[0] = replace(observed[0], target_known_at=calendar[-1] + pd.Timedelta(days=1))
    elif kind == "RF":
        observed[0] = replace(observed[0], risk_free=observed[0].risk_free + 0.01)
    elif kind == "date":
        observed[0] = replace(observed[0], net_returns=observed[0].net_returns.iloc[1:])
    else:
        values = observed[0].net_returns.copy()
        values.iloc[0] = np.nan
        observed[0] = replace(observed[0], net_returns=values)
    with pytest.raises(QuantError):
        rules.choose_allocation(observed, np.full(6, 1 / 6))


def test_policy_cannot_relax_original_gates_or_change_pool():
    policy = read_json(rules.POLICY)
    policy["minimum_economic_factor_families"] = 2
    with pytest.raises(QuantError):
        rules.validate(policy)
