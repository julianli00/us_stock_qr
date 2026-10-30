from __future__ import annotations

import copy
import runpy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_full_covariance_risk_budget_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic():
    helper = runpy.run_path(str(rules.ROOT / "tests/test_expert_dual_relative_selection_v1.py"))
    experts, calendar, universe, _ = helper["synthetic"]()
    return experts, calendar, universe


def test_whole_pool_budgets_are_equal_without_asset_group_constraints():
    covariance = np.diag(np.arange(1, 7, dtype=float) ** 2)
    weights, proof = rules.risk_budget(covariance)
    expected = 1 / np.arange(1, 7, dtype=float)
    expected /= expected.sum()
    assert np.max(abs(weights - expected)) < 1e-10
    assert weights[0] > 0.35 and weights[:4].sum() > 0.7
    assert np.max(abs(np.asarray(proof["signed_variance_contribution_shares"]) - 1 / 6)) < 1e-9
    assert not proof["variance_floor_or_regularization_applied"]
    scaled, _ = rules.risk_budget(covariance * 0.000001)
    assert np.max(abs(scaled - weights)) < 1e-12


def test_singular_positive_covariance_is_not_regularized():
    covariance = np.ones((6, 6), dtype=float)
    weights, proof = rules.risk_budget(covariance)
    assert np.max(abs(weights - 1 / 6)) < 1e-10
    assert proof["scaled_covariance_minimum_eigenvalue"] < 1e-12
    assert proof["variance_floor_or_regularization_applied"] is False


@pytest.mark.parametrize("kind", ("asymmetric", "nonPSD", "nonfinite", "zero", "wrong_shape"))
def test_invalid_covariance_is_not_silently_regularized(kind):
    cov = np.eye(6)
    if kind == "asymmetric":
        cov[0, 1] = 0.1
    elif kind == "nonPSD":
        cov[0, 1] = cov[1, 0] = 2
    elif kind == "nonfinite":
        cov[0, 0] = np.nan
    elif kind == "zero":
        cov[0, 0] = 0
    else:
        cov = np.eye(5)
    with pytest.raises(QuantError):
        rules.risk_budget(cov)


def test_solver_nonconvergence_is_not_equal_weight_success(monkeypatch):
    monkeypatch.setattr(rules, "MAX_ITERATIONS", 0)
    with pytest.raises(QuantError, match="did not converge"):
        rules.risk_budget(np.eye(6))


def test_missing_history_and_zero_risk_are_explicit_not_success():
    experts, calendar, universe = synthetic()
    targets, audits = rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))
    assert targets.index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert not audits[0]["allocation"]["risk_estimate_available"]
    assert audits[0]["allocation"]["safety_or_success_claimed"] is False
    observed = [e.observe(calendar[-1], 126) for e in experts]
    observed[0] = replace(observed[0], net_returns=observed[0].risk_free.copy())
    weights, proof = rules.choose_allocation(observed, np.full(6, 1 / 6))
    assert np.array_equal(weights, np.full(6, 1 / 6))
    assert proof["reason"] == "unscored_nonpositive_individual_variance_equal_weights"
    assert not proof["risk_estimate_available"] and not proof["safety_or_success_claimed"]


def test_full_126_fit_and_84_42_stability_are_not_heldout_validation():
    experts, calendar, universe = synthetic()
    _, audits = rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))
    mature = [a for a in audits if a["allocation"]["risk_estimate_available"]]
    assert mature
    for row in mature:
        allocation = row["allocation"]
        assert allocation["observations_used_for_fit"] == 126
        assert allocation["return_score_or_rank_used"] is False
        assert (
            sum(d["observations"] for d in allocation["subperiod_risk_stability"].values()) == 126
        )
        assert all(
            not d["heldout_validation"] for d in allocation["subperiod_risk_stability"].values()
        )
        assert all(w > 0 for w in row["expert_weights"])


def test_future_source_changes_preserve_prior_allocations():
    experts, calendar, universe = synthetic()
    policy = read_json(rules.POLICY)
    dates = rules.decision_sessions(calendar, policy)
    cutoff = dates[-2]
    before = rules.build_targets(experts, calendar[calendar <= cutoff], universe, policy)
    changed = copy.deepcopy(experts)
    for expert in changed:
        expert._shadow.loc[expert._shadow.index > cutoff, "return"] += 0.15
    after = rules.build_targets(changed, calendar[calendar <= cutoff], universe, policy)
    assert before[0].equals(after[0]) and before[1] == after[1]


@pytest.mark.parametrize("kind", ("identity", "future_target", "RF", "date", "nonfinite"))
def test_missing_or_changed_expert_state_is_not_filled(kind):
    experts, calendar, _ = synthetic()
    observed = [e.observe(calendar[-1], 126) for e in experts]
    if kind == "identity":
        observed.reverse()
    elif kind == "future_target":
        observed[0] = replace(observed[0], target_known_at=calendar[-1] + pd.Timedelta(days=1))
    elif kind == "RF":
        observed[0] = replace(observed[0], risk_free=observed[0].risk_free + 0.001)
    elif kind == "date":
        observed[0] = replace(observed[0], net_returns=observed[0].net_returns.iloc[1:])
    else:
        values = observed[0].net_returns.copy()
        values.iloc[0] = np.nan
        observed[0] = replace(observed[0], net_returns=values)
    with pytest.raises(QuantError):
        rules.choose_allocation(observed, np.full(6, 1 / 6))
