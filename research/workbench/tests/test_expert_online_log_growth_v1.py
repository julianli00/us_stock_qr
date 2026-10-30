from __future__ import annotations

import copy
import runpy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_online_log_growth_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic():
    helper = runpy.run_path(str(rules.ROOT / "tests/test_expert_hierarchical_risk_v1.py"))
    return helper["synthetic"]()


def test_identity_curvature_and_equal_periods_preserve_prior():
    state = rules.AllocationLearningState.initial()
    before = state.weights.copy()
    for _ in range(119):
        state, audit = rules.advance(state, np.full(6, 1.02))
        assert np.max(abs(state.weights - before)) < 1e-12
        assert audit["enumerated_nonempty_faces"] == 63
        assert audit["persistent_array_buffer_bytes"] == 336
        assert audit["buffer_bytes_are_total_process_RSS"] is False
        assert audit["learning_score_is_actual_main_net_wealth"] is False
    assert state.completed_intervals == 119
    with pytest.raises(QuantError):
        rules.advance(state, np.ones(6))


def test_growth_feedback_changes_weights_without_covariance_ranking():
    state = rules.AllocationLearningState.initial()
    for _ in range(18):
        state, audit = rules.advance(state, np.array([1.10, 1.01, 1.02, 0.99, 1.03, 1.0]))
    assert state.weights[0] > 1 / 6 and state.weights[0] == state.weights.max()
    assert (state.weights >= 0).all() and abs(state.weights.sum() - 1) < 1e-12
    assert audit["maximum_complementarity_error"] < 1e-10
    scaled, _ = rules.advance(
        rules.AllocationLearningState.initial(), np.array([1.10, 1.01, 1.02, 0.99, 1.03, 1.0]) * 2
    )
    ordinary, _ = rules.advance(
        rules.AllocationLearningState.initial(), np.array([1.10, 1.01, 1.02, 0.99, 1.03, 1.0])
    )
    assert np.max(abs(scaled.weights - ordinary.weights)) < 1e-12


@pytest.mark.parametrize(
    "linear,expected",
    [
        (np.zeros(6), np.full(6, 1 / 6)),
        (np.array([2, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])),
        (np.array([0.7, 0.3, -1, -1, -1, -1]), np.array([0.7, 0.3, 0, 0, 0, 0])),
    ],
)
def test_analytic_simplex_faces(linear, expected):
    weights, proof = rules.project(np.eye(6), linear)
    assert np.max(abs(weights - expected)) < 1e-12
    assert proof["enumerated_nonempty_faces"] == 63
    assert proof["maximum_positive_stationarity_error"] < 1e-10


@pytest.mark.parametrize("damage", ["shape", "NaN", "asymmetric", "nonpositive"])
def test_invalid_projection_cannot_be_a_neutral_success(damage):
    matrix, linear = np.eye(6), np.zeros(6)
    if damage == "shape":
        matrix = np.eye(5)
    elif damage == "NaN":
        linear[0] = np.nan
    elif damage == "asymmetric":
        matrix[0, 1] = 0.1
    else:
        matrix[0, 0] = 0
    with pytest.raises(QuantError):
        rules.project(matrix, linear)


@pytest.mark.parametrize("damage", ["nonpositive", "NaN", "shape", "state"])
def test_invalid_learning_inputs_have_no_filled_score(damage):
    state, relative = rules.AllocationLearningState.initial(), np.ones(6)
    if damage == "nonpositive":
        relative[0] = 0
    elif damage == "NaN":
        relative[0] = np.nan
    elif damage == "shape":
        relative = np.ones(5)
    else:
        state = replace(state, curvature=np.eye(6) * 0.5)
    with pytest.raises(QuantError):
        rules.advance(state, relative)


def test_full_month_clock_partial_skip_and_maturity_are_explicit():
    experts, calendar, universe = synthetic()
    targets, audits = rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))
    assert targets.index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["online_update"] is None
    assert audits[1]["decision_session"] == "2016-10-31"
    assert audits[1]["initial_partial_source_month_not_assimilated"] is True
    assert audits[1]["online_update"] is None
    assert audits[2]["online_update"]["source_period_first_session"] == "2016-11-01"
    assert audits[2]["complete_learning_intervals"] == 1
    assert audits[2]["source_assessment_mature"] is False
    assert audits[2]["applied_expert_weights"] == np.full(6, 1 / 6).tolist()
    assert sum(not a["source_assessment_mature"] for a in audits) == 7
    assert audits[-1]["complete_learning_intervals"] == 14
    assert all(not a["learning_score_is_actual_main_net_wealth"] for a in audits)


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


@pytest.mark.parametrize("damage", ["identity", "future", "RF", "date", "NaN"])
def test_changed_or_missing_complete_expert_observation_is_rejected(damage):
    experts, calendar, _ = synthetic()
    source = [e.observe(calendar[-1], 126) for e in experts]

    class Fixed:
        def __init__(self, value):
            self.value = value

        def observe(self, day, lookback):
            return self.value

    if damage == "identity":
        source.reverse()
    elif damage == "future":
        source[0] = replace(source[0], target_known_at=calendar[-1] + pd.Timedelta(days=1))
    elif damage == "RF":
        source[0] = replace(source[0], risk_free=source[0].risk_free + 0.1)
    elif damage == "date":
        source[0] = replace(source[0], net_returns=source[0].net_returns.iloc[1:])
    else:
        value = source[0].net_returns.copy()
        value.iloc[0] = np.nan
        source[0] = replace(source[0], net_returns=value)
    with pytest.raises(QuantError):
        rules.observations(tuple(Fixed(v) for v in source), calendar[-1])
