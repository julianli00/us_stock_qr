from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from us_quant import volatility_hmm_v1 as hmm
from us_quant.config import QuantError


def test_forward_backward_matches_exhaustive_state_paths():
    y = np.array([-0.05, 0.01, 0.08, -0.02])
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    v = np.array([0.001, 0.01])
    likelihood, gamma, xi, filtered = hmm.expectation(y, a, v)
    exact_gamma, exact_xi = np.zeros((4, 2)), np.zeros((3, 2, 2))
    total = 0.0
    for states in itertools.product(range(2), repeat=4):
        probability = 0.5
        for i, state in enumerate(states):
            probability *= math.exp(-(y[i] ** 2) / (2 * v[state])) / math.sqrt(
                2 * math.pi * v[state]
            )
            if i:
                probability *= a[states[i - 1], state]
        total += probability
        for i, state in enumerate(states):
            exact_gamma[i, state] += probability
            if i:
                exact_xi[i - 1, states[i - 1], state] += probability
    assert likelihood == pytest.approx(math.log(total), abs=1e-12)
    np.testing.assert_allclose(gamma, exact_gamma / total, rtol=0, atol=1e-12)
    np.testing.assert_allclose(xi, exact_xi / total, rtol=0, atol=1e-12)
    np.testing.assert_allclose(filtered, exact_gamma[-1] / total, rtol=0, atol=1e-12)


def test_independent_scaled_filter_and_return_scale_identity():
    y = np.array([-0.05, 0.01, 0.08, -0.02])
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    v = np.array([0.001, 0.01])
    likelihood, _, _, filtered = hmm.expectation(y, a, v)
    other, score = hmm.independent_filter(y, a, v)
    np.testing.assert_allclose(filtered, other, rtol=0, atol=1e-12)
    assert likelihood == pytest.approx(score, abs=1e-12)
    scaled, scaled_score = hmm.independent_filter(y * 3, a, v * 9)
    np.testing.assert_allclose(other, scaled, rtol=0, atol=1e-12)
    assert scaled_score == pytest.approx(score - len(y) * math.log(3), abs=1e-12)


def test_smoothed_training_states_are_not_past_trading_signals():
    y = np.array([0.002, 0.004, 0.1, -0.12])
    a = np.array([[0.95, 0.05], [0.05, 0.95]])
    v = np.array([0.0001, 0.01])
    _, gamma, _, _ = hmm.expectation(y, a, v)
    _, _, _, early = hmm.expectation(y[:2], a, v)
    assert abs(gamma[1, 0] - early[0]) > 0.01


def test_single_seed_map_fit_is_ordered_finite_and_independently_filtered():
    rng = np.random.default_rng(87431)
    y = np.r_[rng.normal(0, 0.004, 126), rng.normal(0, 0.025, 126)]
    fitted = hmm.fit(y)
    assert fitted["converged"] and 1 <= fitted["iterations"] <= 500
    assert fitted["state_variances_daily"][0] < fitted["state_variances_daily"][1]
    assert fitted["minimum_objective_improvement"] >= -1e-8
    other, likelihood = hmm.independent_filter(
        y, fitted["transition"], fitted["state_variances_daily"]
    )
    np.testing.assert_allclose(
        other, fitted["filtered_probabilities_at_last_observation"], atol=1e-10, rtol=0
    )
    assert likelihood == pytest.approx(fitted["log_likelihood"], abs=1e-8)


@pytest.mark.parametrize(
    "values", [np.zeros(252), np.full(252, 0.01), np.full(252, np.nan), np.ones(251)]
)
def test_invalid_or_unidentified_history_is_not_a_safe_default(values):
    with pytest.raises(QuantError):
        hmm.fit(values)
