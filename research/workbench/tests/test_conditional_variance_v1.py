from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from us_quant import conditional_variance_v1 as model
from us_quant.config import QuantError


def sample():
    rng = np.random.default_rng(952731)
    return np.r_[rng.normal(0, 0.006, 160), rng.normal(0, 0.026, 92)]


@pytest.mark.parametrize("asymmetric", [False, True])
def test_fixed_initialization_and_positive_stationary_domain(asymmetric):
    coordinates = model.initial_coordinates(asymmetric)
    parameters, probabilities = model.decode(coordinates, asymmetric)
    expected = [0.05, 0.05, 0.85, 0.10] if asymmetric else [0.05, 0.05, 0.90, 0.0]
    np.testing.assert_allclose(parameters, expected, rtol=0, atol=1e-15)
    assert probabilities.sum() == pytest.approx(1.0)
    for sign in (-1, 1):
        coordinates[1:] = sign * 20.0
        parameters, probabilities = model.decode(coordinates, asymmetric)
        assert parameters[0] > 0 and (parameters[1:] >= 0).all()
        assert 0 < parameters[1] + parameters[2] + parameters[3] / 2 < 1
        assert probabilities[0] > 0


@pytest.mark.parametrize("asymmetric", [False, True])
def test_transformed_analytic_gradient_matches_finite_differences(asymmetric):
    values = sample()
    values /= math.sqrt(float(np.mean(values**2)))
    coordinates = model.initial_coordinates(asymmetric) + 0.13
    _, gradient = model.likelihood_gradient(coordinates, values, asymmetric)
    step = 1e-6
    for i in range(len(coordinates)):
        delta = np.zeros(len(coordinates))
        delta[i] = step
        upper = model.likelihood_gradient(coordinates + delta, values, asymmetric)[0]
        lower = model.likelihood_gradient(coordinates - delta, values, asymmetric)[0]
        assert gradient[i] == pytest.approx((upper - lower) / (2 * step), abs=1e-7, rel=1e-5)


def test_only_gjr_responds_to_the_sign_of_the_last_shock():
    returns = np.array([0.3, -0.5, 0.2, 1.2])
    flipped = returns.copy()
    flipped[-1] *= -1
    for gamma in (0.0, 0.10):
        parameters = [0.05, 0.05, 0.85, gamma]
        before = model.forecast(returns, parameters, model.variance_path(returns, parameters))
        after = model.forecast(flipped, parameters, model.variance_path(flipped, parameters))
        assert after[0] - before[0] == pytest.approx(gamma * returns[-1] ** 2)
        if gamma == 0:
            np.testing.assert_array_equal(before, after)
        else:
            assert (after > before).all()


def test_gamma_half_expectation_requires_symmetric_innovations():
    omega, alpha, beta, gamma = 0.05, 0.05, 0.85, 0.10
    h = 2.0
    shocks = (-math.sqrt(h), math.sqrt(h))
    expectation = (
        math.fsum(
            omega + alpha * shock**2 + gamma * shock**2 * (shock < 0) + beta * h for shock in shocks
        )
        / 2
    )
    assert expectation == pytest.approx(omega + (alpha + beta + gamma / 2) * h)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_single_fit_matches_independent_native_units_and_finite_forecast(asymmetric):
    values = sample()
    fitted = model.fit(values, asymmetric)
    oracle = model.independent(
        values, fitted["parameters_daily"], fitted["pooled_daily_squared_return"]
    )
    assert fitted["optimizer_success"] and fitted["optimizer_initializations"] == 1
    assert 0 <= fitted["optimizer_iterations"] <= 1000
    assert fitted["mean_negative_log_likelihood_scaled"] <= (
        fitted["initial_mean_negative_log_likelihood_scaled"] + 1e-10
    )
    assert fitted["mean_negative_log_likelihood_daily"] == pytest.approx(
        oracle["mean_negative_log_likelihood"], abs=1e-8
    )
    np.testing.assert_allclose(
        fitted["conditional_variances_daily"], oracle["variances"], rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        fitted["forecast_variances_daily"], oracle["forecast_variances"], rtol=1e-9, atol=0
    )
    assert fitted["mean_forecast_variance_daily"] == pytest.approx(
        oracle["mean_forecast_variance"], rel=1e-9
    )


def test_native_unit_independent_likelihood_scales_without_a_second_fit():
    values = sample()
    coefficients = [0.00001, 0.05, 0.85, 0.10]
    pooled = float(np.mean(values**2))
    before = model.independent(values, coefficients, pooled)
    after = model.independent(values * 3, [0.00009, *coefficients[1:]], pooled * 9)
    np.testing.assert_allclose(
        np.array(before["forecast_variances"]) * 9, after["forecast_variances"], rtol=1e-12
    )
    assert after["mean_negative_log_likelihood"] == pytest.approx(
        before["mean_negative_log_likelihood"] + math.log(3), abs=1e-12
    )


@pytest.mark.parametrize(
    "values",
    [np.zeros(252), np.ones(252), np.full(252, np.nan), np.ones(251), np.tile([-0.01, 0.01], 126)],
)
def test_bad_or_degenerate_observations_do_not_get_a_fallback(values):
    with pytest.raises(QuantError):
        model.fit(values, True)


def test_nonconvergence_is_retained_and_never_uses_an_alternate_seed(monkeypatch):
    calls = []

    def fail(function, initial, **kwargs):
        calls.append(initial.copy())
        return SimpleNamespace(
            x=initial,
            fun=function(initial, *kwargs["args"])[0],
            success=False,
            status=1,
            message="synthetic iteration limit",
            nit=1000,
            nfev=1001,
        )

    monkeypatch.setattr(model, "minimize", fail)
    with pytest.raises(model.FitError, match="did not converge") as caught:
        model.fit(sample(), True)
    assert len(calls) == 1
    assert caught.value.receipt["optimizer_success"] is False
    assert caught.value.receipt["optimizer_iterations"] == 1000


def test_coordinates_outside_preregistered_domain_are_not_clipped():
    coordinates = model.initial_coordinates(True)
    coordinates[1] = 20.001
    with pytest.raises(QuantError, match="domain"):
        model.decode(coordinates, True)
