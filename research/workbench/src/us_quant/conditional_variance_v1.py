from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from us_quant.config import QuantError

LOOKBACK = 252
HORIZON = 21
OMEGA_BOUNDS = (1e-8, 10.0)
LOGIT_LIMIT = 20.0
MAX_ITERATIONS = 1000
FTOL = 1e-12
GTOL = 1e-7
MAX_LINE_SEARCH = 40


class FitError(QuantError):
    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def bounds(asymmetric: bool) -> list[tuple[float, float]]:
    if type(asymmetric) is not bool:
        raise QuantError("The fixed symmetric/asymmetric identity must be boolean.")
    return [
        (math.log(OMEGA_BOUNDS[0]), math.log(OMEGA_BOUNDS[1])),
        *[(-LOGIT_LIMIT, LOGIT_LIMIT)] * (3 if asymmetric else 2),
    ]


def initial_coordinates(asymmetric: bool) -> np.ndarray:
    bounds(asymmetric)
    return np.array(
        [math.log(0.05), 0.0, math.log(17.0), 0.0]
        if asymmetric
        else [math.log(0.05), 0.0, math.log(18.0)]
    )


def decode(coordinates, asymmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    limits = bounds(asymmetric)
    theta = np.asarray(coordinates, dtype=float)
    if (
        theta.shape != (len(limits),)
        or not np.isfinite(theta).all()
        or any(not low <= value <= high for value, (low, high) in zip(theta, limits, strict=True))
    ):
        raise QuantError("Require finite coordinates inside the frozen numerical domain.")
    logits = np.r_[0.0, theta[1:]]
    exponentials = np.exp(logits - logits.max())
    probabilities = exponentials / exponentials.sum()
    omega = math.exp(float(theta[0]))
    alpha, beta = probabilities[1:3]
    gamma = 2 * probabilities[3] if asymmetric else 0.0
    parameters = np.array([omega, alpha, beta, gamma])
    if not 0 < alpha + beta + gamma / 2 < 1:
        raise QuantError("The positive slack must retain strictly subunit persistence.")
    return parameters, probabilities


def variance_path(returns, parameters, initial_variance: float = 1.0) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    coefficients = np.asarray(parameters, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or coefficients.shape != (4,)
        or not np.isfinite(coefficients).all()
        or coefficients[0] <= 0
        or (coefficients[1:] < 0).any()
        or not 0 <= coefficients[1] + coefficients[2] + coefficients[3] / 2 < 1
        or not math.isfinite(initial_variance)
        or initial_variance <= 0
    ):
        raise QuantError("Invalid conditional variance inputs are not a safe-state default.")
    omega, alpha, beta, gamma = coefficients
    variances = np.empty(len(values))
    variances[0] = initial_variance
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for i in range(1, len(values)):
                previous = values[i - 1]
                variances[i] = (
                    omega + (alpha + gamma * (previous < 0)) * previous**2 + beta * variances[i - 1]
                )
        except FloatingPointError as error:
            raise QuantError("Conditional variance arithmetic overflowed.") from error
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise QuantError("Conditional variances must stay finite and strictly positive.")
    return variances


def likelihood_gradient(coordinates, values, asymmetric: bool):
    parameters, probabilities = decode(coordinates, asymmetric)
    values = np.asarray(values, dtype=float)
    variances = variance_path(values, parameters)
    squared = values**2
    derivative = np.zeros(4)
    gradient = np.zeros(4)
    for i in range(1, len(values)):
        derivative = (
            np.array([1.0, squared[i - 1], variances[i - 1], squared[i - 1] * (values[i - 1] < 0)])
            + parameters[2] * derivative
        )
        gradient += 0.5 * (1 - squared[i] / variances[i]) / variances[i] * derivative
    gradient /= len(values)
    objective = float(
        0.5 * np.mean(math.log(2 * math.pi) + np.log(variances) + squared / variances)
    )
    probability_gradient = (
        np.r_[0.0, gradient[1:3], 2 * gradient[3]] if asymmetric else np.r_[0.0, gradient[1:3]]
    )
    logit_gradient = probabilities[1:] * (
        probability_gradient[1:] - float(probabilities @ probability_gradient)
    )
    result = np.r_[parameters[0] * gradient[0], logit_gradient]
    if not math.isfinite(objective) or not np.isfinite(result).all():
        raise QuantError("The likelihood or its analytic gradient is nonfinite.")
    return objective, result


def forecast(returns, parameters, variances) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    variances = np.asarray(variances, dtype=float)
    checked = variance_path(values, parameters)
    if variances.shape != checked.shape or not np.array_equal(variances, checked):
        raise QuantError("Forecast only the variance path from these observed returns.")
    omega, alpha, beta, gamma = parameters
    persistence = alpha + beta + gamma / 2
    result = np.empty(HORIZON)
    result[0] = omega + (alpha + gamma * (values[-1] < 0)) * values[-1] ** 2 + beta * variances[-1]
    for i in range(1, HORIZON):
        result[i] = omega + persistence * result[i - 1]
    if not np.isfinite(result).all() or (result <= 0).any():
        raise QuantError("The finite-horizon forecast must be finite and positive.")
    return result


def independent(returns, parameters, initial_variance: float) -> dict:
    values = [float(value) for value in returns]
    omega, alpha, beta, gamma = [float(value) for value in parameters]
    if (
        len(values) < 2
        or not all(math.isfinite(value) for value in [*values, omega, alpha, beta, gamma])
        or omega <= 0
        or min(alpha, beta, gamma) < 0
        or not 0 <= alpha + beta + gamma / 2 < 1
        or not math.isfinite(initial_variance)
        or initial_variance <= 0
    ):
        raise QuantError("Invalid native-unit inputs for independent convolution.")
    shocks = [omega + (alpha + gamma * (value < 0)) * value**2 for value in values]
    variances = [
        beta**i * initial_variance + math.fsum(beta ** (i - 1 - j) * shocks[j] for j in range(i))
        for i in range(len(values))
    ]
    first = shocks[-1] + beta * variances[-1]
    persistence = alpha + beta + gamma / 2
    future = [
        persistence**i * first + omega * math.fsum(persistence**j for j in range(i))
        for i in range(HORIZON)
    ]
    loss = math.fsum(
        0.5 * (math.log(2 * math.pi) + math.log(h) + value**2 / h)
        for value, h in zip(values, variances, strict=True)
    ) / len(values)
    return {
        "variances": variances,
        "forecast_variances": future,
        "mean_negative_log_likelihood": loss,
        "mean_forecast_variance": math.fsum(future) / HORIZON,
        "pooled_squared_return": math.fsum(value**2 for value in values) / len(values),
    }


def fit(returns, asymmetric: bool) -> dict:
    limits = bounds(asymmetric)
    values = np.asarray(returns, dtype=float)
    if (
        values.shape != (LOOKBACK,)
        or not np.isfinite(values).all()
        or not (values < 0).any()
        or not (values > 0).any()
    ):
        raise QuantError("Require252finite observed returns with both signs.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            pooled = float(np.mean(values**2))
            if not math.isfinite(pooled) or pooled <= 0:
                raise QuantError("A zero or nonfinite pooled second moment cannot initialize risk.")
            normalized = values / math.sqrt(pooled)
        except FloatingPointError as error:
            raise QuantError("Training second-moment normalization failed.") from error
    if float(np.ptp(normalized**2)) <= 1e-12:
        raise QuantError("Constant squared returns do not identify conditional risk dynamics.")
    initial = initial_coordinates(asymmetric)
    initial_objective, _ = likelihood_gradient(initial, normalized, asymmetric)
    result = minimize(
        likelihood_gradient,
        initial,
        args=(normalized, asymmetric),
        method="L-BFGS-B",
        jac=True,
        bounds=limits,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": FTOL,
            "gtol": GTOL,
            "maxls": MAX_LINE_SEARCH,
        },
    )
    receipt = {
        "asymmetric": asymmetric,
        "observations": LOOKBACK,
        "pooled_daily_squared_return": pooled,
        "initial_normalized_variance": 1.0,
        "initial_coordinates": initial.tolist(),
        "optimizer": "L-BFGS-B",
        "optimizer_initializations": 1,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "optimizer_function_evaluations": int(result.nfev),
        "initial_mean_negative_log_likelihood_scaled": initial_objective,
    }
    if not np.isfinite(result.x).all() or not math.isfinite(result.fun):
        raise FitError("The single optimizer returned nonfinite results; no retry.", receipt)
    receipt["coordinates"] = result.x.tolist()
    receipt["optimizer_objective"] = float(result.fun)
    if not result.success:
        raise FitError(
            "The single optimizer did not converge; no alternate initialization.", receipt
        )
    parameters, probabilities = decode(result.x, asymmetric)
    objective, gradient = likelihood_gradient(result.x, normalized, asymmetric)
    if abs(objective - result.fun) > 1e-10 or objective > initial_objective + 1e-10:
        raise FitError(
            "The final likelihood is inconsistent or worse than its initialization.", receipt
        )
    variances = variance_path(normalized, parameters)
    future = forecast(normalized, parameters, variances)
    active_bounds = []
    projected = gradient.copy()
    for i, (value, (low, high)) in enumerate(zip(result.x, limits, strict=True)):
        near_low, near_high = value - low <= 1e-8, high - value <= 1e-8
        if near_low or near_high:
            active_bounds.append({"coordinate": i, "bound": "lower" if near_low else "upper"})
        if (near_low and projected[i] > 0) or (near_high and projected[i] < 0):
            projected[i] = 0.0
    receipt.update(
        {
            "parameters_scaled": parameters.tolist(),
            "parameters_daily": [float(parameters[0] * pooled), *parameters[1:].tolist()],
            "persistence": float(parameters[1] + parameters[2] + parameters[3] / 2),
            "positive_persistence_slack": float(probabilities[0]),
            "active_numerical_bounds": active_bounds,
            "gradient_max_absolute": float(np.max(abs(gradient))),
            "projected_gradient_max_absolute": float(np.max(abs(projected))),
            "mean_negative_log_likelihood_scaled": objective,
            "mean_negative_log_likelihood_daily": objective + 0.5 * math.log(pooled),
            "conditional_variances_daily": (variances * pooled).tolist(),
            "forecast_variances_daily": (future * pooled).tolist(),
            "mean_forecast_variance_daily": float(np.mean(future) * pooled),
            "forecast_sessions": HORIZON,
            "globally_optimal_or_statistically_identified": False,
            "innovation_symmetry_empirically_verified": False,
        }
    )
    return receipt
