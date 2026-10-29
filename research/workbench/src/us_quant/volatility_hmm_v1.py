from __future__ import annotations

import math

import numpy as np

from us_quant.config import QuantError

INITIAL = np.array([0.5, 0.5])
MAX_ITERATIONS = 500
RELATIVE_OBJECTIVE_TOLERANCE = 1e-7
TRANSITION_PSEUDOCOUNT = 1.0
VARIANCE_PSEUDOCOUNT = 6.0


def expectation(returns, transition, variances):
    y, a, v = (np.asarray(value, dtype=float) for value in (returns, transition, variances))
    if (
        y.ndim != 1
        or len(y) == 0
        or a.shape != (2, 2)
        or v.shape != (2,)
        or not all(np.isfinite(value).all() for value in (y, a, v))
        or (a <= 0).any()
        or (v <= 0).any()
        or not np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise QuantError("Require finite observed returns and positive two-state probabilities.")
    log_a = np.log(a)
    emission = -0.5 * (np.log(2 * np.pi * v)[None, :] + y[:, None] ** 2 / v[None, :])
    if not np.isfinite(emission).all():
        raise QuantError("An invalid Gaussian likelihood cannot become a regime signal.")
    forward = np.empty_like(emission)
    forward[0] = np.log(INITIAL) + emission[0]
    for t in range(1, len(y)):
        forward[t, 0] = emission[t, 0] + np.logaddexp(
            forward[t - 1, 0] + log_a[0, 0], forward[t - 1, 1] + log_a[1, 0]
        )
        forward[t, 1] = emission[t, 1] + np.logaddexp(
            forward[t - 1, 0] + log_a[0, 1], forward[t - 1, 1] + log_a[1, 1]
        )
    likelihood = float(np.logaddexp(forward[-1, 0], forward[-1, 1]))
    backward = np.zeros_like(emission)
    for t in range(len(y) - 2, -1, -1):
        continuation = emission[t + 1] + backward[t + 1]
        backward[t, 0] = np.logaddexp(log_a[0, 0] + continuation[0], log_a[0, 1] + continuation[1])
        backward[t, 1] = np.logaddexp(log_a[1, 0] + continuation[0], log_a[1, 1] + continuation[1])
    gamma = np.exp(forward + backward - likelihood)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.exp(
        forward[:-1, :, None]
        + log_a[None, :, :]
        + emission[1:, None, :]
        + backward[1:, None, :]
        - likelihood
    )
    if len(xi):
        xi /= xi.sum(axis=(1, 2), keepdims=True)
    filtered = np.exp(forward[-1] - likelihood)
    filtered /= filtered.sum()
    if not all(np.isfinite(value).all() for value in (gamma, xi, filtered)):
        raise QuantError("Posterior probabilities failed normalization.")
    return likelihood, gamma, xi, filtered


def penalized_objective(likelihood, transition, variances, pooled):
    return float(
        likelihood
        + np.log(transition).sum()
        - 3 * np.log(variances).sum()
        - 3 * pooled * (1 / variances).sum()
    )


def fit(returns):
    y = np.asarray(returns, dtype=float)
    if y.shape != (252,) or not np.isfinite(y).all():
        raise QuantError("Fit only the fixed252observed return window.")
    square = y**2
    pooled = float(square.mean())
    low = square <= np.median(square)
    if pooled <= 0 or not np.isfinite(pooled) or not low.any() or low.all():
        raise QuantError("The history does not identify two volatility regimes.")
    variances = np.array(
        [
            (square[mask].sum() + VARIANCE_PSEUDOCOUNT * pooled)
            / (mask.sum() + VARIANCE_PSEUDOCOUNT)
            for mask in (low, ~low)
        ]
    )
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    likelihood, gamma, xi, filtered = expectation(y, transition, variances)
    objective = penalized_objective(likelihood, transition, variances, pooled)
    minimum_improvement = math.inf
    for iteration in range(1, MAX_ITERATIONS + 1):
        counts = xi.sum(axis=0) + TRANSITION_PSEUDOCOUNT
        new_transition = counts / counts.sum(axis=1, keepdims=True)
        new_variances = (
            np.sum(gamma * square[:, None], axis=0) + VARIANCE_PSEUDOCOUNT * pooled
        ) / (gamma.sum(axis=0) + VARIANCE_PSEUDOCOUNT)
        order = np.argsort(new_variances, kind="stable")
        new_variances = new_variances[order]
        new_transition = new_transition[np.ix_(order, order)]
        new_likelihood, new_gamma, new_xi, new_filtered = expectation(
            y, new_transition, new_variances
        )
        new_objective = penalized_objective(new_likelihood, new_transition, new_variances, pooled)
        improvement = new_objective - objective
        if improvement < -1e-10 * max(1.0, abs(objective)):
            raise QuantError("MAP-EM objective decreased beyond numerical tolerance.")
        minimum_improvement = min(minimum_improvement, improvement)
        scale = max(1.0, abs(objective))
        transition, variances = new_transition, new_variances
        likelihood, gamma, xi, filtered = new_likelihood, new_gamma, new_xi, new_filtered
        objective = new_objective
        if improvement <= RELATIVE_OBJECTIVE_TOLERANCE * scale:
            if variances[1] <= variances[0] * (1 + 1e-8):
                raise QuantError(
                    "Fitted volatility labels are indistinguishable: "
                    f"iteration={iteration}, variances={variances.tolist()}."
                )
            return {
                "iterations": iteration,
                "converged": True,
                "pooled_daily_squared_return": pooled,
                "transition": transition.tolist(),
                "state_variances_daily": variances.tolist(),
                "initial_probabilities": INITIAL.tolist(),
                "filtered_probabilities_at_last_observation": filtered.tolist(),
                "low_variance_state_probability": float(filtered[0]),
                "log_likelihood": likelihood,
                "penalized_log_posterior": objective,
                "last_objective_improvement": improvement,
                "minimum_objective_improvement": minimum_improvement,
                "global_optimum_or_empirical_probability_calibration_claimed": False,
            }
    raise QuantError(
        "Fixed MAP-EM did not converge; no alternate initialization: "
        f"iterations={MAX_ITERATIONS}, objective={objective}, "
        f"variances={variances.tolist()}, transition={transition.tolist()}."
    )


def independent_filter(returns, transition, variances):
    y = list(map(float, returns))
    a = np.asarray(transition, dtype=float)
    v = list(map(float, variances))
    if (
        not y
        or len(v) != 2
        or a.shape != (2, 2)
        or not all(math.isfinite(value) for value in (*y, *v))
        or min(v) <= 0
        or not np.isfinite(a).all()
        or (a <= 0).any()
        or not np.allclose(a.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise QuantError("Independent filtering needs valid observed model inputs.")
    posterior, scales = [0.5, 0.5], []
    for i, value in enumerate(y):
        prior = (
            posterior
            if i == 0
            else [math.fsum(posterior[j] * float(a[j, k]) for j in range(2)) for k in range(2)]
        )
        log_density = [
            -0.5 * (math.log(2 * math.pi * variance) + value * value / variance) for variance in v
        ]
        shift = max(log_density)
        weighted = [prior[k] * math.exp(log_density[k] - shift) for k in range(2)]
        scale = math.fsum(weighted)
        if scale <= 0 or not math.isfinite(scale):
            raise QuantError("Independent likelihood scaling failed.")
        posterior = [value / scale for value in weighted]
        scales.append(shift + math.log(scale))
    return posterior, math.fsum(scales)
