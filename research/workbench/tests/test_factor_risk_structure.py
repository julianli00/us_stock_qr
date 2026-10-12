from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.factor_risk_structure import dependence, market_structure, matrix_structure, validate_policy
from us_quant.storage import read_json


def inputs():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2016-01-04", periods=512)
    market = pd.Series(rng.normal(0.0004, 0.012, len(dates)), index=dates)
    rates = pd.Series(0.00004, index=dates)
    returns = pd.DataFrame({
        name: rates + beta * (market - rates) + rng.normal(0.0001, 0.002, len(dates))
        for name, beta in (("A", 0.8), ("B", 1.1), ("C", 1.3))
    })
    return returns, market, rates


def test_identity_and_rank_one_risk_dimensions_do_not_count_labels_as_independence():
    independent = matrix_structure(np.eye(6))
    identical = matrix_structure(np.ones((6, 6)))
    assert independent["participation_ratio_dimension"] == pytest.approx(6)
    assert independent["leading_eigenvalue_fraction"] == pytest.approx(1 / 6)
    assert identical["participation_ratio_dimension"] == pytest.approx(1)
    assert identical["leading_eigenvalue_fraction"] == pytest.approx(1)
    for scale in (1e-200, 1e200):
        actual = matrix_structure(np.eye(6) * scale)
        assert actual["participation_ratio_dimension"] == pytest.approx(6)
        assert actual["leading_eigenvalue_fraction"] == pytest.approx(1 / 6)


def test_covariance_and_correlation_dimensions_are_distinct_and_scale_sensitive_only_where_expected():
    returns, _, _ = inputs()
    old = dependence(returns)
    rescaled = returns.copy()
    rescaled["C"] *= 20
    new = dependence(rescaled)
    assert old["correlation_structure"]["participation_ratio_dimension"] == pytest.approx(
        new["correlation_structure"]["participation_ratio_dimension"], abs=1e-12
    )
    assert new["covariance_structure"]["participation_ratio_dimension"] < old[
        "covariance_structure"
    ]["participation_ratio_dimension"]
    correlation = returns.corr().to_numpy()
    reference = len(returns.columns) ** 2 / (correlation**2).sum()
    assert old["correlation_structure"]["participation_ratio_dimension"] == pytest.approx(
        reference, abs=1e-12
    )


def test_market_regression_matches_independent_covariance_slope_and_explicit_residuals():
    returns, market, rates = inputs()
    result = market_structure(returns, market, rates)
    x = market - rates
    for name in returns:
        y = returns[name] - rates
        beta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
        intercept = float(y.mean() - beta * x.mean())
        residual = y - (intercept + beta * x)
        r_squared = 1 - float((residual**2).sum() / ((y - y.mean())**2).sum())
        actual = result["in_sample_market_regressions"][name]
        assert actual["beta_vs_spy"] == pytest.approx(beta, abs=1e-12)
        assert actual["market_r_squared"] == pytest.approx(r_squared, abs=1e-12)
    assert result["market_residual_risk"]["estimable"]
    assert not result["market_fit_is_out_of_sample_forecast"]
    assert not result["market_residual_returns_are_investable_strategy"]
    assert not result["independent_alpha_proven"]


def test_perfect_market_fit_explicitly_marks_residual_dependence_unestimable_not_zero_correlation():
    returns, market, rates = inputs()
    exact = pd.DataFrame({
        "A": rates + 0.8 * (market - rates),
        "B": rates + 1.3 * (market - rates),
    })
    result = market_structure(exact, market, rates)
    assert result["market_residual_risk"]["estimable"] is False
    assert result["market_residual_risk"]["symbols"] == ["A", "B"]
    assert "correlation_matrix" not in result["market_residual_risk"]


@pytest.mark.parametrize("matrix", [
    [[1, 2], [2, 1]],
    [[1, 0], [0, 0]],
    [[1, np.nan], [np.nan, 1]],
    [[1, 1], [0, 1]],
])
def test_invalid_risk_matrices_raise_explicitly(matrix):
    with pytest.raises(QuantError):
        matrix_structure(np.asarray(matrix))


@pytest.mark.parametrize("problem", ["missing", "constant", "duplicate", "unsorted", "market"])
def test_bad_or_unaligned_observations_cannot_be_accepted_as_a_valid_diagnostic(problem):
    returns, market, rates = inputs()
    if problem == "missing":
        returns.iloc[0, 0] = np.nan
    elif problem == "constant":
        returns["A"] = 0
    elif problem == "duplicate":
        returns.columns = ["A", "A", "C"]
    elif problem == "unsorted":
        returns = returns.iloc[::-1]
    else:
        market = market.iloc[1:]
    with pytest.raises(QuantError):
        market_structure(returns, market, rates)


@pytest.mark.parametrize("change", ["window", "cohort", "performance"])
def test_fixed_source_study_cannot_become_early_candidate_evaluation(change):
    policy = read_json(Path(__file__).parents[1] / "config/factor-risk-structure.json")
    if change == "window":
        policy["rolling_sessions"] = 126
    elif change == "cohort":
        policy["cohorts"]["six_equity_families"].remove("IJR")
    else:
        policy["pending_candidate_performance_computed"] = True
    with pytest.raises(QuantError):
        validate_policy(policy)
