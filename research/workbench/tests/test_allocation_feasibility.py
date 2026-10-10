from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant.allocation_feasibility import static_sharpe_envelope
from us_quant.config import QuantError


def exact_sample(mean, covariance):
    rows = 2000
    rng = np.random.default_rng(9481)
    observations = rng.normal(size=(rows, len(mean)))
    observations -= observations.mean(axis=0)
    q, _ = np.linalg.qr(observations)
    observations = q * np.sqrt(rows - 1) @ np.linalg.cholesky(covariance / 252).T
    observations += np.asarray(mean) / 252
    return pd.DataFrame(
        observations,
        index=pd.date_range("2010-01-01", periods=rows),
        columns=[f"asset_{i}" for i in range(len(mean))],
    )


def test_diagonal_covariance_matches_analytical_sharpe_envelope():
    mean = np.array([0.10, 0.20])
    covariance = np.diag([0.04, 0.09])
    result = static_sharpe_envelope(exact_sample(mean, covariance))
    expected = float(np.sqrt((mean**2 / np.diag(covariance)).sum()))
    assert result["attained_in_sample_sharpe"] == pytest.approx(expected, abs=1e-7)
    assert result["numerical_upper_bound"] >= expected - 1e-10
    assert result["numerical_upper_bound"] - expected < 1e-6
    assert not result["strategy_qualified"]


def test_negative_mean_assets_are_not_short_sold():
    mean = np.array([0.10, -0.20])
    covariance = np.diag([0.04, 0.09])
    result = static_sharpe_envelope(exact_sample(mean, covariance))
    assert result["attained_in_sample_sharpe"] == pytest.approx(0.50, abs=1e-7)


def test_no_positive_excess_mean_is_an_explicit_non_strategy_result():
    result = static_sharpe_envelope(exact_sample(np.array([-0.10, -0.20]), np.diag([0.04, 0.09])))
    assert result["status"] == "no_positive_sample_excess_mean"
    assert result["numerical_upper_bound"] == 0
    assert not result["strategy_qualified"]


def test_singular_or_missing_data_is_not_silently_fixed():
    frame = exact_sample(np.array([0.10, 0.20]), np.diag([0.04, 0.09]))
    frame.iloc[0, 0] = np.nan
    with pytest.raises(QuantError):
        static_sharpe_envelope(frame)
    frame = frame.dropna()
    frame["asset_1"] = frame["asset_0"]
    with pytest.raises(QuantError, match="positive definite"):
        static_sharpe_envelope(frame)


def test_large_hindsight_sharpe_does_not_qualify_as_a_strategy():
    frame = exact_sample(np.array([0.30, 0.40]), np.diag([0.02, 0.03]))
    result = static_sharpe_envelope(frame)
    assert result["attained_in_sample_sharpe"] > 1
    assert result["strategy_qualified"] is False
    assert result["optimal_weights_not_published_as_trade_recommendations"]
