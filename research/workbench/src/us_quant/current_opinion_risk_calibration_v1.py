from __future__ import annotations

import math

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.growth_macro_defense_hedge_v1 import EXPERT_IDS
from us_quant.retained_pair_equal_allocation_v1 import ASSETS

HISTORY, HORIZON, TOLERANCE = 126, 21, 1e-12
VIEWS = ("equal", "recorded_rolling")
MODELS = ("trailing_expert_NET_RF", "current_opinion_gross_mark_RF")


def require(condition, message):
    if not condition:
        raise QuantError(message)


def _panel(frame, columns, rows, end=None):
    require(
        isinstance(frame, pd.DataFrame)
        and len(frame) == rows
        and tuple(frame.columns) == tuple(columns)
        and frame.index.is_unique
        and frame.index.is_monotonic_increasing
        and frame.index.equals(sessions(frame.index[0], frame.index[-1]))
        and all(dtype.kind in "fiu" for dtype in frame.dtypes)
        and np.isfinite(frame.to_numpy()).all()
        and (frame.to_numpy() > -1).all()
        and (end is None or frame.index[-1] == end),
        "Use complete finite original-calendar return coordinates, never future rows or fills.",
    )


def _rf(series, index):
    require(
        isinstance(series, pd.Series)
        and series.index.equals(index)
        and series.dtype.kind in "fiu"
        and np.isfinite(series.to_numpy()).all(),
        "Keep the exact admitted common assessment RF on every statistical return date.",
    )


def covariance(values):
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 2
        and len(values) >= 2
        and values.shape[1] == 4
        and np.isfinite(values).all(),
        "Covariance needs four complete finite statistical expert coordinates.",
    )
    means = np.array([math.fsum(values[:, i]) / len(values) for i in range(4)])
    centered = values - means
    result = centered.T @ centered / (len(values) - 1)
    require(np.isfinite(result).all(), "A nonfinite covariance is not a forecast.")
    return result


def variance(values):
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(),
        "Variance needs finite complete statistical observations.",
    )
    mean = math.fsum(values) / len(values)
    result = math.fsum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    require(math.isfinite(result) and result >= 0, "Invalid variance cannot be clipped or filled.")
    return result


def forecast(as_of, asset_returns, expert_net_returns, risk_free, opinions, weights):
    as_of = pd.Timestamp(as_of)
    _panel(asset_returns, ASSETS, HISTORY, as_of)
    _panel(expert_net_returns, EXPERT_IDS, HISTORY, as_of)
    require(
        asset_returns.index.equals(expert_net_returns.index),
        "Both risk models need exactly the same mature dates.",
    )
    _rf(risk_free, asset_returns.index)
    opinions = np.asarray(opinions, dtype=float)
    require(
        opinions.shape == (4, len(ASSETS))
        and np.isfinite(opinions).all()
        and (opinions >= 0).all()
        and all(abs(math.fsum(row) - 0.98) <= TOLERANCE for row in opinions),
        "Use the four unchanged known long-only opinions with 98% invested and idle cash.",
    )
    require(tuple(weights) == VIEWS, "Keep both recorded outer allocations, not a new selector.")
    for allocation in weights.values():
        require(
            np.asarray(allocation).shape == (4,)
            and np.isfinite(allocation).all()
            and (np.asarray(allocation) >= 0).all()
            and abs(math.fsum(allocation) - 1) <= TOLERANCE,
            "Statistical expert weights must be the saved funded nonnegative fractions.",
        )
    rf = risk_free.to_numpy()[:, None]
    samples = {
        MODELS[0]: expert_net_returns.to_numpy() - rf,
        MODELS[1]: asset_returns.to_numpy() @ opinions.T - rf,
    }
    results, gap = {}, 0.0
    for model, sample in samples.items():
        matrix = covariance(sample)
        predictions = {}
        for view, allocation in weights.items():
            value = variance(sample @ allocation)
            error = abs(float(np.asarray(allocation) @ matrix @ allocation) - value)
            require(error <= TOLERANCE, "Scalar and matrix forecast variance disagree.")
            gap = max(gap, error)
            predictions[view] = value
        results[model] = {"covariance": matrix.tolist(), "variance_by_recorded_view": predictions}
    return {
        "as_of": str(as_of.date()),
        "history_start": str(asset_returns.index[0].date()),
        "history_end": str(as_of.date()),
        "history_sessions": HISTORY,
        "weights": {k: list(map(float, v)) for k, v in weights.items()},
        "models": results,
        "maximum_scalar_matrix_variance_difference": gap,
        "current_opinion_projection_includes_fees_or_future_policy_changes": False,
        "new_allocation_or_model_selection": False,
    }


def assess(prediction, future_net_returns, future_rf):
    _panel(future_net_returns, EXPERT_IDS, HORIZON)
    _rf(future_rf, future_net_returns.index)
    as_of = pd.Timestamp(prediction["as_of"])
    expected = sessions(as_of, future_net_returns.index[-1])[1:]
    require(
        future_net_returns.index.equals(expected) and future_net_returns.index[0] > as_of,
        "Labels must be exactly the next 21 sessions, not overlapping history or a later crop.",
    )
    sample = future_net_returns.to_numpy() - future_rf.to_numpy()[:, None]
    rows = []
    for view in VIEWS:
        realized = variance(sample @ np.asarray(prediction["weights"][view]))
        for model in MODELS:
            predicted = prediction["models"][model]["variance_by_recorded_view"][view]
            rows.append(
                {
                    "view": view,
                    "model": model,
                    "predicted_daily_variance": predicted,
                    "realized_NET_statistical_mixture_daily_variance": realized,
                    "signed_variance_error": predicted - realized,
                    "absolute_variance_error": abs(predicted - realized),
                    "underforecast": predicted < realized,
                    "forecast_to_realized_variance_ratio": predicted / realized
                    if realized > 0
                    else None,
                    "label_start": str(future_net_returns.index[0].date()),
                    "label_end": str(future_net_returns.index[-1].date()),
                    "label_sessions": HORIZON,
                    "label_is_main_account_or_tradable_portfolio": False,
                }
            )
    return rows
