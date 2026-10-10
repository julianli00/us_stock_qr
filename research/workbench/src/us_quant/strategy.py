from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import Candidate, QuantError, ResearchConfig


def target_weights(
    history: pd.DataFrame, candidate: Candidate | None, config: ResearchConfig
) -> pd.Series:
    weights = pd.Series(0.0, index=history.columns)
    if candidate is None:
        return weights
    warmup = max(
        max(config.momentum_lookbacks) + 1,
        config.trend_lookback,
        config.volatility_lookback + 1,
    )
    if len(history) < warmup:
        return weights
    prices = history.loc[:, list(candidate.universe)]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise QuantError("Signals require complete positive historical prices.")
    latest = prices.iloc[-1]
    trend = prices.tail(config.trend_lookback).mean()
    score = sum(
        latest / prices.iloc[-lookback - 1] - 1 for lookback in config.momentum_lookbacks
    ) / len(config.momentum_lookbacks)
    returns = prices.pct_change(fill_method=None).tail(config.volatility_lookback)
    volatility = returns.std(ddof=1) * np.sqrt(252)
    eligible = [
        symbol
        for symbol in candidate.universe
        if latest[symbol] > trend[symbol]
        and score[symbol] > 0
        and np.isfinite(volatility[symbol])
        and volatility[symbol] > 1e-8
    ]
    if candidate.kind == "momentum":
        eligible.sort(key=lambda symbol: (-float(score[symbol]), symbol))
    else:
        eligible.sort()
    chosen = eligible[: candidate.top_k]
    if not chosen:
        return weights
    inverse_vol = 1.0 / volatility.loc[chosen]
    budget = (1.0 - config.cash_reserve) * len(chosen) / candidate.top_k
    allocation = (inverse_vol / inverse_vol.sum() * budget).clip(upper=candidate.max_weight)
    covariance = returns.loc[:, chosen].cov() * 252
    portfolio_variance = float(
        allocation.to_numpy() @ covariance.to_numpy() @ allocation.to_numpy()
    )
    if not np.isfinite(portfolio_variance) or portfolio_variance < -1e-12:
        raise QuantError("Invalid trailing portfolio variance.")
    predicted_vol = np.sqrt(max(portfolio_variance, 0.0))
    if predicted_vol > candidate.target_volatility:
        allocation *= candidate.target_volatility / predicted_vol
    weights.loc[chosen] = allocation
    if (
        (weights < 0).any()
        or (weights > candidate.max_weight + 1e-10).any()
        or weights.sum() > 1.0 - config.cash_reserve + 1e-10
    ):
        raise QuantError("Strategy produced a short, leveraged, or over-concentrated target.")
    return weights


def monthly_signals(
    close: pd.DataFrame, candidate: Candidate | None, config: ResearchConfig
) -> pd.DataFrame:
    signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for i, day in enumerate(close.index):
        if is_month_end(day):
            signals.loc[day] = target_weights(close.iloc[: i + 1], candidate, config)
    return signals


def buy_and_hold_signals(close: pd.DataFrame, symbol: str, start: str) -> pd.DataFrame:
    signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    first = close.index.searchsorted(pd.Timestamp(start))
    if first < 1 or first >= len(close):
        raise QuantError("A buy-and-hold benchmark needs a prior observed session.")
    signals.iloc[first - 1] = 0.0
    signals.iloc[first - 1, close.columns.get_loc(symbol)] = 1.0
    return signals
