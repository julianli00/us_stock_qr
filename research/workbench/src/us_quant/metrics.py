from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import previous_session
from us_quant.config import QuantError, ResearchConfig


def performance(returns: pd.Series, risk_free: pd.Series) -> dict:
    if len(returns) < 2 or not returns.index.equals(risk_free.index):
        raise QuantError("Metrics need at least two returns and exactly aligned risk-free values.")
    if not np.isfinite(returns).all() or not np.isfinite(risk_free).all() or (returns <= -1).any():
        raise QuantError("Invalid returns in performance calculation.")
    wealth = np.cumprod(1.0 + returns.to_numpy())
    highs = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / highs - 1.0
    days = (returns.index[-1] - previous_session(returns.index[0])).days
    cagr = float(wealth[-1] ** (365.25 / days) - 1.0)
    excess = returns - risk_free
    deviation = float(excess.std(ddof=1))
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = (
        float(excess.mean() / deviation * np.sqrt(252))
        if deviation > 1e-12 and volatility > 1e-12
        else None
    )
    yearly = returns.groupby(returns.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))
    rolling = np.expm1(np.log1p(returns).rolling(756, min_periods=756).sum() / 3.0).dropna()
    underwater = longest = 0
    for value in drawdown:
        underwater = underwater + 1 if value < -1e-10 else 0
        longest = max(longest, underwater)
    return {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "sessions": len(returns),
        "total_return": float(wealth[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "annualized_volatility": volatility,
        "max_drawdown": float(-drawdown.min()),
        "max_underwater_sessions": longest,
        "worst_day": float(returns.min()),
        "expected_shortfall_5pct": float(returns.loc[returns <= returns.quantile(0.05)].mean()),
        "year_returns": {str(year): float(value) for year, value in yearly.items()},
        "positive_year_fraction": float((yearly > 0).mean()),
        "rolling_3y_cagr_min": float(rolling.min()) if len(rolling) else None,
        "rolling_3y_cagr_median": float(rolling.median()) if len(rolling) else None,
        "rolling_3y_cagr_above_20pct_fraction": float((rolling > 0.20).mean())
        if len(rolling)
        else None,
    }


def acceptance(strategy: dict, benchmark: dict, config: ResearchConfig) -> dict[str, bool]:
    if (strategy["start"], strategy["end"]) != (benchmark["start"], benchmark["end"]):
        raise QuantError("Acceptance cannot compare mismatched benchmark windows.")
    targets = config.targets
    return {
        "cagr_above_target": bool(strategy["cagr"] > targets.cagr_strictly_above),
        "sharpe_above_target": bool(
            strategy["sharpe"] is not None and strategy["sharpe"] > targets.sharpe_strictly_above
        ),
        "drawdown_within_limit": bool(strategy["max_drawdown"] <= targets.max_drawdown_at_most),
        "beats_primary_benchmark": bool(
            not targets.beat_primary_benchmark or strategy["cagr"] > benchmark["cagr"]
        ),
        "enough_sessions": bool(strategy["sessions"] >= targets.minimum_holdout_sessions),
    }


def block_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    risk_free: pd.Series,
    *,
    samples: int,
    block: int,
    seed: int,
) -> dict:
    if not strategy.index.equals(benchmark.index) or not strategy.index.equals(risk_free.index):
        raise QuantError("Bootstrap series are misaligned.")
    count = len(strategy)
    if count < block or samples < 1 or block < 2:
        raise QuantError("Insufficient observations or invalid bootstrap settings.")
    values = np.column_stack([strategy, benchmark, risk_free])
    rng = np.random.default_rng(seed)
    cagr_samples, sharpe_samples, excess_samples = [], [], []
    for _ in range(samples):
        starts = rng.integers(0, count, size=(count + block - 1) // block)
        positions = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        selected = values[positions]
        cagr = np.expm1(np.log1p(selected[:, 0]).sum() * 252.0 / count)
        bench_cagr = np.expm1(np.log1p(selected[:, 1]).sum() * 252.0 / count)
        excess = selected[:, 0] - selected[:, 2]
        deviation = excess.std(ddof=1)
        if deviation > 1e-12 and selected[:, 0].std(ddof=1) > 1e-12:
            sharpe_samples.append(float(excess.mean() / deviation * np.sqrt(252)))
        cagr_samples.append(float(cagr))
        excess_samples.append(float(cagr - bench_cagr))

    def interval(items: list[float]) -> list[float] | None:
        return [float(x) for x in np.quantile(items, [0.025, 0.975])] if items else None

    return {
        "method": (
            "paired circular moving-block bootstrap; 252-session CAGR; conditional on history"
        ),
        "samples": samples,
        "block_sessions": block,
        "seed": seed,
        "cagr_95pct_interval": interval(cagr_samples),
        "sharpe_95pct_interval": interval(sharpe_samples),
        "excess_cagr_vs_spy_95pct_interval": interval(excess_samples),
        "warning": "Not a probability of future success; does not remove selection or regime bias.",
    }
