from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, rebalance as legacy_rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import file_digest

ORDER_NOTIONAL_THRESHOLD = 1e-6


def engine_reference() -> dict:
    return {
        "id": "cash_funded_noop_v2",
        "source_sha256": file_digest(Path(__file__)),
    }


def rebalance(
    asset_values: np.ndarray,
    cash: float,
    target: np.ndarray,
    cost_bps: float,
    commission: float,
) -> tuple[np.ndarray, float, float, float, int]:
    if (
        asset_values.ndim != 1
        or asset_values.shape != target.shape
        or not np.isfinite(asset_values).all()
        or not np.isfinite(target).all()
        or (asset_values < 0).any()
        or (target < 0).any()
        or target.sum() > 1 + 1e-10
        or not np.isfinite([cash, cost_bps, commission]).all()
        or cash < 0
        or not 0 <= cost_bps < 100
        or commission < 0
    ):
        raise QuantError("Invalid cash-funded accounting inputs.")
    nav = float(asset_values.sum() + cash)
    if nav <= 0:
        raise QuantError("A funded rebalance requires positive capital.")
    differences = target * nav - asset_values
    # A rounding-only proportional fee must not enter the fixed-commission solver.
    if not (abs(differences) > ORDER_NOTIONAL_THRESHOLD).any():
        return asset_values.copy(), float(cash), 0.0, 0.0, 0
    return legacy_rebalance(asset_values, cash, target, cost_bps, commission)


def simulate(
    data: MarketData,
    signals: pd.DataFrame,
    start: str,
    end: str,
    *,
    initial_capital: float = 100000.0,
    cost_bps: float = 5.0,
    commission: float = 1.0,
    delay: int = 1,
) -> BacktestResult:
    data.validate()
    if (
        not np.isfinite([initial_capital, cost_bps, commission]).all()
        or initial_capital <= 0
        or not 0 <= cost_bps < 100
        or commission < 0
        or type(delay) is not int
        or delay < 1
        or not signals.index.equals(data.close.index)
        or not signals.columns.equals(data.close.columns)
        or (signals.isna().any(axis=1) & ~signals.isna().all(axis=1)).any()
    ):
        raise QuantError("Invalid versioned simulation inputs or execution delay.")
    defined = signals.dropna(how="all")
    if (
        not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or (defined.sum(axis=1) > 1 + 1e-10).any()
    ):
        raise QuantError("Only finite, long-only, cash-funded signals are supported.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("A versioned simulation requires at least two sessions.")
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = float(initial_capital)
    rows, weights = [], []
    for opening, closing, target in zip(
        data.open.loc[dates].to_numpy(),
        data.close.loc[dates].to_numpy(),
        signals.shift(delay).loc[dates].to_numpy(),
        strict=True,
    ):
        cost = turnover = 0.0
        orders = 0
        if not np.isnan(target).all():
            values, cash, cost, turnover, orders = rebalance(
                units * opening, cash, target, cost_bps, commission
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("Versioned accounting became insolvent or borrowed cash.")
        held = values / nav
        rows.append((nav, nav / previous_nav - 1, cash, held.sum(), turnover, cost, orders))
        weights.append(held)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return BacktestResult(
        frame, pd.DataFrame(weights, index=dates, columns=data.close.columns)
    )
