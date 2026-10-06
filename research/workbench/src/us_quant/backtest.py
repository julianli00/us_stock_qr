from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame
    weights: pd.DataFrame


def rebalance(
    asset_values: np.ndarray,
    cash: float,
    target: np.ndarray,
    cost_bps: float,
    commission: float,
) -> tuple[np.ndarray, float, float, float, int]:
    nav = float(asset_values.sum() + cash)
    rate = cost_bps / 10000.0

    def charge(post_cost_nav: float) -> tuple[float, float, int]:
        changes = target * post_cost_nav - asset_values
        traded = float(np.abs(changes).sum())
        orders = int(np.count_nonzero(np.abs(changes) > 1e-6))
        return traded * rate + orders * commission, traded, orders

    if charge(nav)[0] == 0:
        desired = target * nav
        return desired, float(nav - desired.sum()), 0.0, charge(nav)[1] / nav, charge(nav)[2]
    lower, upper = 0.0, nav
    if charge(lower)[0] >= nav:
        raise QuantError("Transaction charges would exhaust the portfolio.")
    for _ in range(80):
        post_cost_nav = (lower + upper) / 2.0
        cost, _, _ = charge(post_cost_nav)
        if post_cost_nav + cost > nav:
            upper = post_cost_nav
        else:
            lower = post_cost_nav
    post_cost_nav = lower
    cost, traded, orders = charge(post_cost_nav)
    desired = target * post_cost_nav
    remaining_cash = nav - float(desired.sum()) - cost
    if remaining_cash < -1e-7 or not np.isfinite(remaining_cash):
        raise QuantError("Rebalance would borrow cash.")
    return desired, max(remaining_cash, 0.0), cost, traded / nav, orders


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
        or not isinstance(delay, int)
        or delay < 1
    ):
        raise QuantError("Invalid simulation capital, costs, or causal execution delay.")
    if not signals.index.equals(data.close.index) or not signals.columns.equals(data.close.columns):
        raise QuantError("Signals are not aligned with the complete price panel.")
    partial = signals.isna().any(axis=1) & ~signals.isna().all(axis=1)
    if partial.any():
        raise QuantError("A signal row must be entirely missing or entirely specified.")
    defined = signals.dropna(how="all")
    if (
        not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or (defined.sum(axis=1) > 1.0 + 1e-10).any()
    ):
        raise QuantError("Only finite, unleveraged, long-only signals are supported.")
    selected = data.close.loc[start:end].index
    if len(selected) < 2:
        raise QuantError("A simulation requires at least two sessions.")
    opens = data.open.loc[selected].to_numpy()
    closes = data.close.loc[selected].to_numpy()
    scheduled = signals.shift(delay).loc[selected].to_numpy()
    units = np.zeros(len(data.close.columns), dtype=float)
    cash = previous_nav = float(initial_capital)
    rows, end_weights = [], []
    for opening, closing, target in zip(opens, closes, scheduled, strict=True):
        costs = turnover = 0.0
        orders = 0
        if not np.isnan(target).all():
            dollars, cash, costs, turnover, orders = rebalance(
                units * opening, cash, target, cost_bps, commission
            )
            units = dollars / opening
        close_values = units * closing
        nav = float(close_values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("Insolvent or nonfinite simulated portfolio.")
        weights = close_values / nav
        rows.append(
            (nav, nav / previous_nav - 1.0, cash, float(weights.sum()), turnover, costs, orders)
        )
        end_weights.append(weights)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=selected,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[selected]
    return BacktestResult(
        frame=frame,
        weights=pd.DataFrame(end_weights, index=selected, columns=data.close.columns),
    )
