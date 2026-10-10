from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData


def independent_equity(
    data: MarketData,
    signals: pd.DataFrame,
    start: str,
    end: str,
    *,
    capital: float,
    cost_bps: float,
    commission: float,
    delay: int,
) -> pd.DataFrame:
    """Rebuild positions/cash using MIT bt, without calling the project's simulator."""
    try:
        import bt
        from scipy.optimize import brentq
    except ImportError as exc:
        raise QuantError(
            "Install the declared research extra to run the independent bt audit."
        ) from exc

    if version("bt") != "1.3.0":
        raise QuantError("The independent audit requires the registered bt 1.3.0 engine.")
    data.validate()
    if (
        type(delay) is not int
        or delay < 1
        or capital <= 0
        or not np.isfinite([capital, cost_bps, commission]).all()
        or cost_bps < 0
        or commission < 0
        or not signals.index.equals(data.close.index)
        or not signals.columns.equals(data.close.columns)
    ):
        raise QuantError("Invalid independent accounting inputs.")
    if (signals.isna().any(axis=1) & ~signals.isna().all(axis=1)).any():
        raise QuantError("Independent audit refuses partially specified allocations.")
    defined = signals.dropna(how="all")
    if (
        not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or (defined.sum(axis=1) > 1 + 1e-10).any()
    ):
        raise QuantError("Independent audit supports only long-only cash-funded allocations.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("Independent accounting needs at least two observed sessions.")
    opening = dates + pd.Timedelta(hours=9, minutes=30)
    closing = dates + pd.Timedelta(hours=16)
    event_index = opening.union(closing).sort_values()
    prices = pd.DataFrame(index=event_index, columns=data.close.columns, dtype=float)
    prices.loc[opening] = data.open.loc[dates].to_numpy()
    prices.loc[closing] = data.close.loc[dates].to_numpy()
    targets = signals.shift(delay).loc[dates].copy()
    targets.index = opening
    targets = targets.dropna(how="all")
    rate = cost_bps / 10000

    def fee(quantity, price):
        notional = abs(float(quantity) * float(price))
        return notional * rate + (commission if notional > 1e-6 else 0.0)

    class CashFundedTargets(bt.Algo):
        def __call__(self, portfolio):
            if portfolio.now not in targets.index:
                return True
            weights = targets.loc[portfolio.now].to_numpy()
            marks = prices.loc[portfolio.now].to_numpy()
            quantities = np.array(
                [
                    portfolio.children[symbol].position if symbol in portfolio.children else 0.0
                    for symbol in prices.columns
                ]
            )
            old_values = quantities * marks
            value = float(portfolio.value)

            def budget_residual(investable_nav):
                changes = weights * investable_nav - old_values
                charges = rate * abs(changes).sum()
                charges += commission * np.count_nonzero(abs(changes) > 1e-6)
                return investable_nav + charges - value

            if budget_residual(value) == 0:
                investable = value
            else:
                if budget_residual(0) >= 0:
                    raise QuantError("Independent accounting cannot fund the transaction charges.")
                investable = brentq(budget_residual, 0, value, xtol=1e-10)
                if abs(budget_residual(investable)) > 1e-5:
                    raise QuantError("Independent rebalance did not satisfy its cash budget.")
            trades = weights * investable / marks - quantities
            # Sell first. bt records the trades, fees, holdings, and cash independently.
            for index in np.argsort(trades):
                if abs(trades[index] * marks[index]) > 1e-6:
                    portfolio.transact(
                        float(trades[index]), child=prices.columns[index], update=False
                    )
            portfolio.update(portfolio.now)
            if portfolio.capital < -1e-5:
                raise QuantError("Independent bt audit detected borrowed cash.")
            return True

    strategy = bt.Strategy("independent_cash_account", [CashFundedTargets()])
    backtest = bt.Backtest(
        strategy,
        prices,
        initial_capital=capital,
        commissions=fee,
        integer_positions=False,
        progress_bar=False,
    )
    backtest.run()
    values = backtest.strategy.values.loc[closing].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise QuantError("Independent bt engine returned invalid equity.")
    return pd.DataFrame(
        {"equity": values, "return": values / np.r_[capital, values[:-1]] - 1},
        index=dates,
    )
