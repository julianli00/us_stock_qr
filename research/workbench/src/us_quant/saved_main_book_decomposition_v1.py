from __future__ import annotations

import math

import numpy as np
import pandas as pd

from us_quant.config import QuantError

CAPITAL = 10000.0
MONEY_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-10
REQUIRED = ("equity", "cash", "gross_exposure", "cost", "orders")


def interval(daily: pd.DataFrame, start: int, stop: int) -> dict:
    if type(start) is not int or type(stop) is not int or not -1 <= start < stop < len(daily):
        raise QuantError("Use exact saved close clocks or the original initial capital.")
    rows = daily.iloc[start + 1 : stop + 1]
    symbols = [c.removeprefix("overnight::") for c in daily if c.startswith("overnight::")]
    beginning = CAPITAL if start == -1 else float(daily["equity"].iloc[start])
    ending = float(daily["equity"].iloc[stop])
    marks = {
        symbol: math.fsum([*rows["overnight::" + symbol], *rows["intraday::" + symbol]])
        for symbol in symbols
    }
    fees = math.fsum(rows["cost"])
    observed = ending - beginning
    error = abs(math.fsum(marks.values()) - fees - observed)
    if error > MONEY_TOLERANCE:
        raise QuantError("The complete stored interval does not reconcile to asset marks and fees.")
    return {
        "start": "initial_USD10000_before_first_formal_return"
        if start == -1
        else str(daily.index[start].date()),
        "end": str(daily.index[stop].date()),
        "start_position": start,
        "end_position": stop,
        "return_sessions": len(rows),
        "start_NAV_USD": beginning,
        "end_NAV_USD": ending,
        "actual_NAV_change_USD": observed,
        "actual_stored_interval_change": ending / beginning - 1,
        "asset_mark_PnL_USD": marks,
        "overnight_mark_PnL_USD": math.fsum(math.fsum(rows["overnight::" + s]) for s in symbols),
        "intraday_mark_PnL_USD": math.fsum(math.fsum(rows["intraday::" + s]) for s in symbols),
        "recorded_fees_USD": fees,
        "model_tickets": int(rows["orders"].sum()),
        "interval_is_net_loss": observed < 0,
        "recorded_fees_fraction_of_actual_net_loss": fees / -observed if observed < 0 else None,
        "maximum_whole_interval_identity_error_USD": error,
        "fee_free_or_expert_causal_counterfactual": False,
    }


def decompose(
    account: pd.DataFrame,
    weights: pd.DataFrame,
    opened: pd.DataFrame,
    closed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if (
        not isinstance(account.index, pd.DatetimeIndex)
        or account.empty
        or not account.index.is_unique
        or not account.index.is_monotonic_increasing
        or not account.columns.is_unique
        or not set(REQUIRED).issubset(account.columns)
        or weights.empty
        or not weights.columns.is_unique
        or any(not isinstance(s, str) or "::" in s for s in weights.columns)
        or any(not p.index.equals(account.index) for p in (weights, opened, closed))
        or any(not p.columns.equals(weights.columns) for p in (opened, closed))
        or any(
            d.kind not in "fiu"
            for frame in (account.loc[:, list(REQUIRED)], weights, opened, closed)
            for d in frame.dtypes
        )
    ):
        raise QuantError("Keep complete, unique and identically aligned saved account coordinates.")
    values = account.loc[:, list(REQUIRED)].to_numpy(dtype=float)
    w, op, cl = (p.to_numpy(dtype=float) for p in (weights, opened, closed))
    if (
        not all(np.isfinite(p).all() for p in (values, w, op, cl))
        or (w < 0).any()
        or (op <= 0).any()
        or (cl <= 0).any()
    ):
        raise QuantError("Stored holdings and adjusted prices must be finite and cash-funded.")
    equity, cash, gross, costs, orders = values.T
    if (
        (equity <= 0).any()
        or (cash < 0).any()
        or (costs < 0).any()
        or (orders < 0).any()
        or not np.array_equal(orders, np.floor(orders))
        or (w.sum(axis=1) > 1 + METRIC_TOLERANCE).any()
        or np.max(np.abs(w.sum(axis=1) - gross)) > METRIC_TOLERANCE
    ):
        raise QuantError(
            "The saved unleveraged book, costs or actual ticket counts are inconsistent."
        )
    units = equity[:, None] * w / cl
    prior_units = np.vstack([np.zeros((1, len(weights.columns))), units[:-1]])
    prior_close = np.vstack([op[0], cl[:-1]])
    overnight = prior_units * (op - prior_close)
    intraday = units * (cl - op)
    observed = np.diff(np.r_[CAPITAL, equity])
    reconstructed = (overnight + intraday).sum(axis=1) - costs
    scalar = np.array(
        [
            math.fsum(
                [
                    *(
                        float(prior_units[t, j]) * (float(op[t, j]) - float(prior_close[t, j]))
                        for j in range(len(weights.columns))
                    ),
                    *(
                        float(units[t, j]) * (float(cl[t, j]) - float(op[t, j]))
                        for j in range(len(weights.columns))
                    ),
                ]
            )
            - float(costs[t])
            for t in range(len(account))
        ]
    )
    expected_cash = np.r_[CAPITAL, cash[:-1]] - ((units - prior_units) * op).sum(axis=1) - costs
    checks = {
        "maximum_daily_NAV_identity_error_USD": float(np.max(np.abs(observed - reconstructed))),
        "maximum_daily_cash_flow_identity_error_USD": float(np.max(np.abs(cash - expected_cash))),
        "maximum_daily_cash_mark_identity_error_USD": float(
            np.max(np.abs(equity - (units * cl).sum(axis=1) - cash))
        ),
        "maximum_independent_scalar_vector_sum_difference_USD": float(
            np.max(np.abs(reconstructed - scalar))
        ),
    }
    if max(checks.values()) > MONEY_TOLERANCE:
        raise QuantError(
            "Saved full-path daily NAV, cash or independent PnL sums do not reconcile."
        )
    fields = {
        "equity": equity,
        "cash": cash,
        "cost": costs,
        "orders": orders,
        "observed_net_change": observed,
        "reconstructed_net_change": reconstructed,
    }
    for j, symbol in enumerate(weights.columns):
        fields["overnight::" + symbol] = overnight[:, j]
        fields["intraday::" + symbol] = intraday[:, j]
    daily = pd.DataFrame(fields, index=account.index)
    peak, peak_position, largest, episode = CAPITAL, -1, 0.0, None
    for i, value in enumerate(equity):
        if value >= peak:
            peak, peak_position = float(value), i
        drawdown = 1 - float(value) / peak
        if drawdown > largest:
            largest, episode = drawdown, (peak_position, i)
    return daily, {
        **checks,
        "full_original_return_sessions": len(account),
        "initial_peak_USD": CAPITAL,
        "maximum_observed_drawdown": largest,
        "episode_status": "observed_drawdown" if episode is not None else "no_observed_drawdown",
        "own_maximum_drawdown_episode": interval(daily, *episode) if episode is not None else None,
        "source_units_are_broker_shares": False,
        "new_account_or_fee_free_counterfactual_run": False,
    }
