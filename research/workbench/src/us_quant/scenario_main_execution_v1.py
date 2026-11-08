from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance, simulate
from us_quant.data import MarketData
from us_quant.main_risk_scenario_control_v1 import ASSETS, Controller, require, target


@dataclass(frozen=True)
class Run:
    result: BacktestResult
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict
    decision_count: int


def run(
    data: MarketData,
    controller: Controller,
    start: str,
    end: str,
    *,
    cost_bps: float,
    delay: int,
    retain: Callable[[dict], None] | None = None,
    before: Callable[[], None] | None = None,
    keep_decisions: bool = True,
) -> Run:
    data.validate()
    dates = data.close.loc[start:end].index
    require(
        tuple(data.close.columns) == ASSETS
        and len(dates) >= 2
        and dates[0] == pd.Timestamp(start)
        and dates[-1] == pd.Timestamp(end)
        and dates.equals(sessions(dates[0], dates[-1]))
        and (cost_bps, delay) in ((5.0, 1), (20.0, 2))
        and (controller.cost_bps, controller.delay) == (cost_bps, delay),
        "Keep complete original dates, assets, fees and one/two-session opening execution.",
    )
    require(
        type(keep_decisions) is bool and (keep_decisions or retain is not None),
        "Large decision records must be retained in memory or streamed, never discarded.",
    )
    anchor = previous_session(dates[0])
    require(anchor in data.close.index, "The genuine fresh-capital anchor is missing.")
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    units = np.zeros(len(ASSETS))
    cash = previous_nav = 10000.0
    pending, decisions, rows, weights = {}, [], [], []

    def record(value):
        if retain is not None:
            retain(value)

    def review(day, values, initial=False):
        if before is not None:
            before()
        remaining = None
        pending_target = None
        if pending:
            require(len(pending) == 1, "Never cancel or overwrite a modeled delayed request.")
            pending_day, pending_target = next(iter(pending.items()))
            current = day
            remaining = 0
            while current < pending_day:
                current = next_session(current)
                remaining += 1
            require(current == pending_day and remaining <= delay, "Invalid pending chronology.")
        record(
            {
                "kind": "main_decision_input",
                "day": str(day.date()),
                "asset_values": values.tolist(),
                "cash": cash,
                "initial": initial,
                "pending_target": None if pending_target is None else pending_target.tolist(),
                "pending_delay": remaining,
            }
        )
        decision = controller.decide(day, values, cash, initial, pending_target, remaining)
        execution = None
        if decision.target is not None:
            require(not pending, "A new decision cannot replace a precommitted request.")
            desired = target(decision.target)
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending[execution] = desired
            issued.loc[day] = desired
        row = {
            "signal_session": str(day.date()),
            "execution_session": None if execution is None else str(execution.date()),
            "initial_capital": initial,
            "main_NAV": float(values.sum() + cash),
            "main_peak": controller.peak,
            "target": None if decision.target is None else decision.target.tolist(),
            "assessment": decision.assessment,
        }
        if keep_decisions:
            decisions.append(row)
        record({"kind": "main_decision_resolved", **row})

    review(anchor, np.zeros(len(ASSETS)), initial=True)
    for day in dates:
        if before is not None:
            before()
        opening = data.open.loc[day].to_numpy()
        closing = data.close.loc[day].to_numpy()
        cost = turnover = 0.0
        orders = 0
        if day in pending:
            desired = pending.pop(day)
            old_values = units * opening
            record(
                {
                    "kind": "modeled_execution_input",
                    "day": str(day.date()),
                    "asset_values": old_values.tolist(),
                    "cash": cash,
                    "target": desired.tolist(),
                }
            )
            values, cash, cost, turnover, orders = rebalance(
                old_values, cash, desired, cost_bps, 1.0
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        require(
            np.isfinite(nav) and nav > 0 and cash >= 0 and (units >= 0).all(),
            "The main book cannot borrow, short or silently become insolvent.",
        )
        held = values / nav
        require(float(held.sum()) <= 1 + 1e-12, "The main portfolio exceeded cash funding.")
        rows.append((nav, nav / previous_nav - 1, cash, held.sum(), turnover, cost, orders))
        weights.append(held)
        record(
            {
                "kind": "main_close",
                "day": str(day.date()),
                "equity": nav,
                "cash": cash,
                "held_weights": held.tolist(),
                "cost": cost,
                "model_tickets": orders,
            }
        )
        review(day, values)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return Run(
        BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns)),
        issued,
        decisions,
        {str(day.date()): value.tolist() for day, value in pending.items()},
        len(dates) + 1,
    )


def cash_replay(data, output, start, end, cost_bps, delay):
    return simulate(
        data,
        output.issued,
        start,
        end,
        initial_capital=10000.0,
        cost_bps=cost_bps,
        commission=1.0,
        delay=delay,
    )
