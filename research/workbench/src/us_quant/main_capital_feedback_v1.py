from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import main_capital_cost_estimate_v1 as costs
from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/main-capital-feedback-v1.json"
ID = "expert_pool_eight_close_mark_capital_feedback_v1"
STUDY = "ssd_main_capital_feedback_20261104_v1"
PRIOR_HEAD = "36c07366bb5f3e77494f7ba85e99d65c58070949950fdbda3e0951f648990513"


@dataclass(frozen=True)
class CapitalSnapshot:
    as_of: pd.Timestamp
    asset_values: pd.Series
    cash: float
    initial_capital: bool


@dataclass(frozen=True)
class AllocationDecision:
    target: pd.Series
    assessment: dict


class DecisionPolicy(Protocol):
    def decide(self, capital: CapitalSnapshot) -> AllocationDecision: ...


class CloseMarkedPairPolicy:
    def __init__(
        self,
        experts: tuple[StrategyExpert, ...],
        universe: tuple[str, ...],
        risk_free: pd.Series,
    ):
        if len(experts) != 8 or not universe or len(set(universe)) != len(universe):
            raise QuantError("Use the fixed eight complete experts and unique asset coordinates.")
        self._experts = experts
        self._universe = universe
        self._rf = risk_free.copy()
        self._previous = np.full(8, 1 / 8)
        self._last = None

    def decide(self, capital: CapitalSnapshot) -> AllocationDecision:
        day = capital.as_of
        if (
            self._last is not None
            and (day <= self._last or capital.initial_capital)
            or tuple(capital.asset_values.index) != self._universe
        ):
            raise QuantError(
                "A fresh main account needs its own ordered allocation learning state."
            )
        observed = [expert.observe(day, 126) for expert in self._experts]
        if (
            tuple(o.identity.identifier for o in observed) != pool.EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != pool.MECHANISMS
            or any(
                o.target is None
                or tuple(o.target.index) != self._universe
                or not o.risk_free.index.isin(self._rf.index).all()
                or not o.risk_free.equals(self._rf.loc[o.risk_free.index])
                for o in observed
            )
        ):
            raise QuantError(
                "A missing source, changed native RF or pool identity cannot be repaired."
            )
        estimator = costs.CloseMarkedCashCost(
            capital.asset_values, capital.cash, initial_capital=capital.initial_capital
        )
        weights, assessment = costs.choose_pair_with_cost(observed, self._previous, estimator)
        target = pool.combine(observed, weights)
        details = {
            "expert_ids": [o.identity.identifier for o in observed],
            "expert_mechanism_groups": [o.identity.mechanism_group for o in observed],
            "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "expert_shadow_observations": [len(o.net_returns) for o in observed],
            "expert_shadow_last_session": [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ],
            "actual_source_introductions": [o.identity.introduced_at for o in observed],
            "previous_statistical_expert_weights": self._previous.tolist(),
            "expert_weights": weights.tolist(),
            "selection": assessment,
            "cost_in_score_is_an_actual_main_account_debit": False,
            "future_execution_price_or_cost_observed": False,
            "counterfactual_historical_expert_availability": True,
            "independent_forward_validation": False,
        }
        self._previous = weights.copy()
        self._last = day
        return AllocationDecision(target.copy(), details)


class ArchivedOpinionPolicy:
    def __init__(self, targets: pd.DataFrame):
        if (
            not isinstance(targets.index, pd.DatetimeIndex)
            or targets.empty
            or targets.index.tz is not None
            or not targets.index.is_unique
            or not targets.index.is_monotonic_increasing
            or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
            or not targets.columns.is_unique
            or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
        ):
            raise QuantError("An archived opinion needs its complete unchanged trading calendar.")
        active = targets.dropna(how="all")
        if (
            active.empty
            or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
            or any(not is_month_end(day) for day in active.index[1:])
            or not active.index[1:].equals(
                pd.DatetimeIndex(
                    [day for day in targets.index if day > active.index[0] and is_month_end(day)]
                )
            )
        ):
            raise QuantError("A monthly opinion cannot fill a missing post-start recommendation.")
        self._active = active.copy()

    def decide(self, capital: CapitalSnapshot) -> AllocationDecision:
        known = self._active.loc[: capital.as_of]
        if known.empty:
            raise QuantError("A future opinion cannot seed earlier capital.")
        target = known.iloc[-1].copy()
        if not target.index.equals(capital.asset_values.index):
            raise QuantError("An archived opinion cannot change the actual asset lattice.")
        return AllocationDecision(target, {"source_opinion_known_at": str(known.index[-1].date())})


@dataclass(frozen=True)
class CapitalRun:
    result: BacktestResult
    desired: pd.DataFrame
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict


def run(
    data: MarketData,
    policy: DecisionPolicy,
    start: str,
    end: str,
    *,
    cost_bps: float,
    delay: int,
    before_decision: Callable[[], None] | None = None,
    retain: Callable[[dict], None] | None = None,
) -> CapitalRun:
    data.validate()
    dates = data.close.loc[start:end].index
    if (
        len(dates) < 2
        or dates[0] != pd.Timestamp(start)
        or dates[-1] != pd.Timestamp(end)
        or not dates.equals(sessions(dates[0], dates[-1]))
        or type(delay) is not int
        or delay not in (1, 2)
        or not np.isfinite(cost_bps)
        or cost_bps not in (5.0, 20.0)
    ):
        raise QuantError("Keep complete formal dates and the original base/stress execution terms.")
    if (cost_bps, delay) not in ((5.0, 1), (20.0, 2)):
        raise QuantError("Do not splice the original execution scenarios.")
    anchor = previous_session(dates[0])
    if anchor not in data.close.index:
        raise QuantError("The real pre-return initialization anchor must remain covered.")
    desired = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    issued = desired.copy()
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = 10000.0
    pending, decisions, rows, weights = {}, [], [], []

    def record(event: dict) -> None:
        if retain is not None:
            retain(event)

    def decide(day: pd.Timestamp, values: np.ndarray, initial: bool = False) -> None:
        if before_decision is not None:
            before_decision()
        snapshot = CapitalSnapshot(
            day, pd.Series(values.copy(), index=data.close.columns), float(cash), initial
        )
        decision = policy.decide(snapshot)
        target = decision.target
        if (
            not isinstance(target, pd.Series)
            or not target.index.equals(data.close.columns)
            or not np.isfinite(target.to_numpy()).all()
            or (target < 0).any()
            or abs(float(target.sum()) - 0.98) > 1e-12
        ):
            raise QuantError(
                "A dynamic decision must deliver one complete 98% funded asset target."
            )
        nav = float(values.sum() + cash)
        gap = float(abs(values / nav - target.to_numpy()).max())
        reason = (
            "initial_capital"
            if initial
            else ("observed_weight_gap" if gap + 1e-12 >= 0.025 else "hold")
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        desired.loc[day] = target
        if reason != "hold":
            if execution in pending:
                raise QuantError("A dynamic policy cannot overwrite a pending modeled request.")
            pending[execution] = target.to_numpy(copy=True)
            issued.loc[day] = target
        row = {
            "signal_session": str(day.date()),
            "execution_session": str(execution.date()),
            "initial_capital": initial,
            "reason": reason,
            "maximum_observed_target_gap": gap,
            "main_asset_values_at_decision": values.tolist(),
            "main_cash_at_decision": float(cash),
            "main_equity_at_decision": nav,
            "desired_target": target.tolist(),
            "assessment": decision.assessment,
        }
        decisions.append(row)
        record({"kind": "decision", **row})

    decide(anchor, np.zeros(len(units)), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        cost = turnover = 0.0
        tickets = 0
        if day in pending:
            target = pending.pop(day)
            old_values = units * opening
            before_nav = float(old_values.sum() + cash)
            record(
                {
                    "kind": "modeled_execution_input",
                    "session": str(day.date()),
                    "opening_asset_values": old_values.tolist(),
                    "cash": float(cash),
                    "target": target.tolist(),
                }
            )
            values, new_cash, cost, turnover, tickets = rebalance(
                old_values, cash, target, cost_bps, 1.0
            )
            post_nav = float(values.sum() + new_cash)
            if (
                not np.isfinite([post_nav, new_cash, cost]).all()
                or post_nav <= 0
                or new_cash < 0
                or not np.isclose(post_nav, before_nav - cost, rtol=1e-12, atol=1e-10)
                or (turnover > 0 or tickets > 0)
                and not np.allclose(values / post_nav, target, rtol=0, atol=1e-12)
            ):
                raise QuantError(
                    "The dynamic main account cannot fund its complete delayed target."
                )
            cash = new_cash
            if turnover > 0 or tickets > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if (
            not np.isfinite(nav)
            or nav <= 0
            or cash < 0
            or (units < 0).any()
            or float(values.sum() / nav) > 1 + 1e-12
        ):
            raise QuantError("The dynamic main book became insolvent, short or borrowed cash.")
        rows.append(
            (nav, nav / previous_nav - 1, cash, values.sum() / nav, turnover, cost, tickets)
        )
        weights.append(values / nav)
        record(
            {
                "kind": "main_close",
                "session": str(day.date()),
                "equity": nav,
                "return": nav / previous_nav - 1,
                "cash": float(cash),
                "asset_values": values.tolist(),
                "cost": float(cost),
                "turnover": float(turnover),
                "model_tickets": int(tickets),
                "risk_free": float(data.risk_free.loc[day]),
            }
        )
        if is_month_end(day):
            decide(day, values)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return CapitalRun(
        BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns)),
        desired,
        issued,
        decisions,
        {str(day.date()): target.tolist() for day, target in pending.items()},
    )


def verify_capital_inputs(run_result: CapitalRun) -> dict:
    frame, held = run_result.result.frame, run_result.result.weights
    for row in run_result.decisions:
        day = pd.Timestamp(row["signal_session"])
        if row["initial_capital"]:
            expected_values, expected_cash, expected_nav = np.zeros(len(held.columns)), 10000, 10000
        else:
            expected_nav = float(frame.loc[day, "equity"])
            expected_cash = float(frame.loc[day, "cash"])
            expected_values = held.loc[day].to_numpy() * expected_nav
        if (
            not np.allclose(
                row["main_asset_values_at_decision"], expected_values, rtol=1e-12, atol=1e-9
            )
            or row["main_cash_at_decision"] != expected_cash
            or not np.isclose(row["main_equity_at_decision"], expected_nav, rtol=0, atol=1e-9)
        ):
            raise QuantError("A decision did not use the actual main book then known.")
    return {
        "decisions_checked": len(run_result.decisions),
        "main_dollar_holdings_cash_and_equity_match_actual_close": True,
        "expert_shadow_wealth_used_as_main_capital": False,
    }
