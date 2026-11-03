from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant import observed_weight_execution as monthly
from us_quant.backtest import BacktestResult
from us_quant.calendar import next_session, previous_session
from us_quant.cash_funded_accounting_v2 import rebalance, simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import file_digest, read_json

POLICY = monthly.ROOT / "config/expert-native-event-execution-v1.json"
BAND_POLICY_SHA = "820cc4e9c2f4be6b286873254f663fba4f018156fb427156e1fc330da53094a7"


def validate(policy):
    expected = {
        "schema_version": 1,
        "execution_id": "ssd_expert_native_opinion_event_band_20261103_v1",
        "source_band_policy": {
            "path": "config/observed-weight-execution.json",
            "sha256": BAND_POLICY_SHA,
        },
        "accounting_engine": "cash_funded_noop_v2",
        "capital_usd": 10000,
        "commission_per_order": 1.0,
        "actual_weight_trade_band": 0.025,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "allowed_cost_delay_pairs": [[5.0, 1], [20.0, 2]],
        "maximum_pending_main_request_batches": 1,
        "source_opinion_refresh_is_automatic_execution": False,
        "pending_request_replaced_or_cancelled": False,
        "unfilled_terminal_requests_retroactively_filled": False,
        "source_positions_or_feedback_floor_copied": False,
        "terminal_liquidation": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze the original funded band, costs, delays and one-request queue.")
    if file_digest(monthly.POLICY) != BAND_POLICY_SHA:
        raise QuantError("The original observed-holding band policy changed.")
    source = read_json(monthly.POLICY)
    monthly.validate(source)
    return monthly.pure_band(source)


def run(data, baseline, policy, start, end, cost, delay):
    band, original_policy = validate(policy)
    data.validate()
    if (
        (cost, delay) not in ((5.0, 1), (20.0, 2))
        or type(delay) is not int
        or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()
    ):
        raise QuantError("Use the complete native target lattice and original base/stress timing.")
    active = baseline.dropna(how="all")
    if (
        active.empty
        or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Native meta targets must remain complete, long-only and cash-funded.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("Keep every original formal return session.")
    anchor = previous_session(dates[0])
    if active.index[0] > anchor:
        raise QuantError("A future opinion cannot fund an earlier main account.")
    known = seed_window(baseline, str(dates[0].date()))
    issued = pd.DataFrame(np.nan, index=baseline.index, columns=baseline.columns)
    units = np.zeros(len(baseline.columns))
    cash = previous_nav = float(policy["capital_usd"])
    pending = None
    rows, weights, decisions = [], [], []

    def decide(day, current, initial=False):
        nonlocal pending
        desired = known.loc[day]
        prior_pending = str(pending[0].date()) if pending is not None else None
        if pending is not None:
            reason = "pending_request_preserved"
        elif initial:
            reason = "initial_capital"
        else:
            covariance = monthly.prior.observed_covariance(
                data, day, original_policy["covariance_sessions"]
            )
            reason = monthly.prior.rebalance_reason(current, desired, covariance, band)
        execution = None
        if reason not in ("hold", "pending_request_preserved"):
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending = (execution, desired.to_numpy().copy())
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()) if execution is not None else None,
                "initial_capital": initial,
                "reason": reason,
                "maximum_observed_target_gap": float(abs(current - desired).max()),
                "pending_execution_before_decision": prior_pending,
            }
        )

    decide(anchor, pd.Series(0.0, index=baseline.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if pending is not None:
            execution, desired = pending
            if execution < day:
                raise QuantError(
                    "A dated pending request was skipped; preserve the partial account."
                )
            if execution == day:
                values, cash, costs, turnover, orders = rebalance(
                    units * opening, cash, desired, cost, policy["commission_per_order"]
                )
                if turnover > 0 or orders > 0:
                    units = values / opening
                pending = None
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The native-event account became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=baseline.columns)
        rows.append(
            (nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders)
        )
        weights.append(current.to_numpy())
        if not known.loc[day].isna().all():
            decide(day, current)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return (
        BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=baseline.columns)),
        issued,
        decisions,
    )


def verify_feedback(data, targets, result, issued, decisions, policy, start, end, cost, delay):
    validate(policy)
    if (cost, delay) not in ((5.0, 1), (20.0, 2)) or type(delay) is not int:
        raise QuantError("Verify only the original matched-cost base/stress delay conventions.")
    dates = data.close.loc[start:end].index
    anchor = previous_session(dates[0])
    known = seed_window(targets, str(dates[0].date()))
    expected_dates = pd.DatetimeIndex([anchor, *known.loc[dates].dropna(how="all").index])
    if (
        not result.frame.index.equals(dates)
        or not result.weights.index.equals(dates)
        or not result.weights.columns.equals(targets.columns)
        or not issued.index.equals(targets.index)
        or not issued.columns.equals(targets.columns)
        or not result.frame["risk_free"].equals(data.risk_free.loc[dates])
        or [row["signal_session"] for row in decisions]
        != expected_dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Feedback proof needs every true decision and original return clock.")
    pending = None
    expected_issued = pd.DataFrame(np.nan, index=targets.index, columns=targets.columns)
    maximum_gap_error = 0.0
    pending_holds = 0
    for day, decision in zip(expected_dates, decisions, strict=True):
        if pending is not None and pending <= day:
            pending = None
        current = (
            pd.Series(0.0, index=targets.columns) if day == anchor else result.weights.loc[day]
        )
        desired = known.loc[day]
        gap = max(abs(float(a) - float(b)) for a, b in zip(current, desired, strict=True))
        old_pending = str(pending.date()) if pending is not None else None
        reason = (
            "pending_request_preserved"
            if pending is not None
            else (
                "initial_capital"
                if day == anchor
                else ("observed_weight_gap" if gap + 1e-12 >= 0.025 else "hold")
            )
        )
        execution = None
        if reason not in ("hold", "pending_request_preserved"):
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            expected_issued.loc[day] = desired
            pending = execution
        elif reason == "pending_request_preserved":
            pending_holds += 1
        error = abs(gap - decision["maximum_observed_target_gap"])
        maximum_gap_error = max(maximum_gap_error, error)
        if (
            decision["reason"] != reason
            or decision["initial_capital"] is not (day == anchor)
            or decision["execution_session"]
            != (str(execution.date()) if execution is not None else None)
            or decision["pending_execution_before_decision"] != old_pending
            or not np.isfinite(error)
            or error > 1e-12
        ):
            raise QuantError(
                "Independent native-event holding-band or preserved-queue proof failed."
            )
    if not expected_issued.equals(issued):
        raise QuantError("The issued matrix differs from the independent native-event replay.")
    if (
        not np.isfinite(result.frame.to_numpy()).all()
        or (result.frame["cash"] < 0).any()
        or (result.frame["gross_exposure"] > 1).any()
    ):
        raise QuantError("A feedback proof cannot waive finite cash-funded accounting.")
    replay = simulate(
        data,
        issued,
        start,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    if not replay.frame.equals(result.frame) or not replay.weights.equals(result.weights):
        raise QuantError("Every native account cash, fee, return and position must reconcile.")
    return {
        "decision_count": len(decisions),
        "pending_refreshes_without_replacement": pending_holds,
        "maximum_observed_gap_error": maximum_gap_error,
        "maximum_pending_main_request_batches": 1,
        "original_band_costs_delays_cash_and_full_return_clock_preserved": True,
        "complete_frame_and_positions_reconciled_to_cash_engine": True,
        "source_feedback_or_held_positions_copied": False,
    }
