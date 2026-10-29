from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant import observed_weight_execution as monthly
from us_quant import ssd_variance_disagreement_research_v1 as original
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import runtime_guard
from us_quant.dual_horizon import seed_window
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, ROOT = original.base, original.ROOT


def run(data, baseline, candidate, policy, start, end, cost, delay):
    monthly.validate(policy)
    data.validate()
    if (
        candidate not in monthly.CANDIDATES
        or type(delay) is not int
        or delay < 1
        or not np.isfinite(cost)
        or not 0 <= cost < 100
        or not isinstance(baseline, pd.DataFrame)
        or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()
    ):
        raise QuantError(
            "Initial-seed execution needs the complete funded target and original policy."
        )
    active = baseline.dropna(how="all")
    if (
        active.empty
        or any(not is_month_end(day) for day in active.index[1:])
        or not np.isfinite(active).all().all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError(
            "Only one real initial seed may precede the original monthly decision lattice."
        )
    band, original_policy = monthly.pure_band(policy)
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("Keep the full original formal account calendar.")
    anchor = previous_session(dates[0])
    if active.index[0] > anchor:
        raise QuantError("A future recommendation cannot initialize earlier main capital.")
    known = seed_window(baseline, str(dates[0].date()))
    issued = pd.DataFrame(np.nan, index=baseline.index, columns=baseline.columns)
    pending, decisions = {}, []
    rows, weights = [], []
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = float(policy["capital_usd"])

    def decide(day, current, initial=False):
        desired = known.loc[day]
        covariance = monthly.prior.observed_covariance(
            data, day, original_policy["covariance_sessions"]
        )
        reason = (
            "initial_capital"
            if initial
            else monthly.prior.rebalance_reason(current, desired, covariance, band)
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if reason != "hold":
            if execution in pending:
                raise QuantError("Never overwrite an original delayed request.")
            pending[execution] = desired.to_numpy().copy()
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()),
                "initial_capital": initial,
                "reason": reason,
                "maximum_observed_target_gap": float(abs(current - desired).max()),
            }
        )

    decide(anchor, pd.Series(0.0, index=baseline.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, costs, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, policy["commission_per_order"]
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The initialized monthly account became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=baseline.columns)
        rows.append(
            (nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders)
        )
        weights.append(current.to_numpy())
        if not known.loc[day].isna().all():
            if not is_month_end(day):
                raise QuantError(
                    "After initialization, only original month ends may inspect holdings."
                )
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


def evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    events = base.journal_events(base.JOURNAL)
    completed = [
        e
        for e in events
        if e["kind"] == "path_completed"
        and e["candidate_id"] == identifier
        and e["body"]["path_key"] == key
    ]
    if completed:
        if len(completed) != 1:
            raise QuantError("Multiple completed paths cannot be selected after outcomes.")
        return base.verify_path(completed[0]["body"]["receipt"], frozen, identifier, key)
    starts = [
        e
        for e in events
        if e["kind"] == "path_started"
        and e["candidate_id"] == identifier
        and e["body"]["path_key"] == key
    ]
    if starts:
        previous = starts[-1]
        receipt_path = ROOT / previous["body"]["directory"] / "complete.json"
        if receipt_path.exists():
            recovered = base.verify_path(base.reference(receipt_path), frozen, identifier, key)
            base.append_event(
                base.JOURNAL,
                "path_completed",
                identifier,
                {
                    "path_key": key,
                    "receipt": base.reference(receipt_path),
                    "recovered_actual_complete_receipt": True,
                },
            )
            return recovered
        if not any(
            e["kind"] == "path_interrupted"
            and e["body"].get("started_event_sha256") == previous["sha256"]
            for e in events
        ):
            base.append_event(
                base.JOURNAL,
                "path_interrupted",
                identifier,
                {
                    "path_key": key,
                    "started_event_sha256": previous["sha256"],
                    "partial_output_preserved": True,
                    "reason": "Exclusive lock reacquired without a complete path receipt.",
                },
            )
    runtime_guard()
    directory = base.ACCOUNTS / identifier / f"{key}-attempt-{len(starts) + 1:03d}"
    base.append_event(
        base.JOURNAL,
        "path_started",
        identifier,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    base.storage_guard()
    new_output_directory(directory)
    execution_policy = read_json(base.checked_reference(frozen["execution_policy"]))
    result, issued, decisions = run(
        data,
        targets,
        monthly.CANDIDATES[0],
        execution_policy,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback = original.verify_feedback(data, targets, result, issued, decisions, window, scenario)
    independent = base.independent_equity(
        data,
        issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(result.frame["return"], result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    criteria = base.gates(
        own,
        lookup[window["years"], scenario["id"], "SPY"],
        lookup[window["years"], scenario["id"], "QQQ"],
        lookup[window["years"], "base", "SPY"],
        scenario["id"],
    )
    execution = base.execution_statistics(result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": result.frame,
        "weights.csv": result.weights,
        "issued.csv": issued,
        "independent-bt.csv": independent,
    }
    base.storage_guard()
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": identifier,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": execution,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"]
        - lookup[window["years"], scenario["id"], "SPY"]["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"]
        - lookup[window["years"], scenario["id"], "QQQ"]["cagr"],
        "stress_vs_base_cost_qqq_diagnostic": own["cagr"]
        - lookup[window["years"], "base", "QQQ"]["cagr"]
        if scenario["id"] == "stress"
        else None,
        "independent_bt_max_equity_difference_usd": difference,
        "independent_monthly_feedback": feedback,
        "explicit_initial_seed_preserves_original_formal_clock": True,
        "artifacts": [base.reference(directory / name) for name in (*frames, "decisions.json")],
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        identifier,
        {
            "path_key": key,
            "receipt": base.reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt
