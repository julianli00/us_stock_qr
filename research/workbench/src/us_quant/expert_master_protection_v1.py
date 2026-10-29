from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.semiconductor_portfolio_protection import protected_target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-master-protection-v1.json"
PRIOR_HEAD = "c504d803b5d630f9ccb994fc0a613a213acd123a2d888d042f88743f2d37b613"
CANDIDATES = (
    {
        "id": "expert_pool_four_equal_master_protection_v1",
        "source_candidate_id": "expert_pool_four_equal_weight_v1",
    },
    {
        "id": "expert_pool_four_rolling_master_protection_v1",
        "source_candidate_id": "expert_pool_four_rolling_pair_v1",
    },
)
NOMINAL_HASHES = (
    "5635ac90a27746d3fa5d58d5af54aeb3ab70638ca728277174814dcc88ed2ec2",
    "80f62c3b27436d1767655f85714e50308592e8e2ec3189a2f3b1d7d2fd153515",
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_expert_master_protection_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 211,
        "prior_new_registered_candidates": 36,
        "prior_new_completed_strategy_evaluations": 33,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 397,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_registration": {
            "path": "evidence/four_expert_meta_20261029_registration_v1.json",
            "sha256": "4e84ee5908ab8046f615dc0f6918aeef73aa9207c95bf58a38876c01c270fcfe",
        },
        "source_results": {
            "path": "evidence/four_expert_meta_20261029_results_v1.json",
            "sha256": "ee877c74e586fa56bd6ed66e79c654ec56980d445ea47af5f3b59ac86ebdcabe",
        },
        "motivation_saved_diagnosis": {
            "path": (
                "reports/ssd-continuous-research-v1/four-expert-meta-20261029-v1/"
                "saved-feedback-selection-risk-v1.json"
            ),
            "sha256": "493ea6a5b974372247ac8f02e4c9f58c1a456b828526f2ef1f811383e4a5b5de",
        },
        "master_initial_capital_usd": 10000.0,
        "ratchet_floor_fraction": 0.85,
        "drawdown_target": 0.15,
        "actual_weight_trade_band": 0.025,
        "cash_reserve": 0.02,
        "invested_target_budget": 0.98,
        "commission_per_order": 1.0,
        "feedback_clock": "each_actual_main_account_close",
        "nominal_clock": "unchanged_initial_seed_then_completed_exchange_month_ends",
        "pending_request_rule": "one_pending_request_never_overwritten",
        "monthly_update_forces_order_without_band": False,
        "final_issued_targets_precomputed_before_accounts": False,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "minimum_economic_factor_families": 3,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        key not in policy or type(policy[key]) is not type(value) or policy[key] != value
        for key, value in expected.items()
    ):
        raise QuantError("Keep the fixed main-wealth rule and original funding contract.")
    if not isinstance(policy.get("methodology"), dict):
        raise QuantError("The fixed protection methodology must be explicit.")
    if policy.get("nominal_sources") != {
        candidate["source_candidate_id"]: {
            "path": (
                "data/ssd-four-expert-meta-20261029-v1/"
                f"{candidate['source_candidate_id']}/targets.csv"
            ),
            "sha256": digest,
        }
        for candidate, digest in zip(CANDIDATES, NOMINAL_HASHES, strict=True)
    } or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "source_wealth_guarantee_transferred",
            "main_drawdown_guarantee",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Use exact old nominal intentions, not promised or precomputed wealth.")


def inputs(data, baseline, start, end):
    if not isinstance(data, MarketData) or not isinstance(baseline, pd.DataFrame):
        raise QuantError("Require admitted market panels and a complete nominal target frame.")
    data.validate()
    if (
        not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or not baseline.columns.is_unique
        or "BIL" not in baseline
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()
    ):
        raise QuantError("The nominal target calendar, assets or missingness changed.")
    active = baseline.dropna(how="all")
    if (
        active.empty
        or any(not is_month_end(day) for day in active.index[1:])
        or not np.isfinite(active).all().all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Keep one real initial seed and the unchanged complete monthly targets.")
    dates = data.close.loc[start:end].index
    if (
        len(dates) < 2
        or dates[0] != pd.Timestamp(start)
        or dates[-1] != pd.Timestamp(end)
        or active.index[0] > previous_session(dates[0])
    ):
        raise QuantError("A full formal path needs a genuinely known prewindow nominal intent.")
    return dates, active


def run(data, baseline, candidate, policy, start, end, cost, delay):
    validate(policy)
    if (
        candidate not in CANDIDATES
        or type(delay) is not int
        or (cost, delay)
        not in (
            (5.0, 1),
            (20.0, 2),
        )
    ):
        raise QuantError("Use only the two registered rules and original execution scenarios.")
    dates, active = inputs(data, baseline, start, end)
    anchor = previous_session(dates[0])
    source_day = active.loc[:anchor].index[-1]
    nominal = active.loc[source_day].copy()
    issued = pd.DataFrame(np.nan, index=baseline.index, columns=baseline.columns)
    units = np.zeros(len(baseline.columns))
    cash = previous_nav = peak = 10000.0
    pending, decisions, rows, weights = {}, [], [], []

    def decide(day, nav, current, initial=False):
        desired = protected_target(nominal, nav, peak)
        gap = float(abs(current - desired).max())
        requested = initial or (not pending and gap + 1e-12 >= 0.025)
        reason = (
            "initial_capital"
            if initial
            else "pending_request"
            if pending
            else "observed_weight_gap"
            if requested
            else "hold"
        )
        execution = None
        if requested:
            if pending:
                raise QuantError("Never replace an already issued delayed request.")
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending[execution] = desired.to_numpy().copy()
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "source_nominal_session": str(source_day.date()),
                "initial_capital": initial,
                "known_nav_usd": nav,
                "known_peak_usd": peak,
                "ratchet_floor_usd": 0.85 * peak,
                "scale": float(np.clip((nav - 0.85 * peak) / (0.15 * nav), 0, 1)),
                "source_bil_weight": float(nominal["BIL"]),
                "desired_bil_weight": float(desired["BIL"]),
                "maximum_observed_target_gap": gap,
                "request_issued": requested,
                "reason": reason,
                "execution_session": str(execution.date()) if execution is not None else None,
                "pending_executions": [str(d.date()) for d in pending],
            }
        )

    decide(anchor, 10000.0, pd.Series(0.0, index=baseline.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, costs, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, 1.0
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < 0:
            raise QuantError("The actual main account became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=baseline.columns)
        rows.append(
            (nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders)
        )
        weights.append(current.to_numpy())
        peak = max(peak, nav)
        if day in active.index:
            source_day, nominal = day, active.loc[day].copy()
        decide(day, nav, current)
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


def scalar_target(nominal, nav, peak):
    if not np.isfinite([nav, peak]).all() or nav <= 0 or peak < nav:
        raise QuantError("The independent transition needs actual positive prefix wealth.")
    scale = max(0.0, min(1.0, (nav - 0.85 * peak) / (0.15 * nav)))
    target = nominal.copy()
    removed = 0.0
    for symbol in nominal.index:
        if symbol != "BIL":
            target[symbol] = float(nominal[symbol]) * scale
            removed += float(nominal[symbol]) - float(target[symbol])
    target["BIL"] = float(nominal["BIL"]) + removed
    return target, scale


def verify_feedback(data, baseline, result, issued, decisions, start, end, delay):
    if type(delay) is not int or delay not in (1, 2):
        raise QuantError("The independent verifier needs an original execution delay.")
    dates, active = inputs(data, baseline, start, end)
    anchor = previous_session(dates[0])
    if (
        not result.frame.index.equals(dates)
        or not result.weights.index.equals(dates)
        or not result.weights.columns.equals(baseline.columns)
        or not issued.index.equals(baseline.index)
        or not issued.columns.equals(baseline.columns)
        or [r["signal_session"] for r in decisions] != [str(day.date()) for day in (anchor, *dates)]
        or not result.frame["risk_free"].equals(data.risk_free.loc[dates])
        or not np.isfinite(result.frame).all().all()
        or not np.isfinite(result.weights).all().all()
        or (result.weights < 0).any().any()
        or (result.frame["cash"] < 0).any()
        or (result.frame["gross_exposure"] > 1 + 1e-12).any()
        or abs(
            float(result.frame.iloc[0]["equity"]) / (1 + float(result.frame.iloc[0]["return"]))
            - 10000.0
        )
        > 1e-8
    ):
        raise QuantError("Preserve every actual main-state opportunity and fresh funded calendar.")
    pending, expected_issued = {}, []
    peak = 10000.0
    maximum_target_error = maximum_state_error = 0.0
    for position, row in enumerate(decisions):
        day = pd.Timestamp(row["signal_session"])
        initial = position == 0
        if not initial:
            pending.pop(day, None)
        known = active.loc[:day]
        source_day, nominal = known.index[-1], known.iloc[-1]
        nav = 10000.0 if initial else float(result.frame.loc[day, "equity"])
        peak = max(peak, nav)
        actual = pd.Series(0.0, index=baseline.columns) if initial else result.weights.loc[day]
        desired, scale = scalar_target(nominal, nav, peak)
        gap = float(abs(actual - desired).max())
        requested = initial or (not pending and gap + 1e-12 >= 0.025)
        reason = (
            "initial_capital"
            if initial
            else "pending_request"
            if pending
            else "observed_weight_gap"
            if requested
            else "hold"
        )
        execution = None
        if requested:
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending[execution] = True
            expected_issued.append(day)
            error = float(abs(issued.loc[day] - desired).max())
            if issued.loc[day].isna().any() or error > 1e-12:
                raise QuantError("An issued target differs from the independent scalar transition.")
            maximum_target_error = max(maximum_target_error, error)
        elif not issued.loc[day].isna().all():
            raise QuantError("A hold or pending request cannot acquire a new issued target.")
        fields = {
            "known_nav_usd": nav,
            "known_peak_usd": peak,
            "ratchet_floor_usd": 0.85 * peak,
            "scale": scale,
            "source_bil_weight": float(nominal["BIL"]),
            "desired_bil_weight": float(desired["BIL"]),
            "maximum_observed_target_gap": gap,
        }
        errors = [abs(float(row[k]) - value) for k, value in fields.items()]
        if (
            not np.isfinite(errors).all()
            or max(errors) > 1e-9
            or row["source_nominal_session"] != str(source_day.date())
            or row["initial_capital"] is not initial
            or row["request_issued"] is not requested
            or row["reason"] != reason
            or row["execution_session"]
            != (str(execution.date()) if execution is not None else None)
            or row["pending_executions"] != [str(d.date()) for d in pending]
        ):
            raise QuantError("The recorded feedback, source clock, band or pending state changed.")
        maximum_state_error = max(maximum_state_error, *errors)
    if not issued.dropna(how="all").index.equals(pd.DatetimeIndex(expected_issued)):
        raise QuantError("Every issued request must retain its actual unique signal date.")
    return {
        "checked_decision_states": len(decisions),
        "checked_issued_requests": len(expected_issued),
        "independent_scalar_transition_and_actual_prefix_wealth_checked": True,
        "single_pending_request_and_original_session_delays_checked": True,
        "maximum_issued_target_difference": maximum_target_error,
        "maximum_recorded_state_difference": maximum_state_error,
        "initial_capital_and_peak_are_fresh_main_10000": True,
        "source_shadow_wealth_or_peak_imported": False,
        "main_drawdown_guarantee": False,
    }
