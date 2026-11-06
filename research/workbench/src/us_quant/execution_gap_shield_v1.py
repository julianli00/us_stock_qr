from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import execution_gap_envelope_v1 as reference
from us_quant.backtest import BacktestResult
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance, simulate
from us_quant.main_capital_feedback_v1 import (
    ArchivedOpinionPolicy,
    CapitalSnapshot,
    DecisionPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/execution-gap-shield-v1.json"
ID = "expert_pool_six_execution_gap_shield_v1"
STUDY = "ssd_execution_gap_shield_20261106_v1"
ASSETS, N, BIL = reference.ASSETS, reference.N, reference.BIL
require = reference.require


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 195752,
        "prior_journal_sequence": 1490,
        "prior_journal_head_sha256": (
            "ed776fb6704f4992abc671865ad2bfe3983ee41baca202b510c1a0d7ce9d31fd"
        ),
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/execution-gap-shield-20261106-v1/"
            "source-inputs-v1.json",
            "sha256": "90d13beea7c84a395f4aeeda29eadf172ce06c7acad78a45241861a7355aa070",
        },
        "nominal_policy": "unchanged_cached_six_confidence_Prod_monthly_opinion",
        "main_review_clock": "each_original_close_after_fills",
        "peak": "fresh_10000_and_all_actual_main_closing_NAVs",
        "scenario_selection": "all_expanding_fully_known_actual_delay_price_paths",
        "scenario_costs": "actual5bp_1session_or20bp_2sessions_plus14_USD1_tickets",
        "reference_peak_fraction": 0.85,
        "nominal_ETF_budget": 0.98,
        "cash_reserve": 0.02,
        "ordinary_actual_holding_band": 0.025,
        "band_override": "certified_full_fill_and_unsafe_unchanged_hold_reference_only",
        "maximum_pending_main_requests": 1,
        "pending_replaced_or_cancelled": False,
        "fraction": "greatest_binary64_inside_exact_represented_affine_interval",
        "uncertified_action": "explicit_BIL_risk_reduction_attempt_subject_to_original_band",
        "uncertified_action_is_successful_certificate": False,
        "money_coherence_USD": 1e-6,
        "target_coherence": 1e-12,
        "notional_noop_threshold_USD": 1e-6,
        "prefix_positions": [125, 503, 1006, 1509, 2012],
        "candidate_count": 1,
        "study_size_is_a_total_research_quota": False,
        "new_source_requests": 0,
        "old_learner_source_target_or_account_replayed": False,
        "old_headroom_V1_repaired_retried_or_executed": False,
        "finite_reference_is_future_or_full_month_guarantee": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Keep the separate delay-aware law, source identity and original financial criteria.",
    )


@dataclass(frozen=True)
class DelayPaths:
    as_of: pd.Timestamp
    delay: int
    starts: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    waiting: np.ndarray
    opening: np.ndarray
    fill_mark: np.ndarray

    def validate(self):
        length = len(self.ends)
        require(
            type(self.delay) is int
            and self.delay in (1, 2)
            and isinstance(self.as_of, pd.Timestamp)
            and self.as_of.tzinfo is None
            and length > 0
            and len(self.starts) == length
            and all(
                d.tz is None and d.is_unique and d.is_monotonic_increasing
                for d in (self.starts, self.ends)
            )
            and (self.starts < self.ends).all()
            and (self.ends <= self.as_of).all()
            and self.waiting.shape == (length, self.delay - 1, N)
            and self.opening.shape == self.fill_mark.shape == (length, N)
            and all(
                a.dtype.kind in "fi" and np.isfinite(a).all() and (a > 0).all()
                for a in (self.waiting, self.opening, self.fill_mark)
            ),
            "Use finite positive fully known paths of the actual delivery delay.",
        )


class ScenarioArchive(Protocol):
    def observe(self, as_of: pd.Timestamp, delay: int) -> DelayPaths: ...


class KnownDelayArchive:
    def __init__(self, opening: pd.DataFrame, closing: pd.DataFrame):
        reference.KnownPriceArchive(opening, closing)
        self._opening, self._closing = opening.copy(), closing.copy()

    def observe(self, as_of, delay):
        day = pd.Timestamp(as_of)
        require(
            type(delay) is int and delay in (1, 2) and day in self._closing.index,
            "No missing information date or invented execution delay.",
        )
        closes = self._closing.loc[:day]
        opens = self._opening.loc[closes.index]
        require(len(closes) > delay, "A complete known delayed path is required.")
        values = closes.to_numpy()
        length = len(values) - delay
        waiting = (
            np.stack([values[j : length + j] / values[:length] for j in range(1, delay)], axis=1)
            if delay > 1
            else np.empty((length, 0, N))
        )
        result = DelayPaths(
            day,
            delay,
            closes.index[:-delay],
            closes.index[delay:],
            waiting,
            opens.to_numpy()[delay:] / values[:-delay],
            values[delay:] / opens.to_numpy()[delay:],
        )
        result.validate()
        return result


def coefficients(nominal, state, sample, cost):
    state.validate()
    sample.validate()
    require(
        sample.as_of == state.as_of
        and (cost, sample.delay) in ((5.0, 1), (20.0, 2))
        and nominal.shape == (N,)
        and np.isfinite(nominal).all()
        and (nominal >= 0).all()
        and abs(math.fsum(nominal) - 0.98) <= 1e-12,
        "Use the known complete nominal opinion and actual path's financial terms.",
    )
    budget = math.fsum(nominal)
    waits = state.cash + sample.waiting @ state.asset_values
    peaks = np.maximum(state.peak, waits.max(axis=1, initial=-np.inf))
    opening_gross = sample.opening @ state.asset_values
    opening_nav = opening_gross + state.cash
    allowance = cost / 10000 * (opening_gross + budget * opening_nav) + N
    lower_nav = opening_nav - allowance
    slack = 1e-6 * abs(sample.fill_mark - 1).sum(axis=1) + 1e-6
    constant = (1 - budget) + budget * sample.fill_mark[:, BIL]
    slope = ((sample.fill_mark - sample.fill_mark[:, BIL, None]) * nominal).sum(axis=1)
    intercepts = lower_nav * constant - slack - 0.85 * peaks
    slopes = lower_nav * slope
    held_final = state.cash + (sample.opening * sample.fill_mark) @ state.asset_values
    prefill_safe = bool((waits >= 0.85 * state.peak).all())
    hold_safe = prefill_safe and bool((held_final >= 0.85 * peaks).all())
    require(
        all(np.isfinite(a).all() for a in (intercepts, slopes, lower_nav, held_final)),
        "A nonfinite reference cannot produce a defensive policy decision.",
    )
    return intercepts, slopes, lower_nav, prefill_safe, hold_safe


def verify_scalar(nominal, state, sample, cost, intercepts, slopes):
    maximum = 0.0
    budget = math.fsum(nominal)
    for i in range(len(sample.ends)):
        waiting_nav = [
            state.cash
            + math.fsum(float(a) * float(b) for a, b in zip(row, state.asset_values, strict=True))
            for row in sample.waiting[i]
        ]
        peak = max(state.peak, *waiting_nav) if waiting_nav else state.peak
        gross = math.fsum(
            float(a) * float(b) for a, b in zip(sample.opening[i], state.asset_values, strict=True)
        )
        nav = gross + state.cash
        lower = nav - (cost / 10000 * (gross + budget * nav) + N)
        bil = float(sample.fill_mark[i, BIL])
        a = lower * ((1 - budget) + budget * bil)
        a -= 1e-6 * math.fsum(abs(float(x) - 1) for x in sample.fill_mark[i]) + 1e-6
        a -= 0.85 * peak
        b = lower * math.fsum(
            float(w) * (float(mark) - bil)
            for w, mark in zip(nominal, sample.fill_mark[i], strict=True)
        )
        maximum = max(maximum, abs(a - float(intercepts[i])), abs(b - float(slopes[i])))
    require(maximum <= 1e-6, "Independent scenario money construction exceeds original tolerance.")
    return maximum


def greatest_binary64(interval):
    if not interval["feasible"]:
        return None
    low, high = (reference.decode(interval[k]) for k in ("lower", "upper"))
    value = float(high)
    if reference.fraction(value) > high:
        value = math.nextafter(value, -math.inf)
    return value if low <= reference.fraction(value) <= high else None


@dataclass(frozen=True)
class ShieldDecision:
    target: np.ndarray | None
    issue: bool
    reason: str
    assessment: dict


class ShieldPolicy:
    def __init__(
        self, nominal: DecisionPolicy, scenarios: ScenarioArchive, cost: float, delay: int
    ):
        require(
            type(delay) is int and (cost, delay) in ((5.0, 1), (20.0, 2)),
            "Keep original matched base/stress costs and delivery timing.",
        )
        self._nominal, self._scenarios, self._cost, self._delay = nominal, scenarios, cost, delay
        self._last = None

    def decide(self, state: reference.MainMark, pending, *, initial=False, retain_inputs=None):
        state.validate()
        require(
            (self._last is None or state.as_of > self._last)
            and (pending is None or pd.Timestamp(pending) > state.as_of),
            "Decision times must advance; a missed pending fill cannot be ignored.",
        )
        self._last = state.as_of
        if pending is not None:
            return ShieldDecision(
                None,
                False,
                "pending_request_preserved",
                {"pending_execution": str(pd.Timestamp(pending).date()), "new_reference": False},
            )
        opinion = self._nominal.decide(
            CapitalSnapshot(
                state.as_of, pd.Series(state.asset_values, index=ASSETS), state.cash, initial
            )
        )
        require(tuple(opinion.target.index) == ASSETS, "Do not replace asset coordinates.")
        nominal = opinion.target.to_numpy(copy=True)
        sample = self._scenarios.observe(state.as_of, self._delay)
        if retain_inputs is not None:
            retain_inputs(state, nominal, sample)
        a, b, lower, prefill_safe, hold_safe = coefficients(nominal, state, sample, self._cost)
        scalar_error = verify_scalar(nominal, state, sample, self._cost, a, b)
        interval = None
        if state.nav < 0.85 * state.peak:
            status = "outside_at_current_mark"
        elif not prefill_safe:
            status = "unavoidable_prefill_reference_breach"
        elif (lower <= 0).any():
            status = "uncertified_conservative_cost_bound"
        else:
            interval = reference.affine_interval(a, b)
            status = (
                "finite_interval_certified"
                if interval["feasible"]
                else "no_finite_sufficient_interval"
            )
        selected = greatest_binary64(interval) if interval is not None else None
        certified = selected is not None
        if status == "finite_interval_certified" and not certified:
            status = "no_representable_binary64_fraction"
        fraction = selected if certified else 0.0
        target = nominal * fraction
        target[BIL] = 0.98 - math.fsum(target[j] for j in range(N) if j != BIL)
        require(
            np.isfinite(target).all()
            and (target >= 0).all()
            and abs(math.fsum(target) - 0.98) <= 1e-12,
            "A defensive attempt must still be a complete funded asset target.",
        )
        gap = float(abs(state.asset_values / state.nav - target).max())
        reason = (
            "initial_capital"
            if initial
            else (
                "observed_weight_gap"
                if gap + 1e-12 >= 0.025
                else (
                    "certified_reference_overrides_band" if certified and not hold_safe else "hold"
                )
            )
        )
        return ShieldDecision(
            target,
            reason != "hold",
            reason,
            {
                "source_opinion": opinion.assessment,
                "reference_status": status,
                "interval": interval,
                "applied_nominal_fraction": fraction,
                "uncertified_defensive_fallback": not certified,
                "hypothetical_full_fill_reference_certified": certified,
                "unchanged_hold_finite_reference_safe": hold_safe,
                "maximum_observed_target_gap": gap,
                "maximum_scalar_money_error_USD": scalar_error,
                "known_scenarios": len(sample.ends),
                "last_scenario_end": str(sample.ends[-1].date()),
                "finite_reference_is_future_or_full_month_guarantee": False,
                "band_hold_is_certified_full_fill": False,
            },
        )


@dataclass(frozen=True)
class ShieldRun:
    result: BacktestResult
    desired: pd.DataFrame
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict


def run(
    data,
    nominal,
    start,
    end,
    *,
    cost_bps,
    delay,
    before_decision=None,
    retain=None,
    retain_inputs=None,
    scenarios=None,
):
    data.validate()
    dates = data.close.loc[start:end].index
    require(
        tuple(data.close.columns) == ASSETS
        and len(dates) >= 2
        and dates[0] == pd.Timestamp(start)
        and dates[-1] == pd.Timestamp(end)
        and dates.equals(sessions(start, end))
        and nominal.index.equals(data.close.index)
        and nominal.columns.equals(data.close.columns),
        "Keep every formal return session and the unchanged fourteen-asset source lattice.",
    )
    policy = ShieldPolicy(
        ArchivedOpinionPolicy(nominal),
        KnownDelayArchive(data.open, data.close) if scenarios is None else scenarios,
        cost_bps,
        delay,
    )
    desired = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    issued = desired.copy()
    units = np.zeros(N)
    cash = previous_nav = peak = 10000.0
    pending = None
    rows, weights, decisions = [], [], []

    def decide(day, values, initial=False):
        nonlocal pending
        if before_decision is not None:
            before_decision()
        nav = float(values.sum() + cash)
        state = reference.MainMark(day, values.copy(), float(cash), nav, float(peak))
        prior_pending = str(pending[0].date()) if pending is not None else None
        decision = policy.decide(
            state,
            pending[0] if pending is not None else None,
            initial=initial,
            retain_inputs=retain_inputs,
        )
        execution = None
        if decision.target is not None:
            desired.loc[day] = decision.target
        if decision.issue:
            require(pending is None, "A new decision cannot overwrite pending main capital.")
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending = (execution, decision.target.copy())
            issued.loc[day] = decision.target
        row = {
            "signal_session": str(day.date()),
            "execution_session": str(execution.date()) if execution is not None else None,
            "initial_capital": initial,
            "reason": decision.reason,
            "pending_execution_before_decision": prior_pending,
            "main_asset_values_at_decision": values.tolist(),
            "main_cash_at_decision": float(cash),
            "main_equity_at_decision": nav,
            "main_peak_at_decision": float(peak),
            "desired_target": decision.target.tolist() if decision.target is not None else None,
            "assessment": decision.assessment,
        }
        decisions.append(row)
        if retain is not None:
            retain({"kind": "decision", **row})

    decide(previous_session(dates[0]), np.zeros(N), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        cost = turnover = 0.0
        orders = 0
        if pending is not None:
            require(
                pending[0] >= day, "A skipped pending fill must preserve a failed partial path."
            )
            if pending[0] == day:
                target = pending[1]
                values, cash, cost, turnover, orders = rebalance(
                    units * opening, cash, target, cost_bps, 1.0
                )
                if turnover > 0 or orders > 0:
                    units = values / opening
                pending = None
        values = units * closing
        nav = float(values.sum() + cash)
        require(
            np.isfinite(nav)
            and nav > 0
            and cash >= 0
            and (units >= 0).all()
            and values.sum() / nav <= 1 + 1e-12,
            "The main account cannot become insolvent, short or cash-borrowed.",
        )
        peak = max(peak, nav)
        rows.append((nav, nav / previous_nav - 1, cash, values.sum() / nav, turnover, cost, orders))
        weights.append(values / nav)
        decide(day, values)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return ShieldRun(
        BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=ASSETS)),
        desired,
        issued,
        decisions,
        {str(pending[0].date()): pending[1].tolist()} if pending is not None else {},
    )


def verify_run(data, output, start, end, cost, delay):
    replay = simulate(
        data,
        output.issued,
        start,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    require(
        replay.frame.equals(output.result.frame) and replay.weights.equals(output.result.weights),
        "Every issued target,fee,position and formal return must match cash-engine replay.",
    )
    dates = output.result.frame.index
    expected_dates = pd.DatetimeIndex([previous_session(dates[0]), *dates])
    require(
        [r["signal_session"] for r in output.decisions]
        == expected_dates.strftime("%Y-%m-%d").tolist(),
        "Every main close,including pending/held and terminal states,must remain visible.",
    )
    pending = None
    peak = 10000.0
    pending_holds = overrides = 0
    expected_issued = output.issued.copy() * np.nan
    for day, row in zip(expected_dates, output.decisions, strict=True):
        initial = day == expected_dates[0]
        nav = 10000.0 if initial else float(output.result.frame.loc[day, "equity"])
        cash = 10000.0 if initial else float(output.result.frame.loc[day, "cash"])
        values = np.zeros(N) if initial else output.result.weights.loc[day].to_numpy() * nav
        peak = max(peak, nav)
        if pending is not None and pending <= day:
            pending = None
        require(
            abs(row["main_peak_at_decision"] - peak) <= 1e-6
            and abs(row["main_equity_at_decision"] - nav) <= 1e-6
            and row["main_cash_at_decision"] == cash
            and np.allclose(row["main_asset_values_at_decision"], values, rtol=0, atol=1e-6)
            and row["pending_execution_before_decision"]
            == (str(pending.date()) if pending is not None else None),
            "A decision must use its own actual main capital and preserved request state.",
        )
        if pending is not None:
            reason = "pending_request_preserved"
            pending_holds += 1
            require(row["desired_target"] is None, "Do not hide replacement targets while pending.")
        else:
            target = np.asarray(row["desired_target"])
            gap = float(abs(values / nav - target).max())
            assessment = row["assessment"]
            reason = (
                "initial_capital"
                if initial
                else "observed_weight_gap"
                if gap + 1e-12 >= 0.025
                else "certified_reference_overrides_band"
                if assessment["hypothetical_full_fill_reference_certified"]
                and not assessment["unchanged_hold_finite_reference_safe"]
                else "hold"
            )
        execution = None
        if reason not in ("hold", "pending_request_preserved"):
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending = execution
            expected_issued.loc[day] = row["desired_target"]
            overrides += reason == "certified_reference_overrides_band"
        require(
            row["reason"] == reason
            and row["execution_session"]
            == (str(execution.date()) if execution is not None else None),
            "Independent daily band,reference override and pending-request verification failed.",
        )
    require(expected_issued.equals(output.issued), "Issued targets differ from the queue proof.")
    return {
        "decision_count": len(output.decisions),
        "pending_refreshes_without_replacement": pending_holds,
        "certified_reference_band_overrides": overrides,
        "all_formal_frames_positions_cash_fees_match_cash_engine": True,
        "own_fresh_capital_peak_and_close_inputs_verified": True,
        "maximum_pending_main_request_batches": 1,
        "broker_orders": 0,
    }
