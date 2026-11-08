from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant.calendar import next_session, sessions
from us_quant.config import QuantError
from us_quant.retained_main_risk_v1 import ASSETS, EXPERTS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/main-risk-scenario-control-v1.json"
STUDY = "ssd_main_risk_scenario_control_20261108_v1"
ID = "expert_pool_six_inventory_scenario_viability_v1"
SCENARIOS, HORIZON = 126, 21
TOLERANCE = 1e-12
BIL = ASSETS.index("BIL")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantError(message)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "actual_native_event_seq": 210356,
        "prior_journal_sequence": 1554,
        "prior_combined_completed": 308,
        "forecast_scenarios": SCENARIOS,
        "forecast_horizon_sessions": HORIZON,
        "opportunity_subperiods": [84, 42],
        "opportunity_score": "minimum_subperiod_mean_terminal_lower_log_wealth_per_session",
        "selection_score_decimal_places": 12,
        "main_drawdown_limit": 0.15,
        "forecast_feasibility_safety_margin": TOLERANCE,
        "forecast_relative_numeric_tolerance": TOLERANCE,
        "forecast_cost_bound": (
            "twice_scenario_rate_times_pre_execution_NAV_plus14_model_commissions"
        ),
        "source_clock": "unchanged_initial_source_anchor_then_true_NYSE_month_ends",
        "main_review_clock": "every_original_main_close_after_delayed_fills",
        "model_pending_rule": "one_pending_request_never_cancelled_or_overwritten",
        "actual_weight_trade_band": 0.025,
        "target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "initial_main_capital_usd": 10000.0,
        "initial_main_drawdown_peak_usd": 10000.0,
        "finite_actions": "six_complete_source_targets_equal_six_blend_BIL_reserve_hold",
        "reserve_or_hold_is_strategy_expert": False,
        "forecast_source_opinions_held_constant": True,
        "prefix_positions": [125, 503, 1006, 1509, 2012],
        "large_daily_decision_records": "streamed_private_chronological_JSONL",
        "opportunity_subperiods_are_heldout_validation": False,
        "historical_scenario_feasibility_is_future_drawdown_guarantee": False,
        "forecast_cost_bound_is_actual_main_debit": False,
        "overlapping_scenarios_are_independent_observations": False,
        "insufficient_forecast_history": (
            "explicit_unavailable_funding_standby_not_feasibility_pass"
        ),
        "no_viable_action": ("retain_witnesses_defensive_BIL_or_pending_hold_not_feasibility_pass"),
        "new_source_learner_or_shadow_runs": 0,
        "new_source_requests": 0,
        "candidate_count": 1,
        "study_size_is_total_research_quota": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze one current-inventory scenario policy before any historical forecasts.",
    )


def numeric(values, shape) -> np.ndarray:
    array = np.asarray(values)
    require(
        array.shape == shape
        and array.dtype.kind in "iuf"
        and np.isfinite(array).all()
        and not any(isinstance(v, bool | np.bool_) for v in np.asarray(values, dtype=object).flat),
        "Require complete finite numeric coordinates without boolean or textual casting.",
    )
    return array.astype(float, copy=True)


def session(day) -> bool:
    return (
        isinstance(day, pd.Timestamp)
        and pd.notna(day)
        and day.tzinfo is None
        and day == day.normalize()
    )


def number(value) -> bool:
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool | np.bool_)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class ScenarioBatch:
    opening: np.ndarray
    closing: np.ndarray
    starts: tuple[pd.Timestamp, ...]
    ends: tuple[pd.Timestamp, ...]

    def check(self, day: pd.Timestamp) -> None:
        opening = numeric(self.opening, (SCENARIOS, HORIZON, len(ASSETS)))
        closing = numeric(self.closing, opening.shape)
        require(
            session(day)
            and (opening > 0).all()
            and (closing > 0).all()
            and len(self.starts) == len(self.ends) == SCENARIOS
            and all(
                session(a) and session(b) and a < b <= day
                for a, b in zip(self.starts, self.ends, strict=True)
            )
            and all(a < b for a, b in zip(self.starts[:-1], self.starts[1:], strict=True))
            and all(a < b for a, b in zip(self.ends[:-1], self.ends[1:], strict=True)),
            "Every complete historical price path must mature before this main close.",
        )


@dataclass(frozen=True)
class UnavailableHistory:
    available_matured_paths: int
    reason: str = "fewer_than126_complete_then_known21_session_paths"


class ScenarioFeed(Protocol):
    def observe(self, day: pd.Timestamp) -> ScenarioBatch | UnavailableHistory: ...


class PriceScenarioFeed:
    def __init__(self, opening: pd.DataFrame, closing: pd.DataFrame):
        require(
            isinstance(opening, pd.DataFrame)
            and isinstance(closing, pd.DataFrame)
            and len(closing) > 0
            and opening.index.equals(closing.index)
            and tuple(opening.columns) == tuple(closing.columns) == ASSETS
            and isinstance(closing.index, pd.DatetimeIndex)
            and closing.index.tz is None
            and closing.index.is_unique
            and closing.index.is_monotonic_increasing
            and closing.index.equals(sessions(closing.index[0], closing.index[-1])),
            "Keep the full original adjusted-open/close calendar and asset coordinates.",
        )
        self.opening, self.closing = opening, closing

    def observe(self, day: pd.Timestamp) -> ScenarioBatch | UnavailableHistory:
        require(session(day) and day in self.closing.index, "The original main close is missing.")
        end = int(self.closing.index.get_loc(day))
        last_start = end - HORIZON
        if last_start + 1 < SCENARIOS:
            return UnavailableHistory(max(0, last_start + 1))
        starts = np.arange(last_start - SCENARIOS + 1, last_start + 1)
        selected = starts[:, None] + np.arange(1, HORIZON + 1)
        last = end + 1
        opening = numeric(self.opening.iloc[:last].to_numpy(), (last, len(ASSETS)))
        closing = numeric(self.closing.iloc[:last].to_numpy(), (last, len(ASSETS)))
        require((opening > 0).all() and (closing > 0).all(), "Never fill missing price history.")
        base = closing[starts, None, :]
        result = ScenarioBatch(
            opening[selected] / base,
            closing[selected] / base,
            tuple(self.closing.index[starts]),
            tuple(self.closing.index[starts + HORIZON]),
        )
        result.check(day)
        return result


@dataclass(frozen=True)
class CapitalState:
    day: pd.Timestamp
    asset_values: np.ndarray
    cash: float
    peak: float
    cost_bps: float
    delay: int
    initial: bool
    pending_target: np.ndarray | None = None
    pending_delay: int | None = None

    @property
    def nav(self) -> float:
        return float(self.asset_values.sum() + self.cash)

    def check(self) -> None:
        values = numeric(self.asset_values, (len(ASSETS),))
        require(
            session(self.day)
            and (values >= 0).all()
            and type(self.initial) is bool
            and type(self.delay) is int
            and all(number(v) for v in (self.cash, self.peak, self.cost_bps))
            and (self.cost_bps, self.delay) in ((5.0, 1), (20.0, 2))
            and np.isfinite([self.cash, self.peak]).all()
            and self.cash >= 0
            and 0 < self.nav <= self.peak
            and self.peak >= 10000
            and (not self.initial or ((values == 0).all() and self.nav == 10000)),
            "Use actual main closing inventory, cash, peak and the original funded scenario.",
        )
        require(
            (self.pending_target is None) == (self.pending_delay is None),
            "A pending modeled target needs its actual remaining execution delay.",
        )
        if self.pending_target is not None:
            target(self.pending_target)
            require(
                type(self.pending_delay) is int and 1 <= self.pending_delay <= self.delay,
                "A pending request cannot be cancelled, advanced or overwritten.",
            )


def target(values) -> np.ndarray:
    result = numeric(values, (len(ASSETS),))
    require(
        (result >= 0).all() and abs(math.fsum(result) - 0.98) <= TOLERANCE,
        "A proposed action must be one complete long-only 98% underlying-asset target.",
    )
    return result


def wealth_envelope(
    batch: ScenarioBatch,
    state: CapitalState,
    proposed: np.ndarray | None,
    *,
    execution_delay: int | None = None,
    independent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    state.check()
    batch.check(state.day)
    values = state.asset_values
    base = np.einsum("sth,h->st", batch.closing, values) + state.cash
    lower, upper = base.copy(), base.copy()
    if proposed is None:
        return lower, upper
    proposed = target(proposed)
    delay = state.delay if execution_delay is None else execution_delay
    require(type(delay) is int and 1 <= delay <= state.delay, "Keep actual remaining delay.")
    marks = batch.opening[:, delay - 1, :]
    before = marks @ values + state.cash
    fee_upper = before * (2 * state.cost_bps / 10000) + len(ASSETS)
    after_lower = before - fee_upper
    ratios = batch.closing[:, delay - 1 :, :] / marks[:, None, :]
    epsilon = np.maximum(1e-6, before * TOLERANCE)
    representation = abs(ratios - 1).sum(axis=2) * epsilon[:, None]
    if independent:
        low_units = proposed[None, :] * after_lower[:, None] / marks
        high_units = proposed[None, :] * before[:, None] / marks
        lower[:, delay - 1 :] = (
            (low_units[:, None, :] * batch.closing[:, delay - 1 :, :]).sum(axis=2)
            + 0.02 * after_lower[:, None]
            - representation
        )
        upper[:, delay - 1 :] = (
            (high_units[:, None, :] * batch.closing[:, delay - 1 :, :]).sum(axis=2)
            + 0.02 * before[:, None]
            + representation
        )
    else:
        relative = 0.02 + np.einsum("sth,h->st", ratios, proposed)
        lower[:, delay - 1 :] = after_lower[:, None] * relative - representation
        upper[:, delay - 1 :] = before[:, None] * relative + representation
    require(
        np.isfinite(lower).all() and np.isfinite(upper).all() and (lower <= upper).all(),
        "The conservative scenario wealth envelope became invalid; do not clip it.",
    )
    return lower, upper


def assessment(batch, state, proposed, *, execution_delay=None):
    lower, upper = wealth_envelope(batch, state, proposed, execution_delay=execution_delay)
    other_low, other_high = wealth_envelope(
        batch, state, proposed, execution_delay=execution_delay, independent=True
    )
    error = float(max(abs(lower - other_low).max(), abs(upper - other_high).max()) / state.nav)
    require(error <= TOLERANCE, "Independent scenario wealth representations disagree.")
    peak = np.maximum.accumulate(np.column_stack([np.full(SCENARIOS, state.peak), upper]), axis=1)[
        :, 1:
    ]
    drawdowns = 1 - lower / peak
    row, column = np.unravel_index(int(np.argmax(drawdowns)), drawdowns.shape)
    terminal = lower[:, -1]
    positive = bool((lower > 0).all())
    logs = np.log(terminal / state.nav) if positive else None
    scores = (
        None
        if logs is None
        else [
            float(logs[:84].mean() / HORIZON),
            float(logs[84:].mean() / HORIZON),
        ]
    )
    risk = float(drawdowns.max())
    return {
        "status": "defined" if positive else "nonpositive_conservative_wealth_not_viable",
        "score": None if scores is None else min(scores),
        "opportunity_subperiod_scores": scores,
        "forecast_feasible": positive and risk <= 0.15 - TOLERANCE,
        "worst_drawdown_upper": risk,
        "scenario_drawdown_upper": drawdowns.max(axis=1).tolist(),
        "scenario_terminal_lower_wealth": terminal.tolist(),
        "worst_scenario_start": str(batch.starts[row].date()),
        "worst_scenario_end": str(batch.ends[row].date()),
        "worst_session_offset": int(column + 1),
        "independent_relative_wealth_error": error,
        "forecast_is_future_drawdown_guarantee": False,
        "forecast_fee_is_actual_debit": False,
    }


@dataclass(frozen=True)
class OpinionSet:
    origin: pd.Timestamp
    targets: np.ndarray
    known_at: tuple[pd.Timestamp, ...]

    def check(self, day):
        rows = numeric(self.targets, (6, len(ASSETS)))
        require(
            session(day)
            and session(self.origin)
            and self.origin <= day
            and len(self.known_at) == 6
            and all(session(d) and d <= self.origin for d in self.known_at)
            and (rows >= 0).all()
            and np.max(abs(rows.sum(axis=1) - 0.98)) <= TOLERANCE,
            "Keep all six complete admitted, then-known opinions; no future source refresh.",
        )


class OpinionFeed(Protocol):
    def observe(self, day: pd.Timestamp) -> OpinionSet: ...


@dataclass(frozen=True)
class Decision:
    target: np.ndarray | None
    assessment: dict


def choose(state: CapitalState, opinions: OpinionSet, history) -> Decision:
    state.check()
    opinions.check(state.day)
    reserve = np.zeros(len(ASSETS))
    reserve[BIL] = 0.98
    if isinstance(history, UnavailableHistory):
        requested = state.pending_target is None and (
            state.initial
            or abs(state.asset_values / state.nav - reserve).max() + TOLERANCE >= 0.025
        )
        return Decision(
            reserve if requested else None,
            {
                "status": "forecast_history_unavailable",
                "history_reason": history.reason,
                "available_matured_paths": history.available_matured_paths,
                "selected_action": "funding_standby_BIL" if requested else "hold_pending_or_band",
                "forecast_feasible": False,
                "reserve_or_hold_is_strategy_expert": False,
            },
        )
    require(isinstance(history, ScenarioBatch), "Require explicit matured scenario delivery.")
    if state.pending_target is not None:
        proof = assessment(
            history, state, state.pending_target, execution_delay=state.pending_delay
        )
        return Decision(
            None,
            {
                "status": "pending_request_preserved",
                "selected_action": "hold_with_committed_pending_target",
                "forecast_feasible": proof["forecast_feasible"],
                "actions": [{"id": "committed_pending", "assessment": proof}],
            },
        )
    rows = opinions.targets
    menu = [
        ("hold", None),
        *[(name, rows[i]) for i, name in enumerate(EXPERTS)],
        ("equal_six_source_blend", rows.mean(axis=0)),
        ("funding_standby_BIL", reserve),
    ]
    actions = []
    for canonical_index, (name, desired) in enumerate(menu):
        requested = desired is not None and (
            state.initial
            or abs(state.asset_values / state.nav - desired).max() + TOLERANCE >= 0.025
        )
        effective = desired if requested else None
        proof = assessment(history, state, effective)
        actions.append(
            {
                "id": name,
                "canonical_index": canonical_index,
                "request": requested,
                "target": effective,
                "assessment": proof,
            }
        )
    viable = [row for row in actions if row["assessment"]["forecast_feasible"]]
    if viable:
        selected = max(
            viable,
            key=lambda row: (
                round(row["assessment"]["score"], 12),
                not row["request"],
                -row["canonical_index"],
            ),
        )
        status = "empirically_viable_action_selected"
    else:
        selected = next(row for row in actions if row["id"] == "funding_standby_BIL")
        status = "no_viable_action_defensive_standby_not_a_pass"
    details = [
        {
            "id": row["id"],
            "request": row["request"],
            "target": None if row["target"] is None else row["target"].tolist(),
            "assessment": row["assessment"],
        }
        for row in actions
    ]
    return Decision(
        selected["target"],
        {
            "status": status,
            "selected_action": selected["id"],
            "forecast_feasible": selected["assessment"]["forecast_feasible"],
            "source_origin": str(opinions.origin.date()),
            "actions": details,
            "source_probabilities_or_regret_recomputed": False,
            "reserve_or_hold_is_strategy_expert": False,
        },
    )


class Controller:
    def __init__(self, scenario_feed: ScenarioFeed, opinion_feed: OpinionFeed, cost_bps, delay):
        self.scenarios, self.opinions = scenario_feed, opinion_feed
        self.cost_bps, self.delay = cost_bps, delay
        self.peak = 10000.0
        self.last = None

    def decide(self, day, values, cash, initial, pending_target=None, pending_delay=None):
        require(
            session(day)
            and type(initial) is bool
            and (
                (self.last is None and initial)
                or (self.last is not None and not initial and day == next_session(self.last))
            ),
            "Observe every actual main close exactly once; no skipped or backfilled NAV.",
        )
        values = numeric(values, (len(ASSETS),))
        require(number(cash), "Actual cash must be numeric, not coerced boolean or text.")
        nav = float(values.sum() + cash)
        peak = max(self.peak, nav)
        state = CapitalState(
            day,
            values.copy(),
            float(cash),
            peak,
            self.cost_bps,
            self.delay,
            initial,
            pending_target,
            pending_delay,
        )
        state.check()
        opinions, history = self.opinions.observe(day), self.scenarios.observe(day)
        result = choose(state, opinions, history)
        detail = {
            **result.assessment,
            "source_origin": str(opinions.origin.date()),
            "source_native_known_at": [str(d.date()) for d in opinions.known_at],
            "source_targets_hex": [[float(v).hex() for v in row] for row in opinions.targets],
            "original_source_clock_refreshed_daily": False,
        }
        if isinstance(history, ScenarioBatch):
            detail["price_path_starts"] = [str(d.date()) for d in history.starts]
            detail["price_path_ends"] = [str(d.date()) for d in history.ends]
        self.peak = peak
        self.last = day
        return Decision(result.target, detail)
