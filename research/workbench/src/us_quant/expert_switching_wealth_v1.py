from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import main_capital_feedback_v1 as monthly
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.execution_gap_shield_v2 import comparison_report
from us_quant.protective_six_adaptive_hedge_v1 import EXPERT_IDS, MECHANISMS
from us_quant.retained_pair_equal_allocation_v1 import ASSETS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-switching-wealth-v1.json"
ID = "expert_pool_six_universal_switching_wealth_v1"
STUDY = "ssd_expert_switching_wealth_20261107_v1"
N, MAX_PERIODS, PRECISION, QUADRATURE = 6, 119, 160, 60
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 199485,
        "prior_journal_sequence": 1518,
        "prior_journal_head_sha256": (
            "4e4d7d6f1d8f1627ee4fca861fa95d6fc8bbfc60cca9eaca825fe499b0edfaca"
        ),
        "prior_combined_completed": 305,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/expert-switching-wealth-20261107-v1/"
            "source-inputs-v1.json",
            "sha256": "1a5fdd36318a276d18556050f6acd9931eed24338e7bbd5d18f09a675b172297",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_expert_prior": "uniform_six",
        "switch_prior": "Jeffreys_Beta_half_half",
        "switch_destination": "uniform_other_five_not_same_expert",
        "score": "positive_original_NET_complete_source_month_relatives",
        "learning_clock": "matured_full_exchange_months",
        "allocation_clock": "original_initial_then_true_NYSE_month_ends",
        "initial_decision": "2016-10-05",
        "first_full_month_origin": "2016-10-31",
        "maximum_complete_source_months": MAX_PERIODS,
        "primary_decimal_precision": PRECISION,
        "maximum_primary_mass_cells": MAX_PERIODS * N,
        "independent_hazard_quadrature_nodes": QUADRATURE,
        "forecast_polynomial_degree_covered": MAX_PERIODS,
        "independent_coherence": TOLERANCE,
        "primary_mass_coherence": "1e-120",
        "initial_partial_month_assimilated": False,
        "shared_global_source_learner": True,
        "five_year_source_learning_state_reset": False,
        "capital_seed": "last_known_saved_master_not_native_opinion_refresh",
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_band": 0.025,
        "canonical_gross": "sum_actual_held_weight_vector",
        "comparison_operands_retained_before_assertions": True,
        "prefix_positions": [125, 503, 1006, 1509, 2012],
        "candidate_count": 1,
        "study_size_is_a_total_research_quota": False,
        "switch_rate_is_calibrated_market_probability_or_turnover": False,
        "source_game_score_is_main_net_wealth": False,
        "source_wealth_or_regret_guarantee_transferred_to_main": False,
        "fixed_policy_integral_repaired_or_reexecuted": False,
        "old_learner_source_generator_shadow_or_account_replayed": False,
        "new_source_requests": 0,
        "new_shadow_accounts": 0,
        "new_RF_or_event_Adapter": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze one expert-sequence prior,source clock and unchanged financial contract.",
    )


def logs_vector(values):
    result = np.asarray(values)
    require(
        result.shape == (N,) and result.dtype.kind in "fi" and np.isfinite(result).all(),
        "Keep six finite original matured source log-relatives,not a filled loss vector.",
    )
    return result.astype(float, copy=False)


class SwitchCountWealth:
    def __init__(self):
        with localcontext() as context:
            context.prec = PRECISION
            self.mass = [[Decimal(1) / N] * N]
        self.periods = 0
        self.log_score = Decimal(0)

    def check(self):
        with localcontext() as context:
            context.prec = PRECISION
            require(
                type(self.periods) is int
                and 0 <= self.periods <= MAX_PERIODS
                and len(self.mass) == max(1, self.periods)
                and all(len(row) == N for row in self.mass)
                and all(
                    isinstance(v, Decimal) and v.is_finite() and v > 0
                    for row in self.mass
                    for v in row
                )
                and abs(sum(sum(row) for row in self.mass) - 1) <= Decimal("1e-120")
                and self.log_score.is_finite(),
                "Invalid expert/switch-count posterior cannot become neutral allocation.",
            )

    def _prediction(self):
        self.check()
        if self.periods == 0:
            return tuple(self.mass[0])
        result = [Decimal(0)] * N
        for switches, row in enumerate(self.mass):
            hazard = (Decimal(switches) + Decimal("0.5")) / self.periods
            total = sum(row)
            for i in range(N):
                result[i] += (1 - hazard) * row[i] + hazard * (total - row[i]) / (N - 1)
        require(
            all(v > 0 for v in result) and abs(sum(result) - 1) <= Decimal("1e-120"),
            "The next-identity forecast must preserve positive six-source mass.",
        )
        return tuple(result)

    def predict(self):
        with localcontext() as context:
            context.prec = PRECISION
            result = np.array([float(v) for v in self._prediction()])
        require(
            np.isfinite(result).all()
            and (result > 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "The complete-policy capital fractions cannot underflow into source exclusion.",
        )
        return result

    def expected_hazard(self):
        self.check()
        if self.periods == 0:
            return 0.5
        with localcontext() as context:
            context.prec = PRECISION
            return float(
                sum(
                    sum(row) * (Decimal(s) + Decimal("0.5")) / self.periods
                    for s, row in enumerate(self.mass)
                )
            )

    def snapshot(self):
        self.check()
        return {
            "complete_source_intervals": self.periods,
            "switch_count_expert_mass_decimal": [[str(v) for v in row] for row in self.mass],
            "source_game_log_score_decimal": str(self.log_score),
            "posterior_mean_latent_switch_rate": self.expected_hazard(),
            "persistent_primary_mass_cells": len(self.mass) * N,
            "source_game_score_is_actual_main_net_wealth": False,
            "hazard_is_market_probability_or_physical_turnover": False,
        }

    def advance(self, source_logs):
        values = logs_vector(source_logs)
        require(self.periods < MAX_PERIODS, "Keep the original119complete source intervals.")
        with localcontext() as context:
            context.prec = PRECISION
            forecast = self._prediction()
            exact_logs = [Decimal.from_float(float(v)) for v in values]
            shift = max(exact_logs)
            relative = [(v - shift).exp() for v in exact_logs]
            if self.periods == 0:
                predicted = [list(self.mass[0])]
            else:
                predicted = [[Decimal(0)] * N for _ in range(self.periods + 1)]
                for switches, row in enumerate(self.mass):
                    hazard = (Decimal(switches) + Decimal("0.5")) / self.periods
                    total = sum(row)
                    for i in range(N):
                        predicted[switches][i] += (1 - hazard) * row[i]
                        predicted[switches + 1][i] += hazard * (total - row[i]) / (N - 1)
            weighted = [[v * g for v, g in zip(row, relative, strict=True)] for row in predicted]
            normalizer = sum(sum(row) for row in weighted)
            prior_score = sum(w * g for w, g in zip(forecast, relative, strict=True))
            require(
                normalizer.is_finite()
                and normalizer > 0
                and abs(normalizer - prior_score) <= Decimal("1e-120"),
                "Prequential source score and switch-count recursion disagree.",
            )
            log_relative = shift + normalizer.ln()
            self.mass = [[v / normalizer for v in row] for row in weighted]
            self.periods += 1
            self.log_score += log_relative
            self.check()
            return {
                "original_source_log_relatives_hex": [float(v).hex() for v in values],
                "forecast_before_assimilation": [float(v) for v in forecast],
                "prequential_source_game_log_relative_decimal": str(log_relative),
                "state_after": self.snapshot(),
                "score_is_actual_main_NET_return": False,
            }


class HazardQuadrature:
    def __init__(self):
        require(
            2 * QUADRATURE - 1 >= MAX_PERIODS, "Cover the forecast numerator,not just evidence."
        )
        nodes, _ = np.polynomial.chebyshev.chebgauss(QUADRATURE)
        self.hazard = (1 + nodes) / 2
        self.mass = np.full((QUADRATURE, N), 1 / N)
        self.log_evidence = np.zeros(QUADRATURE)
        self.periods = 0

    def _node_prediction(self):
        return (
            self.mass
            if self.periods == 0
            else (
                (1 - self.hazard[:, None]) * self.mass
                + self.hazard[:, None] * (1 - self.mass) / (N - 1)
            )
        )

    def _evidence_weights(self):
        shifted = np.exp(self.log_evidence - self.log_evidence.max())
        require(
            np.isfinite(shifted).all() and (shifted > 0).all(),
            "The independent hazard evidence cannot silently discard a prior node.",
        )
        return shifted / math.fsum(shifted)

    def predict(self):
        result = self._evidence_weights() @ self._node_prediction()
        require(
            np.isfinite(result).all()
            and (result > 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "Independent hazard integration did not preserve complete-source fractions.",
        )
        return result

    def expected_hazard(self):
        return float(self._evidence_weights() @ self.hazard)

    def game_log_score(self):
        peak = float(self.log_evidence.max())
        return peak + math.log(float(np.mean(np.exp(self.log_evidence - peak))))

    def advance(self, source_logs):
        logs = logs_vector(source_logs)
        require(self.periods < MAX_PERIODS, "Independent integration retains the same119months.")
        shift = float(logs.max())
        weighted = self._node_prediction() * np.exp(logs - shift)[None, :]
        normalizers = weighted.sum(axis=1)
        require(
            np.isfinite(weighted).all()
            and (weighted > 0).all()
            and np.isfinite(normalizers).all()
            and (normalizers > 0).all(),
            "A failed independent node cannot turn into uniform successful mass.",
        )
        self.mass = weighted / normalizers[:, None]
        self.log_evidence += np.log(normalizers) + shift
        self.periods += 1


@dataclass(frozen=True)
class OpinionSnapshot:
    as_of: pd.Timestamp
    targets: np.ndarray
    native_known_at: tuple[pd.Timestamp, ...]

    def check(self, day):
        require(
            self.as_of == day
            and self.targets.shape == (N, len(ASSETS))
            and self.targets.dtype.kind in "fi"
            and np.isfinite(self.targets).all()
            and (self.targets >= 0).all()
            and np.allclose(self.targets.sum(axis=1), 0.98, rtol=0, atol=TOLERANCE)
            and len(self.native_known_at) == N
            and all(
                isinstance(d, pd.Timestamp) and d.tzinfo is None and d <= day
                for d in self.native_known_at
            ),
            "Use all six complete then-known opinions,not a missing-source allocation fallback.",
        )


class OpinionFeed(Protocol):
    def observe(self, day: pd.Timestamp) -> OpinionSnapshot: ...


@dataclass(frozen=True)
class PreparedSourceTargets:
    targets: pd.DataFrame
    audits: list[dict]
    proof: dict


def build_targets(calendar, feed: OpinionFeed, periods, *, before=None, retain=None):
    require(
        isinstance(calendar, pd.DatetimeIndex)
        and len(calendar) > 0
        and calendar.tz is None
        and calendar.is_unique
        and calendar.is_monotonic_increasing
        and calendar.equals(sessions(calendar[0], calendar[-1]))
        and pd.Timestamp("2016-10-05") in calendar,
        "Retain the original full market lattice and actual pre-return source anchor.",
    )
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in calendar if d >= pd.Timestamp("2016-10-06") and is_month_end(d)],
    ]
    by_end = {row["end"]: row for row in periods}
    require(len(by_end) == len(periods), "An original completed source month cannot repeat.")
    target = pd.DataFrame(np.nan, index=calendar, columns=ASSETS)
    primary, independent = SwitchCountWealth(), HazardQuadrature()
    anchor, audits = None, []
    max_weight = max_target = max_score = max_hazard = 0.0
    for day in days:
        if before is not None:
            before()
        source = feed.observe(day)
        source.check(day)
        labels = None
        interval = None
        if anchor is not None:
            require(str(day.date()) in by_end, "The original matured six-source month is missing.")
            interval = by_end[str(day.date())]
            require(
                interval["start_after"] == str(anchor.date()),
                "No partial or differently anchored source month may train the sequence mixture.",
            )
            labels = logs_vector(interval["canonical_log_relatives"])
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": str(day.date()),
                    "state": primary.snapshot(),
                    "forecast_before_new_labels": primary.predict().tolist(),
                    "source_native_known_at": [str(d.date()) for d in source.native_known_at],
                    "source_targets_hex": [[float(v).hex() for v in row] for row in source.targets],
                    "matured_source_interval": None
                    if interval is None
                    else {
                        "start_after": interval["start_after"],
                        "end": interval["end"],
                        "original_log_relatives_hex": [float(v).hex() for v in labels],
                    },
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        update = None
        if labels is not None:
            update = primary.advance(labels)
            independent.advance(labels)
        weights, oracle = primary.predict(), independent.predict()
        max_weight = max(max_weight, float(abs(weights - oracle).max()))
        max_score = max(max_score, abs(float(primary.log_score) - independent.game_log_score()))
        max_hazard = max(max_hazard, abs(primary.expected_hazard() - independent.expected_hazard()))
        row = weights @ source.targets
        rational = [
            float(
                sum(
                    Fraction.from_float(float(weights[i]))
                    * Fraction.from_float(float(source.targets[i, j]))
                    for i in range(N)
                )
            )
            for j in range(len(ASSETS))
        ]
        max_target = max(max_target, float(abs(row - rational).max()))
        require(
            max(max_weight, max_target, max_score, max_hazard) <= TOLERANCE
            and np.isfinite(row).all()
            and (row >= 0).all()
            and abs(math.fsum(row) - 0.98) <= TOLERANCE,
            "Independent switching weights/evidence/hazard or rational target proof failed.",
        )
        target.loc[day] = row
        audit = {
            "decision_session": str(day.date()),
            "expert_weights": weights.tolist(),
            "complete_source_intervals": primary.periods,
            "state": primary.snapshot(),
            "period_update": update,
            "source_native_known_at": [str(d.date()) for d in source.native_known_at],
            "source_game_score_is_main_NET_wealth": False,
            "hazard_is_market_probability_or_physical_turnover": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "phase": "after_decision",
                    "decision_session": str(day.date()),
                    "audit": audit,
                    "unified_target_hex": [float(v).hex() for v in row],
                }
            )
        if is_month_end(day):
            anchor = day
    return PreparedSourceTargets(
        target,
        audits,
        {
            "source_decisions": len(audits),
            "complete_source_intervals": primary.periods,
            "maximum_independent_forecast_difference": max_weight,
            "maximum_independent_source_log_score_difference": max_score,
            "maximum_independent_hazard_mean_difference": max_hazard,
            "maximum_independent_rational_target_difference": max_target,
            "maximum_primary_mass_cells": max(
                a["state"]["persistent_primary_mass_cells"] for a in audits
            ),
            "all_source_inputs_matured_before_next_identity_forecast": True,
            "independent_forecast_polynomial_degree_covered": 2 * QUADRATURE - 1,
            "source_score_is_actual_main_wealth_or_bound": False,
            "old_learner_or_policy_replayed": False,
        },
    )


class ArchivedSwitchPolicy:
    def __init__(self, prepared):
        active = prepared.targets.dropna(how="all")
        self.targets = active
        self.audits = {row["decision_session"]: row for row in prepared.audits}
        require(
            not active.empty
            and active.index.is_unique
            and active.index.is_monotonic_increasing
            and active.index.strftime("%Y-%m-%d").tolist() == list(self.audits),
            "A prepared source stream needs every actual source opinion and learning audit.",
        )
        self.last = None

    def decide(self, capital):
        day = capital.as_of
        require(
            self.last is None or day > self.last, "A main account cannot reuse earlier reviews."
        )
        known = self.targets.loc[:day]
        require(not known.empty, "A future source mixture cannot seed earlier main capital.")
        origin = known.index[-1]
        row = known.iloc[-1].copy()
        require(row.index.equals(capital.asset_values.index), "Keep the original fourteen assets.")
        audit = self.audits[str(origin.date())]
        self.last = day
        return monthly.AllocationDecision(
            row,
            {
                "source_master_known_at": str(origin.date()),
                "source_expert_weights": audit["expert_weights"],
                "complete_source_intervals": audit["complete_source_intervals"],
                "source_learning_is_actual_main_capital": False,
                "capital_initialization_is_native_source_refresh": False,
            },
        )


@dataclass(frozen=True)
class SwitchingRun:
    result: BacktestResult
    desired: pd.DataFrame
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict
    raw_kernel_auxiliary_gross: pd.Series


def run(data, prepared, start, end, *, cost_bps, delay, before_decision=None, retain=None):
    output = monthly.run(
        data,
        ArchivedSwitchPolicy(prepared),
        start,
        end,
        cost_bps=cost_bps,
        delay=delay,
        before_decision=before_decision,
        retain=retain,
    )
    frame = output.result.frame.copy()
    raw = frame["gross_exposure"].copy()
    frame["gross_exposure"] = [float(row.copy().sum()) for row in output.result.weights.to_numpy()]
    require(
        frame.drop(columns="gross_exposure").equals(
            output.result.frame.drop(columns="gross_exposure")
        ),
        "Canonical gross representation cannot modify a single financial primitive.",
    )
    return SwitchingRun(
        BacktestResult(frame, output.result.weights),
        output.desired,
        output.issued,
        output.decisions,
        output.pending_after_window,
        raw,
    )


def verify_run(data, prepared, output, start, end, cost, delay, *, retain_comparison):
    require(callable(retain_comparison), "Preserve every cash and position comparison operand.")
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
    comparison = comparison_report(output.result, replay)
    retain_comparison(output.result, replay, comparison)
    require(
        comparison["full_frame_equals"] and comparison["full_positions_equal"],
        "Actual switching main book must exactly match original cash-engine replay.",
    )
    dates = output.result.frame.index
    reviews = pd.DatetimeIndex([previous_session(dates[0]), *[d for d in dates if is_month_end(d)]])
    require(
        [r["signal_session"] for r in output.decisions] == reviews.strftime("%Y-%m-%d").tolist(),
        "Preserve every actual initial/monthly capital review.",
    )
    active = prepared.targets.dropna(how="all")
    expected_issued = output.issued.copy() * np.nan
    pending = None
    for i, (day, record) in enumerate(zip(reviews, output.decisions, strict=True)):
        if pending is not None and pending <= day:
            pending = None
        require(pending is None, "Never replace a pending original delayed request.")
        known = active.loc[:day]
        require(not known.empty, "No known original source mixture at capital review.")
        desired = known.iloc[-1].to_numpy()
        held = np.zeros(len(ASSETS)) if i == 0 else output.result.weights.loc[day].to_numpy()
        gap = float(abs(desired - held).max())
        reason = (
            "initial_capital"
            if i == 0
            else ("observed_weight_gap" if gap + TOLERANCE >= 0.025 else "hold")
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        require(
            np.max(abs(output.desired.loc[day].to_numpy() - desired)) <= TOLERANCE
            and record["assessment"]["source_master_known_at"] == str(known.index[-1].date())
            and record["reason"] == reason
            and record["execution_session"] == str(execution.date())
            and record["initial_capital"] is (i == 0)
            and abs(record["maximum_observed_target_gap"] - gap) <= TOLERANCE,
            "Original source seed,holding band or delay disagrees with independent main review.",
        )
        if reason != "hold":
            pending = execution
            expected_issued.loc[day] = desired
    require(
        expected_issued.equals(output.issued), "Issued switching requests differ from actual band."
    )
    return {
        **monthly.verify_capital_inputs(output),
        "actual_source_mixture_selection_and_seed_verified": True,
        "all_cash_financial_columns_and_positions_exact": True,
        "comparison_operands_retained_before_assertions": True,
        "original_cost_delay_RF_and_full_calendars_retained": True,
        "shared_source_history_is_not_main_capital_history": True,
        "source_game_score_is_actual_main_wealth_or_bound": False,
        "broker_orders": 0,
    }
