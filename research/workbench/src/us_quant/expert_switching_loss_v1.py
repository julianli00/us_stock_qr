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
POLICY = ROOT / "config/expert-switching-loss-v1.json"
ID = "expert_pool_six_gap_adaptive_switching_loss_v1"
STUDY = "ssd_expert_switching_loss_20261107_v1"
N, MAX_PERIODS, PRECISION, QUADRATURE, TOLERANCE = 6, 119, 160, 60, 1e-12


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 200235,
        "prior_journal_sequence": 1530,
        "prior_journal_head_sha256": (
            "06bb6711232195b5f8967142affb1796dbf4e40718021f718d8d9f6b077d38c6"
        ),
        "prior_combined_completed": 306,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/expert-switching-loss-20261107-v1/"
            "source-inputs-v1.json",
            "sha256": "04dcd922d57d5ff26c2610249174c4fc8c6c1973c0611b63bcf96161ca7524eb",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "switch_prior": "Jeffreys_Beta_half_half",
        "initial_expert_prior": "uniform_six",
        "switch_destination": "uniform_other_five",
        "source_loss_labels": "unchanged_sealed_six_binary64_input_loss_labels",
        "rate": "log6_over_PAST_cumulative_mixability_gap",
        "zero_gap_rate": "analytic_infinity_constant_past_prior",
        "equal_current_losses": "analytic_played_equals_mix_loss_zero_gap",
        "current_rate_partition": "refilter_only_NEW_source_loss_history_at_current_rate",
        "negative_gap_clipped": False,
        "gap_floor": None,
        "common_loss_centering_changes_source_labels": False,
        "static_AdaHedge_regret_bound_claimed_for_sequence_class": False,
        "primary_decimal_precision": PRECISION,
        "independent_hazard_nodes": QUADRATURE,
        "forecast_polynomial_degree_covered": MAX_PERIODS,
        "maximum_complete_source_months": MAX_PERIODS,
        "maximum_primary_mass_cells": MAX_PERIODS * N,
        "independent_coherence": TOLERANCE,
        "initial_decision": "2016-10-05",
        "first_full_month_origin": "2016-10-31",
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
        "source_gap_rate_posterior_is_main_NET_wealth_or_market_probability": False,
        "source_regret_or_wealth_guarantee_transferred_to_main": False,
        "old_learner_source_generator_shadow_or_account_replayed": False,
        "old_gap_repaired_clipped_or_floored": False,
        "new_source_requests": 0,
        "new_shadow_accounts": 0,
        "new_RF_or_event_Adapter": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze one gap-adaptive source-sequence law and unchanged financial acceptance.",
    )


def checked_labels(values):
    array = np.asarray(values)
    require(
        array.shape == (N,)
        and array.dtype.kind in "fi"
        and np.isfinite(array).all()
        and (array >= 0).all()
        and (array <= 1).all(),
        "Preserve six original bounded source labels without recoding or filling.",
    )
    return tuple(Decimal.from_float(float(v)) for v in array)


@dataclass(frozen=True)
class PathPartition:
    mass: list[list[Decimal]]
    forecast: tuple[Decimal, ...]
    centered_log_partition: Decimal | None
    mean_hazard: Decimal


def path_partition(history, rate):
    require(len(history) <= MAX_PERIODS, "Keep the original119complete source intervals.")
    with localcontext() as context:
        context.prec = PRECISION
        if rate is None:
            require(
                all(len(row) == N and min(row) == max(row) for row in history),
                "An infinite rate cannot discard a nonconstant observed source loss.",
            )
            degree = max(0, len(history) - 1)
            masses = []
            for switches in range(degree + 1):
                probability = Fraction(
                    math.comb(2 * switches, switches)
                    * math.comb(2 * (degree - switches), degree - switches),
                    4**degree,
                )
                value = Decimal(probability.numerator) / probability.denominator / N
                masses.append([value] * N)
            return PathPartition(
                masses,
                (Decimal(1) / N,) * N,
                None,
                Decimal("0.5"),
            )
        require(
            isinstance(rate, Decimal) and rate.is_finite() and rate > 0,
            "Use a finite positive rate.",
        )
        mass, partition = [[Decimal(1) / N] * N], Decimal(0)
        for index, row in enumerate(history):
            require(
                len(row) == N and all(v.is_finite() and 0 <= v <= 1 for v in row),
                "Invalid cached loss history.",
            )
            minimum = min(row)
            relative = [(-rate * (v - minimum)).exp() for v in row]
            if index == 0:
                prediction = [mass[0].copy()]
            else:
                prediction = [[Decimal(0)] * N for _ in range(index + 1)]
                for switches, previous in enumerate(mass):
                    hazard = (Decimal(switches) + Decimal("0.5")) / index
                    total = sum(previous)
                    for i in range(N):
                        prediction[switches][i] += (1 - hazard) * previous[i]
                        prediction[switches + 1][i] += hazard * (total - previous[i]) / (N - 1)
            weighted = [
                [v * g for v, g in zip(previous, relative, strict=True)] for previous in prediction
            ]
            normalizer = sum(sum(previous) for previous in weighted)
            require(
                normalizer.is_finite() and normalizer > 0, "Source partition normalization failed."
            )
            mass = [[v / normalizer for v in previous] for previous in weighted]
            partition += normalizer.ln()
        if not history:
            forecast, hazard_mean = tuple(mass[0]), Decimal("0.5")
        else:
            forecast, hazard_mean = [Decimal(0)] * N, Decimal(0)
            for switches, previous in enumerate(mass):
                hazard = (Decimal(switches) + Decimal("0.5")) / len(history)
                total = sum(previous)
                hazard_mean += total * hazard
                for i in range(N):
                    forecast[i] += (1 - hazard) * previous[i] + hazard * (total - previous[i]) / (
                        N - 1
                    )
            forecast = tuple(forecast)
        require(
            all(v > 0 for v in forecast)
            and abs(sum(forecast) - 1) <= Decimal("1e-120")
            and abs(sum(sum(row) for row in mass) - 1) <= Decimal("1e-120"),
            "Refitting the current-rate source prior cannot exclude or invent an expert.",
        )
        return PathPartition(mass, forecast, partition, hazard_mean)


def independent_partition(history, rate):
    require(2 * QUADRATURE - 1 >= MAX_PERIODS, "Cover the degree119forecast numerator.")
    values = np.array([[float(v) for v in row] for row in history])
    if rate is None:
        require(
            not len(values) or np.all(values.max(axis=1) == values.min(axis=1)),
            "The independent infinite limit needs genuinely constant source losses.",
        )
        return np.full(N, 1 / N), None, 0.5
    eta = float(rate)
    nodes, _ = np.polynomial.chebyshev.chebgauss(QUADRATURE)
    hazard = (1 + nodes) / 2
    mass, evidence = np.full((QUADRATURE, N), 1 / N), np.zeros(QUADRATURE)
    for index, row in enumerate(values):
        prediction = (
            mass
            if index == 0
            else ((1 - hazard[:, None]) * mass + hazard[:, None] * (1 - mass) / (N - 1))
        )
        centered = row - float(row.min())
        weighted = prediction * np.exp(-eta * centered)[None, :]
        normalizer = weighted.sum(axis=1)
        require(
            np.isfinite(weighted).all()
            and (weighted > 0).all()
            and np.isfinite(normalizer).all()
            and (normalizer > 0).all(),
            "The independent current-rate filter cannot drop a failed node or source.",
        )
        mass = weighted / normalizer[:, None]
        evidence += np.log(normalizer)
    predictive = (
        mass
        if not history
        else ((1 - hazard[:, None]) * mass + hazard[:, None] * (1 - mass) / (N - 1))
    )
    peak = float(evidence.max())
    raw = np.exp(evidence - peak)
    weights = raw / math.fsum(raw)
    forecast = weights @ predictive
    require(
        np.isfinite(forecast).all()
        and (forecast > 0).all()
        and abs(math.fsum(forecast) - 1) <= TOLERANCE,
        "The independent current-rate identity forecast is not a funded simplex.",
    )
    return forecast, peak + math.log(float(np.mean(raw))), float(weights @ hazard)


class AdaptiveSwitchingLoss:
    def __init__(self):
        self.history = []
        self.gap, self.played = Decimal(0), Decimal(0)
        self.rate = None
        self.partition = path_partition([], None)

    def check(self):
        with localcontext() as context:
            context.prec = PRECISION
            require(
                len(self.history) <= MAX_PERIODS
                and self.gap.is_finite()
                and self.gap >= 0
                and self.played.is_finite()
                and 0 <= self.played <= len(self.history)
                and ((self.rate is None) is (self.gap == 0))
                and len(self.partition.forecast) == N
                and all(v.is_finite() and v > 0 for v in self.partition.forecast)
                and abs(sum(self.partition.forecast) - 1) <= Decimal("1e-120"),
                "A failed gap/rate history cannot be repaired into a neutral source forecast.",
            )

    def predict(self):
        self.check()
        result = np.array([float(v) for v in self.partition.forecast])
        require(
            np.isfinite(result).all()
            and (result > 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "Keep all complete experts in the predictive source distribution.",
        )
        return result

    def snapshot(self):
        self.check()
        return {
            "complete_source_intervals": len(self.history),
            "cumulative_source_mixability_gap_decimal": str(self.gap),
            "cumulative_source_played_loss_decimal": str(self.played),
            "learning_rate_decimal": "analytic_infinity" if self.rate is None else str(self.rate),
            "switch_count_identity_mass_decimal": [
                [str(v) for v in row] for row in self.partition.mass
            ],
            "centered_source_log_partition_decimal": (
                None
                if self.partition.centered_log_partition is None
                else str(self.partition.centered_log_partition)
            ),
            "posterior_mean_latent_switch_rate": float(self.partition.mean_hazard),
            "persistent_primary_mass_cells": len(self.partition.mass) * N,
            "negative_gap_clipped": False,
            "gap_floor_applied": False,
            "source_loss_gap_is_actual_main_NET_return": False,
            "static_AdaHedge_reference_bound_claimed": False,
        }

    def advance(self, values):
        self.check()
        labels = checked_labels(values)
        require(len(self.history) < MAX_PERIODS, "Keep the original119complete source months.")
        with localcontext() as context:
            context.prec = PRECISION
            before = self.partition.forecast
            previous_rate = self.rate
            minimum = min(labels)
            centered = [v - minimum for v in labels]
            played_centered = sum(w * v for w, v in zip(before, centered, strict=True))
            if not any(centered):
                played_centered = mix_centered = increment = Decimal(0)
            else:
                mix_centered = (
                    Decimal(0)
                    if previous_rate is None
                    else -sum(
                        w * (-previous_rate * v).exp()
                        for w, v in zip(before, centered, strict=True)
                    ).ln()
                    / previous_rate
                )
                increment = played_centered - mix_centered
                require(increment >= 0, "Negative source Jensen gap;do not clip or floor it.")
            played, mix = minimum + played_centered, minimum + mix_centered
            self.history.append(labels)
            self.played += played
            self.gap += increment
            self.rate = None if self.gap == 0 else Decimal(N).ln() / self.gap
            self.partition = path_partition(self.history, self.rate)
            self.check()
            return {
                "original_source_loss_labels_hex": [float(v).hex() for v in labels],
                "prequential_forecast_before_assimilation": [float(v) for v in before],
                "prior_learning_rate_decimal": (
                    "analytic_infinity" if previous_rate is None else str(previous_rate)
                ),
                "source_played_loss_decimal": str(played),
                "source_mix_loss_decimal": str(mix),
                "source_mixability_gap_increment_decimal": str(increment),
                "state_after": self.snapshot(),
                "source_gap_is_main_NET_performance": False,
            }


class IndependentAdaptiveLoss:
    def __init__(self):
        self.history, self.gap = [], 0.0
        self.rate, self.forecast, self.partition, self.hazard = None, np.full(N, 1 / N), None, 0.5

    def advance(self, values):
        labels = np.array([float(v) for v in checked_labels(values)])
        centered = labels - labels.min()
        played = math.fsum(float(w * v) for w, v in zip(self.forecast, centered, strict=True))
        if not np.any(centered):
            gap = 0.0
        else:
            mix = (
                0.0
                if self.rate is None
                else -math.log1p(
                    math.fsum(
                        float(w) * math.expm1(-self.rate * float(v))
                        for w, v in zip(self.forecast, centered, strict=True)
                    )
                )
                / self.rate
            )
            gap = played - mix
            require(gap >= 0, "The independent gap is negative;no clipped source success.")
        self.history.append(tuple(Decimal.from_float(float(v)) for v in labels))
        self.gap = math.fsum((self.gap, gap))
        self.rate = None if self.gap == 0 else math.log(N) / self.gap
        self.forecast, self.partition, self.hazard = independent_partition(self.history, self.rate)


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
            "Keep all six complete then-known source opinions,not neutral missing histories.",
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
        "Keep the full original source calendar and real initialization anchor.",
    )
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in calendar if d >= pd.Timestamp("2016-10-06") and is_month_end(d)],
    ]
    by_end = {row["end"]: row for row in periods}
    require(len(by_end) == len(periods), "No duplicated original source-loss month.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=ASSETS)
    primary, independent = AdaptiveSwitchingLoss(), IndependentAdaptiveLoss()
    anchor, audits = None, []
    max_weight = max_target = max_gap = max_rate = max_partition = max_hazard = 0.0
    for day in days:
        if before is not None:
            before()
        source = feed.observe(day)
        source.check(day)
        interval, labels = None, None
        if anchor is not None:
            require(str(day.date()) in by_end, "A matured original loss month is missing.")
            interval = by_end[str(day.date())]
            require(
                interval["start_after"] == str(anchor.date()),
                "Do not assimilate a partial source month.",
            )
            labels = np.array([float(v) for v in checked_labels(interval["input_loss_labels"])])
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
                        "original_loss_labels_hex": [float(v).hex() for v in labels],
                    },
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        update = None
        if labels is not None:
            update = primary.advance(labels)
            independent.advance(labels)
        weights = primary.predict()
        max_weight = max(max_weight, float(abs(weights - independent.forecast).max()))
        max_gap = max(max_gap, abs(float(primary.gap) - independent.gap))
        require((primary.rate is None) is (independent.rate is None), "Rate-limit status differs.")
        if primary.rate is not None:
            max_rate = max(max_rate, abs(float(primary.rate) - independent.rate))
            max_partition = max(
                max_partition,
                abs(float(primary.partition.centered_log_partition) - independent.partition),
            )
        max_hazard = max(max_hazard, abs(float(primary.partition.mean_hazard) - independent.hazard))
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
            max(max_weight, max_target, max_gap, max_rate, max_partition, max_hazard) <= TOLERANCE
            and np.isfinite(row).all()
            and (row >= 0).all()
            and abs(math.fsum(row) - 0.98) <= TOLERANCE,
            "Independent current-rate/gap/partition/forecast or target source proof failed.",
        )
        targets.loc[day] = row
        audit = {
            "decision_session": str(day.date()),
            "expert_weights": weights.tolist(),
            "complete_source_intervals": len(primary.history),
            "state": primary.snapshot(),
            "period_update": update,
            "source_native_known_at": [str(d.date()) for d in source.native_known_at],
            "source_loss_game_is_main_NET_wealth": False,
            "static_AdaHedge_regret_bound_claimed": False,
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
        targets,
        audits,
        {
            "source_decisions": len(audits),
            "complete_source_intervals": len(primary.history),
            "maximum_independent_forecast_difference": max_weight,
            "maximum_independent_source_gap_difference": max_gap,
            "maximum_independent_rate_difference": max_rate,
            "maximum_independent_centered_partition_difference": max_partition,
            "maximum_independent_hazard_mean_difference": max_hazard,
            "maximum_independent_rational_target_difference": max_target,
            "maximum_primary_mass_cells": max(
                a["state"]["persistent_primary_mass_cells"] for a in audits
            ),
            "all_source_inputs_matured_before_next_identity_forecast": True,
            "independent_forecast_polynomial_degree_covered": 2 * QUADRATURE - 1,
            "negative_gap_clipped_or_floored": False,
            "source_loss_gap_is_actual_main_wealth_or_bound": False,
            "old_learner_or_policy_replayed": False,
        },
    )


class ArchivedAdaptivePolicy:
    def __init__(self, prepared):
        self.targets = prepared.targets.dropna(how="all")
        self.audits = {row["decision_session"]: row for row in prepared.audits}
        require(
            not self.targets.empty
            and self.targets.index.is_unique
            and self.targets.index.is_monotonic_increasing
            and self.targets.index.strftime("%Y-%m-%d").tolist() == list(self.audits),
            "Preserve every actual source mixture and its matured learning audit.",
        )
        self.last = None

    def decide(self, capital):
        day = capital.as_of
        require(self.last is None or day > self.last, "No repeated original capital review.")
        known = self.targets.loc[:day]
        require(not known.empty, "A future adaptive mixture cannot seed earlier capital.")
        source_day = known.index[-1]
        target = known.iloc[-1].copy()
        require(
            target.index.equals(capital.asset_values.index), "Keep the fourteen underlying assets."
        )
        self.last = day
        audit = self.audits[str(source_day.date())]
        return monthly.AllocationDecision(
            target,
            {
                "source_master_known_at": str(source_day.date()),
                "source_expert_weights": audit["expert_weights"],
                "complete_source_intervals": audit["complete_source_intervals"],
                "source_learning_is_actual_main_capital": False,
                "capital_initialization_is_native_source_refresh": False,
            },
        )


@dataclass(frozen=True)
class AdaptiveRun:
    result: BacktestResult
    desired: pd.DataFrame
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict
    raw_kernel_auxiliary_gross: pd.Series


def run(data, prepared, start, end, *, cost_bps, delay, before_decision=None, retain=None):
    output = monthly.run(
        data,
        ArchivedAdaptivePolicy(prepared),
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
        "Canonical gross representation cannot change a financial primitive.",
    )
    return AdaptiveRun(
        BacktestResult(frame, output.result.weights),
        output.desired,
        output.issued,
        output.decisions,
        output.pending_after_window,
        raw,
    )


def verify_run(data, prepared, output, start, end, cost, delay, *, retain_comparison):
    require(callable(retain_comparison), "Retain complete independent cash/position operands.")
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
    report = comparison_report(output.result, replay)
    retain_comparison(output.result, replay, report)
    require(
        report["full_frame_equals"] and report["full_positions_equal"],
        "Adaptive switching main book differs from original cash-engine replay.",
    )
    dates = output.result.frame.index
    reviews = pd.DatetimeIndex([previous_session(dates[0]), *[d for d in dates if is_month_end(d)]])
    require(
        [r["signal_session"] for r in output.decisions] == reviews.strftime("%Y-%m-%d").tolist(),
        "Keep every original initial/monthly capital review.",
    )
    active, issued, pending = (
        prepared.targets.dropna(how="all"),
        output.issued.copy() * np.nan,
        None,
    )
    for i, (day, record) in enumerate(zip(reviews, output.decisions, strict=True)):
        if pending is not None and pending <= day:
            pending = None
        require(pending is None, "No replacement of the original pending request.")
        known = active.loc[:day]
        require(not known.empty, "No then-known adaptive source opinion at capital review.")
        target = known.iloc[-1].to_numpy()
        held = np.zeros(len(ASSETS)) if i == 0 else output.result.weights.loc[day].to_numpy()
        gap = float(abs(target - held).max())
        reason = (
            "initial_capital"
            if i == 0
            else ("observed_weight_gap" if gap + TOLERANCE >= 0.025 else "hold")
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        require(
            np.max(abs(output.desired.loc[day].to_numpy() - target)) <= TOLERANCE
            and record["assessment"]["source_master_known_at"] == str(known.index[-1].date())
            and record["reason"] == reason
            and record["execution_session"] == str(execution.date())
            and record["initial_capital"] is (i == 0)
            and abs(record["maximum_observed_target_gap"] - gap) <= TOLERANCE,
            "Independent original source seed,actual band or execution delay differs.",
        )
        if reason != "hold":
            pending = execution
            issued.loc[day] = target
    require(
        issued.equals(output.issued),
        "Adaptive source requests do not match the actual holding band.",
    )
    return {
        **monthly.verify_capital_inputs(output),
        "all_cash_financial_columns_and_positions_exact": True,
        "comparison_operands_retained_before_assertions": True,
        "original_source_seed_band_cost_delay_RF_verified": True,
        "shared_source_history_is_not_main_capital_history": True,
        "source_gap_rate_is_actual_main_wealth_or_bound": False,
        "broker_orders": 0,
    }
