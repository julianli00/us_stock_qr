from __future__ import annotations

import math
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import expert_adanormalhedge_v1 as scalar
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-dyadic-meta-v1.json"
ID = "expert_pool_six_dyadic_interval_meta_v1"
STUDY = "ssd_expert_dyadic_meta_20261108_v1"
N, MAX_PERIODS, PRECISION, INDEPENDENT_PRECISION = 6, 119, 320, 160
LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128)
TOLERANCE = 1e-12
EXPERT_IDS, MECHANISMS, ASSETS = scalar.EXPERT_IDS, scalar.MECHANISMS, scalar.ASSETS
OpinionSnapshot, PreparedSourceTargets = scalar.OpinionSnapshot, scalar.PreparedSourceTargets


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "actual_native_event_seq": 211975,
        "prior_journal_sequence": 1558,
        "prior_journal_head_sha256": (
            "0e63b8fc8d86baca7259c82112628c4b1b15913bc8394c346600f3fb64973675"
        ),
        "prior_combined_completed": 308,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/expert-dyadic-meta-20261108-v1/"
            "source-inputs-v1.json",
            "sha256": "339f098b4b820e6de40276374043db6452bb5b3e73d39c57fae1e8b70e3b30be",
        },
        "interval_lengths": list(LENGTHS),
        "interval_start": "1_plus_floor_completed_rounds_over_length_times_length",
        "base_learner": "six_source_exponential_loss_weights",
        "base_learning_rate": "sqrt_8_log_six_over_declared_interval_length",
        "meta_score": "original_shifted_regret_potential_scalar_times_interval_prior",
        "interval_prior": "one_over_start_startplus1_length_lengthplus1",
        "all_zero_meta_scores": "explicit_normalized_active_priors_without_state_reset",
        "source_loss_labels": "unchanged_sealed_six_binary64_input_loss_labels",
        "primary_decimal_precision": PRECISION,
        "independent_decimal_precision": INDEPENDENT_PRECISION,
        "independent_method": "multiplicative_base_evidence_and_recomputed_epoch_regrets",
        "maximum_complete_source_months": MAX_PERIODS,
        "maximum_active_meta_learners": len(LENGTHS),
        "independent_coherence": TOLERANCE,
        "initial_partial_month_assimilated": False,
        "shared_global_source_learner": True,
        "five_year_source_learning_state_reset": False,
        "allocation_clock": "original_initial_then_true_NYSE_month_ends",
        "capital_seed": "last_known_new_meta_target_not_native_source_refresh",
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_band": 0.025,
        "prefix_positions": [125, 503, 1006, 1509, 2012],
        "candidate_count": 1,
        "study_size_is_total_research_quota": False,
        "auxiliary_interval_learners_are_strategy_experts_or_independent_trials": False,
        "faithful_Daniely_SAOL_algorithm_or_regret_guarantee": False,
        "source_game_guarantee_is_actual_main_wealth_or_drawdown_bound": False,
        "failed_scenario_V1_repaired_or_replayed": False,
        "old_source_learner_or_account_replayed": False,
        "new_source_requests": 0,
        "new_shadow_accounts": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze one interval-calendar law, not a post-outcome rate or horizon sweep.",
    )


def active(completed):
    require(type(completed) is int and 0 <= completed <= MAX_PERIODS, "Invalid source clock.")
    return tuple((1 + completed // length * length, length) for length in LENGTHS)


def rate(length):
    require(type(length) is int and length in LENGTHS, "Use the declared dyadic lengths.")
    return (8 * Decimal(N).ln() / length).sqrt()


def checked_labels(values):
    array = np.asarray(values)
    require(
        array.shape == (N,)
        and array.dtype.kind in "iuf"
        and np.isfinite(array).all()
        and not any(isinstance(v, bool | np.bool_) for v in np.asarray(values, dtype=object).flat)
        and (array >= 0).all()
        and (array <= 1).all(),
        "Use six finite bounded source numbers without boolean or text coercion.",
    )
    return scalar.capital.checked_labels(array)


@dataclass(frozen=True)
class Epoch:
    start: int
    length: int
    age: int = 0
    losses: tuple[Decimal, ...] = (Decimal(0),) * N
    regret: Decimal = Decimal(0)
    magnitude: Decimal = Decimal(0)

    def check(self):
        require(
            type(self.start) is int
            and self.start >= 1
            and self.length in LENGTHS
            and type(self.age) is int
            and 0 <= self.age <= self.length
            and len(self.losses) == N
            and all(v.is_finite() and 0 <= v <= self.age for v in self.losses)
            and self.regret.is_finite()
            and self.magnitude.is_finite()
            and self.regret.copy_abs() <= self.magnitude <= self.age <= MAX_PERIODS,
            "An epoch must retain its own bounded matured losses and signed regret.",
        )

    @property
    def prior(self):
        return Decimal(1) / (self.start * (self.start + 1) * self.length * (self.length + 1))

    def forecast(self):
        self.check()
        minimum = min(self.losses)
        eta = rate(self.length)
        mass = tuple((-eta * (v - minimum)).exp() for v in self.losses)
        total = sum(mass)
        require(total.is_finite() and total > 0, "Invalid base evidence is not uniform success.")
        return tuple(v / total for v in mass)

    def record(self):
        return {
            "start_round": self.start,
            "declared_length": self.length,
            "end_round": self.start + self.length - 1,
            "matured_rounds": self.age,
            "prior_decimal": str(self.prior),
            "base_rate_decimal": str(rate(self.length)),
            "source_cumulative_losses_decimal": [str(v) for v in self.losses],
            "regret_decimal": str(self.regret),
            "absolute_regret_decimal": str(self.magnitude),
            "base_forecast_decimal": [str(v) for v in self.forecast()],
        }


def mixture(epochs, *, independent=False):
    evaluator = scalar.independent_score if independent else scalar.potential_score
    scores = tuple(e.prior * evaluator(e.regret, e.magnitude) for e in epochs)
    require(all(v.is_finite() and v >= 0 for v in scores), "Invalid meta potential.")
    normalizer = sum(scores)
    all_zero = normalizer == 0
    if all_zero:
        scores = tuple(e.prior for e in epochs)
        normalizer = sum(scores)
    meta = tuple(v / normalizer for v in scores)
    forecasts = [e.forecast() for e in epochs]
    source = tuple(sum(meta[j] * q[i] for j, q in enumerate(forecasts)) for i in range(N))
    require(
        all(v.is_finite() and 0 <= v <= 1 for v in source)
        and abs(sum(source) - 1) <= Decimal(1).scaleb(-140),
        "The interval mixture must conserve six-source mass without clipping.",
    )
    return source, meta, all_zero


class DyadicLearner:
    def __init__(self):
        self.periods = 0
        self.epochs = {key: Epoch(*key) for key in active(0)}
        self.played_loss = Decimal(0)
        self.retired = 0

    def check(self):
        require(set(self.epochs) == set(active(self.periods)), "Epoch calendar was reset.")
        for key, epoch in self.epochs.items():
            epoch.check()
            require(epoch.age == self.periods - key[0] + 1, "An unmatured label was consumed.")
        require(
            self.played_loss.is_finite() and 0 <= self.played_loss <= self.periods,
            "Source-game loss is bounded, not main capital.",
        )

    def prediction(self):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            self.check()
            return mixture([self.epochs[k] for k in active(self.periods)])

    def predict(self):
        weights, _, _ = self.prediction()
        result = np.array([float(v) for v in weights])
        require(
            np.isfinite(result).all()
            and (result >= 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "Binary64 source mass is invalid; do not normalize a failed forecast.",
        )
        return result

    def snapshot(self):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            source, meta, zero = self.prediction()
            return {
                "complete_source_intervals": self.periods,
                "active_intervals": [self.epochs[k].record() for k in active(self.periods)],
                "meta_forecast_decimal": [str(v) for v in meta],
                "source_forecast_decimal": [str(v) for v in source],
                "all_zero_meta_scores_use_active_priors": zero,
                "all_zero_scores_reset_state": False,
                "retired_interval_count": self.retired,
                "cumulative_played_source_loss_decimal": str(self.played_loss),
                "auxiliary_learners_are_complete_strategy_experts": False,
                "source_game_is_actual_main_wealth_or_bound": False,
            }

    def advance(self, values):
        labels = checked_labels(values)
        require(self.periods < MAX_PERIODS, "Keep the original finite matured source stream.")
        with localcontext() as ctx:
            ctx.prec = PRECISION
            weights, _, _ = self.prediction()
            minimum = min(labels)
            played = minimum + sum(w * (v - minimum) for w, v in zip(weights, labels, strict=True))
            expired, updated = [], {}
            for key in active(self.periods):
                epoch = self.epochs[key]
                q = epoch.forecast()
                base_loss = minimum + sum(w * (v - minimum) for w, v in zip(q, labels, strict=True))
                delta = played - base_loss
                following = replace(
                    epoch,
                    age=epoch.age + 1,
                    losses=tuple(a + b for a, b in zip(epoch.losses, labels, strict=True)),
                    regret=epoch.regret + delta,
                    magnitude=epoch.magnitude + delta.copy_abs(),
                )
                following.check()
                if following.age == following.length:
                    expired.append(following.record())
                else:
                    updated[key] = following
            self.periods += 1
            self.retired += len(expired)
            self.played_loss += played
            self.epochs = {
                key: updated[key] if key in updated else Epoch(*key) for key in active(self.periods)
            }
            self.check()
            return {
                "original_source_labels_hex": [float(v).hex() for v in labels],
                "forecast_before_labels": [float(v) for v in weights],
                "played_source_loss_decimal": str(played),
                "retired_intervals": expired,
                "state_after": self.snapshot(),
                "original_source_clock_or_labels_changed": False,
            }


class RecomputedOracle:
    def __init__(self):
        self.labels = []
        self.played = []
        self.cache = {}

    def _base(self, start, length, stop):
        require(start - 1 <= stop <= len(self.labels), "Independent base uses future labels.")
        key = (start, length, stop)
        if key not in self.cache:
            if stop == start - 1:
                value = (Decimal(1) / N,) * N
            else:
                previous = self._base(start, length, stop - 1)
                eta = rate(length)
                mass = tuple(
                    w * (-eta * v).exp()
                    for w, v in zip(previous, self.labels[stop - 1], strict=True)
                )
                total = sum(mass)
                require(total.is_finite() and total > 0, "Independent evidence failed.")
                value = tuple(v / total for v in mass)
            self.cache[key] = value
        return self.cache[key]

    def epochs(self):
        result = []
        for start, length in active(len(self.labels)):
            losses = tuple(
                sum((v[i] for v in self.labels[start - 1 :]), Decimal(0)) for i in range(N)
            )
            deltas = []
            for stop in range(start - 1, len(self.labels)):
                q, labels = self._base(start, length, stop), self.labels[stop]
                minimum = min(labels)
                own = minimum + sum(q[i] * (labels[i] - minimum) for i in range(N))
                deltas.append(self.played[stop] - own)
            result.append(
                Epoch(
                    start,
                    length,
                    len(self.labels) - start + 1,
                    losses,
                    sum(deltas, Decimal(0)),
                    sum((v.copy_abs() for v in deltas), Decimal(0)),
                )
            )
        return result

    def forecast(self):
        with localcontext() as ctx:
            ctx.prec = INDEPENDENT_PRECISION
            epochs = self.epochs()
            evaluator = scalar.independent_score
            scores = [e.prior * evaluator(e.regret, e.magnitude) for e in epochs]
            if sum(scores) == 0:
                scores = [e.prior for e in epochs]
            total = sum(scores)
            meta = [v / total for v in scores]
            source = tuple(
                sum(
                    meta[j] * self._base(e.start, e.length, len(self.labels))[i]
                    for j, e in enumerate(epochs)
                )
                for i in range(N)
            )
            return source

    def predict(self):
        return np.array([float(v) for v in self.forecast()])

    def advance(self, values):
        checked = checked_labels(values)
        with localcontext() as ctx:
            ctx.prec = INDEPENDENT_PRECISION
            prediction = self.forecast()
            labels = tuple(Decimal.from_float(float(v)) for v in checked)
            minimum = min(labels)
            played = minimum + sum(prediction[i] * (labels[i] - minimum) for i in range(N))
            self.played.append(played)
            self.labels.append(labels)


class OpinionFeed(Protocol):
    def observe(self, day: pd.Timestamp) -> OpinionSnapshot: ...


def build_targets(calendar, feed: OpinionFeed, periods, *, before=None, retain=None):
    require(
        isinstance(calendar, pd.DatetimeIndex)
        and len(calendar) > 0
        and calendar.tz is None
        and calendar.is_unique
        and calendar.is_monotonic_increasing
        and calendar.equals(sessions(calendar[0], calendar[-1]))
        and pd.Timestamp("2016-10-05") in calendar,
        "Keep every original session and the genuine source initialization anchor.",
    )
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in calendar if d >= pd.Timestamp("2016-10-06") and is_month_end(d)],
    ]
    by_end = {row["end"]: row for row in periods}
    require(len(by_end) == len(periods), "Original matured months cannot repeat.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=ASSETS)
    primary, oracle = DyadicLearner(), RecomputedOracle()
    anchor, audits = None, []
    maxima = {"forecast": 0.0, "regret": 0.0, "magnitude": 0.0, "played": 0.0, "target": 0.0}
    for day in days:
        if before is not None:
            before()
        source = feed.observe(day)
        source.check(day)
        interval, labels = None, None
        if anchor is not None:
            require(str(day.date()) in by_end, "A full original matured source month is missing.")
            interval = by_end[str(day.date())]
            require(interval["start_after"] == str(anchor.date()), "No partial-month learning.")
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
            oracle.advance(labels)
        weights = primary.predict()
        maxima["forecast"] = max(maxima["forecast"], float(abs(weights - oracle.predict()).max()))
        with localcontext() as ctx:
            ctx.prec = INDEPENDENT_PRECISION
            for epoch in oracle.epochs():
                actual = primary.epochs[epoch.start, epoch.length]
                maxima["regret"] = max(
                    maxima["regret"], float((actual.regret - epoch.regret).copy_abs())
                )
                maxima["magnitude"] = max(
                    maxima["magnitude"], float((actual.magnitude - epoch.magnitude).copy_abs())
                )
            maxima["played"] = max(
                maxima["played"], float((primary.played_loss - sum(oracle.played)).copy_abs())
            )
        row = weights @ source.targets
        rational = np.array(
            [
                float(
                    sum(
                        Fraction.from_float(float(weights[i]))
                        * Fraction.from_float(float(source.targets[i, j]))
                        for i in range(N)
                    )
                )
                for j in range(len(ASSETS))
            ]
        )
        maxima["target"] = max(maxima["target"], float(abs(row - rational).max()))
        require(
            max(maxima.values()) <= TOLERANCE
            and np.isfinite(row).all()
            and (row >= 0).all()
            and abs(math.fsum(row) - 0.98) <= TOLERANCE,
            "Independent epoch, source-weight or rational asset-target acceptance failed.",
        )
        audit = {
            "decision_session": str(day.date()),
            "expert_weights": weights.tolist(),
            "complete_source_intervals": primary.periods,
            "state": primary.snapshot(),
            "period_update": update,
            "source_native_known_at": [str(d.date()) for d in source.native_known_at],
            "source_game_is_actual_main_wealth_or_bound": False,
        }
        targets.loc[day] = row
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
            "complete_source_intervals": primary.periods,
            **{f"maximum_independent_{k}_difference": v for k, v in maxima.items()},
            "maximum_active_meta_learners": len(LENGTHS),
            "retired_auxiliary_interval_learners": primary.retired,
            "all_source_inputs_matured_before_next_forecast": True,
            "auxiliary_interval_learners_are_independent_economic_trials": False,
            "source_game_score_is_actual_main_net_wealth": False,
            "old_learner_or_policy_replayed": False,
        },
    )


# Only the original capital Interface is reused, not either completed source learner.
run = scalar.capital.run
AdaptiveRun = scalar.capital.AdaptiveRun


def verify_run(data, prepared, output, start, end, cost, delay, *, retain_comparison):
    report = scalar.capital.verify_run(
        data, prepared, output, start, end, cost, delay, retain_comparison=retain_comparison
    )
    return {
        **{
            k: v for k, v in report.items() if k != "source_gap_rate_is_actual_main_wealth_or_bound"
        },
        "dyadic_source_game_is_actual_main_wealth_or_bound": False,
    }
