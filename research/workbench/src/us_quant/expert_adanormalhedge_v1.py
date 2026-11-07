from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import expert_switching_loss_v1 as capital
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.protective_six_adaptive_hedge_v1 import EXPERT_IDS, MECHANISMS
from us_quant.retained_pair_equal_allocation_v1 import ASSETS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-adanormalhedge-v1.json"
ID = "expert_pool_six_adanormalhedge_v1"
STUDY = "ssd_expert_adanormalhedge_20261107_v1"
N, MAX_PERIODS, PRECISION, INDEPENDENT_PRECISION = 6, 119, 320, 160
MAX_TERMS, MAX_EXPONENT, TOLERANCE = 4096, Decimal(40), 1e-12
DecimalVector = tuple[Decimal, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantError(message)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 202229,
        "resumed_native_event_seq": 207389,
        "prior_journal_sequence": 1542,
        "prior_journal_head_sha256": (
            "efa86b900b0a476f517519a429a5604fd2c109e242b59d6e17534fb90da2c8a0"
        ),
        "prior_combined_completed": 307,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/expert-adanormalhedge-20261107-v1/"
            "source-inputs-v1.json",
            "sha256": "974ad12b3307f0ef91ee387d6a4092fa7834c161278d43c51e713eb264d77069",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "source_loss_labels": "unchanged_sealed_six_binary64_input_loss_labels",
        "algorithm": "AdaNormalHedge_original_fixed_expert_Algorithm1",
        "initial_expert_prior": "uniform_six",
        "instantaneous_regret": "prior_played_loss_minus_expert_loss",
        "cumulative_magnitude": "sum_absolute_instantaneous_regret",
        "potential": "exp_positive_R_squared_over_3C",
        "score": "half_shifted_potential_difference_R_plus_minus_1_C_plus_1",
        "all_zero_scores": "logged_uniform_source_prior_without_state_reset",
        "primary_decimal_precision": PRECISION,
        "independent_decimal_precision": INDEPENDENT_PRECISION,
        "independent_score": "positive_polynomial_difference_exponential_series",
        "maximum_series_terms": MAX_TERMS,
        "maximum_shifted_exponent": 40,
        "maximum_complete_source_months": MAX_PERIODS,
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
        "old_source_learner_or_account_replayed": False,
        "original_source_labels_or_normalization_changed": False,
        "source_regret_potential_is_actual_main_wealth_or_bound": False,
        "new_source_requests": 0,
        "new_shadow_accounts": 0,
        "new_RF_or_event_Adapter": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze original AdaNormalHedge Algorithm1 and unchanged source/financial contracts.",
    )


@dataclass(frozen=True)
class RegretState:
    intervals: int = 0
    regrets: DecimalVector = (Decimal(0),) * N
    magnitudes: DecimalVector = (Decimal(0),) * N
    played_loss: Decimal = Decimal(0)

    def check(self) -> None:
        require(
            type(self.intervals) is int
            and 0 <= self.intervals <= MAX_PERIODS
            and len(self.regrets) == len(self.magnitudes) == N
            and isinstance(self.played_loss, Decimal)
            and self.played_loss.is_finite()
            and 0 <= self.played_loss <= self.intervals
            and all(
                isinstance(r, Decimal)
                and isinstance(c, Decimal)
                and r.is_finite()
                and c.is_finite()
                and r.copy_abs() <= c <= self.intervals
                and r <= self.played_loss
                and 0 <= self.played_loss - r <= self.intervals
                for r, c in zip(self.regrets, self.magnitudes, strict=True)
            ),
            "Invalid source regret/magnitude state; do not reset, clip or invent a forecast.",
        )


def shifted_exponents(regret: Decimal, magnitude: Decimal) -> tuple[Decimal, Decimal]:
    require(
        regret.is_finite()
        and magnitude.is_finite()
        and regret.copy_abs() <= magnitude <= MAX_PERIODS,
        "Shifted potential requires a valid bounded source regret and magnitude.",
    )
    denominator = 3 * (magnitude + 1)
    high = max(regret + 1, Decimal(0)) ** 2 / denominator
    low = max(regret - 1, Decimal(0)) ** 2 / denominator
    require(0 <= low <= high <= MAX_EXPONENT, "Keep the original119-round exponent bound40.")
    return high, low


def positive_expm1(value: Decimal) -> Decimal:
    require(value.is_finite() and 0 <= value <= MAX_EXPONENT, "Invalid potential increment.")
    if value == 0:
        return Decimal(0)
    if value > Decimal("0.5"):
        return value.exp() - 1
    total = term = value
    tolerance = Decimal(1).scaleb(-(PRECISION - 10))
    for degree in range(1, MAX_TERMS):
        following = term * value / (degree + 1)
        tail = following / (1 - value / (degree + 2))
        if tail <= tolerance * total:
            return total
        total += following
        term = following
    raise QuantError("Primary positive potential series did not reach its declared precision.")


def potential_score(regret: Decimal, magnitude: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = PRECISION
        high, low = shifted_exponents(regret, magnitude)
        return low.exp() * positive_expm1(high - low) / 2


def independent_score(regret: Decimal, magnitude: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = INDEPENDENT_PRECISION
        high, low = shifted_exponents(regret, magnitude)
        difference = high - low
        if difference == 0:
            return Decimal(0)
        total, polynomial, high_power, factorial = Decimal(0), Decimal(1), Decimal(1), 1
        tolerance = Decimal(1).scaleb(-(INDEPENDENT_PRECISION - 10))
        for degree in range(1, MAX_TERMS + 1):
            total += difference * polynomial / factorial
            high_power *= high
            if degree + 1 > high:
                tail = difference * high_power / factorial / (1 - high / (degree + 1))
                if tail <= tolerance * total:
                    return total / 2
            polynomial = high_power + low * polynomial
            factorial *= degree + 1
        raise QuantError("Independent positive polynomial series exceeded its numerical limit.")


def source_forecast(
    state: RegretState, *, independent: bool = False
) -> tuple[DecimalVector, DecimalVector, bool]:
    with localcontext() as context:
        context.prec = INDEPENDENT_PRECISION if independent else PRECISION
        state.check()
        evaluator = independent_score if independent else potential_score
        scores = tuple(
            evaluator(r, c) for r, c in zip(state.regrets, state.magnitudes, strict=True)
        )
        require(
            all(value.is_finite() and value >= 0 for value in scores),
            "A failed potential cannot become an expert-exclusion or uniform remedy.",
        )
        normalizer = sum(scores)
        all_zero = normalizer == 0
        weights = (Decimal(1) / N,) * N if all_zero else tuple(v / normalizer for v in scores)
        require(
            all(v.is_finite() and 0 <= v <= 1 for v in weights)
            and abs(sum(weights) - 1) <= Decimal(1).scaleb(-(context.prec - 10)),
            "The source potential forecast must be a finite simplex.",
        )
        return weights, scores, all_zero


class AdaNormalHedge:
    def __init__(self, state: RegretState | None = None):
        self.state = RegretState() if state is None else state
        self.state.check()

    def predict(self) -> np.ndarray:
        weights, _, _ = source_forecast(self.state)
        result = np.array([float(v) for v in weights])
        require(
            np.isfinite(result).all()
            and (result >= 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "Keep the original source simplex without weight clipping or a floor.",
        )
        return result

    def snapshot(self) -> dict:
        weights, scores, all_zero = source_forecast(self.state)
        return {
            "complete_source_intervals": self.state.intervals,
            "cumulative_source_regret_decimal": [str(v) for v in self.state.regrets],
            "cumulative_absolute_source_regret_decimal": [str(v) for v in self.state.magnitudes],
            "cumulative_source_played_loss_decimal": str(self.state.played_loss),
            "source_potential_scores_decimal": [str(v) for v in scores],
            "source_forecast_decimal": [str(v) for v in weights],
            "all_zero_scores_use_frozen_uniform_prior": all_zero,
            "all_zero_scores_reset_source_state": False,
            "source_regret_potential_is_actual_main_wealth_or_bound": False,
            "weight_floor_or_regret_clipping": False,
        }

    def advance(self, values) -> dict:
        labels = capital.checked_labels(values)
        with localcontext() as context:
            context.prec = PRECISION
            self.state.check()
            require(self.state.intervals < MAX_PERIODS, "Keep original119complete source months.")
            weights, _, uniform = source_forecast(self.state)
            minimum = min(labels)
            played = minimum + sum(
                w * (loss - minimum) for w, loss in zip(weights, labels, strict=True)
            )
            regrets = tuple(played - loss for loss in labels)
            self.state = RegretState(
                self.state.intervals + 1,
                tuple(a + b for a, b in zip(self.state.regrets, regrets, strict=True)),
                tuple(
                    a + b.copy_abs() for a, b in zip(self.state.magnitudes, regrets, strict=True)
                ),
                self.state.played_loss + played,
            )
            self.state.check()
            return {
                "original_source_loss_labels_hex": [float(v).hex() for v in labels],
                "prequential_forecast_before_assimilation": [float(v) for v in weights],
                "prior_all_zero_scores_uniform_case": uniform,
                "source_played_loss_decimal": str(played),
                "instantaneous_source_regret_decimal": [str(v) for v in regrets],
                "absolute_instantaneous_source_regret_decimal": [
                    str(v.copy_abs()) for v in regrets
                ],
                "state_after": self.snapshot(),
                "source_regret_is_main_NET_performance": False,
            }


class IndependentAdaNormalHedge:
    def __init__(self):
        self.state = RegretState()

    @property
    def forecast(self) -> np.ndarray:
        weights, _, _ = source_forecast(self.state, independent=True)
        return np.array([float(v) for v in weights])

    def advance(self, values) -> None:
        capital.checked_labels(values)
        labels = tuple(Decimal.from_float(float(v)) for v in np.asarray(values))
        with localcontext() as context:
            context.prec = INDEPENDENT_PRECISION
            self.state.check()
            require(self.state.intervals < MAX_PERIODS, "Independent source horizon exceeded.")
            prediction, _, _ = source_forecast(self.state, independent=True)
            minimum = min(labels)
            played = minimum + sum(prediction[i] * (labels[i] - minimum) for i in range(N))
            regret, magnitude = [], []
            for i in range(N):
                increment = played - labels[i]
                regret.append(self.state.regrets[i] + increment)
                magnitude.append(self.state.magnitudes[i] + increment.copy_abs())
            self.state = RegretState(
                self.state.intervals + 1,
                tuple(regret),
                tuple(magnitude),
                self.state.played_loss + played,
            )
            self.state.check()


OpinionSnapshot = capital.OpinionSnapshot
PreparedSourceTargets = capital.PreparedSourceTargets


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
        "Keep the complete original calendar and genuine source initialization anchor.",
    )
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in calendar if d >= pd.Timestamp("2016-10-06") and is_month_end(d)],
    ]
    by_end = {row["end"]: row for row in periods}
    require(len(by_end) == len(periods), "No duplicated original source-loss month.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=ASSETS)
    primary, independent = AdaNormalHedge(), IndependentAdaNormalHedge()
    anchor, audits = None, []
    maxima = {"forecast": 0.0, "regret": 0.0, "magnitude": 0.0, "played": 0.0, "target": 0.0}
    for day in days:
        if before is not None:
            before()
        source = feed.observe(day)
        source.check(day)
        interval, labels = None, None
        if anchor is not None:
            require(str(day.date()) in by_end, "A matured original source-loss month is missing.")
            interval = by_end[str(day.date())]
            require(
                interval["start_after"] == str(anchor.date()),
                "Do not assimilate the partial initial source month.",
            )
            labels = np.array(
                [float(v) for v in capital.checked_labels(interval["input_loss_labels"])]
            )
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": str(day.date()),
                    "state": primary.snapshot(),
                    "forecast_before_new_labels": primary.predict().tolist(),
                    "source_native_known_at": [str(d.date()) for d in source.native_known_at],
                    "source_targets_hex": [[float(v).hex() for v in row] for row in source.targets],
                    "matured_source_interval": (
                        None
                        if interval is None
                        else {
                            "start_after": interval["start_after"],
                            "end": interval["end"],
                            "original_loss_labels_hex": [float(v).hex() for v in labels],
                        }
                    ),
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        update = None
        if labels is not None:
            update = primary.advance(labels)
            independent.advance(labels)
        weights = primary.predict()
        maxima["forecast"] = max(
            maxima["forecast"], float(abs(weights - independent.forecast).max())
        )
        for name, actual, other in (
            ("regret", primary.state.regrets, independent.state.regrets),
            ("magnitude", primary.state.magnitudes, independent.state.magnitudes),
        ):
            maxima[name] = max(
                maxima[name],
                max(abs(float(a) - float(b)) for a, b in zip(actual, other, strict=True)),
            )
        maxima["played"] = max(
            maxima["played"], abs(float(primary.state.played_loss - independent.state.played_loss))
        )
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
        maxima["target"] = max(maxima["target"], float(abs(row - rational).max()))
        require(
            max(maxima.values()) <= TOLERANCE
            and np.isfinite(row).all()
            and (row >= 0).all()
            and abs(math.fsum(row) - 0.98) <= TOLERANCE,
            "Independent source regret, magnitude, played loss, forecast or target differs.",
        )
        audit = {
            "decision_session": str(day.date()),
            "expert_weights": weights.tolist(),
            "complete_source_intervals": primary.state.intervals,
            "state": primary.snapshot(),
            "period_update": update,
            "source_native_known_at": [str(d.date()) for d in source.native_known_at],
            "source_regret_potential_is_actual_main_wealth_or_bound": False,
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
            "complete_source_intervals": primary.state.intervals,
            **{f"maximum_independent_{name}_difference": value for name, value in maxima.items()},
            "all_source_inputs_matured_before_next_forecast": True,
            "independent_score_form": "positive_polynomial_difference_exponential_series",
            "maximum_shifted_exponent": 40,
            "weight_floor_or_regret_clipping": False,
            "source_regret_potential_is_actual_main_wealth_or_bound": False,
            "old_learner_or_policy_replayed": False,
        },
    )


# Reuse only the unchanged capital Interface; neither prior source learner is invoked.
AdaptiveRun = capital.AdaptiveRun
run = capital.run
comparison_report = capital.comparison_report


def verify_run(data, prepared, output, start, end, cost, delay, *, retain_comparison):
    report = capital.verify_run(
        data, prepared, output, start, end, cost, delay, retain_comparison=retain_comparison
    )
    return {
        **{
            k: v for k, v in report.items() if k != "source_gap_rate_is_actual_main_wealth_or_bound"
        },
        "source_regret_potential_is_actual_main_wealth_or_bound": False,
    }
