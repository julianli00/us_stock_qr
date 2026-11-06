from __future__ import annotations

import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import protective_six_adaptive_hedge_v1 as prior
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/confidence-rated-prod-v1.json"
N, MAX_PERIODS, PRECISION, TOLERANCE = 6, 119, 160, 1e-12
ID = "expert_pool_six_confidence_rated_prod_v1"
CANDIDATES = ({"id": ID, "weighting": "confidence_rated_adaptive_multi_rate_Prod"},)
EXPERT_IDS, MECHANISMS, CLOCK = prior.EXPERT_IDS, prior.MECHANISMS, prior.CLOCK
PRIOR_HEAD = "d0a7e820e4d503d82b18f88652b7a17c99ee60b99e4665fe7879d5167fd6f08d"
observations, combine, risk_cost_audit = prior.observations, prior.combine, prior.risk_cost_audit


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_confidence_rated_prod_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 193715,
        "prior_journal_sequence": 1478,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 302,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/confidence-rated-prod-20261106-v1/"
            "source-inputs-v1.json",
            "sha256": "b54b4e6897ee93ee552de85dcd73609e5bb6229739bf3a8db83f70e02f515e2a",
        },
        "expert_ids": list(EXPERT_IDS),
        "initial_decision_session": "2016-10-05",
        "first_complete_month_anchor": "2016-10-31",
        "maximum_complete_learning_intervals": MAX_PERIODS,
        "learning_clock": "completed_full_exchange_month_intervals",
        "learning_memory": "all_expanding_completed_full_months",
        "initial_partial_source_month_assimilated": False,
        "learning_resets_for_five_year_account": False,
        "learning_rule": "Adapt_ML_Prod_Corollary4_with_Eq9_confidence_reduction",
        "fixed_prior": "uniform_six_raw_masses",
        "learning_rate": "min_half_sqrt_log6_over_one_plus_past_squared_confidence_excess",
        "raw_mass_update": (
            "old_mass_times_one_plus_old_rate_excess_then_new_rate_over_old_rate_power"
        ),
        "raw_mass_renormalized_after_update": False,
        "source_loss_labels": "sealed_original_six_binary64_bounded_month_losses_unchanged",
        "nonfeedback_confidence": 1.0,
        "feedback_confidence": "one_minus_original_own_shadow_protection_scale",
        "confidence_clock": "prior_month_origin_held_for_that_month_loss_update",
        "confidence_is_probability_or_source_admission": False,
        "prequential_validation_before_assimilation": True,
        "primary_decimal_precision": PRECISION,
        "independent_decimal_precision": 100,
        "independent_numeric_tolerance": TOLERANCE,
        "rate_or_window_or_budget_floor_sweep": False,
        "source_wealth_guarantee_transferred_to_main": False,
        "normalized_confidence_regret_is_main_wealth": False,
        "old_learner_or_source_or_control_reexecuted": False,
        "main_decision_clock": "unchanged_initial_then_true_month_ends",
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "admitted_legacy_defensive_views_reused_unchanged": True,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "global_pool_changed": False,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze one confidence-rated learner,source clock and unchanged financial contract.",
    )


def vector(value, confidence=False):
    rows = np.asarray(value)
    require(
        rows.shape == (N,)
        and rows.dtype.kind in "fi"
        and np.isfinite(rows).all()
        and (rows >= 0).all()
        and (rows <= 1).all()
        and (not confidence or (rows > 0).any()),
        "Use six finite [0,1] values and at least one explicitly active confidence.",
    )
    return [Decimal.from_float(float(v)) for v in rows]


def rates(squares):
    return [min(Decimal("0.5"), (Decimal(N).ln() / (1 + value)).sqrt()) for value in squares]


class ConfidenceProd:
    def __init__(self):
        with localcontext() as context:
            context.prec = PRECISION
            self.mass = [Decimal(1) / N] * N
        self.squares = [Decimal(0)] * N
        self.regrets = [Decimal(0)] * N
        self.periods = 0

    def check(self):
        require(
            type(self.periods) is int
            and 0 <= self.periods <= MAX_PERIODS
            and all(len(row) == N for row in (self.mass, self.squares, self.regrets))
            and all(isinstance(v, Decimal) and v.is_finite() and v > 0 for v in self.mass)
            and all(
                isinstance(v, Decimal) and v.is_finite() and 0 <= v <= self.periods
                for v in self.squares
            )
            and all(
                isinstance(v, Decimal) and v.is_finite() and abs(v) <= self.periods
                for v in self.regrets
            ),
            "Invalid Prod mass or excess-loss state cannot become neutral allocations.",
        )

    def _probabilities(self, confidence):
        self.check()
        rated = vector(confidence, confidence=True)
        mass = [
            c * eta * weight
            for c, eta, weight in zip(rated, rates(self.squares), self.mass, strict=True)
        ]
        total = sum(mass)
        require(
            total.is_finite() and total > 0, "No positive active mass;do not invent a fallback."
        )
        return [v / total for v in mass]

    def weights(self, confidence):
        with localcontext() as context:
            context.prec = PRECISION
            result = np.array([float(v) for v in self._probabilities(confidence)])
        active = np.asarray(confidence) > 0
        require(
            np.isfinite(result).all()
            and (result[active] > 0).all()
            and (result[~active] == 0).all()
            and abs(math.fsum(result) - 1) <= TOLERANCE,
            "Binary64 capital fractions must preserve active and sleeping source roles.",
        )
        return result

    def snapshot(self):
        self.check()
        with localcontext() as context:
            context.prec = PRECISION
            return {
                "complete_intervals": self.periods,
                "unnormalized_mass_decimal": [str(v) for v in self.mass],
                "squared_confidence_excess_decimal": [str(v) for v in self.squares],
                "confidence_regret_decimal": [str(v) for v in self.regrets],
                "learning_rates_decimal": [str(v) for v in rates(self.squares)],
                "arithmetic_precision": PRECISION,
            }

    def update(self, labels, prior_confidence):
        self.check()
        require(self.periods < MAX_PERIODS, "The original history contains at most119full months.")
        with localcontext() as context:
            context.prec = PRECISION
            losses, confidence = vector(labels), vector(prior_confidence, confidence=True)
            before = self._probabilities(prior_confidence)
            old_rates = rates(self.squares)
            played = sum(w * value for w, value in zip(before, losses, strict=True))
            excess = [c * (played - value) for c, value in zip(confidence, losses, strict=True)]
            squares = [old + value * value for old, value in zip(self.squares, excess, strict=True)]
            after_rates = rates(squares)
            factors = [1 + eta * value for eta, value in zip(old_rates, excess, strict=True)]
            require(
                all(Decimal("0.5") <= f <= Decimal("1.5") for f in factors)
                and all(new <= old for new, old in zip(after_rates, old_rates, strict=True)),
                "The original positive-factor and nonincreasing-rate conditions must hold.",
            )
            self.mass = [
                old
                if factor == 1 and new_rate == old_rate
                else ((old * factor).ln() * new_rate / old_rate).exp()
                for old, factor, new_rate, old_rate in zip(
                    self.mass, factors, after_rates, old_rates, strict=True
                )
            ]
            self.squares = squares
            self.regrets = [old + value for old, value in zip(self.regrets, excess, strict=True)]
            self.periods += 1
            constant = (
                3 * Decimal(N).ln()
                + (
                    1 + Decimal(N) / (2 * Decimal(1).exp()) * (1 + Decimal(self.periods + 1).ln())
                ).ln()
            )
            bounds = [
                constant * (Decimal(N).ln() * (1 + value)).sqrt() + 2 * constant
                for value in squares
            ]
            require(
                all(
                    r <= b + Decimal(str(TOLERANCE))
                    for r, b in zip(self.regrets, bounds, strict=True)
                ),
                "Confidence-game regret reference failed;it is not a main wealth bound.",
            )
            return {
                "input_loss_labels": [float(v) for v in losses],
                "confidence_held_from_prior_origin": [float(v) for v in confidence],
                "prequential_capital_weights_before_assimilation": [float(v) for v in before],
                "played_loss_decimal": str(played),
                "confidence_excess_decimal": [str(v) for v in excess],
                "raw_mass_multiplier_decimal": [str(v) for v in factors],
                "source_game_reference_bounds_decimal": [str(v) for v in bounds],
                "state_after": self.snapshot(),
                "confidence_regret_is_actual_main_performance": False,
            }


class OwnShadowConfidenceArchive:
    def __init__(self, equity):
        require(
            isinstance(equity, pd.Series)
            and not equity.empty
            and isinstance(equity.index, pd.DatetimeIndex)
            and equity.index.tz is None
            and equity.index.equals(sessions(equity.index[0], equity.index[-1]))
            and np.isfinite(equity).all()
            and (equity > 0).all(),
            "Confidence reads an existing complete positive own-shadow NAV archive.",
        )
        self.equity = equity.copy()
        self.anchor = previous_session(equity.index[0])

    def observe(self, day):
        day = pd.Timestamp(day)
        require(
            day == self.anchor or day in self.equity.index,
            "Missing or future confidence wealth cannot be carried or fabricated.",
        )
        if day == self.anchor:
            nav = peak = 10000.0
            count = 0
        else:
            known = self.equity.loc[:day]
            nav, peak, count = float(known.iloc[-1]), max(10000.0, float(known.max())), len(known)
        return {
            "as_of": str(day.date()),
            "own_shadow_nav_hex": nav.hex(),
            "own_shadow_peak_hex": peak.hex(),
            "past_saved_return_rows": count,
            "initial_source_wealth_seed": day == self.anchor,
            "main_wealth_or_future_shadow_used": False,
        }


def confidences(wealth):
    nav, peak = (
        float.fromhex(wealth["own_shadow_nav_hex"]),
        float.fromhex(wealth["own_shadow_peak_hex"]),
    )
    require(
        math.isfinite(nav)
        and math.isfinite(peak)
        and 0 < nav <= peak
        and peak >= 10000
        and wealth["main_wealth_or_future_shadow_used"] is False,
        "Only the expert's own known net NAV and original prefix peak define engagement.",
    )
    scale = float(np.clip((nav - 0.85 * peak) / (0.15 * nav), 0, 1))
    return np.array([1.0] * (N - 1) + [1.0 - scale])


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    require(
        isinstance(calendar, pd.DatetimeIndex)
        and not calendar.empty
        and calendar.equals(sessions(calendar[0], calendar[-1]))
        and initial in calendar,
        "Keep the complete original exchange-calendar and actual initial source anchor.",
    )
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def build_targets(
    experts,
    calendar,
    assets,
    policy,
    cached_periods,
    confidence_archive,
    before_decision=None,
    retain=None,
):
    dates = decision_sessions(calendar, policy)
    require(
        len(experts) == N and len(set(assets)) == len(assets), "Keep six complete expert policies."
    )
    cached = {p["end"]: p for p in cached_periods}
    require(len(cached) == len(cached_periods), "A cached source month cannot occur twice.")
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    state, anchor, previous_confidence = ConfidenceProd(), None, None
    previous_weights = np.full(N, 1 / N)
    audits, updates = [], []
    first = pd.Timestamp(policy["first_complete_month_anchor"])
    for day in dates:
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        require(
            all(tuple(o.target.index) == tuple(assets) for o in observed),
            "Each known source opinion must retain every admitted asset coordinate.",
        )
        key, wealth = str(day.date()), confidence_archive.observe(day)
        require(wealth["as_of"] == key, "Confidence cannot acquire an end-month lookahead clock.")
        labels, logs, period_dates = None, None, pd.DatetimeIndex([])
        if anchor is not None:
            period_dates = sessions(next_session(anchor), day)
            require(key in cached, "A complete source month is missing;do not fill its loss.")
            saved = cached[key]
            require(
                saved["start_after"] == str(anchor.date()),
                "Keep exact complete source-month edges.",
            )
            logs, checked_dates = prior.period_logs(observed, anchor, day)
            labels = prior.checked_losses(saved["input_loss_labels"])
            require(
                checked_dates.equals(period_dates)
                and [float(v).hex() for v in logs]
                == [float(v).hex() for v in saved["canonical_log_relatives"]]
                and [float(v).hex() for v in labels]
                == [float(v).hex() for v in prior.bounded_losses(logs)],
                "Reuse unchanged sealed labels and exact native NET months,not an old learner.",
            )
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": key,
                    "state": state.snapshot(),
                    "confidence_wealth_input": wealth,
                    "previous_confidences_hex": None
                    if previous_confidence is None
                    else [float(v).hex() for v in previous_confidence],
                    "known_target_dates": [str(o.target_known_at.date()) for o in observed],
                    "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
                    "matured_period_dates": period_dates.strftime("%Y-%m-%d").tolist(),
                    "matured_NET_returns_hex": [
                        [float(v).hex() for v in o.net_returns.loc[period_dates]] for o in observed
                    ],
                    "canonical_RF_hex": [
                        float(v).hex() for v in observed[0].risk_free.loc[period_dates]
                    ],
                    "cached_loss_labels_hex": None
                    if labels is None
                    else [float(v).hex() for v in labels],
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        current_confidence = confidences(wealth)
        update = None
        if labels is not None:
            require(
                previous_confidence is not None, "A new label needs its actual prior confidence."
            )
            update = {
                "start_after": str(anchor.date()),
                "end": key,
                "canonical_log_relatives": logs.tolist(),
                **state.update(labels, previous_confidence),
            }
            updates.append(update)
        weights = state.weights(current_confidence)
        row = combine(observed, weights)
        target.loc[day] = row
        audit = {
            "decision_session": key,
            "expert_weights": weights.tolist(),
            "previous_expert_weights": previous_weights.tolist(),
            "confidence_wealth_input": wealth,
            "current_confidences": current_confidence.tolist(),
            "complete_learning_intervals": state.periods,
            "score_mature": state.periods > 0,
            "learner_state": state.snapshot(),
            "period_update": update,
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "risk_and_cost": risk_cost_audit(observed, weights, previous_weights, first, day),
            "confidence_is_probability_or_technical_source_eligibility": False,
            "source_wealth_guarantee_transferred_to_main": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "phase": "after_decision",
                    "decision_session": key,
                    "audit": audit,
                    "unified_target_hex": [float(v).hex() for v in row],
                }
            )
        if is_month_end(day):
            anchor = day
        previous_confidence, previous_weights = current_confidence, weights
    return target, audits, updates


def independent_trace(history):
    results = []
    with localcontext() as context:
        context.prec = 100
        mass = [Decimal(1) / N] * N
        accumulated = [Decimal(0)] * N
        regrets = [Decimal(0)] * N
        for labels, confidence in history:
            losses = [Decimal.from_float(float(v)) for v in labels]
            rated = [Decimal.from_float(float(v)) for v in confidence]
            eta = [
                min(Decimal(1) / 2, (Decimal(N).ln() / (1 + value)).sqrt()) for value in accumulated
            ]
            total = sum(e * w for e, w in zip(eta, mass, strict=True))
            standard = [e * w / total for e, w in zip(eta, mass, strict=True)]
            active = sum(c * w for c, w in zip(rated, standard, strict=True))
            require(active > 0, "The independent confidence reduction has no active mass.")
            weights = [c * w / active for c, w in zip(rated, standard, strict=True)]
            played = sum(w * value for w, value in zip(weights, losses, strict=True))
            modified = [
                c * value + (1 - c) * played for c, value in zip(rated, losses, strict=True)
            ]
            reduced_loss = sum(w * value for w, value in zip(standard, modified, strict=True))
            excess = [played - value for value in modified]
            reduction_error = max(
                abs(reduced_loss - played),
                *[
                    abs(r - c * (played - value))
                    for r, c, value in zip(excess, rated, losses, strict=True)
                ],
            )
            require(
                reduction_error <= Decimal("1e-80"),
                "The generic modified-loss reduction must retain the confidence game.",
            )
            accumulated = [old + value**2 for old, value in zip(accumulated, excess, strict=True)]
            regrets = [old + value for old, value in zip(regrets, excess, strict=True)]
            after = [
                min(Decimal(1) / 2, (Decimal(N).ln() / (1 + value)).sqrt()) for value in accumulated
            ]
            mass = [
                old
                if value == 0 and new_rate == old_rate
                else context.power(old * (1 + old_rate * value), new_rate / old_rate)
                for old, value, new_rate, old_rate in zip(mass, excess, after, eta, strict=True)
            ]
            results.append(
                {
                    "weights_before": [float(v) for v in weights],
                    "played_loss": float(played),
                    "modified_losses": [float(v) for v in modified],
                    "confidence_excess": [float(v) for v in excess],
                    "mass_after_decimal": [str(v) for v in mass],
                    "squares_after_decimal": [str(v) for v in accumulated],
                    "regrets_after_decimal": [str(v) for v in regrets],
                    "rates_after": [float(v) for v in after],
                    "confidence_reduction_error": float(reduction_error),
                }
            )
    return results
