from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from us_quant import benchmark_opportunity_quintet_v1 as pool
from us_quant import expert_net_wealth_mixture_v1 as calendar_source
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quintet-anytime-hedge-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = pool.EXPERT_IDS, pool.MECHANISMS, pool.CLOCK
ID = "expert_pool_quintet_anytime_hedge_v1"
CANDIDATES = ({"id": ID, "weighting": "bounded_mature_month_decreasing_hedge"},)
PRIOR_HEAD = "e50d5c627e252346930fc126c0a454494522a2b4a923a6e1fb9ad258cd573157"
N, MAX_PERIODS, TOLERANCE = 5, 119, 1e-12
combine, risk_cost_audit = pool.combine, calendar_source.risk_cost_audit


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_quintet_anytime_hedge_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 187094,
        "prior_journal_sequence": 1416,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 297,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-anytime-hedge-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "c4ff0d85b83a938f666e3af3b9160604acee286d1994076e25d74046834bccb7",
        },
        "academic_reference": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-anytime-hedge-20261106-v1/"
                "academic-reference-v1.json"
            ),
            "sha256": "2518e369b0237b0f0e2520e39c1fd4e7bb432cf1c6f6efba675dc58d4ef6755d",
        },
        "original_source_slots": [4, 5, 7, 0, 1],
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "first_complete_month_anchor": "2016-10-31",
        "source_history_delivery_sessions": 2512,
        "maximum_complete_learning_intervals": MAX_PERIODS,
        "learning_clock": "completed_full_exchange_month_intervals",
        "learning_memory": "all_expanding_completed_full_months",
        "initial_partial_source_month_assimilated": False,
        "learning_resets_for_five_year_account": False,
        "canonical_period_log": "math_fsum_of_math_log1p_recorded_source_NET_returns",
        "bounded_loss": "exact_binary64_log_fraction_from_best_to_worst",
        "equal_log_relative_month_loss": "all_zero",
        "neutral_month_advances_learning_round": True,
        "learning_rate_coefficient": 2,
        "learning_rate_round": "mature_months_plus_one",
        "learning_rate": "2_sqrt_log5_over_learning_round",
        "fixed_prior": [0.2] * N,
        "prequential_validation_before_assimilation": True,
        "state_loss_coordinates": N,
        "independent_decimal_precision": 80,
        "independent_numeric_tolerance": TOLERANCE,
        "incremental_switch_proxy_affects_allocation": False,
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": TOLERANCE,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "decision_information_clock": CLOCK,
        "admitted_legacy_defensive_views_reused_unchanged": True,
        "all_source_native_RF_claimed_equal_without_view": False,
        "mechanism_labels_are_independent_risks": False,
        "risk_or_wealth_guarantee_transferred": False,
        "normalized_regret_is_actual_main_wealth": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != fixed or any(type(policy[k]) is not type(v) for k, v in fixed.items()):
        raise QuantError("Freeze one fixed-pool bounded-loss learner and unchanged capital rules.")


def bounded_losses(log_relatives):
    values = np.asarray(log_relatives)
    if values.shape != (N,) or not np.isfinite(values).all():
        raise QuantError("Five finite canonical matured log relatives are required.")
    exact = [Fraction.from_float(float(value)) for value in values]
    highest, lowest = max(exact), min(exact)
    if highest == lowest:
        return np.zeros(N)
    return np.array([float((highest - value) / (highest - lowest)) for value in exact])


class AnytimeHedge:
    def __init__(self):
        self.periods = 0
        self.loss = np.zeros(N)
        self.played_loss = 0.0

    def weights(self):
        if (
            not 0 <= self.periods <= MAX_PERIODS
            or self.loss.shape != (N,)
            or not np.isfinite(self.loss).all()
            or (self.loss < 0).any()
            or (self.loss > self.periods).any()
            or not math.isfinite(self.played_loss)
        ):
            raise QuantError("Invalid bounded cumulative state cannot become neutral weights.")
        eta = 2 * math.sqrt(math.log(N) / (self.periods + 1))
        baseline = float(self.loss.min())
        mass = [math.exp(-eta * math.fsum((float(value), -baseline))) for value in self.loss]
        normalizer = math.fsum(mass)
        weights = np.array([value / normalizer for value in mass])
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise QuantError("Five positive finite Hedge fractions are required.")
        return weights

    def update(self, log_relatives):
        if self.periods >= MAX_PERIODS:
            raise QuantError("The original full-month history contains at most119intervals.")
        before = self.weights()
        losses = bounded_losses(log_relatives)
        current_loss = math.fsum(
            float(w) * float(loss) for w, loss in zip(before, losses, strict=True)
        )
        self.played_loss = math.fsum((self.played_loss, current_loss))
        self.loss = np.array(
            [
                math.fsum((float(old), float(new)))
                for old, new in zip(self.loss, losses, strict=True)
            ]
        )
        self.periods += 1
        regret = math.fsum((self.played_loss, -float(self.loss.min())))
        reference_bound = math.sqrt(self.periods * math.log(N))
        if regret > reference_bound + TOLERANCE:
            raise QuantError("The bounded-loss learner's own reference regret identity failed.")
        return self.weights(), {
            "losses": losses.tolist(),
            "prequential_weights_before_assimilation": before.tolist(),
            "prequential_bounded_loss": current_loss,
            "cumulative_expert_losses": self.loss.tolist(),
            "cumulative_played_bounded_loss": self.played_loss,
            "bounded_loss_regret": regret,
            "bounded_loss_reference_bound": reference_bound,
            "learning_round_after_assimilation": self.periods + 1,
            "learning_rate_after_assimilation": 2 * math.sqrt(math.log(N) / (self.periods + 1)),
            "source_loss_or_regret_is_actual_main_performance": False,
        }


def independent_weights(history):
    values = np.asarray(history)
    if values.size == 0:
        return np.full(N, 1 / N)
    if values.ndim != 2 or values.shape[1] != N or len(values) > MAX_PERIODS:
        raise QuantError("Independent weights require the exact bounded five-source history.")
    if not np.isfinite(values).all():
        raise QuantError("Invalid history cannot produce independent neutral fractions.")
    losses = []
    with localcontext() as context:
        context.prec = 80
        for row in values:
            exact = [Decimal.from_float(float(value)) for value in row]
            maximum, minimum = max(exact), min(exact)
            losses.append(
                [
                    0.0 if maximum == minimum else float((maximum - value) / (maximum - minimum))
                    for value in exact
                ]
            )
        cumulative = [math.fsum(row[i] for row in losses) for i in range(N)]
        eta = Decimal(2) * (Decimal(N).ln() / Decimal(len(values) + 1)).sqrt()
        baseline = min(cumulative)
        mass = [(-eta * Decimal.from_float(value - baseline)).exp() for value in cumulative]
        total = sum(mass)
        return np.array([float(value / total) for value in mass])


def observations(experts, day):
    return FunctionType(
        calendar_source.observations.__code__,
        {**calendar_source.observations.__globals__, **globals()},
        argdefs=calendar_source.observations.__defaults__,
    )(experts, day)


def period_logs(observed, first, last):
    dates = sessions(next_session(first), last)
    values = []
    for observation in observed:
        rows = observation.net_returns.loc[dates]
        if not rows.index.equals(dates) or not np.isfinite(rows).all() or (rows <= -1).any():
            raise QuantError("A missing matured source month cannot be cropped or filled.")
        values.append(math.fsum(math.log1p(float(value)) for value in rows))
    return np.array(values), dates


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the original initial opinion and full exchange-calendar rows.")
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def build_targets(experts, calendar, assets, policy, before_decision=None, retain=None):
    dates = decision_sessions(calendar, policy)
    if len(experts) != N or len(set(assets)) != len(assets):
        raise QuantError("Keep exactly five complete policies and unique asset coordinates.")
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    state, previous = AnytimeHedge(), np.full(N, 1 / N)
    anchor = first_anchor = None
    audits, periods = [], []
    for day in dates:
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(o.target.index) != tuple(assets) for o in observed):
            raise QuantError("Current known recommendations must use the exact admitted assets.")
        logs, period_dates = None, pd.DatetimeIndex([])
        if is_month_end(day) and anchor is not None:
            logs, period_dates = period_logs(observed, anchor, day)
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": str(day.date()),
                    "mature_periods_before": state.periods,
                    "cumulative_losses_hex": [float(v).hex() for v in state.loss],
                    "previous_weights_hex": [float(v).hex() for v in previous],
                    "known_target_dates": [str(o.target_known_at.date()) for o in observed],
                    "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
                    "matured_period_dates": period_dates.strftime("%Y-%m-%d").tolist(),
                    "matured_NET_returns_hex": [
                        [float(v).hex() for v in o.net_returns.loc[period_dates]] for o in observed
                    ],
                    "canonical_period_logs_hex": None
                    if logs is None
                    else [float(v).hex() for v in logs],
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        validation, period = None, None
        if is_month_end(day):
            if anchor is None:
                first_anchor = day
            else:
                validation = math.fsum(
                    float(w) * math.exp(float(value))
                    for w, value in zip(previous, logs, strict=True)
                )
                if not math.isfinite(validation) or validation <= 0:
                    raise QuantError(
                        "A matured statistical mixture score must be finite and positive."
                    )
                _, update = state.update(logs)
                period = {
                    "start_after": str(anchor.date()),
                    "end": str(day.date()),
                    "canonical_log_relatives": logs.tolist(),
                    "prequential_source_mixture_relative": validation,
                    **update,
                }
                periods.append(period)
            anchor = day
        weights = state.weights()
        row = combine(observed, weights)
        target.loc[day] = row
        audit = {
            "decision_session": str(day.date()),
            "expert_weights": weights.tolist(),
            "previous_expert_weights": previous.tolist(),
            "complete_learning_intervals": state.periods,
            "cumulative_expert_losses": state.loss.tolist(),
            "learning_round": state.periods + 1,
            "learning_rate": 2 * math.sqrt(math.log(N) / (state.periods + 1)),
            "score_mature": state.periods > 0,
            "period_update": period,
            "prequential_source_mixture_relative": validation,
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "risk_and_cost": risk_cost_audit(observed, weights, previous, first_anchor, day),
            "normalized_source_loss_is_actual_main_performance": False,
            "expert_wealth_guarantee_transferred": False,
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
        previous = weights
    return target, audits, periods
