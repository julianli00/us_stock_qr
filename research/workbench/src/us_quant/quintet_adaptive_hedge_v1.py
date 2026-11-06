from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import quintet_anytime_hedge_v1 as prior
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quintet-adaptive-hedge-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = prior.EXPERT_IDS, prior.MECHANISMS, prior.CLOCK
ID = "expert_pool_quintet_adaptive_hedge_v1"
CANDIDATES = ({"id": ID, "weighting": "AdaHedge_past_mixability_gap"},)
PRIOR_HEAD = "10f84ebd6d31b9ef0b0a118c80a82f984ccf8620c387f17f9f34fbf8270f60c1"
N, MAX_PERIODS, PRECISION, TOLERANCE = 5, 119, 512, 1e-12
observations, combine, risk_cost_audit = (
    prior.observations,
    prior.combine,
    prior.risk_cost_audit,
)


def expected_policy():
    return {
        "schema_version": 1,
        "study_id": "ssd_quintet_adaptive_hedge_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 188833,
        "prior_journal_sequence": 1440,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 299,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-adaptive-hedge-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "44748807bd3bda5eef3b7f7e523ed6e07b75562544dd0311b5e710a74ad19262",
        },
        "academic_reference": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-adaptive-hedge-20261106-v1/"
                "academic-reference-v1.json"
            ),
            "sha256": "8fbb4d0a24719163ba2f076315793e3ea39759e8f1866ddc9c51a55a4f00b968",
        },
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
        "learning_rule": "AdaHedge2014_Eq8_9",
        "learning_rate": "log5_divided_by_past_cumulative_mixability_gap",
        "zero_gap_learning_rate": "analytic_infinity_uniform_exact_past_leaders",
        "infinite_rate_mix_loss": "minimum_cumulative_loss_after_minus_before",
        "source_loss_labels": "sealed_original_binary64_bounded_month_losses_unchanged",
        "neutral_month_advances_maturity_not_learning_rate": True,
        "cumulative_loss_representation": "exact_Fraction_of_binary64_labels",
        "primary_decimal_precision": PRECISION,
        "independent_decimal_precision": 128,
        "independent_numeric_tolerance": TOLERANCE,
        "negative_Jensen_gap_clipped": False,
        "mixability_gap_floor": None,
        "rate_or_window_sweep": False,
        "prequential_validation_before_assimilation": True,
        "normalized_regret_reference": "R_at_most_2_Delta_source_game_only",
        "normalized_regret_is_actual_main_wealth": False,
        "old_learner_or_source_or_control_reexecuted": False,
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
        "main_drawdown_guarantee": False,
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


def validate(policy):
    fixed = expected_policy()
    if policy != fixed or any(type(policy[k]) is not type(v) for k, v in fixed.items()):
        raise QuantError("Freeze one original gap-adaptive learner and unchanged capital rules.")


def checked_losses(losses):
    values = np.asarray(losses)
    if (
        values.shape != (N,)
        or values.dtype.kind not in "fi"
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not ((values == 0).all() or values.min() == 0 and values.max() == 1)
    ):
        raise QuantError("Use the original five normalized source labels,not a filled loss vector.")
    return values


def cached_losses(period):
    values = checked_losses(period["losses"])
    expected = prior.bounded_losses(period["canonical_log_relatives"])
    if [float(v).hex() for v in values] != [float(v).hex() for v in expected]:
        raise QuantError("The sealed binary64 source loss labels changed.")
    return values


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def mix(losses, eta):
    with localcontext() as context:
        context.prec = PRECISION
        minimum = min(losses)
        if eta is None:
            mass = [Decimal(int(value == minimum)) for value in losses]
        else:
            mass = [(-eta * decimal_fraction(value - minimum)).exp() for value in losses]
        total = sum(mass)
        weights = [value / total for value in mass]
        cumulative_mix = decimal_fraction(minimum)
        if eta is not None:
            cumulative_mix -= (total / Decimal(N)).ln() / eta
        return weights, cumulative_mix


class AdaHedge:
    def __init__(self):
        self.periods = 0
        self.losses = (Fraction(0),) * N
        self.gap = Decimal(0)
        self.played = Decimal(0)
        self.mixed = Decimal(0)

    def check_state(self):
        if (
            type(self.periods) is not int
            or not 0 <= self.periods <= MAX_PERIODS
            or len(self.losses) != N
            or any(not isinstance(v, Fraction) for v in self.losses)
            or any(v < 0 or v > self.periods for v in self.losses)
            or any(
                not isinstance(v, Decimal) or not v.is_finite() or v < 0 or v > self.periods
                for v in (self.gap, self.played, self.mixed)
            )
            or self.mixed > self.played
            or self.gap == 0
            and any(v != 0 for v in self.losses)
        ):
            raise QuantError("Invalid exact AdaHedge state cannot become neutral or infinite-rate.")

    def rate(self):
        self.check_state()
        with localcontext() as context:
            context.prec = PRECISION
            return None if self.gap == 0 else Decimal(N).ln() / self.gap

    def weights(self):
        probabilities, _ = mix(self.losses, self.rate())
        weights = np.array([float(v) for v in probabilities])
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or abs(float(weights.sum()) - 1) > TOLERANCE
        ):
            raise QuantError("Finite capital weights must follow the unchanged mathematical law.")
        return weights

    def snapshot(self):
        eta = self.rate()
        return {
            "complete_intervals": self.periods,
            "cumulative_loss_fractions": [
                {"numerator": str(v.numerator), "denominator": str(v.denominator)}
                for v in self.losses
            ],
            "cumulative_mixability_gap_decimal": str(self.gap),
            "cumulative_played_loss_decimal": str(self.played),
            "cumulative_mix_loss_decimal": str(self.mixed),
            "learning_rate_decimal": None if eta is None else str(eta),
            "learning_rate_is_infinite": eta is None,
            "arithmetic_precision": PRECISION,
        }

    def update(self, labels):
        self.check_state()
        if self.periods >= MAX_PERIODS:
            raise QuantError("The original history contains at most119complete source months.")
        values = checked_losses(labels)
        eta = self.rate()
        before = self.weights()
        with localcontext() as context:
            context.prec = PRECISION
            probabilities, old_mix = mix(self.losses, eta)
            updated = tuple(
                old + Fraction.from_float(float(new))
                for old, new in zip(self.losses, values, strict=True)
            )
            _, new_mix = mix(updated, eta)
            hedge_loss = sum(
                w * Decimal.from_float(float(loss))
                for w, loss in zip(probabilities, values, strict=True)
            )
            mix_loss = new_mix - old_mix
            delta = hedge_loss - mix_loss
            if delta < 0 or delta > 1 or mix_loss < 0 or mix_loss > hedge_loss:
                raise QuantError("A Jensen violation stops;no numerical clipping or gap floor.")
            gap, played, mixed = (
                self.gap + delta,
                self.played + hedge_loss,
                self.mixed + mix_loss,
            )
            regret = played - decimal_fraction(min(updated))
            if regret > 2 * gap + Decimal(str(TOLERANCE)):
                raise QuantError(
                    "The source-loss AdaHedge regret check failed,not a wealth guarantee."
                )
            self.losses, self.gap, self.played, self.mixed = updated, gap, played, mixed
            self.periods += 1
            audit = {
                "input_loss_labels": values.tolist(),
                "prequential_capital_weights_before_assimilation": before.tolist(),
                "played_statistical_weights_decimal": [str(v) for v in probabilities],
                "played_loss_decimal": str(hedge_loss),
                "mix_loss_decimal": str(mix_loss),
                "mixability_gap_increment_decimal": str(delta),
                "normalized_source_regret_decimal": str(regret),
                "normalized_source_regret_reference_decimal": str(2 * gap),
                "learning_rate_before_decimal": None if eta is None else str(eta),
                "learning_rate_before_is_infinite": eta is None,
                "state_after": self.snapshot(),
                "source_loss_or_regret_is_actual_main_performance": False,
            }
        return self.weights(), audit


def independent_trace(history):
    cumulative, gap, trace = [Decimal(0)] * N, Decimal(0), []
    with localcontext() as context:
        context.prec = 128
        for labels in history:
            row = [Decimal.from_float(float(v)) for v in checked_losses(labels)]
            eta = None if gap == 0 else Decimal(N).ln() / gap
            if eta is None:
                mask = [Decimal(int(v == min(cumulative))) for v in cumulative]
                weights = [v / sum(mask) for v in mask]
            else:
                mass = [(-eta * value).exp() for value in cumulative]
                weights = [value / sum(mass) for value in mass]
            updated = [old + value for old, value in zip(cumulative, row, strict=True)]
            played = sum(w * value for w, value in zip(weights, row, strict=True))
            mixture = (
                min(updated) - min(cumulative)
                if eta is None
                else -sum(
                    w * (-eta * value).exp() for w, value in zip(weights, row, strict=True)
                ).ln()
                / eta
            )
            gap += played - mixture
            cumulative = updated
            next_eta = None if gap == 0 else Decimal(N).ln() / gap
            if next_eta is None:
                mask = [Decimal(int(v == min(cumulative))) for v in cumulative]
                after = [v / sum(mask) for v in mask]
            else:
                mass = [(-next_eta * v).exp() for v in cumulative]
                after = [v / sum(mass) for v in mass]
            trace.append(
                {
                    "weights_before": [float(v) for v in weights],
                    "weights_after": [float(v) for v in after],
                    "gap": float(gap),
                    "played_loss": float(played),
                    "mix_loss": float(mixture),
                    "learning_rate_after": None if next_eta is None else float(next_eta),
                }
            )
    return trace


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the full admitted calendar and original initial decision.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts, cached_periods, calendar, assets, policy, before_decision=None, retain=None
):
    dates = decision_sessions(calendar, policy)
    if len(experts) != N or len(set(assets)) != len(assets):
        raise QuantError("Keep five complete sources and unique original underlying coordinates.")
    by_end = {row["end"]: row for row in cached_periods}
    if len(by_end) != len(cached_periods):
        raise QuantError("The original cached full-month labels require unique end dates.")
    first_anchor = pd.Timestamp(policy["first_complete_month_anchor"])
    state, previous = AdaHedge(), np.full(N, 0.2)
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    audits, updates, anchor = [], [], None
    for day in dates:
        if before_decision is not None:
            before_decision()
        key = str(day.date())
        observed = observations(experts, day)
        if any(tuple(o.target.index) != tuple(assets) for o in observed):
            raise QuantError("Preserve every original source recommendation coordinate.")
        period, labels = None, None
        period_dates = pd.DatetimeIndex([])
        if anchor is not None:
            if key not in by_end:
                raise QuantError("A missing matured label cannot be filled or regenerated.")
            period = by_end[key]
            if period["start_after"] != str(anchor.date()):
                raise QuantError("Use the actual adjacent complete source month,not a crop.")
            labels = cached_losses(period)
            period_dates = sessions(next_session(anchor), day)
            actual_logs = [
                math.fsum(math.log1p(float(v)) for v in o.net_returns.loc[period_dates])
                for o in observed
            ]
            if [v.hex() for v in actual_logs] != [
                float(v).hex() for v in period["canonical_log_relatives"]
            ]:
                raise QuantError("Cached labels must belong to the exact known native NET month.")
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": key,
                    "state": state.snapshot(),
                    "previous_weights_hex": [float(v).hex() for v in previous],
                    "known_target_dates": [str(o.target_known_at.date()) for o in observed],
                    "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
                    "matured_period_dates": period_dates.strftime("%Y-%m-%d").tolist(),
                    "matured_NET_returns_hex": [
                        [float(v).hex() for v in o.net_returns.loc[period_dates]] for o in observed
                    ],
                    "cached_loss_labels_hex": None
                    if labels is None
                    else [float(v).hex() for v in labels],
                    "cached_canonical_log_relatives_hex": None
                    if period is None
                    else [float(v).hex() for v in period["canonical_log_relatives"]],
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        update = None
        if labels is not None:
            _, learned = state.update(labels)
            update = {
                "start_after": str(anchor.date()),
                "end": key,
                "canonical_log_relatives": period["canonical_log_relatives"],
                **learned,
            }
            updates.append(update)
        weights = state.weights()
        row = combine(observed, weights)
        target.loc[day] = row
        audit = {
            "decision_session": key,
            "expert_weights": weights.tolist(),
            "previous_expert_weights": previous.tolist(),
            "complete_learning_intervals": state.periods,
            "score_mature": state.periods > 0,
            "adaptive_state": state.snapshot(),
            "period_update": update,
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "risk_and_cost": risk_cost_audit(observed, weights, previous, first_anchor, day),
            "cached_loss_labels_changed": False,
            "old_learner_reexecuted": False,
            "normalized_source_regret_is_main_wealth": False,
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
        previous = weights
    return target, audits, updates
