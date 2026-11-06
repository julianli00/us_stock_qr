from __future__ import annotations

from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from us_quant import quintet_adaptive_hedge_v1 as prior
from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.calendar import is_month_end
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/protective-six-adaptive-hedge-v1.json"
SLOTS = (4, 5, 7, 0, 1, 3)
EXPERT_IDS = tuple(pool.EXPERT_IDS[i] for i in SLOTS)
MECHANISMS = tuple(pool.MECHANISMS[i] for i in SLOTS)
CLOCK = prior.CLOCK
ID = "expert_pool_protective_six_adaptive_hedge_v1"
CANDIDATES = (
    {
        "id": ID,
        "weighting": "unchanged_AdaHedge_with_existing_wealth_feedback_source",
    },
)
N, MAX_PERIODS, PRECISION, TOLERANCE = 6, 119, 768, 1e-12
PRIOR_HEAD = "b7f953c4190694b0b27751d7941599ddc0d243115cc190390224c13e9d930f2f"
combine, risk_cost_audit = prior.combine, prior.risk_cost_audit


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


checked_losses = bound(prior.checked_losses)
mix = bound(prior.mix)
independent_trace = bound(prior.independent_trace)
bounded_losses = bound(prior.prior.bounded_losses)
observations = bound(prior.prior.calendar_source.observations)
period_logs = prior.prior.period_logs


class AdaHedge:
    __init__ = bound(prior.AdaHedge.__init__)
    check_state = bound(prior.AdaHedge.check_state)
    rate = bound(prior.AdaHedge.rate)
    weights = bound(prior.AdaHedge.weights)
    snapshot = bound(prior.AdaHedge.snapshot)
    update = bound(prior.AdaHedge.update)


def expected_policy():
    return {
        "schema_version": 1,
        "study_id": "ssd_protective_six_adaptive_hedge_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 190233,
        "prior_journal_sequence": 1452,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 300,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/protective-six-adaptive-hedge-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "7db879697264482fa5cac20bcdee07cb1e2d6c62755a789e47b63af673232923",
        },
        "source_five_math_code": {
            "path": "src/us_quant/quintet_adaptive_hedge_v1.py",
            "sha256": "730a154f0a1ab7abaecece20dbd5235e4a7b4efc852e2b7a856b8e53bcf8cf0b",
        },
        "original_source_slots": list(SLOTS),
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
        "learning_rule": "unchanged_AdaHedge2014_Eq8_9_at_N6",
        "learning_rate": "log6_divided_by_past_cumulative_mixability_gap",
        "zero_gap_learning_rate": "analytic_infinity_uniform_exact_past_leaders",
        "infinite_rate_mix_loss": "minimum_cumulative_loss_after_minus_before",
        "canonical_period_log": "math_fsum_of_math_log1p_recorded_source_NET_returns",
        "bounded_loss": "exact_binary64_log_fraction_from_six_source_best_to_worst",
        "equal_log_relative_month_loss": "all_zero",
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
        "source_wealth_guarantee_transferred_to_main": False,
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
        "global_pool_changed": False,
        "new_expert_admissions": 0,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Only one existing complete source is added;keep the original adaptive law."
        )


decision_sessions = bound(prior.decision_sessions)


def build_targets(experts, calendar, assets, policy, before_decision=None, retain=None):
    dates = decision_sessions(calendar, policy)
    if len(experts) != N or len(set(assets)) != len(assets):
        raise QuantError("Use six real complete sources and unique original asset coordinates.")
    first_anchor = pd.Timestamp(policy["first_complete_month_anchor"])
    state, previous = AdaHedge(), np.full(N, 1 / N)
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    audits, updates, anchor = [], [], None
    for day in dates:
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(o.target.index) != tuple(assets) for o in observed):
            raise QuantError("The complete six recommendations require every original coordinate.")
        logs, labels, period_dates = None, None, pd.DatetimeIndex([])
        if anchor is not None:
            logs, period_dates = period_logs(observed, anchor, day)
            labels = bounded_losses(logs)
        key = str(day.date())
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
                    "canonical_RF_hex": [
                        float(v).hex() for v in observed[0].risk_free.loc[period_dates]
                    ],
                    "canonical_log_relatives_hex": None
                    if logs is None
                    else [float(v).hex() for v in logs],
                    "normalized_loss_labels_hex": None
                    if labels is None
                    else [float(v).hex() for v in labels],
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        update = None
        if labels is not None:
            _, learned = state.update(labels)
            update = {
                "start_after": str(anchor.date()),
                "end": key,
                "canonical_log_relatives": logs.tolist(),
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
            "six_sources_are_five_padded_coordinates": False,
            "old_learner_or_source_or_control_reexecuted": False,
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
        previous = weights
    return target, audits, updates
