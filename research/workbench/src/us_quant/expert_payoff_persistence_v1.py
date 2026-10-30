from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as pool
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.expert_contextual_contest_v1 import co_loss_weights, exact, rational_record
from us_quant.expert_online_log_growth_v1 import observations
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-payoff-persistence-v1.json"
PRIOR_HEAD = "89350f4299a96e640a9267bc8d508028b8538e0f53113871ba891404f1c655b8"
EXPERT_IDS, MECHANISMS = pool.EXPERT_IDS, pool.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_payoff_continuation_v1",
        "weighting": "positive_payoff_mass_continuation_same_joint_loss_tilt",
    },
    {
        "id": "expert_pool_six_annual_refresh_v1",
        "weighting": "positive_payoff_mass_uniform_annual_refresh_same_joint_loss_tilt",
    },
)
TOLERANCE = 1e-12


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_payoff_persistence_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_continuation_event": 55981,
        "prior_completed_configurations": 226,
        "prior_new_registered_candidates": 56,
        "prior_new_completed_strategy_evaluations": 48,
        "prior_new_technical_failures": 8,
        "prior_journal_sequence": 583,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "shadow_reference_path": "10y-base",
        "lookback_sessions": 126,
        "learning_clock": "completed_full_exchange_month_intervals",
        "learning_state_resets_for_five_year_account": False,
        "initial_partial_source_month_assimilated": False,
        "maximum_complete_learning_intervals": 119,
        "initial_learning_mass_per_expert": "1/6",
        "refresh_numerator": 1,
        "refresh_denominator": 12,
        "monthly_relative_input": (
            "exact_product_of_saved_binary64_NET_daily_relatives_then_one_binary64_rounding"
        ),
        "learning_state_arithmetic": "exact_rational_saved_binary64_month_relatives",
        "risk_tilt_coefficient": 1,
        "risk_transform": "unchanged_exact_joint_loss_exponential_transform",
        "independent_risk_tilt_decimal_precision": 50,
        "independent_weight_and_target_tolerance": TOLERANCE,
        "main_decision_clock": "actual_initial_anchor_and_genuine_month_ends",
        "actual_weight_trade_band": 0.025,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "reference_capital_usd": 10000,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        key not in policy or type(policy[key]) is not type(value) or policy[key] != value
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze both payoff rules,their exact clock and original execution.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "pool_selected_from_complete_returns",
            "refresh_frequency_selected_from_outcomes",
            "learning_mass_is_calibrated_probability",
            "learning_score_is_actual_main_net_wealth",
            "source_shadow_wealth_or_protection_floor_transferred",
            "uniform_simplex_wealth_integral_repaired_or_approximated",
            "failed_OLMAR_PAMR_V1_repaired_retried_or_recertified",
            "new_standalone_expert_added",
            "new_risk_tilt_coefficient_selected",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError(
            "A persistence prior cannot imply calibrated probabilities or main wealth."
        )


@dataclass(frozen=True)
class PayoffMassState:
    mass: tuple[Fraction, ...]
    completed_intervals: int

    @classmethod
    def initial(cls):
        return cls((Fraction(1, 6),) * 6, 0)

    def validate(self):
        if (
            len(self.mass) != 6
            or any(not isinstance(value, Fraction) or value <= 0 for value in self.mass)
            or sum(self.mass) != 1
            or type(self.completed_intervals) is not int
            or not 0 <= self.completed_intervals <= 119
        ):
            raise QuantError("Require exact positive unit learning mass,not invented main capital.")

    def advance(self, relatives, refresh: bool):
        self.validate()
        if (
            type(refresh) is not bool
            or len(relatives) != 6
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in relatives)
            or self.completed_intervals >= 119
        ):
            raise QuantError("Advance once with six saved positive complete-month relatives.")
        values = tuple(self.mass[i] * exact(relatives[i]) for i in range(6))
        score = sum(values)
        posterior = tuple(value / score for value in values)
        mass = (
            tuple(Fraction(11, 12) * value + Fraction(1, 72) for value in posterior)
            if refresh
            else posterior
        )
        result = PayoffMassState(mass, self.completed_intervals + 1)
        result.validate()
        return result, score


def monthly_relatives(observed, period):
    if (
        len(observed) != 6
        or period.empty
        or any(not period.isin(item.net_returns.index).all() for item in observed)
    ):
        raise QuantError("Keep every matured NET source observation of a genuine full month.")
    result = []
    for item in observed:
        value = Fraction(1)
        for net in item.net_returns.loc[period]:
            daily = 1 + exact(net)
            if daily <= 0:
                raise QuantError("A source month cannot include a nonpositive daily relative.")
            value *= daily
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise QuantError("The fixed binary64 monthly input is not positive and representable.")
        result.append(number)
    return result


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or initial not in calendar
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
    ):
        raise QuantError("Retain the original initial anchor and complete exchange calendar.")
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def build_targets(
    experts: tuple[StrategyExpert, ...],
    close: pd.DataFrame,
    policy: dict,
    before_decision=None,
    record_learning=None,
):
    dates = decision_sessions(close.index, policy)
    if len(experts) != 6 or not close.columns.is_unique:
        raise QuantError("Keep the six complete experts and original unique asset coordinates.")
    targets = {
        candidate["id"]: pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for candidate in CANDIDATES
    }
    state = {candidate["id"]: PayoffMassState.initial() for candidate in CANDIDATES}
    previous_day, mature_started = None, False
    audits = []
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(item.target.index) != tuple(close.columns) for item in observed):
            raise QuantError("A source recommendation changed the admitted asset lattice.")
        period, relatives = None, None
        initial_partial = previous_day is not None and not is_month_end(previous_day)
        if previous_day is not None and not initial_partial:
            if day.year * 12 + day.month - previous_day.year * 12 - previous_day.month != 1:
                raise QuantError("Do not skip a completed source month.")
            period = sessions(next_session(previous_day), day)
            relatives = monthly_relatives(observed, period)
        if record_learning is not None:
            record_learning(
                {
                    "stage": "input",
                    "session": str(day.date()),
                    "previous_decision": None if previous_day is None else str(previous_day.date()),
                    "mass_before": {
                        name: [rational_record(value) for value in current_state.mass]
                        for name, current_state in state.items()
                    },
                    "completed_intervals_before": {
                        name: value.completed_intervals for name, value in state.items()
                    },
                    "monthly_relative_inputs": relatives,
                    "monthly_relative_input_hex": None
                    if relatives is None
                    else [value.hex() for value in relatives],
                    "source_return_sessions": observed[0]
                    .net_returns.index.strftime("%Y-%m-%d")
                    .tolist(),
                    "source_NET_returns": [item.net_returns.tolist() for item in observed],
                    "source_RF": observed[0].risk_free.tolist(),
                }
            )
        mature = all(item.score_mature for item in observed)
        if mature_started and not mature:
            raise QuantError("Post-maturity missing risk history cannot be a neutral success.")
        mature_started = mature_started or mature
        allocations = {}
        for index, candidate in enumerate(CANDIDATES):
            name, score = candidate["id"], None
            if relatives is not None:
                state[name], score = state[name].advance(relatives, index == 1)
            if mature:
                weights, risk = co_loss_weights(observed, state[name].mass)
            else:
                weights, risk = np.full(6, 1 / 6), None
            targets[name].loc[day] = pool.combine(observed, weights)
            allocation = {
                "learning_mass_after_update": [rational_record(x) for x in state[name].mass],
                "completed_intervals": state[name].completed_intervals,
                "annual_refresh_applied": index == 1 and relatives is not None,
                "prequential_source_score_relative": None
                if score is None
                else rational_record(score),
                "applied_expert_weights": weights.tolist(),
                "co_loss_assessment": risk,
                "reason": "mature_payoff_mass_and_joint_loss_assessment"
                if mature
                else "unscored_initial_equal_weights",
            }
            allocations[name] = allocation
            if record_learning is not None:
                record_learning(
                    {
                        "stage": "allocation_resolved",
                        "session": str(day.date()),
                        "candidate_id": name,
                        "allocation": allocation,
                        "unified_target": targets[name].loc[day].tolist(),
                    }
                )
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if position == 0 else "completed_exchange_month_end",
            "previous_decision": None if previous_day is None else str(previous_day.date()),
            "initial_partial_source_month_not_assimilated": initial_partial,
            "source_period_first_session": None if period is None else str(period[0].date()),
            "source_period_last_session": None if period is None else str(period[-1].date()),
            "source_period_sessions": 0 if period is None else len(period),
            "monthly_relative_inputs": relatives,
            "complete_learning_intervals": state[CANDIDATES[0]["id"]].completed_intervals,
            "source_observation_counts": [len(item.net_returns) for item in observed],
            "source_assessment_mature": mature,
            "expert_target_known_at": [str(item.target_known_at.date()) for item in observed],
            "preserved_archive_identity_introduced_at": [
                item.identity.introduced_at for item in observed
            ],
            "allocations": allocations,
            "learning_state_is_main_wealth": False,
            "learning_mass_is_calibrated_probability": False,
            "source_shadow_wealth_or_protection_floor_transferred": False,
            "counterfactual_historical_expert_availability": True,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if record_learning is not None:
            record_learning({"stage": "resolved", "session": str(day.date()), "audit": audit})
        previous_day = day
    return targets, audits
