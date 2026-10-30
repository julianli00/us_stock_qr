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
from us_quant.expert_online_log_growth_v1 import observations
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-contextual-contest-v1.json"
PRIOR_HEAD = "08418b982a7ee8ab32fd013fcaa17cda5b5c34e4affe2132d105443514b6bd21"
EXPERT_IDS, MECHANISMS = pool.EXPERT_IDS, pool.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_contest_frequency_v1",
        "weighting": "unconditional_monthly_contest_with_joint_loss_tilt",
    },
    {
        "id": "expert_pool_six_contextual_contest_v1",
        "weighting": "matched_context_monthly_contest_with_same_joint_loss_tilt",
    },
)
LOOKBACK, CREDIT, SCORE_SCALE = 126, 60, 10**12
TOLERANCE = 1e-10


def exact(value) -> Fraction:
    number = float(value)
    if not math.isfinite(number):
        raise QuantError("An exact assessment still requires finite observed binary64 values.")
    return Fraction.from_float(number)


def rational_record(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_contextual_contest_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_continuation_event": 54426,
        "prior_completed_configurations": 224,
        "prior_new_registered_candidates": 54,
        "prior_new_completed_strategy_evaluations": 46,
        "prior_new_technical_failures": 8,
        "prior_journal_sequence": 560,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "shadow_reference_path": "10y-base",
        "lookback_sessions": LOOKBACK,
        "context_lookback_sessions": LOOKBACK,
        "context_symbols": ["SPY", "IEF", "BIL"],
        "market_contexts": 4,
        "learning_clock": "completed_full_exchange_month_intervals",
        "learning_memory": "expanding_completed_months",
        "initial_partial_source_month_assimilated": False,
        "learning_state_resets_for_five_year_account": False,
        "maximum_complete_learning_intervals": 119,
        "posterior_pseudocount_per_expert": 1,
        "winner_relative_score_scale": SCORE_SCALE,
        "equal_winner_credit_units_per_month": CREDIT,
        "exact_input_interpretation": (
            "binary64 values treated as exact rationals for context,"
            "month ranking and co-loss assessment"
        ),
        "risk_tilt_coefficient": 1,
        "no_common_excess_loss_days_rule": "retain posterior fractions without claiming safety",
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
        raise QuantError(
            "Freeze the categorical context study,source clock and original execution."
        )
    methodology = policy.get("methodology")
    if not isinstance(methodology, dict) or any(
        methodology.get(key) is not False
        for key in (
            "pool_selected_from_complete_returns",
            "failed_OLMAR_PAMR_V1_repaired_retried_or_recertified",
            "old_uniform_simplex_wealth_integral_repaired_or_approximated",
            "new_standalone_expert_added",
            "new_contexts_are_independent_economic_factors",
            "learning_state_is_main_wealth",
            "source_shadow_wealth_or_protection_floor_transferred",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Observable contexts and co-loss assessments are not wealth guarantees.")


def context_inputs(close: pd.DataFrame, day: pd.Timestamp) -> dict:
    if (
        not close.index.is_unique
        or day not in close.index
        or not {"SPY", "IEF", "BIL"}.issubset(close.columns)
    ):
        raise QuantError("Market context needs the known admitted SPY/IEF/BIL closes.")
    position = close.index.get_loc(day)
    if not isinstance(position, int | np.integer) or position < LOOKBACK:
        raise QuantError("A context needs127observed closes before the actual decision.")
    earlier = close.index[position - LOOKBACK]
    before = {symbol: float(close.loc[earlier, symbol]) for symbol in ("SPY", "IEF", "BIL")}
    after = {symbol: float(close.loc[day, symbol]) for symbol in before}
    if any(not math.isfinite(value) or value <= 0 for value in [*before.values(), *after.values()]):
        raise QuantError("Market-context prices must be observed,finite and positive.")
    return {
        "as_of": str(day.date()),
        "earlier_session": str(earlier.date()),
        "earlier_prices": before,
        "current_prices": after,
    }


def market_context(record: dict) -> int:
    before, after = record["earlier_prices"], record["current_prices"]
    values = [before[name] for name in ("SPY", "IEF", "BIL")] + [
        after[name] for name in ("SPY", "IEF", "BIL")
    ]
    if any(exact(value) <= 0 for value in values):
        raise QuantError("Classify only positive finite context observations.")
    positive = [
        exact(after[symbol]) * exact(before["BIL"]) > exact(after["BIL"]) * exact(before[symbol])
        for symbol in ("SPY", "IEF")
    ]
    return 2 * int(positive[0]) + int(positive[1])


def winner_credit(observed, period_dates: pd.DatetimeIndex) -> dict:
    if (
        len(observed) != 6
        or period_dates.empty
        or any(not period_dates.isin(o.net_returns.index).all() for o in observed)
    ):
        raise QuantError("Credit requires every genuinely matured day of a complete source month.")
    keys = []
    for item in observed:
        relative = Fraction(1)
        for value in item.net_returns.loc[period_dates]:
            daily = Fraction(1) + exact(value)
            if daily <= 0:
                raise QuantError("A complete source NET relative must remain positive.")
            relative *= daily
        keys.append(round(relative * SCORE_SCALE))
    winners = [i for i, value in enumerate(keys) if value == max(keys)]
    if not winners or CREDIT % len(winners):
        raise QuantError("The fixed integer representation cannot split this winner set.")
    units = [CREDIT // len(winners) if i in winners else 0 for i in range(6)]
    return {"winner_relative_keys": keys, "winner_indices": winners, "credit_units": units}


@dataclass(frozen=True)
class ContestState:
    credits: tuple[tuple[int, ...], ...]
    complete_months: int

    @classmethod
    def initial(cls):
        return cls(tuple((0,) * 6 for _ in range(4)), 0)

    def validate(self):
        if (
            len(self.credits) != 4
            or any(len(row) != 6 for row in self.credits)
            or any(type(value) is not int or value < 0 for row in self.credits for value in row)
            or type(self.complete_months) is not int
            or not 0 <= self.complete_months <= 119
            or sum(sum(row) for row in self.credits) != CREDIT * self.complete_months
            or any(sum(row) % CREDIT for row in self.credits)
        ):
            raise QuantError("Categorical counts need exact nonnegative completed-month credit.")

    def advance(self, context: int, credit: dict):
        self.validate()
        units = credit["credit_units"]
        if (
            type(context) is not int
            or not 0 <= context < 4
            or self.complete_months >= 119
            or len(units) != 6
            or any(type(value) is not int or value < 0 for value in units)
            or sum(units) != CREDIT
        ):
            raise QuantError("Advance exactly one complete source month in its preceding context.")
        values = [list(row) for row in self.credits]
        values[context] = [values[context][i] + units[i] for i in range(6)]
        result = ContestState(tuple(tuple(row) for row in values), self.complete_months + 1)
        result.validate()
        return result

    def posterior(self, context: int | None):
        self.validate()
        if context is not None and (type(context) is not int or not 0 <= context < 4):
            raise QuantError("A conditional posterior needs its fixed observable context.")
        counts = (
            [sum(row[i] for row in self.credits) for i in range(6)]
            if context is None
            else list(self.credits[context])
        )
        denominator = sum(counts) + 6 * CREDIT
        return tuple(Fraction(value + CREDIT, denominator) for value in counts), sum(
            counts
        ) // CREDIT


def co_loss_weights(observed, fractions: tuple[Fraction, ...]):
    if (
        len(fractions) != 6
        or any(not isinstance(value, Fraction) or value <= 0 for value in fractions)
        or sum(fractions) != 1
        or any(len(o.net_returns) != LOOKBACK for o in observed)
    ):
        raise QuantError(
            "Co-loss assessment needs a positive posterior and126complete source days."
        )
    excess = [
        [exact(o.net_returns.iloc[j]) - exact(o.risk_free.iloc[j]) for j in range(LOOKBACK)]
        for o in observed
    ]
    losing = [
        j
        for j in range(LOOKBACK)
        if sum((fractions[i] * excess[i][j] for i in range(6)), Fraction(0)) < 0
    ]
    if losing:
        losses = [
            sum((max(-excess[i][j], Fraction(0)) for j in losing), Fraction(0)) / len(losing)
            for i in range(6)
        ]
        reference = sum((fractions[i] * losses[i] for i in range(6)), Fraction(0))
        if reference <= 0:
            raise QuantError(
                "Nonempty common loss dates require a strictly positive co-loss proxy."
            )
        ratios = tuple(value / reference for value in losses)
        if any(value < 0 or value > 1 / fractions[i] for i, value in enumerate(ratios)):
            raise QuantError("A co-loss ratio exceeded its exact positive-budget bound.")
        raw = [float(fractions[i]) * math.exp(-float(ratios[i])) for i in range(6)]
        reason = "observed_joint_excess_loss_tilt"
    else:
        losses = [Fraction(0)] * 6
        reference = Fraction(0)
        ratios = (Fraction(0),) * 6
        raw = [float(value) for value in fractions]
        reason = "no_common_excess_loss_dates_posterior_unchanged_no_safety_claim"
    mass = math.fsum(raw)
    if not math.isfinite(mass) or mass <= 0:
        raise QuantError("The bounded risk transform cannot invent a neutral funded result.")
    result = np.array([value / mass for value in raw])
    if not np.isfinite(result).all() or (result <= 0).any() or abs(float(result.sum()) - 1) > 1e-12:
        raise QuantError("The co-loss transform must retain a positive unit statistical budget.")
    return result, {
        "posterior_rational_fractions": [rational_record(value) for value in fractions],
        "common_loss_source_sessions": [
            str(observed[0].net_returns.index[j].date()) for j in losing
        ],
        "component_nonoffsetting_loss_means": [rational_record(value) for value in losses],
        "reference_nonoffsetting_loss": rational_record(reference),
        "component_relative_losses": [rational_record(value) for value in ratios],
        "expert_weights_after_tilt": result.tolist(),
        "risk_tilt_reason": reason,
        "source_risk_proxy_is_main_loss_or_guarantee": False,
        "positive_hedge_gains_credited_in_proxy": False,
    }


def decision_sessions(calendar: pd.DatetimeIndex, policy: dict):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Retain the real initial anchor and every original exchange session.")
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
    universe = tuple(close.columns)
    if len(experts) != 6 or len(universe) != len(set(universe)):
        raise QuantError("Keep the six complete experts and original unique asset coordinates.")
    targets = {
        c["id"]: pd.DataFrame(np.nan, index=close.index, columns=universe) for c in CANDIDATES
    }
    state = ContestState.initial()
    previous_day = previous_context = None
    previous_main = {c["id"]: np.full(6, 1 / 6) for c in CANDIDATES}
    audits = []
    mature_started = False
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = observations(experts, day)
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("A complete expert recommendation changed the admitted asset lattice.")
        context_record = context_inputs(close, day)
        if record_learning is not None:
            record_learning(
                {
                    "stage": "input",
                    "session": str(day.date()),
                    "context_input": context_record,
                    "previous_decision": None if previous_day is None else str(previous_day.date()),
                    "previous_context": previous_context,
                    "credit_state_before": [list(row) for row in state.credits],
                    "complete_months_before": state.complete_months,
                    "source_return_sessions": observed[0]
                    .net_returns.index.strftime("%Y-%m-%d")
                    .tolist(),
                    "source_NET_returns": [o.net_returns.tolist() for o in observed],
                    "source_RF": observed[0].risk_free.tolist(),
                }
            )
        context = market_context(context_record)
        update = None
        initial_partial = previous_day is not None and not is_month_end(previous_day)
        if previous_day is not None and is_month_end(previous_day):
            if day.year * 12 + day.month - previous_day.year * 12 - previous_day.month != 1:
                raise QuantError("A full learning month cannot skip a genuine main decision.")
            known = sessions(next_session(previous_day), day)
            update = winner_credit(observed, known)
            update.update(
                previous_context=previous_context,
                first_source_session=str(known[0].date()),
                last_source_session=str(known[-1].date()),
                source_sessions=len(known),
            )
            state = state.advance(previous_context, update)
        mature = all(o.score_mature for o in observed)
        if mature_started and not mature:
            raise QuantError("Post-maturity missing risk history cannot become a neutral success.")
        mature_started = mature_started or mature
        allocations = {}
        for index, candidate in enumerate(CANDIDATES):
            name = candidate["id"]
            posterior, counted_months = state.posterior(None if index == 0 else context)
            if mature:
                weights, risk = co_loss_weights(observed, posterior)
            else:
                weights, risk = np.full(6, 1 / 6), None
            targets[name].loc[day] = pool.combine(observed, weights)
            allocations[name] = {
                "counted_complete_months": counted_months,
                "categorical_forecast_is_prior_only": counted_months == 0,
                "posterior_rational_fractions": [rational_record(value) for value in posterior],
                "applied_expert_weights": weights.tolist(),
                "previous_applied_expert_weights": previous_main[name].tolist(),
                "co_loss_assessment": risk,
                "reason": "mature_categorical_and_joint_loss_assessment"
                if mature
                else "unscored_initial_equal_weights",
            }
            if record_learning is not None:
                record_learning(
                    {
                        "stage": "allocation_resolved",
                        "session": str(day.date()),
                        "candidate_id": name,
                        "allocation": allocations[name],
                        "unified_target": targets[name].loc[day].tolist(),
                    }
                )
            previous_main[name] = weights.copy()
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if position == 0 else "completed_exchange_month_end",
            "context_input": context_record,
            "current_context": context,
            "initial_partial_source_month_not_assimilated": initial_partial,
            "monthly_credit_update": update,
            "credit_state_after": [list(row) for row in state.credits],
            "complete_learning_months": state.complete_months,
            "source_observation_counts": [len(o.net_returns) for o in observed],
            "source_assessment_mature": mature,
            "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "preserved_archive_identity_introduced_at": [
                o.identity.introduced_at for o in observed
            ],
            "allocations": allocations,
            "learning_state_is_main_wealth": False,
            "source_shadow_wealth_or_protection_floor_transferred": False,
            "counterfactual_historical_expert_availability": True,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if record_learning is not None:
            record_learning({"stage": "resolved", "session": str(day.date()), "audit": audit})
        previous_day, previous_context = day, context
    return targets, audits
