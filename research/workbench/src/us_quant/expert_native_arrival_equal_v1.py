from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-native-arrival-equal-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = pool.EXPERT_IDS, pool.MECHANISMS, pool.CLOCK
PRIOR_HEAD = "6bb375ba455b4aab3d34a3d11e6ea2b01788b23cdb90b93bca22c616349340d0"
CLOCK_CONTRACT_SHA = "c3ed9b485e248a3cbe240d105f0b696d5af526a7ac4582b126c1147e58f826d7"
SOURCE_CLOCKS = ("monthly", "monthly", "monthly", "daily_feedback", *("monthly",) * 4)
CANDIDATES = (
    {
        "id": "expert_pool_eight_native_arrival_equal_v1",
        "weighting": "equal_eighths_at_native_opinion_refresh_union",
    },
)
combine = pool.combine


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_native_arrival_equal_20261103_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 737,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_registered": 70,
        "prior_completed": 61,
        "prior_technical_failures": 9,
        "prior_funded_paths": 244,
        "prior_combined_completed": 239,
        "initial_decision_session": "2016-10-05",
        "decision_information_clock": CLOCK,
        "source_clock_contract_sha256": CLOCK_CONTRACT_SHA,
        "expert_ids": list(EXPERT_IDS),
        "source_opinion_clocks": list(SOURCE_CLOCKS),
        "expert_fractions": [0.125] * 8,
        "source_observation_lookback_not_a_scoring_parameter": 2,
        "source_NET_or_RF_used_to_rank_or_allocate": False,
        "source_native_request_or_execution_dates_copied": False,
        "source_feedback_floor_or_shadow_wealth_guarantee_transferred": False,
        "raw_dense_known_date_interpreted_as_native_event": False,
        "initial_capital_is_ordinary_source_refresh": False,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "minimum_economic_factor_families": 3,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze the native opinion clocks, equal eighths and original funding.")


@dataclass(frozen=True)
class ClockedOpinion:
    observation: ExpertObservation
    meta_as_of: pd.Timestamp
    policy_refresh_at: pd.Timestamp
    arrived_today: bool
    native_clock: str


class NativeOpinionAdapter:
    def __init__(self, expert: StrategyExpert, calendar: pd.DatetimeIndex):
        if (
            expert.identity.identifier not in EXPERT_IDS
            or calendar.empty
            or not calendar.equals(sessions(calendar[0], calendar[-1]))
        ):
            raise QuantError("Use admitted complete experts and the full genuine source calendar.")
        position = EXPERT_IDS.index(expert.identity.identifier)
        if expert.identity.mechanism_group != MECHANISMS[position]:
            raise QuantError("A source clock cannot substitute another economic mechanism.")
        self._expert = expert
        self._calendar = calendar
        self._month_ends = pd.DatetimeIndex([day for day in calendar if is_month_end(day)])
        self._clock = SOURCE_CLOCKS[position]

    def _query_day(self, as_of):
        day = pd.Timestamp(as_of)
        if day not in self._calendar:
            raise QuantError("Observe only an actual covered information session.")
        if day < pd.Timestamp("2016-10-05"):
            raise QuantError(
                "This Adapter covers the formal master range; raw source prehistory is unchanged."
            )
        if self._clock == "daily_feedback":
            return day
        known = self._month_ends[self._month_ends <= day]
        if known.empty:
            raise QuantError("No previous genuine month-end opinion is available.")
        return known[-1]

    def observe(self, as_of: pd.Timestamp) -> ClockedOpinion:
        day = pd.Timestamp(as_of)
        query = self._query_day(day)
        observation = self._expert.observe(query, 2)
        if (
            observation.identity != self._expert.identity
            or observation.as_of != query
            or observation.target is None
            or observation.target_known_at != query
            or query > day
            or not np.isfinite(observation.target).all()
            or (observation.target < 0).any()
            or abs(float(observation.target.sum()) - 0.98) > 1e-12
        ):
            raise QuantError(
                f"Unavailable or invalid native opinion for {self._expert.identity.identifier} "
                f"at {query.date()}; no carried safe fallback."
            )
        return ClockedOpinion(observation, day, query, query == day, self._clock)


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    retain=None,
) -> tuple[pd.DataFrame, list[dict]]:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        len(experts) != 8
        or tuple(expert.identity.identifier for expert in experts) != EXPERT_IDS
        or not universe
        or len(set(universe)) != len(universe)
        or initial not in calendar
    ):
        raise QuantError("Keep the fixed complete pool, asset order and real initial anchor.")
    adapters = tuple(NativeOpinionAdapter(expert, calendar) for expert in experts)
    target = pd.DataFrame(np.nan, index=calendar, columns=universe)
    weights = np.full(8, 0.125)
    audits = []
    for day in calendar[calendar >= initial]:
        if before_decision is not None:
            before_decision()
        if retain is not None:
            retain(
                {
                    "record": "opinion_query",
                    "meta_as_of": str(day.date()),
                    "source_ids": list(EXPERT_IDS),
                    "native_query_sessions": [
                        str(adapter._query_day(day).date()) for adapter in adapters
                    ],
                    "information_clock": CLOCK,
                }
            )
        opinions = [adapter.observe(day) for adapter in adapters]
        native = [opinion.observation for opinion in opinions]
        if any(tuple(item.target.index) != universe for item in native):
            raise QuantError("All complete opinions need the unchanged common asset lattice.")
        sources = [
            {
                "id": item.identity.identifier,
                "mechanism_group": item.identity.mechanism_group,
                "introduced_at": item.identity.introduced_at,
                "source_observed_as_of": str(item.as_of.date()),
                "source_known_at": str(item.target_known_at.date()),
                "policy_refreshed_at": str(opinion.policy_refresh_at.date()),
                "arrived_today": opinion.arrived_today,
                "native_opinion_clock": opinion.native_clock,
                "asset_order": list(item.target.index),
                "target_binary64_hex": [float(x).hex() for x in item.target],
                "source_native_order_or_execution_claimed": False,
            }
            for item, opinion in zip(native, opinions, strict=True)
        ]
        if retain is not None:
            retain(
                {
                    "record": "opinion_input",
                    "meta_as_of": str(day.date()),
                    "source_opinions": sources,
                    "equal_expert_weights_binary64_hex": [float(x).hex() for x in weights],
                    "source_NET_or_RF_used_to_rank_or_allocate": False,
                }
            )
        refreshed = [row["id"] for row in sources if row["arrived_today"]]
        initial_role = day == initial
        if not initial_role and not refreshed:
            if retain is not None:
                retain(
                    {
                        "record": "opinion_resolved",
                        "meta_as_of": str(day.date()),
                        "meta_target_emitted": False,
                        "reason": "no_new_native_opinion",
                    }
                )
            continue
        combined = combine(native, weights)
        target.loc[day] = combined
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if initial_role else "native_opinion_refresh",
            "decision_information_clock": CLOCK,
            "source_opinions": sources,
            "fresh_source_opinion_ids": refreshed,
            "expert_weights": weights.tolist(),
            "source_scores_used": False,
            "source_requests_or_held_positions_copied": False,
            "shadow_wealth_or_protection_guarantee_transferred": False,
            "counterfactual_historical_expert_availability": True,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "opinion_resolved",
                    "meta_as_of": str(day.date()),
                    "meta_target_emitted": True,
                    "audit": audit,
                    "target_binary64_hex": [float(x).hex() for x in combined],
                }
            )
    return target, audits
