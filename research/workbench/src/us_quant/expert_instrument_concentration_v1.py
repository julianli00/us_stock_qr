from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_native_arrival_equal_v1 import NativeOpinionAdapter
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-instrument-concentration-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = pool.EXPERT_IDS, pool.MECHANISMS, pool.CLOCK
PRIOR_HEAD = "d19188bd8e6c5ea2943ad56326303f1e03316f1848fae2b0ea08b10e1fef61ca"
CANDIDATES = (
    {
        "id": "expert_pool_eight_instrument_concentration_pair_v1",
        "weighting": "equal_pair_minimum_current_ETF_concentration_cost_ties",
    },
)
combine = pool.combine


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_instrument_concentration_20261103_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 749,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_registered": 71,
        "prior_completed": 62,
        "prior_technical_failures": 9,
        "prior_funded_paths": 248,
        "prior_combined_completed": 240,
        "initial_decision_session": "2016-10-05",
        "decision_schedule": "initial_capital_and_genuine_completed_month_ends",
        "decision_information_clock": CLOCK,
        "source_opinion_observation_lookback_not_a_scoring_parameter": 2,
        "pair_menu_size": 28,
        "pair_fractions": [0.5, 0.5],
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "instrument_concentration": "sum_of_squared_pair_ETF_targets_normalized_by_0.98",
        "concentration_score_decimal_places": 12,
        "cost_tie_decimal_places": 12,
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": 1e-12,
        "initial_previous_expert_allocation": None,
        "source_return_RF_or_covariance_scores_used": False,
        "concentration_is_stock_overlap_independent_risk_or_drawdown_bound": False,
        "source_native_orders_positions_or_feedback_guarantees_copied": False,
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
    if (
        not isinstance(policy, dict)
        or policy != fixed
        or any(type(policy[key]) is not type(value) for key, value in fixed.items())
    ):
        raise QuantError("Freeze one current instrument concentration rule, not a risk-cap scan.")


def choose_pair(opinions: np.ndarray, previous: np.ndarray | None):
    if (
        opinions.ndim != 2
        or opinions.shape[0] != 8
        or opinions.shape[1] == 0
        or not np.isfinite(opinions).all()
        or (opinions < 0).any()
        or any(abs(math.fsum(row) - 0.98) > 1e-12 for row in opinions)
        or previous is not None
        and (
            previous.shape != (8,)
            or not np.isfinite(previous).all()
            or (previous < 0).any()
            or abs(math.fsum(previous) - 1) > 1e-12
        )
    ):
        raise QuantError(
            "Concentration needs all eight valid funded opinions and a real prior state."
        )
    records = []
    for first, second in itertools.combinations(range(8), 2):
        weights = np.zeros(8)
        weights[[first, second]] = 0.5
        target = np.array(
            [
                math.fsum((0.5 * float(a), 0.5 * float(b)))
                for a, b in zip(opinions[first], opinions[second], strict=True)
            ]
        )
        concentration = math.fsum((float(value) / 0.98) ** 2 for value in target)
        cost = 0.0
        changed = 0
        if previous is not None:
            change = (weights - previous) @ opinions
            changed = int((abs(change) > 1e-12).sum())
            cost = float(5 * abs(change).sum() + changed)
        if not math.isfinite(concentration) or not math.isfinite(cost) or cost < 0:
            raise QuantError("Nonfinite concentration or cost cannot receive a successful rank.")
        records.append(
            {
                "pair": sorted((EXPERT_IDS[first], EXPERT_IDS[second])),
                "expert_weights": weights.tolist(),
                "combined_target_binary64_hex": [float(x).hex() for x in target],
                "instrument_concentration": concentration,
                "concentration_key_12dp": round(concentration, 12),
                "switch_cost_proxy_usd": cost,
                "switch_cost_key_12dp": round(cost, 12),
                "switch_changed_asset_coordinates": changed,
                "initial_no_previous_allocation": previous is None,
            }
        )
    selected = min(
        records,
        key=lambda row: (row["concentration_key_12dp"], row["switch_cost_key_12dp"], row["pair"]),
    )
    return np.asarray(selected["expert_weights"]), {
        "pair_records": records,
        "selected_pair": selected["pair"],
        "selected_concentration_key_12dp": selected["concentration_key_12dp"],
        "selected_switch_cost_key_12dp": selected["switch_cost_key_12dp"],
        "composition_assessment_available": True,
        "source_return_or_covariance_history_used": False,
        "instrument_HHI_is_economic_independence_or_future_risk": False,
    }


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    retain=None,
):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
        or len(experts) != 8
        or tuple(e.identity.identifier for e in experts) != EXPERT_IDS
        or not universe
        or len(set(universe)) != len(universe)
    ):
        raise QuantError("Keep the complete pool, initial anchor and genuine original calendar.")
    adapters = tuple(NativeOpinionAdapter(e, calendar) for e in experts)
    dates = [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    target = pd.DataFrame(np.nan, index=calendar, columns=universe)
    previous = None
    audits = []
    for day in dates:
        if before_decision is not None:
            before_decision()
        views = [adapter.observe(day) for adapter in adapters]
        observed = [view.observation for view in views]
        sources = [
            {
                "id": item.identity.identifier,
                "mechanism_group": item.identity.mechanism_group,
                "introduced_at": item.identity.introduced_at,
                "source_query_as_of": str(item.as_of.date()),
                "source_known_at": str(item.target_known_at.date()),
                "policy_refreshed_at": str(view.policy_refresh_at.date()),
                "asset_order": list(item.target.index),
                "target_binary64_hex": [float(x).hex() for x in item.target],
                "source_native_order_or_execution_claimed": False,
            }
            for item, view in zip(observed, views, strict=True)
        ]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "as_of": str(day.date()),
                    "decision_information_clock": CLOCK,
                    "previous_expert_weights_binary64_hex": [float(x).hex() for x in previous]
                    if previous is not None
                    else None,
                    "source_opinions": sources,
                    "new_source_HHI_or_score_not_yet_computed": True,
                }
            )
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("Concentration cannot ignore or reorder an underlying instrument.")
        weights, selection = choose_pair(
            np.stack([o.target.to_numpy() for o in observed]), previous
        )
        combined = combine(observed, weights)
        target.loc[day] = combined
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital"
            if day == initial
            else "completed_exchange_month_end",
            "decision_information_clock": CLOCK,
            "source_opinions": sources,
            "previous_expert_weights": previous.tolist() if previous is not None else None,
            "expert_weights": weights.tolist(),
            "selection": selection,
            "source_return_scores_used": False,
            "source_orders_positions_or_wealth_floor_copied": False,
            "counterfactual_historical_expert_availability": True,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "target_binary64_hex": [float(x).hex() for x in combined],
                }
            )
        previous = weights
    return target, audits
