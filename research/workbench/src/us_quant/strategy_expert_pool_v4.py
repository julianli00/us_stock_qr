from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v3 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/strategy-expert-pool-six-v1.json"
PRIOR_HEAD = "4dd16707904ec6b8b2fc1aa3442ffd31af3372fadfc63b512ed11deff42b81d6"
EXPERT_IDS = (*prior.EXPERT_IDS, "four_factor_defense_diversified")
MECHANISMS = (*prior.MECHANISMS, "cross_asset_defensive_relative_strength")
CANDIDATES = (
    {"id": "expert_pool_six_equal_weight_v1", "weighting": "equal"},
    {"id": "expert_pool_six_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair = prior.choose_pair
combine = prior.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_strategy_expert_pool_six_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 217,
        "prior_new_registered_candidates": 42,
        "prior_new_completed_strategy_evaluations": 39,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 467,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "selection_score_decimal_places": 12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "switch_cost_amortization_sessions": 21,
        "changed_asset_numeric_tolerance": 1e-12,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()
    ):
        raise QuantError("Freeze six admitted experts and the unchanged original pair law.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A sixth mechanism is not qualification or authority to trade.")


def decision_sessions(calendar: pd.DatetimeIndex, policy: dict) -> pd.DatetimeIndex:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.is_monotonic_increasing
        or not calendar.is_unique
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the real initial anchor and every original source session.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    dates = decision_sessions(calendar, policy)
    if len(experts) != 6 or len(universe) != len(set(universe)):
        raise QuantError("Use the fixed six complete experts and unique asset coordinates.")
    targets = {c["id"]: pd.DataFrame(np.nan, index=calendar, columns=universe) for c in CANDIDATES}
    audits = []
    previous = np.full(6, 1 / 6)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
            or any(o.target is None for o in observed)
            or any(tuple(o.target.index) != universe for o in observed)
            or position > 0
            and any(not len(o.net_returns) for o in observed)
        ):
            raise QuantError("A missing post-start expert cannot be dropped or silently replaced.")
        equal = np.full(6, 1 / 6)
        rolling, selection = choose_pair(observed, previous)
        targets[CANDIDATES[0]["id"]].loc[day] = combine(observed, equal)
        targets[CANDIDATES[1]["id"]].loc[day] = combine(observed, rolling)
        audits.append(
            {
                "decision_session": str(day.date()),
                "decision_role": "initial_capital"
                if position == 0
                else "completed_exchange_month_end",
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
                "preserved_archive_identity_introduced_at": [
                    o.identity.introduced_at for o in observed
                ],
                "expert_shadow_last_observation": [
                    str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                    for o in observed
                ],
                "equal_expert_weights": equal.tolist(),
                "previous_rolling_expert_weights": previous.tolist(),
                "rolling_expert_weights": rolling.tolist(),
                "selection": selection,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
        )
        previous = rolling
    return targets, audits
