from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/strategy-expert-pool-seven-v1.json"
PRIOR_HEAD = "079d350ca6788e089f8f07253da6037b0aed8fab865f0c54b498d358bbf9d879"
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
EXPERT_IDS = (*prior.EXPERT_IDS, "two_segment_vvix_uncertainty_confirmation_v1")
MECHANISMS = (*prior.MECHANISMS, "option_market_uncertainty_confirmation")
CANDIDATES = (
    {"id": "expert_pool_seven_equal_weight_v1", "weighting": "equal"},
    {"id": "expert_pool_seven_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_strategy_expert_pool_seven_20261031_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 228,
        "prior_new_registered_candidates": 58,
        "prior_new_completed_strategy_evaluations": 50,
        "prior_new_technical_failures": 8,
        "prior_journal_sequence": 606,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "decision_information_clock": CLOCK,
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
        policy.get(k) != value or type(policy.get(k)) is not type(value)
        for k, value in fixed.items()
    ):
        raise QuantError(
            "Freeze seven admitted experts,after-index-close clock and unchanged pair law."
        )
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError(
            "Study-specific pool membership is not qualification or trading authority."
        )


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
    retain=None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    dates = decision_sessions(calendar, policy)
    n = len(EXPERT_IDS)
    if len(experts) != n or len(universe) != len(set(universe)):
        raise QuantError("Use the fixed seven complete experts and unique asset coordinates.")
    targets = {c["id"]: pd.DataFrame(np.nan, index=calendar, columns=universe) for c in CANDIDATES}
    audits = []
    previous = np.full(n, 1 / n)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": position,
                    "as_of": str(day.date()),
                    "information_clock": CLOCK,
                    "previous_weights": previous.tolist(),
                    "previous_weights_binary64_hex": [float(x).hex() for x in previous],
                    "observations": [
                        {
                            "id": o.identity.identifier,
                            "mechanism_group": o.identity.mechanism_group,
                            "introduced_at": o.identity.introduced_at,
                            "as_of": str(o.as_of.date()),
                            "target_known_at": str(o.target_known_at.date())
                            if o.target_known_at is not None
                            else None,
                            "target": o.target.tolist() if o.target is not None else None,
                            "asset_order": list(o.target.index) if o.target is not None else None,
                            "return_dates": o.net_returns.index.strftime("%Y-%m-%d").tolist(),
                            "NET_returns": o.net_returns.tolist(),
                            "RF": o.risk_free.tolist(),
                            "score_mature": o.score_mature,
                        }
                        for o in observed
                    ],
                    "binary64_values_truncated_to_fixed_decimals": False,
                }
            )
        if (
            tuple(o.identity.identifier for o in observed) != EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
            or any(o.target is None for o in observed)
            or any(tuple(o.target.index) != universe for o in observed)
            or position > 0
            and any(not len(o.net_returns) for o in observed)
        ):
            raise QuantError("A missing post-start expert cannot be dropped or silently replaced.")
        equal = np.full(n, 1 / n)
        rolling, selection = choose_pair(observed, previous)
        targets[CANDIDATES[0]["id"]].loc[day] = combine(observed, equal)
        targets[CANDIDATES[1]["id"]].loc[day] = combine(observed, rolling)
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if position == 0 else "completed_exchange_month_end",
            "decision_information_clock": CLOCK,
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
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "decision_index": position,
                    "audit": audit,
                    "asset_order": list(universe),
                    "combined_targets": {
                        name: matrix.loc[day].tolist() for name, matrix in targets.items()
                    },
                }
            )
        previous = rolling
    return targets, audits
