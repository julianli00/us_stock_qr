from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar
from us_quant.conditional_hedge_expert_v1 import CLOCK, CORE, UNIVERSE, checked_close
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/fixed-four-sleeve-expert-v1.json"
ID = "fixed_four_sleeve_rebalance_v1"
CANDIDATE = {"id": ID, "construction": "monthly_constant_four_bucket_capital_rebalance"}
DEFENSIVE = ("TLT", "GLD", "BIL")


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_fixed_four_sleeve_complete_policy_20261105_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "candidate_id": ID,
        "candidates": [CANDIDATE],
        "mechanism_group": "fixed_cross_asset_capital_rebalancing",
        "prior_completed_configurations": 261,
        "prior_journal_sequence": 992,
        "prior_journal_head_sha256": (
            "1586cb6695a5f09652895362b77e3b4afc2db36e8cfecf19a8f49a625bec1332"
        ),
        "source_market_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "reversal-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "b67a17a0a2d1aeef0034720c9f8a313d03e12bb02ab75c35d53674bff6ef9d56",
        },
        "source_market_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/reversal-expert-inclusion-20261105-v1/"
                "financial-implementation-freeze-v1.json"
            ),
            "sha256": "00cbdd54ff37e4ee9546c81dd090a5d9d4bec2b06b6acc7e93f37a1d91483367",
        },
        "universe": list(UNIVERSE),
        "core_symbols": list(CORE),
        "capital_buckets": ["six_style_core", *DEFENSIVE],
        "bucket_weights_within_ETF_budget": [0.25] * 4,
        "style_core_weighting": "equal_capital_over_six_existing_style_ETFs",
        "cash_asset": "BIL",
        "decision_lattice": "every_genuine_completed_month_end_in_whole_admitted_history",
        "decision_information_clock": CLOCK,
        "execution": "unchanged_actual_weight_band_and_original_one_or_two_session_later_open",
        "price_return_covariance_or_forecast_used_to_choose_weights": False,
        "source_price_warmup_sessions": 0,
        "score_maturity_is_separate_from_policy_availability": True,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_financial_source_participation_family_count": 3,
        "new_registered_factor_families": 0,
        "new_traded_assets": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "parameter_search": False,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | {"rationale", "methodology"}
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in fixed.items()
        )
        or not isinstance(policy.get("rationale"), dict)
        or not policy["rationale"]
    ):
        raise QuantError(
            "Preserve one fixed four-bucket complete policy and original source scope."
        )
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "real_source_targets_accounts_or_financial_outcomes_computed",
            "new_expert_technically_admitted",
            "existing_pool_changed",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Preparation is not source admission, pool membership or qualification.")


def decision(as_of: pd.Timestamp) -> tuple[pd.Series, dict]:
    if (
        not isinstance(as_of, pd.Timestamp)
        or pd.isna(as_of)
        or as_of.tzinfo is not None
        or as_of != as_of.normalize()
    ):
        raise QuantError("A fixed allocation opinion needs a genuine normalized NYSE month end.")
    calendar = market_calendar()
    if (
        as_of < calendar.first_session
        or as_of >= calendar.last_session
        or not calendar.is_session(as_of)
        or not is_month_end(as_of)
    ):
        raise QuantError("A fixed allocation opinion needs a covered month end and next session.")
    target = pd.Series(0.0, index=UNIVERSE)
    target.loc[list(CORE)] = 0.98 / (4 * len(CORE))
    target.loc[list(DEFENSIVE)] = 0.98 / 4
    return target, {
        "decision_session": str(as_of.date()),
        "decision_information_clock": CLOCK,
        "fixed_capital_buckets": ["six_style_core", *DEFENSIVE],
        "bucket_weights_within_ETF_budget": [0.25] * 4,
        "selected_assets": [*CORE, *DEFENSIVE],
        "BIL_is_actual_ETF_not_idle_cash": True,
        "idle_cash_reserve": 0.02,
        "price_return_covariance_or_forecast_used_to_choose_weights": False,
        "capital_buckets_claimed_independent_strategy_experts": False,
        "counterfactual_historical_policy_availability": True,
        "actual_policy_introduction_or_admission_backdated": False,
        "mature_NET_score_inferred_from_fixed_target_availability": False,
        "independent_forward_validation": False,
    }


def build_targets(close: pd.DataFrame, policy: dict, before_decision=None, retain=None):
    validate(policy)
    checked_close(close)
    if close.index[-1] >= market_calendar().last_session:
        raise QuantError("The calendar must cover the following session without guessing.")
    targets = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for day in close.index:
        if not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": CLOCK,
                    "fixed_capital_buckets": ["six_style_core", *DEFENSIVE],
                    "bucket_weights_within_ETF_budget": [0.25] * 4,
                    "price_or_future_return_observed_to_choose_weight": False,
                }
            )
        target, audit = decision(day)
        targets.loc[day] = target
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "asset_order": list(close.columns),
                    "complete_target": target.tolist(),
                }
            )
    return targets, audits
