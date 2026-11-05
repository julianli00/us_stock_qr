from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar
from us_quant.conditional_hedge_expert_v1 import CLOCK, CORE, UNIVERSE, checked_close
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/calendar-seasonal-expert-v1.json"
ID = "calendar_seasonal_six_core_bil_v1"
CANDIDATE = {"id": ID, "construction": "monthly_calendar_seasonal_core_or_bill"}
EQUITY_MONTHS = (11, 12, 1, 2, 3, 4)


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_calendar_seasonal_complete_policy_20261105_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "candidate_id": ID,
        "candidates": [CANDIDATE],
        "mechanism_group": "fixed_calendar_seasonal_equity_participation",
        "prior_completed_configurations": 255,
        "prior_journal_sequence": 922,
        "prior_journal_head_sha256": (
            "7913272587b33cd7573578cfed3fc34519174690caf877533e875117f36a52e8"
        ),
        "universe": list(UNIVERSE),
        "core_symbols": list(CORE),
        "cash_asset": "BIL",
        "intended_equity_calendar_months": list(EQUITY_MONTHS),
        "season_input": "next_calendar_month_known_at_genuine_current_month_end",
        "stock_weighting": "equal_capital_over_six_existing_equity_families",
        "other_months": "98_percent_actual_BIL_not_free_idle_cash",
        "decision_lattice": "every_genuine_completed_month_end_in_whole_admitted_history",
        "decision_information_clock": CLOCK,
        "execution": "unchanged_actual_weight_band_and_original_one_or_two_session_later_open",
        "price_return_or_volatility_used_to_choose_season": False,
        "source_price_warmup_sessions": 0,
        "score_maturity_is_separate_from_policy_availability": True,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "new_economic_factor_definitions": 0,
        "new_traded_assets": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "parameter_search": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Preserve the single calendar law, source clock, universe and budget.")
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
        raise QuantError("Calendar preparation is not admission, qualification or order authority.")


def decision(as_of: pd.Timestamp) -> tuple[pd.Series, dict]:
    if (
        not isinstance(as_of, pd.Timestamp)
        or pd.isna(as_of)
        or as_of.tzinfo is not None
        or as_of != as_of.normalize()
    ):
        raise QuantError("A calendar opinion requires a genuine normalized NYSE month end.")
    calendar = market_calendar()
    if (
        as_of < calendar.first_session
        or as_of >= calendar.last_session
        or not calendar.is_session(as_of)
        or not is_month_end(as_of)
    ):
        raise QuantError("A calendar opinion requires a covered NYSE month end and next session.")
    intended_month = as_of.month % 12 + 1
    equity = intended_month in EQUITY_MONTHS
    selected = list(CORE) if equity else ["BIL"]
    target = pd.Series(0.0, index=UNIVERSE)
    target.loc[selected] = 0.98 / len(selected)
    return target, {
        "decision_session": str(as_of.date()),
        "decision_information_clock": CLOCK,
        "intended_calendar_month": intended_month,
        "state": "six_family_equity_participation" if equity else "actual_BIL_participation",
        "selected_assets": selected,
        "season_is_future_return_or_crash_forecast": False,
        "counterfactual_historical_policy_availability": True,
        "actual_policy_introduction_or_admission_backdated": False,
        "mature_NET_score_inferred_from_calendar_availability": False,
        "independent_forward_validation": False,
    }


def build_targets(close: pd.DataFrame, policy: dict, before_decision=None, retain=None):
    validate(policy)
    checked_close(close)
    if close.index[-1] >= market_calendar().last_session:
        raise QuantError("The native calendar must cover the following session without guessing.")
    targets = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for day in close.index:
        if not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": CLOCK,
            "intended_calendar_month": day.month % 12 + 1,
            "future_price_or_return_observed": False,
        }
        if retain is not None:
            retain(before)
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
