from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar
from us_quant.conditional_hedge_expert_v1 import CLOCK, CORE, UNIVERSE, checked_close
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/monthly-style-reversal-expert-v1.json"
ID = "monthly_style_reversal_v1"
CANDIDATE = {
    "id": ID,
    "construction": "monthly_equal_capital_two_recently_weakest_styles",
}
LOOKBACK = 21
SELECTED = 2
RANK_DECIMALS = 12


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_monthly_style_reversal_complete_policy_20261105_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "candidate_id": ID,
        "candidates": [CANDIDATE],
        "mechanism_group": "monthly_cross_sectional_style_reversal",
        "prior_completed_configurations": 258,
        "prior_journal_sequence": 957,
        "prior_journal_head_sha256": (
            "d66608671751e0527858252fb3b7744bfb065fa65e238b78bfcb4358f2c8371a"
        ),
        "source_market_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "calendar-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "ec373f28b3432f578f1331e85d36565562d1ebcc9d7eba7a5bc528507b09b2e3",
        },
        "source_market_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/calendar-expert-inclusion-20261105-v1/"
                "financial-implementation-freeze-v1.json"
            ),
            "sha256": "2ab68d14ec2dca60ce9c534fb8961a7b8153d9552819e354512f999e79a3ed22",
        },
        "universe": list(UNIVERSE),
        "core_symbols": list(CORE),
        "lookback_return_sessions": LOOKBACK,
        "source_price_warmup_levels": LOOKBACK + 1,
        "ranking": "ascending_21_session_simple_total_return_rounded_then_lexical_symbol",
        "ranking_decimal_places": RANK_DECIMALS,
        "selected_styles": SELECTED,
        "selected_style_weighting": "equal_capital_0.49_each",
        "decision_lattice": "genuine_completed_month_ends_after_full_price_warmup",
        "decision_information_clock": CLOCK,
        "execution": "unchanged_actual_weight_band_and_original_one_or_two_session_later_open",
        "missing_history": "unavailable_until_22_consecutive_price_levels_not_BIL_or_zero_score",
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "score_maturity_is_separate_from_policy_availability": True,
        "minimum_financial_source_participation_family_count": 3,
        "new_registered_factor_families": 0,
        "new_traded_assets": 0,
        "new_asset_price_ranking_rule": True,
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
        raise QuantError("Preserve the single reversal law, source scope, clock and budget.")
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
        raise QuantError("Preparation is not admission, qualification or order authority.")


def _checked_market(close: pd.DataFrame) -> None:
    calendar = market_calendar()
    if (
        not isinstance(close, pd.DataFrame)
        or close.empty
        or not isinstance(close.index, pd.DatetimeIndex)
        or close.index.hasnans
        or close.index.tz is not None
        or close.index[0] < calendar.first_session
        or close.index[-1] > calendar.last_session
        or any(not isinstance(dtype, np.dtype) for dtype in close.dtypes)
    ):
        raise QuantError(
            "Use covered original numeric market rows, not inferred or filled history."
        )
    checked_close(close)


def decision(close: pd.DataFrame, position: int) -> tuple[pd.Series, dict]:
    if (
        not isinstance(close, pd.DataFrame)
        or type(position) is not int
        or position < LOOKBACK
        or position >= len(close)
    ):
        raise QuantError("A reversal opinion needs22 actual price levels at a declared position.")
    window = close.iloc[position - LOOKBACK : position + 1]
    _checked_market(window)
    day = window.index[-1]
    if day >= market_calendar().last_session or not is_month_end(day):
        raise QuantError("A reversal opinion needs a genuine covered month end and next session.")
    prices = window.loc[:, list(CORE)]
    with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
        try:
            returns = prices.iloc[-1].to_numpy() / prices.iloc[0].to_numpy() - 1
        except FloatingPointError as error:
            raise QuantError(
                "Invalid reversal price-ratio arithmetic; do not normalize it."
            ) from error
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise QuantError("Reversal assessments require finite, positive price relatives.")
    assessed = {symbol: float(value) for symbol, value in zip(CORE, returns, strict=True)}
    rounded = {symbol: round(value, RANK_DECIMALS) for symbol, value in assessed.items()}
    ranked = sorted(CORE, key=lambda symbol: (rounded[symbol], symbol))
    selected = ranked[:SELECTED]
    target = pd.Series(0.0, index=UNIVERSE)
    target.loc[selected] = 0.98 / SELECTED
    return target, {
        "decision_session": str(day.date()),
        "decision_information_clock": CLOCK,
        "source_window_start": str(window.index[0].date()),
        "source_window_end": str(day.date()),
        "source_price_levels": LOOKBACK + 1,
        "return_observation_sessions": LOOKBACK,
        "simple_total_returns": assessed,
        "rounded_ranking_values": rounded,
        "ranking_order": ranked,
        "selected_assets": selected,
        "observed_weakness_is_verified_future_recovery": False,
        "source_missingness_or_undefined_NET_score_used_as_safe_success": False,
        "counterfactual_historical_policy_availability": True,
        "actual_policy_introduction_or_admission_backdated": False,
        "independent_forward_validation": False,
    }


def build_targets(close: pd.DataFrame, policy: dict, before_decision=None, retain=None):
    validate(policy)
    _checked_market(close)
    if close.index[-1] >= market_calendar().last_session:
        raise QuantError("The calendar must cover the following session without guessing.")
    targets = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for position, day in enumerate(close.index):
        if position < LOOKBACK or not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        window = close.iloc[position - LOOKBACK : position + 1].loc[:, list(CORE)]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": CLOCK,
                    "window_dates": window.index.strftime("%Y-%m-%d").tolist(),
                    "used_source_prices_binary64_hex": {
                        symbol: [float(value).hex() for value in window[symbol]] for symbol in CORE
                    },
                    "future_price_or_return_observed": False,
                }
            )
        target, audit = decision(close, position)
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
