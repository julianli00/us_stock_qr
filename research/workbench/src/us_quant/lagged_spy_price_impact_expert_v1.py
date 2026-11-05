from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar
from us_quant.conditional_hedge_expert_v1 import CLOCK, CORE, UNIVERSE, checked_close
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/lagged-spy-price-impact-expert-v1.json"
ID = "lagged_spy_price_impact_participation_v1"
CANDIDATE = {"id": ID, "construction": "monthly_lagged_price_impact_core_or_actual_BIL"}


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_lagged_spy_price_impact_complete_policy_20261105_v1",
        "candidate_id": ID,
        "candidates": [CANDIDATE],
        "mechanism_group": "lagged_market_price_impact_participation",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 264,
        "prior_journal_sequence": 1027,
        "prior_journal_head_sha256": (
            "94a0ae53b6a9138e6b1776e9955491030e369a3bc1f2d840532d6d823804ede5"
        ),
        "source_market_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/fixed-four-sleeve-expert-inclusion-"
                "20261105-v1/financial-implementation-freeze-v1.json"
            ),
            "sha256": "783e373593ab4109672af0324ec22827dc7d91604b0b96558f97abe6c4f27ed7",
        },
        "source_market_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "fixed-four-sleeve-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "c94d61f4d9d57d6caeaa55a2221587dc5f4b439ba98bb368e91177877263aa40",
        },
        "universe": list(UNIVERSE),
        "core_symbols": list(CORE),
        "input_symbol": "SPY",
        "cash_asset": "BIL",
        "input_lag_sessions": 1,
        "current_impact_sessions": 21,
        "nonoverlapping_reference_impact_sessions": 252,
        "required_adjusted_price_levels": 274,
        "source_price_warmup_position": 274,
        "relative_ratio_decimal_places": 12,
        "risk_off_ratio_strictly_greater_than": 1.0,
        "degenerate_reference_behavior": "explicit_fatal_error_not_BIL_or_neutral_fallback",
        "decision_lattice": "genuine_completed_month_ends_after_full_lagged_input_warmup",
        "decision_information_clock": CLOCK,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "new_registered_factor_families": 0,
        "new_traded_assets": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "parameter_search": False,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | {"rationale", "methodology"}
        or any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items())
        or not isinstance(policy.get("rationale"), dict)
        or not policy["rationale"]
    ):
        raise QuantError("Freeze one lagged price-impact participation law and original scope.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "real_source_indicators_targets_accounts_or_financial_outcomes_computed",
            "new_expert_technically_admitted",
            "existing_pool_changed",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError(
            "Source preparation is not financial admission, membership or qualification."
        )


def checked_market(data: MarketData) -> None:
    if not isinstance(data, MarketData):
        raise QuantError("This Module needs the admitted OHLCV panels, not closes alone.")
    data.validate()
    checked_close(data.close)
    if any(frame["SPY"].dtype.kind not in "fi" for frame in (data.raw_close, data.volume)):
        raise QuantError("Reported SPY price and volume must be actual numeric observations.")


def price_impact_state(closes, reported_closes, reported_volumes) -> dict:
    arrays = [np.asarray(value) for value in (closes, reported_closes, reported_volumes)]
    if (
        [array.shape for array in arrays] != [(274,), (273,), (273,)]
        or any(array.dtype.kind not in "fi" for array in arrays)
        or not all(np.isfinite(array).all() and (array > 0).all() for array in arrays)
    ):
        raise QuantError(
            "Require274actual closes and273matched positive price/volume observations."
        )
    close, raw, volume = [array.astype(float, copy=False) for array in arrays]
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            activity = raw * volume
            impact = np.abs(close[1:] / close[:-1] - 1.0) / activity
        prior = math.fsum(float(v) for v in impact[:252]) / 252
        current = math.fsum(float(v) for v in impact[252:]) / 21
        if not math.isfinite(prior) or prior <= 0 or not math.isfinite(current):
            raise QuantError("A degenerate price-impact reference cannot become a safe opinion.")
        ratio = current / prior
    except (FloatingPointError, OverflowError) as error:
        raise QuantError("Nonfinite price-impact arithmetic cannot receive a fallback.") from error
    if (
        not np.isfinite(activity).all()
        or (activity <= 0).any()
        or not np.isfinite(impact).all()
        or (impact < 0).any()
        or not math.isfinite(ratio)
        or ratio < 0
    ):
        raise QuantError("Invalid reported activity or impact cannot be defaulted.")
    key = round(ratio, 12)
    return {
        "reference_impact_mean": prior,
        "current_impact_mean": current,
        "relative_impact_ratio": ratio,
        "rounded_relative_ratio": key,
        "risk_off_triggered": key > 1.0,
    }


def _window(data: MarketData, position: int):
    return (
        data.close["SPY"].iloc[position - 274 : position],
        data.raw_close["SPY"].iloc[position - 273 : position],
        data.volume["SPY"].iloc[position - 273 : position],
    )


def decision(data: MarketData, as_of: pd.Timestamp) -> tuple[pd.Series, dict]:
    checked_market(data)
    if (
        not isinstance(as_of, pd.Timestamp)
        or pd.isna(as_of)
        or as_of.tzinfo is not None
        or as_of != as_of.normalize()
        or as_of not in data.close.index
    ):
        raise QuantError("Require one genuine covered normalized NYSE month end.")
    if as_of >= market_calendar().last_session or not is_month_end(as_of):
        raise QuantError("The opinion needs a genuine month end and covered following session.")
    position = int(data.close.index.get_loc(as_of))
    if position < 274:
        raise QuantError("No price-impact opinion before complete lagged input warmup.")
    close, raw, volume = _window(data, position)
    state = price_impact_state(close.to_numpy(), raw.to_numpy(), volume.to_numpy())
    selected = ["BIL"] if state["risk_off_triggered"] else list(CORE)
    target = pd.Series(0.0, index=UNIVERSE)
    target.loc[selected] = 0.98 / len(selected)
    return target, {
        "decision_session": str(as_of.date()),
        "decision_information_clock": CLOCK,
        "first_prior_adjusted_close": str(close.index[0].date()),
        "reference_input_start": str(raw.index[0].date()),
        "reference_input_end": str(raw.index[251].date()),
        "current_input_start": str(raw.index[252].date()),
        "last_input_session": str(raw.index[-1].date()),
        "reference_impact_observations": 252,
        "current_impact_observations": 21,
        **state,
        "selected_assets": selected,
        "decision_session_own_market_bar_used_to_choose_weight": False,
        "proxy_is_measured_market_depth_flow_or_execution_cost": False,
        "historical_counterfactual_availability": True,
        "actual_policy_introduction_or_admission_backdated": False,
        "mature_NET_score_inferred_from_price_warmup": False,
        "independent_forward_validation": False,
    }


def build_targets(data: MarketData, policy: dict, before_decision=None, retain=None):
    validate(policy)
    checked_market(data)
    if data.close.index[-1] >= market_calendar().last_session:
        raise QuantError("The calendar must cover the following session without guessing.")
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    audits = []
    for position, day in enumerate(data.close.index):
        if position < 274 or not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        close, raw, volume = _window(data, position)
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": CLOCK,
                    "input_symbol": "SPY",
                    "impact_return_session_dates": raw.index.strftime("%Y-%m-%d").tolist(),
                    "adjusted_close_binary64_hex": [float(v).hex() for v in close],
                    "reported_close_binary64_hex": [float(v).hex() for v in raw],
                    "reported_share_volume_binary64_hex": [float(v).hex() for v in volume],
                    "decision_session_own_or_future_bar_used_to_choose_weight": False,
                }
            )
        target, audit = decision(data, day)
        targets.loc[day] = target
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "asset_order": list(data.close.columns),
                    "complete_target": target.tolist(),
                }
            )
    return targets, audits
