from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/conditional-hedge-expert-v1.json"
ID = "conditional_covariance_hedge_expert_v1"
CORE = ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
DEFENSE = ("TLT", "IEF", "GLD")
UNIVERSE = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_conditional_hedge_expert_20261103_v1",
        "candidate_id": ID,
        "mechanism_group": "conditional_cross_asset_hedge_suitability",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "market_symbol": "SPY",
        "stock_context_sessions": 252,
        "hedge_observation_sessions": 63,
        "core_symbols": list(CORE),
        "defensive_symbols": list(DEFENSE),
        "cash_asset": "BIL",
        "universe": list(UNIVERSE),
        "decision_lattice": "genuine_completed_month_ends_after_full_price_warmup",
        "decision_information_clock": CLOCK,
        "stock_context": "SPY_close_strictly_greater_than_252_session_prior_close",
        "hedge_eligibility": (
            "positive_63_session_own_momentum_AND_negative_observed_63_return_stock_covariance"
        ),
        "covariance_sign_definition": (
            "exact_centered_cross_product_of_original_binary64_daily_total_returns"
        ),
        "eligible_hedge_weighting": "equal_capital_over_all_eligible_complete_defensive_assets",
        "no_eligible_hedge_behavior": "explicit_actual_BIL_state_not_missing_data_default",
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze one complete conditional-hedge law, source clock and budget.")
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
        raise QuantError(
            "A prepared hedge policy is not admission, qualification or order authority."
        )


def checked_close(close: pd.DataFrame) -> None:
    if (
        not isinstance(close, pd.DataFrame)
        or close.empty
        or not isinstance(close.index, pd.DatetimeIndex)
        or close.index.tz is not None
        or not close.index.is_unique
        or not close.index.is_monotonic_increasing
        or tuple(close.columns) != UNIVERSE
        or any(dtype.kind not in "fi" for dtype in close.dtypes)
        or not np.isfinite(close.to_numpy()).all()
        or (close <= 0).any().any()
        or not close.index.equals(sessions(close.index[0], close.index[-1]))
    ):
        raise QuantError(
            "Use the full admitted fourteen-asset calendar without filling or remapping."
        )


def covariance_numerator(left: np.ndarray, right: np.ndarray) -> Fraction:
    if (
        left.shape != (63,)
        or right.shape != (63,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise QuantError("The fixed hedge assessment needs exactly63 finite mature return pairs.")
    x = [Fraction.from_float(float(value)) for value in left]
    y = [Fraction.from_float(float(value)) for value in right]
    return 63 * sum(a * b for a, b in zip(x, y, strict=True)) - sum(x) * sum(y)


def decision(close: pd.DataFrame, position: int) -> tuple[pd.Series, dict]:
    if type(position) is not int or position < 252 or position >= len(close):
        raise QuantError("A decision needs full source warmup, not filled history.")
    window = close.iloc[position - 63 : position + 1].loc[:, ["SPY", *DEFENSE]]
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            values = window.to_numpy(dtype=float)
            returns = values[1:] / values[:-1] - 1
        except FloatingPointError as error:
            raise QuantError(
                "Invalid source price-ratio arithmetic; do not normalize it."
            ) from error
    eligible, evidence = [], []
    for column, symbol in enumerate(DEFENSE, start=1):
        numerator = covariance_numerator(returns[:, 0], returns[:, column])
        momentum = bool(window[symbol].iloc[-1] > window[symbol].iloc[0])
        accepted = momentum and numerator < 0
        if accepted:
            eligible.append(symbol)
        evidence.append(
            {
                "symbol": symbol,
                "own_momentum_positive": momentum,
                "observed_stock_covariance_sign": -1
                if numerator < 0
                else 1
                if numerator > 0
                else 0,
                "exact_centered_cross_product": {
                    "numerator": str(numerator.numerator),
                    "denominator": str(numerator.denominator),
                },
                "eligible": bool(accepted),
            }
        )
    stock_positive = bool(close["SPY"].iloc[position] > close["SPY"].iloc[position - 252])
    selected = list(CORE) if stock_positive else eligible or ["BIL"]
    row = pd.Series(0.0, index=close.columns)
    row.loc[selected] = 0.98 / len(selected)
    if (row < 0).any() or abs(float(row.sum()) - 0.98) > 1e-12:
        raise QuantError("The complete recommendation must retain the original funded ETF budget.")
    return row, {
        "source_price_history_mature": True,
        "stock_year_context_positive": stock_positive,
        "state": "six_factor_core"
        if stock_positive
        else ("compatible_rising_hedges" if eligible else "explicit_BIL_no_eligible_hedge"),
        "eligible_defensive_assets": eligible,
        "selected_assets": selected,
        "defensive_assessments": evidence,
        "negative_covariance_is_future_hedge_or_drawdown_guarantee": False,
        "source_missingness_or_undefined_NET_score_used_as_safe_success": False,
    }


def independent_decision(close: pd.DataFrame, position: int) -> tuple[pd.Series, list[str]]:
    window = close.iloc[position - 63 : position + 1]
    returns = {
        symbol: [
            float(window[symbol].iloc[i]) / float(window[symbol].iloc[i - 1]) - 1
            for i in range(1, 64)
        ]
        for symbol in ("SPY", *DEFENSE)
    }
    ratios = {
        name: [float(value).as_integer_ratio() for value in row] for name, row in returns.items()
    }
    denominator = max(bottom for row in ratios.values() for _, bottom in row)
    integers = {
        name: [top * (denominator // bottom) for top, bottom in row] for name, row in ratios.items()
    }
    x = integers["SPY"]
    selected = []
    for symbol in DEFENSE:
        y = integers[symbol]
        centered = 63 * sum(x[i] * y[i] for i in range(63)) - sum(x) * sum(y)
        if window[symbol].iloc[-1] > window[symbol].iloc[0] and centered < 0:
            selected.append(symbol)
    if close["SPY"].iloc[position] > close["SPY"].iloc[position - 252]:
        selected = list(CORE)
    elif not selected:
        selected = ["BIL"]
    row = pd.Series(
        [0.98 / len(selected) if name in selected else 0.0 for name in close.columns],
        index=close.columns,
    )
    return row, selected


def build_targets(close: pd.DataFrame, policy: dict, before_decision=None, retain=None):
    validate(policy)
    checked_close(close)
    target = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for position, day in enumerate(close.index):
        if position < 252 or not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        window = close.iloc[position - 63 : position + 1].loc[:, ["SPY", *DEFENSE]]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": CLOCK,
                    "window_dates": window.index.strftime("%Y-%m-%d").tolist(),
                    "used_source_prices_binary64_hex": {
                        symbol: [float(value).hex() for value in window[symbol]]
                        for symbol in window.columns
                    },
                    "SPY_year_prior_source_date": str(close.index[position - 252].date()),
                    "SPY_year_prior_price_binary64_hex": float(
                        close["SPY"].iloc[position - 252]
                    ).hex(),
                }
            )
        row, state = decision(close, position)
        target.loc[day] = row
        audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": CLOCK,
            **state,
            "counterfactual_historical_policy_availability": True,
            "actual_policy_introduction_or_admission_backdated": False,
            "shadow_wealth_or_feedback_guarantee_transferred_to_main_capital": False,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "asset_order": list(close.columns),
                    "complete_target": row.tolist(),
                }
            )
    return target, audits
