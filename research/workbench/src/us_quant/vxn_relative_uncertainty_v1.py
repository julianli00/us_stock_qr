from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.relative_equity_odds_v1 import CORE, FACTOR_IDS, STOCKS, UNIVERSE, source_equity
from us_quant.split_session_variation_v1 import measure

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/vxn-relative-uncertainty-v1.json"
PRIOR_HEAD = "e627683ada2fc5ed539c0676aee694c7b62a76f9567e2ad8cfc9435006ab26e7"
CANDIDATES = (
    {
        "id": "hmm_vxn_relative_uncertainty_v1",
        "construction": "underlying_matched_relative_option_uncertainty_odds",
    },
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_vxn_relative_uncertainty_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 204,
        "prior_registered_ssd_configurations": 27,
        "prior_journal_sequence": 309,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "causal_two_state_growth_bil_v1",
        "realized_return_sessions": 21,
        "annualization_sessions": 252,
        "segments_per_session": 2,
        "market_proxy_symbols": ["SPY", "QQQ"],
        "option_column_order": ["VXN", "VIX"],
        "source_target_months": 122,
        "additional_source_months_dropped": 0,
        "undefined_prehistory_calendar_months": 9,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict)
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in expected.items()
        )
        or any(
            policy.get("methodology", {}).get(key) is not False
            for key in (
                "source_model_refitting",
                "new_source_requests",
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError(
            "Preserve the fixed relative-uncertainty question and original acceptance."
        )


def growth_share(vxn, vix, qqq_variation, spy_variation):
    if not all(
        math.isfinite(value) and value > 0 for value in (vxn, vix, qqq_variation, spy_variation)
    ):
        raise QuantError("Every native quote and measured variation must be finite and positive.")
    ndx_implied, spx_implied = (vxn / 100) * (vxn / 100), (vix / 100) * (vix / 100)
    ndx, spx = ndx_implied / qqq_variation, spx_implied / spy_variation
    if not all(
        math.isfinite(value) and value > 0 for value in (ndx_implied, spx_implied, ndx, spx)
    ):
        raise QuantError("Invalid unit conversion or ratios cannot become a neutral share.")
    scale = max(ndx, spx)
    share = (spx / scale) / (spx / scale + ndx / scale)
    if not math.isfinite(share) or not 0 < share < 1:
        raise QuantError("The fixed positive odds must retain both stock sleeves without clipping.")
    return share, {
        "NDX_implied_variance_proxy": ndx_implied,
        "SPX_implied_variance_proxy": spx_implied,
        "NDX_implied_to_observed_ratio": ndx,
        "SPX_implied_to_observed_ratio": spx,
        "SOXX_share_of_stock_budget": share,
    }


def observed_variation(data, symbol, day):
    if symbol not in {"SPY", "QQQ"}:
        raise QuantError("Use only the declared matching underlying-market ETF proxies.")
    location = data.close.index.get_loc(day)
    if location < 21:
        raise QuantError("The relative uncertainty rule needs all21observed return sessions.")
    closes = data.close[symbol].iloc[location - 21 : location + 1]
    opens = data.open[symbol].iloc[location - 20 : location + 1]
    if not opens.index.equals(closes.index[1:]):
        raise QuantError(
            "Both observed market proxies need matching dated opening/closing intervals."
        )
    value, details = measure(closes.to_numpy(), opens.to_numpy())
    return value, {
        "symbol": symbol,
        "first_prior_close": str(closes.index[0].date()),
        "first_open_session": str(opens.index[0].date()),
        "last_observation": str(day.date()),
        "return_sessions": 21,
        "segments": 42,
        "two_segment_squared_log_variation_annualized": value,
        "close_only_squared_log_variation_annualized": details[
            "SPY_close_only_squared_log_variation_annualized"
        ],
        "overnight_squared_log_component_annualized": details[
            "SPY_overnight_squared_log_component_annualized"
        ],
        "open_to_close_squared_log_component_annualized": details[
            "SPY_open_to_close_squared_log_component_annualized"
        ],
        "signed_cross_term_annualized": details["signed_cross_term_annualized"],
        "squared_variation_identity_error": details["squared_variation_identity_error"],
    }


def eligible_dates(data, desired):
    dates = desired.dropna(how="all").index
    if not len(dates) or any(data.close.index.get_loc(day) < 21 for day in dates):
        raise QuantError("Do not remove source dates to manufacture the declared price prehistory.")
    return pd.DatetimeIndex(dates)


def build_targets(data: MarketData, desired: pd.DataFrame, information: pd.DataFrame, policy: dict):
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != tuple(UNIVERSE)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
        or not all(is_month_end(day) for day in desired.dropna(how="all").index)
        or not information.index.equals(data.close.index)
        or list(information.columns) != ["VXN", "VIX"]
        or not np.isfinite(information).all().all()
        or (information <= 0).any().any()
    ):
        raise QuantError(
            "Retain the complete original market, source calendar and admitted native quotes."
        )
    dates = eligible_dates(data, desired)
    target = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    records = []
    nonstock = desired.columns.difference(STOCKS)
    for day in dates:
        row = desired.loc[day]
        equity = source_equity(row)
        broad, broad_details = observed_variation(data, "SPY", day)
        growth, growth_details = observed_variation(data, "QQQ", day)
        vxn, vix = map(float, information.loc[day])
        share, details = growth_share(vxn, vix, growth, broad)
        result = row.copy()
        result["SOXX"] = equity * share
        result.loc[list(CORE)] = equity * (1 - share) / 4
        if (
            not result[nonstock].equals(row[nonstock])
            or abs(math.fsum(float(result[name]) for name in STOCKS) - equity) > 1e-12
            or abs(float(result.sum()) - 0.98) > 1e-12
            or (result.loc[list(STOCKS)] <= 0).any()
        ):
            raise QuantError(
                "Composition cannot change source stock capital, defense "
                "or genuine factor participation."
            )
        target.loc[day] = result
        records.append(
            {
                "decision_session": str(day.date()),
                "VXN_quote_session": str(day.date()),
                "VIX_quote_session": str(day.date()),
                "VXN_native_quote": vxn,
                "VIX_native_quote": vix,
                "observed_variation": {"SPY": broad_details, "QQQ": growth_details},
                **details,
                "source_equity_budget_preserved": equity,
                "BIL_target_unchanged": float(row["BIL"]),
                "GLD_target_unchanged": float(row["GLD"]),
                "information_clock_after_16_15_ET": True,
                "hmm_refitted_or_states_recomputed": False,
                "exact_portfolio_or_SOXX_implied_variance_claimed": False,
                "true_variance_risk_premium_claimed": False,
            }
        )
    return {CANDIDATES[0]["id"]: target}, {CANDIDATES[0]["id"]: records}


def independent_row(data, original, information, day):
    location = data.close.index.get_loc(day)
    if location < 21:
        raise QuantError("Independent target verification requires full declared prehistory.")
    variations = {}
    for symbol in ("SPY", "QQQ"):
        squares = []
        for i in range(location - 20, location + 1):
            previous = float(data.close[symbol].iloc[i - 1])
            opening = float(data.open[symbol].iloc[i])
            closing = float(data.close[symbol].iloc[i])
            if not all(
                math.isfinite(value) and value > 0 for value in (previous, opening, closing)
            ):
                raise QuantError("Unknown source prices cannot become a variance default.")
            squares.append(math.log(opening / previous) ** 2 + math.log(closing / opening) ** 2)
        variations[symbol] = 252 * math.fsum(squares) / 21
    vxn, vix = float(information["VXN"]), float(information["VIX"])
    numerator = (vix / 100) ** 2 * variations["QQQ"]
    other = (vxn / 100) ** 2 * variations["SPY"]
    if not all(math.isfinite(value) and value > 0 for value in (numerator, other)):
        raise QuantError("Independent cross-product odds cannot accept zero or invalid variation.")
    share = numerator / (numerator + other)
    if not 0 < share < 1:
        raise QuantError("Independent odds must preserve both positive stock sleeves.")
    equity = math.fsum(float(original[name]) for name in STOCKS)
    target = original.copy()
    target["SOXX"] = equity * share
    for name in CORE:
        target[name] = equity * (1 - share) / 4
    return target, share, variations
