from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.anfci_residual_confirmation_v1 import CORE, STOCKS, UNIVERSE, donor_budget
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import FACTOR_IDS
from us_quant.data import MarketData
from us_quant.split_session_variation_v1 import equity_share

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/overnight-range-product-v1.json"
FIELDS = ("open", "high", "low", "close")
PRIOR_HEAD = "47da54d89093b8897a6c1522e6dfa05f4ae1d5c6429c5e4c528e6a1e2c6eec65"
CANDIDATES = (
    {
        "id": "dollar_option_overnight_range_product_v1",
        "construction": "unchanged_overnight_square_plus_raw_intraday_range_products",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_overnight_range_product_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 197,
        "prior_journal_sequence": 228,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_two_segment_variation_v1",
        "realized_return_sessions": 21,
        "annualization_sessions": 252,
        "monthly_decisions": 131,
        "intraday_fields": list(FIELDS),
        "equity_share_min": 0.30,
        "equity_share_max": 0.70,
        "growth_share_of_equity": 0.50,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "primary_full_paper_verified",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed field-based proxy and original acceptance boundaries.")


def measure(closes, opens, raw_bars):
    closes, opens, bars = (np.asarray(value, dtype=float) for value in (closes, opens, raw_bars))
    if (
        closes.shape != (22,)
        or opens.shape != (21,)
        or bars.shape != (21, 4)
        or not all(
            np.isfinite(value).all() and (value > 0).all() for value in (closes, opens, bars)
        )
        or (bars[:, 2] > np.minimum(bars[:, 0], bars[:, 3])).any()
        or (bars[:, 1] < np.maximum(bars[:, 0], bars[:, 3])).any()
    ):
        raise QuantError("Require22closes/21opens and21valid native OHLC observations.")
    overnight = np.log(opens) - np.log(closes[:-1])
    u = np.log(bars[:, 1]) - np.log(bars[:, 0])
    low = np.log(bars[:, 2]) - np.log(bars[:, 0])
    c = np.log(bars[:, 3]) - np.log(bars[:, 0])
    range_product = u * (u - c) + low * (low - c)
    if (range_product < 0).any() or not np.isfinite(range_product).all():
        raise QuantError("Valid raw bar products cannot become negative or nonfinite.")
    overnight_part = float(252 * np.mean(overnight**2))
    range_part = float(252 * np.mean(range_product))
    total = overnight_part + range_part
    if not np.isfinite(total) or total <= 0:
        raise QuantError("Zero or invalid total range proxy is not a safe equity-risk default.")
    return total, {
        "overnight_squared_component_annualized": overnight_part,
        "intraday_range_product_component_annualized": range_part,
        "combined_proxy_annualized": total,
        "minimum_daily_range_product": float(range_product.min()),
    }


def build_targets(data: MarketData, desired: pd.DataFrame, information: dict, policy: dict):
    validate(policy)
    data.validate()
    bars, options = information["bars"], information["options"]
    if (
        tuple(data.close.columns) != UNIVERSE
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
        or list(bars.columns) != list(FIELDS)
        or not bars.index.equals(data.close.index)
    ):
        raise QuantError("Preserve the full registered market and exact source range calendar.")
    active = desired.dropna(how="all")
    if (
        active.empty
        or not active.index.equals(options.index)
        or list(options.columns) != ["GVZ", "VIX"]
        or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(options).all().all()
        or (options <= 0).any().any()
    ):
        raise QuantError("Every original month requires its actual complete option observations.")
    target = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    audits = []
    for day, old in active.iterrows():
        location = data.close.index.get_loc(day)
        if location < 21:
            raise QuantError("Keep all21complete sessions and their actual previous close.")
        closes = data.close["SPY"].iloc[location - 21 : location + 1]
        opens = data.open["SPY"].iloc[location - 20 : location + 1]
        raw = bars.iloc[location - 20 : location + 1]
        if not raw.index.equals(opens.index) or not opens.index.equals(closes.index[1:]):
            raise QuantError(
                "The current/prior prices and intraday extrema belong to different dates."
            )
        variance, details = measure(closes, opens, raw)
        budget, _ = donor_budget(old)
        gvz, vix = map(float, options.loc[day])
        share = equity_share(gvz, vix, variance)
        row = old.copy()
        row["SOXX"] = budget * share / 2
        row.loc[list(CORE)] = budget * share / 8
        row["GLD"] = budget * (1 - share)
        if row["BIL"] != old["BIL"] or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12):
            raise QuantError("A measurement change cannot release the original BIL defense.")
        target.loc[day] = row
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_prior_close": str(closes.index[0].date()),
                "first_intraday_session": str(raw.index[0].date()),
                "last_price_observation": str(day.date()),
                "source_BIL_target_preserved": float(old["BIL"]),
                "source_non_BIL_budget_preserved": budget,
                "equity_share": share,
                "GVZ_native_quote": gvz,
                "VIX_native_quote": vix,
                "raw_intraday_common_adjustment_invariance": True,
                "unbiased_variance_or_true_premium_claimed": False,
                **details,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(data: MarketData, old: pd.Series, information: dict, day: pd.Timestamp):
    index = data.close.index
    position = index.get_loc(day)
    if position < 21:
        raise QuantError("Independent range verification needs the actual prior21sessions.")
    gaps, ranges = [], []
    for i in range(position - 20, position + 1):
        o, h, low, c = map(float, information["bars"].loc[index[i], list(FIELDS)])
        gap = math.log(
            float(data.open.loc[index[i], "SPY"]) / float(data.close.loc[index[i - 1], "SPY"])
        )
        gaps.append(gap * gap)
        ranges.append(math.log(h / o) * math.log(h / c) + math.log(low / o) * math.log(low / c))
    overnight_part = 252 * math.fsum(gaps) / 21
    range_part = 252 * math.fsum(ranges) / 21
    variance = overnight_part + range_part
    if not math.isfinite(variance) or variance <= 0:
        raise QuantError("Independent proxy is invalid; do not substitute a neutral allocation.")
    gvz, vix = map(float, information["options"].loc[day])
    odds = (gvz / vix) * (vix / 100) ** 2 / variance
    share = max(0.30, min(0.70, odds / (1 + odds)))
    budget = math.fsum(float(old[name]) for name in (*STOCKS, "GLD"))
    result = old.copy()
    result["SOXX"] = budget * share / 2
    for name in CORE:
        result[name] = budget * share / 8
    result["GLD"] = budget * (1 - share)
    return result, {
        "overnight_squared_component_annualized": overnight_part,
        "intraday_range_product_component_annualized": range_part,
        "combined_proxy_annualized": variance,
    }
