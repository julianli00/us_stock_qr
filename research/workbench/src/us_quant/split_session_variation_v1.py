from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import daily_variance_disagreement_v1 as original
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/split-session-variation-v1.json"
CORE, STOCKS, UNIVERSE = original.CORE, original.STOCKS, original.UNIVERSE
PRIOR_HEAD = "e2d1ef0ff76db4e232dfd0d0a0ae15bcdccfbcba335b606867c7160094344793"
CANDIDATES = (
    {
        "id": "dollar_option_two_segment_variation_v1",
        "construction": "overnight_and_open_to_close_squared_returns",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_split_session_variation_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 192,
        "prior_journal_sequence": 166,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "semiconductor_option_proxy_dollar_observed_band025",
        "realized_return_sessions": 21,
        "annualization_sessions": 252,
        "segments_per_session": 2,
        "monthly_decisions": 131,
        "equity_share_min": 0.30,
        "equity_share_max": 0.70,
        "growth_share_of_equity": 0.50,
        "core_symbols": list(CORE),
        "factor_ids": list(original.FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
        "daily_proxy_registration": {
            "path": "evidence/daily_variance_disagreement_20261028_registration_v1.json",
            "sha256": "fe9d39e5bab50f33a71c36a303893e7ff6e9e2450bb39750fdd5855a4ee3d176",
        },
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed two-segment information partition and acceptance scope.")


def measure(closes: np.ndarray, opens: np.ndarray) -> tuple[float, dict]:
    close, open_ = np.asarray(closes, dtype=float), np.asarray(opens, dtype=float)
    if (
        close.shape != (22,)
        or open_.shape != (21,)
        or not np.isfinite(close).all()
        or not np.isfinite(open_).all()
        or (close <= 0).any()
        or (open_ <= 0).any()
    ):
        raise QuantError("Require22positive adjusted closes and21same-session adjusted opens.")
    overnight = np.log(open_) - np.log(close[:-1])
    intraday = np.log(close[1:]) - np.log(open_)
    daily = np.diff(np.log(close))
    split = float(252 * np.mean(overnight**2 + intraday**2))
    old = float(252 * np.mean(daily**2))
    cross = float(504 * np.mean(overnight * intraday))
    identity_error = abs(old - split - cross)
    if not np.isfinite(split + old + cross) or split <= 0:
        raise QuantError("Zero or invalid two-segment variation is not a safe allocation default.")
    if identity_error > 1e-12 * max(1.0, old, split, abs(cross)):
        raise QuantError("The exact close-only/two-segment/cross-term identity failed.")
    return split, {
        "SPY_two_segment_squared_log_variation_annualized": split,
        "SPY_close_only_squared_log_variation_annualized": old,
        "SPY_overnight_squared_log_component_annualized": float(252 * np.mean(overnight**2)),
        "SPY_open_to_close_squared_log_component_annualized": float(252 * np.mean(intraday**2)),
        "signed_cross_term_annualized": cross,
        "maximum_log_additivity_error": float(np.max(np.abs(daily - overnight - intraday))),
        "squared_variation_identity_error": identity_error,
    }


def equity_share(gvz: float, vix: float, variation: float) -> float:
    if (
        not np.isfinite(gvz)
        or not np.isfinite(vix)
        or not np.isfinite(variation)
        or min(gvz, vix, variation) <= 0
    ):
        raise QuantError("Require positive native option quotes and measured variation.")
    product = (gvz / 100) * (vix / 100)
    if product <= 0 or not np.isfinite(product + variation):
        raise QuantError("Option-unit conversion overflow or underflow cannot become a target.")
    return float(np.clip(product / (product + variation), 0.30, 0.70))


def build_targets(
    data: MarketData, desired: pd.DataFrame, options: pd.DataFrame, policy: dict
) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
    ):
        raise QuantError("Keep the full original calendar and sparse monthly source targets.")
    active = desired.dropna(how="all")
    if (
        active.empty
        or not active.index.equals(options.index)
        or list(options.columns) != ["GVZ", "VIX"]
        or not options.index.is_unique
        or not options.index.is_monotonic_increasing
        or not np.isfinite(options).all().all()
        or (options <= 0).any().any()
        or not all(is_month_end(day) for day in active.index)
    ):
        raise QuantError("Preserve every aligned completed month and its actual option quotes.")
    result = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    audits = []
    for day, row in active.iterrows():
        location = data.close.index.get_loc(day)
        if location < 21:
            raise QuantError("The first original decision requires its full21-session prehistory.")
        closes = data.close["SPY"].iloc[location - 21 : location + 1]
        opens = data.open["SPY"].iloc[location - 20 : location + 1]
        if not opens.index.equals(closes.index[1:]):
            raise QuantError("Open and close intervals cannot cross different session labels.")
        variation, details = measure(closes.to_numpy(), opens.to_numpy())
        gvz, vix = map(float, options.loc[day])
        risky, _ = original.source_row(row, gvz, vix)
        share = equity_share(gvz, vix, variation)
        result.loc[day] = original.transform(row, share, risky)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_prior_close": str(closes.index[0].date()),
                "first_open_session": str(opens.index[0].date()),
                "last_price_observation": str(day.date()),
                "sessions": 21,
                "segments": 42,
                "source_non_BIL_budget_preserved": risky,
                "source_BIL_target_preserved": float(row["BIL"]),
                "equity_share": share,
                "GVZ_native_quote": gvz,
                "VIX_native_quote": vix,
                "true_variance_risk_premium_or_high_frequency_replication_claimed": False,
                **details,
            }
        )
    return {CANDIDATES[0]["id"]: result}, audits


def independent_row(
    data: MarketData, row: pd.Series, option: pd.Series, day: pd.Timestamp
) -> pd.Series:
    location = data.close.index.get_loc(day)
    if location < 21:
        raise QuantError("Independent reconstruction needs all actual interval endpoints.")
    squares = []
    for position in range(location - 20, location + 1):
        prior = float(data.close.iloc[position - 1]["SPY"])
        open_ = float(data.open.iloc[position]["SPY"])
        close = float(data.close.iloc[position]["SPY"])
        if min(prior, open_, close) <= 0 or not np.isfinite(prior + open_ + close):
            raise QuantError("An independent interval endpoint is invalid.")
        squares.extend((math.log(open_ / prior) ** 2, math.log(close / open_) ** 2))
    variation = 252 * math.fsum(squares) / 21
    gvz, vix = float(option["GVZ"]), float(option["VIX"])
    if min(gvz, vix, variation) <= 0 or not np.isfinite(gvz + vix + variation):
        raise QuantError("Independent information inputs must be finite and positive.")
    odds = (gvz / vix) * (vix / 100) ** 2 / variation
    share = max(0.30, min(0.70, odds / (1 + odds)))
    risky = math.fsum(float(row[name]) for name in (*STOCKS, "GLD"))
    result = row.copy()
    result["SOXX"] = risky * share / 2
    for name in CORE:
        result[name] = risky * share / 8
    result["GLD"] = risky * (1 - share)
    return result
