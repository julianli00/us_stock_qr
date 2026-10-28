from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.ssd_factor_catalog_v1 import EXISTING

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/daily-variance-disagreement-v1.json"
CORE = ("MTUM", "VLUE", "QUAL", "USMV")
STOCKS = (*CORE, "SOXX")
UNIVERSE = tuple(EXISTING)
FACTOR_IDS = ("price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure")
UNUSED = ("SPY", "IEF", "TLT", "QQQ", "XLK")
PRIOR_HEAD = "dc097d8f64e361c68f1c6ac8d9cbc6d16129aaeff3423633faf3b8c78702badd"
CANDIDATES = (
    {
        "id": "dollar_option_daily_variance_disagreement_v1",
        "construction": "daily_variance_disagreement_odds_tilt",
    },
)
SOURCE_TARGET_SHA = "bd1c3ba7b9a1e5d1ace69a8e4d17b67b6ba8aafeab32ec2271e112bfdc957261"


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_daily_variance_disagreement_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 188,
        "prior_journal_sequence": 118,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "semiconductor_option_proxy_dollar_observed_band025",
        "realized_return_sessions": 21,
        "annualization_sessions": 252,
        "monthly_decisions": 131,
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
    expected_refs = {
        "source_registration": {
            "path": "evidence/implied_execution_20261012_registration.json",
            "sha256": "5a52f3bafc13358933c67af51d2625455ac0bc4a56e5a79a555ed1dca80222bb",
        },
        "source_targets": {
            "path": (
                "data/implied-risk-execution-20261012/"
                "semiconductor_option_proxy_dollar_observed_band025-desired-targets.csv"
            ),
            "sha256": SOURCE_TARGET_SHA,
        },
        "monthly_options": {
            "path": "data/gold-volatility-inputs-20261011/monthly-closes.csv",
            "sha256": "f4afd690798d9eaa46da84674a9bf1ec47e54e84d82cd77cc41f666e14d74a79",
        },
    }
    if (
        any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in expected.items()
        )
        or any(policy.get(key) != value for key, value in expected_refs.items())
        or any(
            policy.get("methodology", {}).get(key) is not False
            for key in (
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError(
            "Keep the single fixed daily-proxy rule and exact preserved source identities."
        )


def equity_share(gvz: float, vix: float, log_returns: np.ndarray) -> tuple[float, dict]:
    observed = np.asarray(log_returns, dtype=float)
    if (
        observed.shape != (21,)
        or not np.isfinite(observed).all()
        or not np.isfinite(gvz)
        or not np.isfinite(vix)
        or gvz <= 0
        or vix <= 0
    ):
        raise QuantError(
            "Require actual positive monthly option quotes and 21 complete daily returns."
        )
    realized = float(252 * np.mean(observed**2))
    if realized <= 0 or not np.isfinite(realized):
        raise QuantError("Zero or invalid realized variance is not a safe cap or cash default.")
    g, v = gvz / 100, vix / 100
    share = float(np.clip(g * v / (g * v + realized), 0.30, 0.70))
    return share, {
        "realized_SPY_daily_log_variance_annualized": realized,
        "VIX_implied_variance_proxy": v * v,
        "implied_over_backward_realized_variance_ratio": v * v / realized,
        "unclipped_equity_share": g * v / (g * v + realized),
        "equity_share": share,
        "true_variance_risk_premium_estimated": False,
        "intraday_paper_prediction_claimed": False,
    }


def source_row(row: pd.Series, gvz: float, vix: float) -> tuple[float, float]:
    if (
        tuple(row.index) != UNIVERSE
        or not np.isfinite(row).all()
        or (row < 0).any()
        or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
        or (row.loc[list(UNUSED)].abs() > 1e-12).any()
        or not np.isfinite(gvz + vix)
        or min(gvz, vix) <= 0
    ):
        raise QuantError("The unchanged source allocation and native option units are required.")
    equity = float(row.loc[list(STOCKS)].sum())
    risky = equity + float(row["GLD"])
    if (
        not any(np.isclose(risky, scale, rtol=0, atol=1e-12) for scale in (0.49, 0.98))
        or not np.isclose(row["BIL"], 0.98 - risky, rtol=0, atol=1e-12)
        or not np.isclose(
            equity / risky, np.clip(gvz / (gvz + vix), 0.30, 0.70), rtol=0, atol=1e-12
        )
        or not np.isclose(row["SOXX"], equity / 2, rtol=0, atol=1e-12)
        or not np.allclose(row.loc[list(CORE)], equity / 8, rtol=0, atol=1e-12)
    ):
        raise QuantError(
            "Do not reconstruct or change the original option budget/dollar guard/core."
        )
    return risky, equity


def transform(row: pd.Series, share: float, risky: float) -> pd.Series:
    result = row.copy()
    equity = risky * share
    result.loc[list(CORE)] = equity / 8
    result["SOXX"] = equity / 2
    result["GLD"] = risky - equity
    if (
        (result < 0).any()
        or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)
        or result["BIL"] != row["BIL"]
        or not result.loc[list(UNUSED)].equals(row.loc[list(UNUSED)])
    ):
        raise QuantError(
            "The daily-proxy transformation changed funding or unrelated source targets."
        )
    return result


def build_targets(
    data: MarketData,
    source_targets: pd.DataFrame,
    options: pd.DataFrame,
    policy: dict,
) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not source_targets.index.equals(data.close.index)
        or not source_targets.columns.equals(data.close.columns)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or (source_targets.isna().any(axis=1) & ~source_targets.isna().all(axis=1)).any()
    ):
        raise QuantError(
            "Keep the original complete twelve-asset calendar and sparse monthly source."
        )
    active = source_targets.dropna(how="all")
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
        raise QuantError("Original monthly source/option rows must remain aligned without fill.")
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    audits = []
    for day, row in active.iterrows():
        location = data.close.index.get_loc(day)
        if location < 21:
            raise QuantError(
                "The new proxy requires 22 actual SPY closes before every original date."
            )
        past = data.close["SPY"].iloc[location - 21 : location + 1]
        log_returns = np.diff(np.log(past.to_numpy()))
        gvz, vix = map(float, options.loc[day, ["GVZ", "VIX"]])
        risky, old_equity = source_row(row, gvz, vix)
        share, proof = equity_share(gvz, vix, log_returns)
        targets.loc[day] = transform(row, share, risky)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_price_observation": str(past.index[0].date()),
                "last_price_observation": str(day.date()),
                "realized_return_observations": 21,
                "source_non_BIL_budget_preserved": risky,
                "source_BIL_target_preserved": float(row["BIL"]),
                "source_equity_share": old_equity / risky,
                "GVZ_native_quote": gvz,
                "VIX_native_quote": vix,
                **proof,
            }
        )
    return {CANDIDATES[0]["id"]: targets}, audits


def independent_row(
    close: pd.DataFrame, row: pd.Series, option: pd.Series, day: pd.Timestamp
) -> pd.Series:
    location = close.index.get_loc(day)
    if location < 21:
        raise QuantError("Independent proxy reconstruction lacks its actual price window.")
    squares = [
        math.log(float(close.iloc[i]["SPY"]) / float(close.iloc[i - 1]["SPY"])) ** 2
        for i in range(location - 20, location + 1)
    ]
    realized = 252 * math.fsum(squares) / 21
    gvz, vix = float(option["GVZ"]), float(option["VIX"])
    if realized <= 0 or min(gvz, vix) <= 0 or not np.isfinite(realized + gvz + vix):
        raise QuantError("Invalid observations cannot yield an independent proxy allocation.")
    risky = math.fsum(float(row[name]) for name in (*STOCKS, "GLD"))
    original_odds = gvz / vix
    variance_ratio = (vix / 100) ** 2 / realized
    tilted_odds = original_odds * variance_ratio
    share = max(0.30, min(0.70, tilted_odds / (1 + tilted_odds)))
    result = row.copy()
    result["SOXX"] = risky * share * 0.5
    for name in CORE:
        result[name] = risky * share * 0.125
    result["GLD"] = risky * (1 - share)
    return result
