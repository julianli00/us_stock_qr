from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import CORE, FACTOR_IDS, STOCKS, UNIVERSE
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/relative-equity-odds-v1.json"
LOOKBACK, SKIP = 126, 21
PRIOR_HEAD = "3c68f3bc663ea90c9cce0e3ad44f32c4e0d24ca3597ed62d0151992d13e3eb2b"
CANDIDATES = (
    {
        "id": "daily_proxy_relative_equity_odds_v1",
        "construction": "SOXX_vs_equal_start_core_relative_wealth",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_relative_equity_odds_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 191,
        "prior_journal_sequence": 154,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_daily_variance_disagreement_v1",
        "lookback_sessions": LOOKBACK,
        "skip_recent_sessions": SKIP,
        "original_source_months": 131,
        "eligible_source_months": 128,
        "undefined_prehistory_source_months": 3,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
        "source_registration": {
            "path": "evidence/daily_variance_disagreement_20261028_registration_v1.json",
            "sha256": "fe9d39e5bab50f33a71c36a303893e7ff6e9e2450bb39750fdd5855a4ee3d176",
        },
        "source_targets": {
            "path": (
                "data/ssd-daily-variance-disagreement-20261028-v1/"
                "dollar_option_daily_variance_disagreement_v1/targets.csv"
            ),
            "sha256": "665f4a7c8e43fc6030eddf6b6c0bb4b342beeef9caeb683dd87271008a49ec8d",
        },
    }
    if any(
        policy.get(k) != x or type(policy.get(k)) is not type(x) for k, x in expected.items()
    ) or any(
        policy.get("methodology", {}).get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError(
            "Preserve the fixed relative-wealth construction, inherited horizon and source."
        )


def relative_share(close: pd.DataFrame, day: pd.Timestamp) -> tuple[float, dict]:
    location = close.index.get_loc(day)
    if location < LOOKBACK:
        raise QuantError("Relative equity odds require their full actual prehistory.")
    start, end = close.iloc[location - LOOKBACK], close.iloc[location - SKIP]
    endpoints = close.iloc[[location - LOOKBACK, location - SKIP]][list(STOCKS)]
    if not np.isfinite(endpoints).all().all() or (endpoints <= 0).any().any():
        raise QuantError("Each actual fund needs finite positive historical endpoints.")
    gross = end[list(STOCKS)] / start[list(STOCKS)]
    core = float(gross[list(CORE)].mean())
    semiconductor = float(gross["SOXX"])
    share = semiconductor / (semiconductor + core)
    if not np.isfinite(share) or not 0 < share < 1:
        raise QuantError("Invalid relative wealth cannot become a neutral allocation fallback.")
    return share, {
        "first_price_observation": str(close.index[location - LOOKBACK].date()),
        "last_price_observation": str(close.index[location - SKIP].date()),
        "observed_return_sessions": LOOKBACK - SKIP,
        "SOXX_gross_wealth_factor": semiconductor,
        "equal_start_core_gross_wealth_factor": core,
        "SOXX_share_of_unchanged_equity": share,
        "synthetic_daily_rebalanced_core_index_claimed": False,
    }


def source_equity(row: pd.Series) -> float:
    if (
        tuple(row.index) != tuple(UNIVERSE)
        or not np.isfinite(row).all()
        or (row < 0).any()
        or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Use the complete unchanged funded source target.")
    equity = float(row[list(STOCKS)].sum())
    if (
        equity <= 0
        or not np.isclose(row["SOXX"], equity / 2, rtol=0, atol=1e-12)
        or not np.allclose(row[list(CORE)], equity / 8, rtol=0, atol=1e-12)
    ):
        raise QuantError("The fixed original half-sector/equal-core source is required.")
    return equity


def build_targets(data: MarketData, desired: pd.DataFrame, policy: dict) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != tuple(UNIVERSE)
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
    ):
        raise QuantError("Keep the full original market calendar and sparse source target schema.")
    active = desired.dropna()
    if active.empty or not all(is_month_end(day) for day in active.index):
        raise QuantError("Only completed original monthly decisions are eligible.")
    target = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    audits = []
    for day, row in active.iterrows():
        equity = source_equity(row)
        if data.close.index.get_loc(day) < LOOKBACK:
            continue
        share, details = relative_share(data.close, day)
        result = row.copy()
        result["SOXX"] = equity * share
        result.loc[list(CORE)] = equity * (1 - share) / 4
        unchanged = [name for name in desired.columns if name not in STOCKS]
        if (
            not result[unchanged].equals(row[unchanged])
            or not np.isclose(result[list(STOCKS)].sum(), equity, rtol=0, atol=1e-12)
            or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)
            or (result[list(STOCKS)] <= 0).any()
        ):
            raise QuantError(
                "Relative allocation changed total equity, defense, or genuine fund participation."
            )
        target.loc[day] = result
        audits.append(
            {
                "decision_session": str(day.date()),
                "source_equity_budget_preserved": equity,
                "GLD_target_unchanged": float(row["GLD"]),
                "BIL_target_unchanged": float(row["BIL"]),
                **details,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(close: pd.DataFrame, row: pd.Series, day: pd.Timestamp) -> pd.Series:
    location = close.index.get_loc(day)
    if location < LOOKBACK:
        raise QuantError("The independent relative-wealth proof lacks full prehistory.")
    factors = {
        name: math.prod(
            float(close.iloc[i][name]) / float(close.iloc[i - 1][name])
            for i in range(location - LOOKBACK + 1, location - SKIP + 1)
        )
        for name in STOCKS
    }
    core = math.fsum(factors[name] for name in CORE) / 4
    share = 1 / (1 + core / factors["SOXX"])
    equity = math.fsum(float(row[name]) for name in STOCKS)
    result = row.copy()
    result["SOXX"] = equity * share
    for name in CORE:
        result[name] = equity * (1 - share) / 4
    return result
