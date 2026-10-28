from __future__ import annotations

import math

import numpy as np
import pandas as pd

from us_quant import daily_variance_disagreement_v1 as previous
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = previous.ROOT
POLICY = ROOT / "config/symmetric-variance-disagreement-v1.json"
CORE, STOCKS, UNIVERSE, FACTOR_IDS = (
    previous.CORE,
    previous.STOCKS,
    previous.UNIVERSE,
    previous.FACTOR_IDS,
)
PRIOR_HEAD = "2f5ea6b5667c855f1c41966ca95a833c781e650928bb11ec477718e40b98d3cb"
CANDIDATES = (
    {
        "id": "dollar_option_symmetric_variance_disagreement_v1",
        "construction": "symmetric_daily_disagreement",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_symmetric_variance_disagreement_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 189,
        "prior_journal_sequence": 130,
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
            "Preserve the single symmetric information extension and all original bounds."
        )


def equity_share(gvz: float, vix: float, stock_logs: np.ndarray, gold_logs: np.ndarray) -> tuple:
    stock_logs, gold_logs = np.asarray(stock_logs, dtype=float), np.asarray(gold_logs, dtype=float)
    if (
        stock_logs.shape != (21,)
        or gold_logs.shape != (21,)
        or not np.isfinite(stock_logs).all()
        or not np.isfinite(gold_logs).all()
        or not np.isfinite(gvz)
        or not np.isfinite(vix)
        or min(gvz, vix) <= 0
    ):
        raise QuantError(
            "Both actual daily return windows and positive monthly quotes are required."
        )
    rv_stock = float(252 * np.mean(stock_logs**2))
    rv_gold = float(252 * np.mean(gold_logs**2))
    if min(rv_stock, rv_gold) <= 0 or not np.isfinite(rv_stock + rv_gold):
        raise QuantError("Invalid realized variance cannot become a safe target or standby.")
    g, v = gvz / 100, vix / 100
    raw = v * rv_gold / (v * rv_gold + g * rv_stock)
    share = float(np.clip(raw, 0.30, 0.70))
    return share, {
        "RV_SPY_annualized_daily_log_squares": rv_stock,
        "RV_GLD_annualized_daily_log_squares": rv_gold,
        "equity_implied_over_backward_realized_variance_ratio": v * v / rv_stock,
        "gold_implied_over_backward_realized_variance_ratio": g * g / rv_gold,
        "unclipped_equity_share": raw,
        "equity_share": share,
        "true_variance_risk_premia_or_intraday_paper_claimed": False,
    }


def build_targets(
    data: MarketData,
    source: pd.DataFrame,
    options: pd.DataFrame,
    policy: dict,
) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not source.index.equals(data.close.index)
        or not source.columns.equals(data.close.columns)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or (source.isna().any(axis=1) & ~source.isna().all(axis=1)).any()
    ):
        raise QuantError("Preserve the complete original twelve-asset calendar and sparse source.")
    active = source.dropna(how="all")
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
        raise QuantError(
            "Original monthly source and option observations cannot be filled or dropped."
        )
    target = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    audits = []
    for day, row in active.iterrows():
        position = data.close.index.get_loc(day)
        if position < 21:
            raise QuantError("Every monthly decision needs22actualSPY/GLDcloses.")
        past = data.close.loc[:, ["SPY", "GLD"]].iloc[position - 21 : position + 1]
        logs = np.diff(np.log(past.to_numpy()), axis=0)
        g, v = map(float, options.loc[day, ["GVZ", "VIX"]])
        risky, old_stock = previous.source_row(row, g, v)
        share, proof = equity_share(g, v, logs[:, 0], logs[:, 1])
        target.loc[day] = previous.transform(row, share, risky)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_price_observation": str(past.index[0].date()),
                "last_price_observation": str(day.date()),
                "realized_return_observations": 21,
                "source_non_BIL_budget_preserved": risky,
                "source_BIL_target_preserved": float(row["BIL"]),
                "source_equity_share": old_stock / risky,
                "GVZ_native_quote": g,
                "VIX_native_quote": v,
                **proof,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(
    close: pd.DataFrame, row: pd.Series, option: pd.Series, day: pd.Timestamp
) -> pd.Series:
    position = close.index.get_loc(day)
    if position < 21:
        raise QuantError("Independent symmetric proxy lacks its required actual price window.")
    rv = {
        name: 252
        * math.fsum(
            math.log(float(close.iloc[i][name]) / float(close.iloc[i - 1][name])) ** 2
            for i in range(position - 20, position + 1)
        )
        / 21
        for name in ("SPY", "GLD")
    }
    g, v = float(option["GVZ"]) / 100, float(option["VIX"]) / 100
    if min(g, v, rv["SPY"], rv["GLD"]) <= 0 or not np.isfinite(g + v + sum(rv.values())):
        raise QuantError("Independent proxy rejects invalid quotes or realized variances.")
    odds = (g / v) * ((v * v / rv["SPY"]) / (g * g / rv["GLD"]))
    share = max(0.30, min(0.70, odds / (1 + odds)))
    risky = math.fsum(float(row[name]) for name in (*STOCKS, "GLD"))
    result = row.copy()
    result["SOXX"] = risky * share / 2
    for name in CORE:
        result[name] = risky * share / 8
    result["GLD"] = risky * (1 - share)
    return result
