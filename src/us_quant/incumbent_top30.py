"""Frozen Top30 baseline extracted from run_all_stock_pool_iteration.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.us_quant.paths import CONFIG_DIR
from src.us_quant.us_realistic_backtest_engine import ExecutionConfig


STRATEGY_ID = "user_nolev_top30_m126_invvol_m"
REFERENCE_SOURCE_SHA256 = "2eb267f05330c425d3b5409e3c38269ff7bee9a7fccec95c5841eaf450711e0a"


def load_config(path: Path = CONFIG_DIR / "incumbent_top30.json") -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version_name") != STRATEGY_ID:
        raise ValueError("Unexpected incumbent strategy identity")
    return config


def execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_capital=20_000.0,
        broker="IBKR_PRO",
        ibkr_pro_pricing="tiered",
        max_participation_rate=0.05,
        fractional_shares=True,
        rebalance_only_on_weight_change=True,
        allow_leverage=False,
        account_type="margin",
        broker_rule_uncertainty=False,
        margin_interest_rate_annual=0.0514,
    )


def rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    valid = prices.dropna(how="all")
    period = "W-FRI" if freq == "W" else "M"
    return pd.DatetimeIndex(valid.groupby(valid.index.to_period(period)).tail(1).index)


def tradable_on_date(prices: pd.DataFrame, returns: pd.DataFrame, tickers: list[str], date: pd.Timestamp, min_price: float) -> list[str]:
    px = prices.loc[date, tickers]
    ok_price = px >= min_price
    ok_hist = returns.loc[:date, tickers].tail(126).notna().sum() >= 100
    return px[ok_price & ok_hist].index.tolist()


def inverse_vol_weights(returns: pd.DataFrame, selected: list[str], date: pd.Timestamp, lookback: int) -> pd.Series:
    vol = returns.loc[:date, selected].tail(lookback).std(ddof=0).replace(0, np.nan)
    inv = (1 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    if inv.sum() <= 0:
        return pd.Series(1 / len(selected), index=selected)
    return inv / inv.sum()


def build_all_stock_weights(prices: pd.DataFrame, config: dict, all_trade_cols: list[str]) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    equities = [ticker for ticker in config["equities"] if ticker in prices.columns]
    weights = pd.DataFrame(np.nan, index=prices.index, columns=all_trade_cols)
    for date in rebalance_dates(prices, config["freq"]):
        if prices.loc[:date].shape[0] <= max(config["lookback"], config["market_ma"], config["stock_ma"], 126):
            continue
        qqq_ok = prices.loc[date, "QQQ"] > prices["QQQ"].loc[:date].tail(config["market_ma"]).mean()
        spy_ok = prices.loc[date, "SPY"] > prices["SPY"].loc[:date].tail(config["market_ma"]).mean()
        vix = float(prices.loc[date, "^VIX"]) if "^VIX" in prices.columns and pd.notna(prices.loc[date, "^VIX"]) else np.nan
        vix_ok = bool(np.isnan(vix) or vix <= config["vix_threshold"])
        market_drawdown = prices.loc[date, "QQQ"] / prices["QQQ"].loc[:date].tail(config["dd_window"]).max() - 1
        dd_ok = market_drawdown >= config["dd_limit"]
        risk_on = (qqq_ok or spy_ok) and vix_ok and dd_ok
        row = pd.Series(0.0, index=all_trade_cols)
        if risk_on:
            universe_today = tradable_on_date(prices, returns, equities, date, config["min_price"])
            if universe_today:
                mom = prices.loc[date, universe_today] / prices.shift(config["lookback"]).loc[date, universe_today] - 1.0
                trend = prices.loc[date, universe_today] > prices[universe_today].loc[:date].tail(config["stock_ma"]).mean()
                recent_vol = returns.loc[:date, universe_today].tail(63).std(ddof=0) * np.sqrt(252)
                vol_ok = recent_vol <= config["max_stock_vol"]
                selected = (
                    mom[(mom > config["min_momentum"]) & trend & vol_ok]
                    .sort_values(ascending=False)
                    .head(config["top_n"])
                    .index.tolist()
                )
            else:
                selected = []
            if selected:
                if config["weighting"] == "inverse_vol":
                    stock_weights = inverse_vol_weights(returns, selected, date, config["vol_lookback"])
                else:
                    stock_weights = pd.Series(1 / len(selected), index=selected)
                for ticker, weight in stock_weights.items():
                    row[ticker] = weight * config["stock_sleeve"]
            for ticker, weight in config.get("overlay_weights", {}).items():
                if ticker in row.index:
                    row[ticker] += weight
            if row.sum() < config["gross_exposure"] and "BIL" in row.index:
                row["BIL"] += max(0.0, min(1.0, config["gross_exposure"] - row.sum()))
        else:
            for ticker, weight in config["defense_weights"].items():
                if ticker in row.index:
                    row[ticker] += weight
        if row.sum() > 0:
            if not config["allow_margin"] and row.sum() > 1.0:
                row = row / row.sum()
            elif config["allow_margin"] and row.sum() > config["gross_exposure"]:
                row = row / row.sum() * config["gross_exposure"]
        weights.loc[date] = row
    return weights.ffill().fillna(0.0)
