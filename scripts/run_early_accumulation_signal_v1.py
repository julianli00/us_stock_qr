from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.paths import ARTIFACT_DIR  # noqa: E402


PREFIX = "early_accumulation_signal_v1"
NO_LIVE_ORDER_REASON = (
    "NO LIVE ORDER: Early-accumulation signal is a research/shadow decision layer. "
    "PIT/survivorship, live NBBO, broker position reconciliation, and production "
    "forward-OOS are not fully proven."
)
INITIAL_EQUITY = 10_000.0
TOP_N = 20
OUTPUT_SIGNAL_N = 12
V123_BEST_CANDIDATE = "v123_b04_alpha_reboost_rel42_flat_safe_h10_alpha01_a160_stop_04800_trail_07000"
V123_STATUS_PATH = ARTIFACT_DIR / "v123_forced_early_alpha_reboost_status.json"
V123_DAILY_STATUS_PATH = ARTIFACT_DIR / "v123_daily_shadow_status.json"
V123_DAILY_SIGNAL_EVENTS_PATH = ARTIFACT_DIR / "v123_daily_shadow_signal_events.csv"
V123_BEST_TARGET_WEIGHTS_PATH = ARTIFACT_DIR / "v123_forced_early_alpha_reboost_best_target_weights.csv"
V123_REFERENCE_MAX_STALE_DAYS = 7


def _safe_rank(s: pd.Series, higher_better: bool = True) -> pd.Series:
    r = s.replace([np.inf, -np.inf], np.nan).rank(pct=True)
    if not higher_better:
        r = 1 - r
    return r.fillna(0.0)


def _pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}%}"


def _money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_data() -> dict[str, Any]:
    stock_bars = pd.read_parquet(ARTIFACT_DIR / "us_all_stock_daily_bars.parquet")
    stock_bars["date"] = pd.to_datetime(stock_bars["date"]).dt.tz_localize(None)
    universe = pd.read_csv(ARTIFACT_DIR / "us_all_stock_tradable_liquid_universe.csv")
    universe["first_date"] = pd.to_datetime(universe["first_date"], errors="coerce")
    universe["last_date"] = pd.to_datetime(universe["last_date"], errors="coerce")
    universe = universe.loc[
        universe["stock_candidate"].astype(bool)
        & (~universe["is_etf"].astype(bool))
        & (universe["latest_adj_close"].fillna(0) >= 5)
        & (universe["median_daily_dollar_volume_63d"].fillna(0) >= 10_000_000)
    ].copy()

    tickers = sorted(set(universe["ticker"]).intersection(stock_bars["ticker"].unique()))
    stock_bars = stock_bars[stock_bars["ticker"].isin(tickers)].copy()

    etf_bars = pd.read_parquet(ROOT / "data" / "processed" / "us_daily_bars.parquet")
    etf_bars["date"] = pd.to_datetime(etf_bars["date"]).dt.tz_localize(None)
    etf_keep = ["SPY", "QQQ", "IWM", "BIL", "XLK", "SMH", "SOXX", "^VIX"]
    etf_bars = etf_bars[etf_bars["ticker"].isin(etf_keep)].copy()

    stock_raw_latest = pd.Timestamp(stock_bars["date"].max())
    etf_raw_latest = pd.Timestamp(etf_bars["date"].max())

    bars = pd.concat([stock_bars, etf_bars], ignore_index=True)
    prices = bars.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    high = bars.pivot(index="date", columns="ticker", values="adj_high").sort_index()
    low = bars.pivot(index="date", columns="ticker", values="adj_low").sort_index()
    volume = bars.pivot(index="date", columns="ticker", values="volume").sort_index()
    if "SPY" not in prices.columns:
        raise RuntimeError("SPY benchmark data is required for market-day alignment.")
    market_idx = prices.index[prices["SPY"].notna()]
    prices = prices.reindex(market_idx)
    high = high.reindex(market_idx)
    low = low.reindex(market_idx)
    volume = volume.reindex(market_idx)
    stock_cols = [t for t in tickers if t in prices.columns and prices[t].notna().sum() >= 120]
    stock_coverage = prices[stock_cols].notna().sum(axis=1) if stock_cols else pd.Series(dtype=float)
    min_stock_coverage = int(max(1, len(stock_cols) * 0.90))
    valid_latest_dates = prices.index[
        prices.get("SPY", pd.Series(index=prices.index, dtype=float)).notna()
        & stock_coverage.ge(min_stock_coverage)
    ]
    if len(valid_latest_dates) == 0:
        raise RuntimeError("No date has sufficient stock coverage and SPY data for early accumulation scanning.")
    latest = pd.Timestamp(valid_latest_dates.max())
    return {
        "bars": bars,
        "universe": universe.set_index("ticker"),
        "prices": prices,
        "high": high,
        "low": low,
        "volume": volume,
        "stock_cols": stock_cols,
        "latest": latest,
        "stock_raw_latest": stock_raw_latest,
        "etf_raw_latest": etf_raw_latest,
        "latest_stock_coverage": int(stock_coverage.loc[latest]),
        "latest_stock_coverage_ratio": float(stock_coverage.loc[latest] / max(len(stock_cols), 1)),
    }


def _write_external_template() -> Path:
    path = ARTIFACT_DIR / f"{PREFIX}_external_signal_template.csv"
    if not path.exists():
        pd.DataFrame(
            [
                {
                    "asof_date": "2026-06-01",
                    "ticker": "EXAMPLE",
                    "source_type": "reddit|news|filing|analyst|insider|custom",
                    "source_title": "short human-readable catalyst",
                    "source_url": "https://example.com/source",
                    "mentions_24h": 0,
                    "mentions_7d": 0,
                    "sentiment_score": 0.0,
                    "relevance_score": 0.0,
                    "novelty_score": 0.0,
                    "notes": "optional",
                }
            ]
        ).to_csv(path, index=False)
    return path


def _load_external_signals(latest: pd.Timestamp) -> pd.DataFrame:
    _write_external_template()
    candidates = [
        ROOT / "data" / "external" / "early_signal_catalysts.csv",
        ROOT / "data" / "external" / "reddit_mentions.csv",
        ROOT / "data" / "external" / "news_mentions.csv",
        ARTIFACT_DIR / "manual_external_signal_intake.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or "ticker" not in frame.columns:
            continue
        frame = frame.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        if "asof_date" in frame.columns:
            frame["asof_date"] = pd.to_datetime(frame["asof_date"], errors="coerce")
            frame = frame[frame["asof_date"].isna() | (frame["asof_date"] >= latest - pd.Timedelta(days=10))]
        for col in ["mentions_24h", "mentions_7d", "sentiment_score", "relevance_score", "novelty_score"]:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        if "source_type" not in frame.columns:
            frame["source_type"] = path.stem
        if "source_title" not in frame.columns:
            frame["source_title"] = ""
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "external_catalyst_score",
                "external_source_count",
                "external_reason",
                "external_sources",
            ]
        )
    raw = pd.concat(frames, ignore_index=True)
    raw["mentions_score"] = np.log1p(raw["mentions_24h"] + 0.35 * raw["mentions_7d"])
    raw["source_score"] = (
        0.35 * raw["mentions_score"].clip(0, 5) / 5
        + 0.25 * raw["sentiment_score"].clip(-1, 1).add(1).div(2)
        + 0.25 * raw["relevance_score"].clip(0, 1)
        + 0.15 * raw["novelty_score"].clip(0, 1)
    )
    grouped = (
        raw.groupby("ticker")
        .agg(
            external_catalyst_score=("source_score", "max"),
            external_source_count=("source_score", "count"),
            external_reason=("source_title", lambda x: "；".join([str(v) for v in x.dropna().astype(str).head(3) if str(v).strip()])),
            external_sources=("source_type", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
        )
        .reset_index()
    )
    return grouped


def _load_v123_reference(latest: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    status = _read_json_file(V123_STATUS_PATH)
    daily_status = _read_json_file(V123_DAILY_STATUS_PATH)
    best_candidate = str(status.get("best_candidate") or V123_BEST_CANDIDATE)
    rows = pd.DataFrame()
    source = ""
    source_path: Path | None = None

    if V123_DAILY_SIGNAL_EVENTS_PATH.exists():
        try:
            frame = pd.read_csv(V123_DAILY_SIGNAL_EVENTS_PATH)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        required = {"ticker", "signal_date", "target_weight"}
        if required.issubset(frame.columns):
            frame = frame.copy()
            frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
            frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.tz_localize(None)
            frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
            frame = frame[
                frame["signal_date"].notna()
                & frame["signal_date"].le(latest)
                & frame["target_weight"].gt(0)
            ].copy()
            if not frame.empty:
                ref_date = pd.Timestamp(frame["signal_date"].max())
                rows = frame.loc[frame["signal_date"].eq(ref_date), ["ticker", "target_weight", "signal_date"]].copy()
                source = "v123_daily_shadow_signal_events"
                source_path = V123_DAILY_SIGNAL_EVENTS_PATH

    if rows.empty and V123_BEST_TARGET_WEIGHTS_PATH.exists():
        try:
            weights = pd.read_csv(V123_BEST_TARGET_WEIGHTS_PATH)
        except pd.errors.EmptyDataError:
            weights = pd.DataFrame()
        if not weights.empty and "date" in weights.columns:
            weights = weights.copy()
            weights["date"] = pd.to_datetime(weights["date"], errors="coerce").dt.tz_localize(None)
            weights = weights[weights["date"].notna() & weights["date"].le(latest)].copy()
            if not weights.empty:
                latest_row = weights.sort_values("date").tail(1)
                ref_date = pd.Timestamp(latest_row["date"].iloc[0])
                melted = latest_row.drop(columns=["date"]).T.reset_index()
                melted.columns = ["ticker", "target_weight"]
                melted["ticker"] = melted["ticker"].astype(str).str.upper().str.strip()
                melted["target_weight"] = pd.to_numeric(melted["target_weight"], errors="coerce").fillna(0.0)
                rows = melted[melted["target_weight"].gt(0)].copy()
                rows["signal_date"] = ref_date
                source = "v123_best_target_weights"
                source_path = V123_BEST_TARGET_WEIGHTS_PATH

    if rows.empty:
        return (
            pd.DataFrame(
                columns=[
                    "ticker",
                    "v123_target_weight",
                    "v123_reference_score",
                    "v123_reference_active",
                    "v123_reference_date",
                    "v123_reference_stale_days",
                    "v123_reference_source",
                    "v123_best_candidate",
                    "v123_best_cagr",
                    "v123_best_mdd",
                ]
            ),
            {
                "available": False,
                "best_candidate": best_candidate,
                "source": None,
                "source_path": None,
                "reason": "no_v123_reference_rows",
            },
        )

    rows = rows.sort_values("target_weight", ascending=False).drop_duplicates("ticker", keep="first").copy()
    reference_date = pd.Timestamp(rows["signal_date"].max())
    stale_days = int((latest.normalize() - reference_date.normalize()).days)
    active = stale_days <= V123_REFERENCE_MAX_STALE_DAYS
    positive = rows["target_weight"].clip(lower=0)
    rank = positive.rank(pct=True)
    rows["v123_reference_score"] = np.where(active & positive.gt(0), 0.60 + 0.40 * rank, 0.0)
    rows["v123_reference_active"] = active & positive.gt(0)
    rows["v123_reference_date"] = reference_date.date().isoformat()
    rows["v123_reference_stale_days"] = stale_days
    rows["v123_reference_source"] = source
    rows["v123_best_candidate"] = best_candidate
    rows["v123_best_cagr"] = status.get("best_CAGR")
    rows["v123_best_mdd"] = status.get("best_MDD")
    rows = rows.rename(columns={"target_weight": "v123_target_weight"})
    out_cols = [
        "ticker",
        "v123_target_weight",
        "v123_reference_score",
        "v123_reference_active",
        "v123_reference_date",
        "v123_reference_stale_days",
        "v123_reference_source",
        "v123_best_candidate",
        "v123_best_cagr",
        "v123_best_mdd",
    ]
    meta = {
        "available": True,
        "active": bool(active),
        "best_candidate": best_candidate,
        "best_CAGR": status.get("best_CAGR"),
        "best_MDD": status.get("best_MDD"),
        "best_profit_factor": status.get("best_profit_factor"),
        "daily_shadow_date": daily_status.get("date"),
        "daily_shadow_final_decision": daily_status.get("final_decision"),
        "reference_date": reference_date.date().isoformat(),
        "reference_stale_days": stale_days,
        "max_stale_days": V123_REFERENCE_MAX_STALE_DAYS,
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "target_count": int(len(rows)),
        "target_tickers": rows["ticker"].head(20).tolist(),
    }
    return rows[out_cols], meta


def _market_regime(data: dict[str, Any]) -> dict[str, Any]:
    prices = data["prices"]
    latest = data["latest"]
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna() if "QQQ" in prices else pd.Series(dtype=float)
    vix = prices["^VIX"].dropna() if "^VIX" in prices else pd.Series(dtype=float)
    spy_latest = float(spy.loc[latest])
    ma50 = float(spy.rolling(50).mean().loc[latest])
    ma200 = float(spy.rolling(200).mean().loc[latest])
    dd = float(spy_latest / spy.cummax().loc[latest] - 1)
    qqq_rs = qqq / spy.reindex(qqq.index)
    qqq_rs_positive = bool(qqq_rs.loc[latest] > qqq_rs.rolling(63).mean().loc[latest]) if not qqq_rs.empty else False
    vix_latest = float(vix.loc[latest]) if not vix.empty and latest in vix.index else np.nan

    if spy_latest < ma200 or dd < -0.08 or (not pd.isna(vix_latest) and vix_latest > 28):
        regime = "risk_off"
        allow_new_positions = False
        risk_budget = 0.35
    elif spy_latest > ma50 > ma200 and qqq_rs_positive and (pd.isna(vix_latest) or vix_latest < 22):
        regime = "risk_on"
        allow_new_positions = True
        risk_budget = 1.00
    else:
        regime = "neutral"
        allow_new_positions = True
        risk_budget = 0.60

    return {
        "date": latest.date().isoformat(),
        "market_regime": regime,
        "allow_new_positions": allow_new_positions,
        "risk_budget": risk_budget,
        "spy_close": spy_latest,
        "spy_ma50": ma50,
        "spy_ma200": ma200,
        "spy_drawdown": dd,
        "qqq_rs_positive": qqq_rs_positive,
        "vix": vix_latest,
    }


def _candidate_frame(data: dict[str, Any], asof: pd.Timestamp | None = None) -> pd.DataFrame:
    prices = data["prices"][data["stock_cols"]]
    high = data["high"][data["stock_cols"]]
    low = data["low"][data["stock_cols"]]
    volume = data["volume"][data["stock_cols"]]
    universe = data["universe"]
    asof = pd.Timestamp(asof or data["latest"])

    if asof not in prices.index:
        asof = pd.Timestamp(prices.index[prices.index <= asof].max())
    p = prices.loc[asof]
    ret = prices.pct_change(fill_method=None)
    spy = data["prices"]["SPY"]
    spy_ret20 = float(spy.loc[asof] / spy.shift(20).loc[asof] - 1)
    spy_ret63 = float(spy.loc[asof] / spy.shift(63).loc[asof] - 1)

    mom5 = prices.loc[asof] / prices.shift(5).loc[asof] - 1
    mom20 = prices.loc[asof] / prices.shift(20).loc[asof] - 1
    mom42 = prices.loc[asof] / prices.shift(42).loc[asof] - 1
    mom63 = prices.loc[asof] / prices.shift(63).loc[asof] - 1
    mom126 = prices.loc[asof] / prices.shift(126).loc[asof] - 1
    rel20 = mom20 - spy_ret20
    rel63 = mom63 - spy_ret63

    ma20 = prices.rolling(20).mean().loc[asof]
    ma50 = prices.rolling(50).mean().loc[asof]
    ma120 = prices.rolling(120).mean().loc[asof]
    vol20 = ret.rolling(20).std().loc[asof] * np.sqrt(252)
    vol63 = ret.rolling(63).std().loc[asof] * np.sqrt(252)
    compression = 1 - (vol20 / vol63.replace(0, np.nan))
    dollar20 = (prices * volume).rolling(20).mean().loc[asof]
    volume_ratio = volume.rolling(10).mean().loc[asof] / volume.rolling(63).mean().loc[asof] - 1
    pos_volume = ((ret > 0).astype(float) * volume).rolling(20).sum().loc[asof] / volume.rolling(20).sum().loc[asof]
    high252 = high.rolling(252).max().loc[asof]
    low20 = low.rolling(20).min().loc[asof]
    atr20 = (high - low).rolling(20).mean().loc[asof]
    gap_to_52w_high = p / high252 - 1
    history_days = prices.loc[:asof].notna().sum()
    first_seen = prices.apply(lambda col: col.first_valid_index())

    frame = pd.DataFrame(
        {
            "ticker": p.index,
            "asof_date": asof.date().isoformat(),
            "price": p,
            "history_days": history_days,
            "first_seen": first_seen,
            "dollar_volume_20d": dollar20,
            "mom_5d": mom5,
            "mom_20d": mom20,
            "mom_42d": mom42,
            "mom_63d": mom63,
            "mom_126d": mom126,
            "rel_spy_20d": rel20,
            "rel_spy_63d": rel63,
            "ma20": ma20,
            "ma50": ma50,
            "ma120": ma120,
            "vol20": vol20,
            "vol63": vol63,
            "compression": compression,
            "volume_ratio_10_63": volume_ratio,
            "positive_volume_ratio_20d": pos_volume,
            "gap_to_52w_high": gap_to_52w_high,
            "atr20": atr20,
            "low20": low20,
        }
    ).reset_index(drop=True)
    frame["security_name"] = frame["ticker"].map(universe["security_name"])
    frame["listing_exchange"] = frame["ticker"].map(universe["listing_exchange"])
    frame["universe_first_date"] = frame["ticker"].map(universe["first_date"])
    frame["latest_adj_close_universe"] = frame["ticker"].map(universe["latest_adj_close"])
    if asof == data["latest"]:
        external = _load_external_signals(asof)
        if not external.empty:
            frame = frame.merge(external, on="ticker", how="left")
    if "external_catalyst_score" not in frame.columns:
        frame["external_catalyst_score"] = 0.0
        frame["external_source_count"] = 0
        frame["external_reason"] = ""
        frame["external_sources"] = ""
    frame["external_catalyst_score"] = frame["external_catalyst_score"].fillna(0.0)
    frame["external_source_count"] = frame["external_source_count"].fillna(0).astype(int)
    frame["external_reason"] = frame["external_reason"].fillna("")
    frame["external_sources"] = frame["external_sources"].fillna("")
    v123_meta: dict[str, Any] = {"available": False, "reason": "historical_sample_or_missing"}
    if asof == data["latest"]:
        v123_reference, v123_meta = _load_v123_reference(asof)
        if not v123_reference.empty:
            frame = frame.merge(v123_reference, on="ticker", how="left")
    v123_defaults: dict[str, Any] = {
        "v123_target_weight": 0.0,
        "v123_reference_score": 0.0,
        "v123_reference_active": False,
        "v123_reference_date": "",
        "v123_reference_stale_days": np.nan,
        "v123_reference_source": "",
        "v123_best_candidate": "",
        "v123_best_cagr": np.nan,
        "v123_best_mdd": np.nan,
    }
    for col, default in v123_defaults.items():
        if col not in frame.columns:
            frame[col] = default
    frame["v123_target_weight"] = pd.to_numeric(frame["v123_target_weight"], errors="coerce").fillna(0.0)
    frame["v123_reference_score"] = pd.to_numeric(frame["v123_reference_score"], errors="coerce").fillna(0.0)
    frame["v123_reference_active"] = np.where(
        frame["v123_reference_active"].isna(),
        False,
        frame["v123_reference_active"],
    ).astype(bool)
    frame["v123_reference_date"] = frame["v123_reference_date"].fillna("")
    frame["v123_reference_source"] = frame["v123_reference_source"].fillna("")
    frame["v123_best_candidate"] = frame["v123_best_candidate"].fillna("")

    tradable = (
        frame["price"].ge(5)
        & frame["history_days"].ge(120)
        & frame["dollar_volume_20d"].ge(10_000_000)
        & frame["ma20"].notna()
        & frame["ma50"].notna()
        & frame["ma120"].notna()
    )
    early_setup = (
        frame["price"].gt(frame["ma20"] * 0.95)
        & frame["ma20"].gt(frame["ma50"] * 0.98)
        & frame["ma50"].gt(frame["ma120"] * 0.92)
        & frame["rel_spy_20d"].gt(-0.05)
        & frame["rel_spy_63d"].gt(-0.08)
        & frame["mom_5d"].lt(0.18)
        & frame["mom_20d"].lt(0.35)
        & frame["mom_63d"].lt(0.80)
        & frame["gap_to_52w_high"].between(-0.55, 0.03)
    )
    frame["screen_pass"] = tradable & early_setup

    rel_strength_score = 0.45 * _safe_rank(frame["rel_spy_20d"]) + 0.35 * _safe_rank(frame["rel_spy_63d"]) + 0.20 * _safe_rank(frame["mom_42d"])
    accumulation_score = 0.60 * _safe_rank(frame["volume_ratio_10_63"]) + 0.40 * _safe_rank(frame["positive_volume_ratio_20d"])
    compression_score = _safe_rank(frame["compression"])
    trend_score = 0.45 * _safe_rank(frame["price"] / frame["ma50"] - 1) + 0.35 * _safe_rank(frame["ma20"] / frame["ma50"] - 1) + 0.20 * _safe_rank(frame["ma50"] / frame["ma120"] - 1)
    not_extended_score = 0.50 * _safe_rank(-frame["mom_5d"]) + 0.30 * _safe_rank(-frame["mom_20d"]) + 0.20 * _safe_rank(frame["gap_to_52w_high"].clip(-0.45, 0.02))
    under_reacted_score = 0.55 * not_extended_score + 0.25 * compression_score + 0.20 * _safe_rank(-frame["mom_63d"])
    frame["rel_strength_score"] = rel_strength_score
    frame["accumulation_score"] = accumulation_score
    frame["compression_score"] = compression_score
    frame["trend_score"] = trend_score
    frame["not_extended_score"] = not_extended_score
    frame["under_reacted_score"] = under_reacted_score
    frame["early_accumulation_score"] = (
        0.22 * rel_strength_score
        + 0.21 * accumulation_score
        + 0.17 * under_reacted_score
        + 0.15 * trend_score
        + 0.09 * compression_score
        + 0.08 * frame["external_catalyst_score"]
        + 0.08 * frame["v123_reference_score"]
    )
    frame.loc[~frame["screen_pass"], "early_accumulation_score"] *= 0.35
    frame["signal_bucket"] = np.where(
        frame["screen_pass"] & frame["early_accumulation_score"].ge(frame["early_accumulation_score"].quantile(0.92)),
        "buy_watch",
        np.where(frame["screen_pass"], "watch", "reject"),
    )
    frame["entry_low"] = frame["price"] * 0.97
    frame["entry_high"] = frame["price"] * 1.01
    frame["initial_stop"] = np.maximum(frame["price"] * 0.90, frame["low20"] * 0.98)
    frame["trim_zone_1"] = frame["price"] + 2.0 * frame["atr20"].fillna(0)
    frame["trim_zone_2"] = frame["price"] + 3.5 * frame["atr20"].fillna(0)
    frame["starter_weight"] = np.where(frame["signal_bucket"].eq("buy_watch"), 0.02, 0.0)
    frame["max_shadow_weight"] = np.where(frame["signal_bucket"].eq("buy_watch"), 0.06, 0.0)
    result = frame.sort_values("early_accumulation_score", ascending=False)
    result.attrs["v123_reference_meta"] = v123_meta
    return result


def _reason(row: pd.Series) -> str:
    tags: list[str] = []
    if bool(row.get("v123_reference_active", False)):
        tags.append(f"v123最佳策略同向权重 {_pct(row.get('v123_target_weight'))}")
    if row.get("external_catalyst_score", 0) > 0:
        source = str(row.get("external_sources", "external"))
        tags.append(f"外部催化信号 {source}")
    if row["rel_spy_20d"] > 0:
        tags.append(f"20日相对SPY {_pct(row['rel_spy_20d'])}")
    if row["volume_ratio_10_63"] > 0.2:
        tags.append("量能温和放大")
    if row["compression"] > 0:
        tags.append("短期波动收缩")
    if row["mom_5d"] < 0.12 and row["mom_20d"] < 0.35:
        tags.append("尚未短线过热")
    if row["gap_to_52w_high"] < -0.05:
        tags.append(f"距52周高点 {_pct(row['gap_to_52w_high'])}")
    if str(row.get("external_reason", "")).strip():
        tags.append(str(row["external_reason"]).strip())
    return "；".join(tags[:5]) or "早期积累综合评分靠前"


def _build_signal_outputs(candidates: pd.DataFrame, regime: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = candidates[candidates["signal_bucket"].eq("buy_watch")].head(OUTPUT_SIGNAL_N).copy()
    if not regime["allow_new_positions"]:
        selected["action"] = "watch"
        selected["target_weight"] = 0.0
    else:
        selected["action"] = "buy_watch"
        selected["target_weight"] = selected["starter_weight"] * float(regime["risk_budget"])
    selected["delta_weight"] = selected["target_weight"]
    selected["estimated_trade_notional"] = selected["target_weight"] * INITIAL_EQUITY
    selected["estimated_qty_at_reference_close"] = selected["estimated_trade_notional"] / selected["price"].replace(0, np.nan)
    selected["reference_close_price"] = selected["price"]
    selected["signal_date"] = selected["asof_date"]
    selected["event_type"] = "early_accumulation_setup"
    selected["status"] = "shadow_only_not_live_valid"
    selected["live_order_allowed"] = False
    selected["no_live_order_reason"] = NO_LIVE_ORDER_REASON
    selected["reason"] = selected.apply(_reason, axis=1)
    selected["signal_id"] = selected.apply(
        lambda r: f"{r['ticker']}_{r['signal_date']}_early_accumulation_v1",
        axis=1,
    )
    signal_cols = [
        "signal_id",
        "signal_date",
        "ticker",
        "action",
        "target_weight",
        "delta_weight",
        "estimated_trade_notional",
        "estimated_qty_at_reference_close",
        "reference_close_price",
        "entry_low",
        "entry_high",
        "initial_stop",
        "trim_zone_1",
        "trim_zone_2",
        "early_accumulation_score",
        "rel_spy_20d",
        "rel_spy_63d",
        "mom_20d",
        "mom_63d",
        "gap_to_52w_high",
        "dollar_volume_20d",
        "external_catalyst_score",
        "external_source_count",
        "external_sources",
        "v123_reference_score",
        "v123_target_weight",
        "v123_reference_active",
        "v123_reference_date",
        "v123_reference_stale_days",
        "v123_reference_source",
        "v123_best_candidate",
        "reason",
        "live_order_allowed",
        "no_live_order_reason",
    ]
    events = selected[
        [
            "signal_id",
            "ticker",
            "signal_date",
            "status",
            "action",
            "event_type",
            "target_weight",
            "live_order_allowed",
            "no_live_order_reason",
        ]
    ].copy()
    return selected[signal_cols], events


def _historical_quality(data: dict[str, Any]) -> pd.DataFrame:
    prices = data["prices"]
    spy = prices["SPY"]
    latest = data["latest"]
    start = max(pd.Timestamp("2024-01-02"), latest - pd.Timedelta(days=730))
    sample_dates = prices.index[(prices.index >= start) & (prices.index <= latest - pd.Timedelta(days=25))]
    if len(sample_dates) == 0:
        return pd.DataFrame()
    # Monthly fixed-horizon checks keep the daily scanner fast while still
    # covering different market states. Deep weekly validation can be added once
    # the source mix is stable.
    sample_dates = pd.DatetimeIndex(pd.Series(sample_dates, index=sample_dates).groupby(sample_dates.to_period("M")).tail(1).values)
    rows = []
    for dt in sample_dates:
        frame = _candidate_frame(data, pd.Timestamp(dt))
        picks = frame[frame["signal_bucket"].eq("buy_watch")].head(OUTPUT_SIGNAL_N)
        if picks.empty:
            continue
        for _, pick in picks.iterrows():
            for horizon in [3, 7, 14]:
                loc = prices.index.get_indexer([pd.Timestamp(dt)])[0]
                end_loc = loc + horizon
                if end_loc >= len(prices.index):
                    continue
                end = prices.index[end_loc]
                ticker = pick["ticker"]
                if ticker not in prices.columns or pd.isna(prices.loc[dt, ticker]) or pd.isna(prices.loc[end, ticker]):
                    continue
                ret = float(prices.loc[end, ticker] / prices.loc[dt, ticker] - 1)
                spy_ret = float(spy.loc[end] / spy.loc[dt] - 1)
                rows.append(
                    {
                        "signal_date": pd.Timestamp(dt).date().isoformat(),
                        "horizon": f"T+{horizon}",
                        "ticker": ticker,
                        "score": float(pick["early_accumulation_score"]),
                        "return": ret,
                        "SPY_return": spy_ret,
                        "excess_vs_SPY": ret - spy_ret,
                        "win": ret > 0,
                        "beats_SPY": ret > spy_ret,
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail.to_csv(ARTIFACT_DIR / f"{PREFIX}_signal_forward_detail.csv", index=False)
    agg = (
        detail.groupby("horizon")
        .agg(
            sample=("return", "count"),
            win_rate=("win", "mean"),
            beat_spy_rate=("beats_SPY", "mean"),
            avg_return=("return", "mean"),
            avg_spy_return=("SPY_return", "mean"),
            avg_excess=("excess_vs_SPY", "mean"),
        )
        .reset_index()
    )
    return agg


def _write_report(
    candidates: pd.DataFrame,
    signal: pd.DataFrame,
    events: pd.DataFrame,
    quality: pd.DataFrame,
    regime: dict[str, Any],
) -> Path:
    path = ARTIFACT_DIR / f"{PREFIX}_daily_report.md"
    v123_meta = candidates.attrs.get("v123_reference_meta", {})
    v123_signal_overlap = int(signal["v123_reference_active"].sum()) if "v123_reference_active" in signal.columns and not signal.empty else 0
    lines = [
        "# Early Accumulation Signal V1 Daily Report",
        "",
        f"- Date: {regime['date']}",
        f"- Market regime: {regime['market_regime']}",
        f"- New positions allowed: {regime['allow_new_positions']}",
        f"- Risk budget: {_pct(regime['risk_budget'])}",
        f"- Candidate rows scanned: {len(candidates):,}",
        f"- Screen-passed rows: {int(candidates['screen_pass'].sum()):,}",
        f"- Shadow signal rows: {len(signal):,}",
        "",
        "## QuantReview-style Interpretation",
        "",
        "This layer is designed for early accumulation setups: relative strength is improving, volume is warming up, volatility is not yet explosive, external catalysts can be blended in, and the name has not already made an extreme short-term move.",
        "",
        "## V123 Reference Overlay",
        "",
        f"- Best v123 candidate: `{v123_meta.get('best_candidate', V123_BEST_CANDIDATE)}`",
        f"- Reference source/date: {v123_meta.get('source', 'n/a')} / {v123_meta.get('reference_date', 'n/a')}",
        f"- Reference stale days: {v123_meta.get('reference_stale_days', 'n/a')}",
        f"- Best v123 CAGR/MDD/profit factor: {_pct(v123_meta.get('best_CAGR'))} / {_pct(v123_meta.get('best_MDD'))} / {v123_meta.get('best_profit_factor', 'n/a')}",
        f"- Top-signal overlap with v123 target: {v123_signal_overlap}",
        "",
        "## Top Shadow Signals",
        "",
    ]
    if signal.empty:
        lines.append("No early-accumulation shadow signals passed today.")
    else:
        lines.append("| ticker | action | score | v123 | entry zone | stop | trim 1 | trim 2 | reason |")
        lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
        for _, row in signal.head(OUTPUT_SIGNAL_N).iterrows():
            v123_weight = _pct(row.get("v123_target_weight")) if bool(row.get("v123_reference_active", False)) else ""
            lines.append(
                "| {ticker} | {action} | {score:.3f} | {v123_weight} | {entry_low}-{entry_high} | {stop} | {trim1} | {trim2} | {reason} |".format(
                    ticker=row["ticker"],
                    action=row["action"],
                    score=float(row["early_accumulation_score"]),
                    v123_weight=v123_weight,
                    entry_low=_money(row["entry_low"]),
                    entry_high=_money(row["entry_high"]),
                    stop=_money(row["initial_stop"]),
                    trim1=_money(row["trim_zone_1"]),
                    trim2=_money(row["trim_zone_2"]),
                    reason=row["reason"],
                )
            )

    lines.extend(["", "## Fixed-horizon Signal Quality", ""])
    lines.append(
        "Scope note: this fixed-horizon table validates the historical quant core. "
        "Current external news/Reddit and v123 overlays are tracked as live attribution inputs; "
        "the daily external-history archive must accumulate enough observations before full external-overlay ablation is reliable."
    )
    lines.append("")
    if quality.empty:
        lines.append("No completed fixed-horizon samples yet.")
    else:
        lines.append("| horizon | sample | win rate | beat SPY | avg return | avg SPY | avg excess |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in quality.iterrows():
            lines.append(
                f"| {row['horizon']} | {int(row['sample'])} | {_pct(row['win_rate'])} | {_pct(row['beat_spy_rate'])} | {_pct(row['avg_return'])} | {_pct(row['avg_spy_return'])} | {_pct(row['avg_excess'])} |"
            )

    lines.extend(
        [
            "",
            "## Execution Guard",
            "",
            "NO LIVE ORDER.",
            "",
            NO_LIVE_ORDER_REASON,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    regime = _market_regime(data)
    candidates = _candidate_frame(data)
    candidates.to_csv(ARTIFACT_DIR / f"{PREFIX}_candidates.csv", index=False)
    signal, events = _build_signal_outputs(candidates, regime)
    signal.to_csv(ARTIFACT_DIR / f"{PREFIX}_latest_signal.csv", index=False)
    events.to_csv(ARTIFACT_DIR / f"{PREFIX}_signal_events.csv", index=False)
    quality = _historical_quality(data)
    quality.to_csv(ARTIFACT_DIR / f"{PREFIX}_signal_quality.csv", index=False)
    report = _write_report(candidates, signal, events, quality, regime)
    v123_meta = candidates.attrs.get("v123_reference_meta", {})
    v123_signal_overlap = (
        signal.loc[signal["v123_reference_active"].astype(bool), "ticker"].tolist()
        if "v123_reference_active" in signal.columns and not signal.empty
        else []
    )

    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_family": PREFIX,
        "best_version": PREFIX,
        "shadow_test_ready": True,
        "live_trading_ready": False,
        "latest_price_date": data["latest"].date().isoformat(),
        "stock_raw_latest_date": data["stock_raw_latest"].date().isoformat(),
        "etf_raw_latest_date": data["etf_raw_latest"].date().isoformat(),
        "latest_stock_coverage": data["latest_stock_coverage"],
        "latest_stock_coverage_ratio": data["latest_stock_coverage_ratio"],
        "market_regime": regime["market_regime"],
        "allow_new_positions": regime["allow_new_positions"],
        "risk_budget": regime["risk_budget"],
        "candidate_rows_scanned": int(len(candidates)),
        "screen_pass_count": int(candidates["screen_pass"].sum()),
        "shadow_signal_count": int(len(signal)),
        "top_signal_tickers": signal["ticker"].head(OUTPUT_SIGNAL_N).tolist() if not signal.empty else [],
        "v123_reference": v123_meta,
        "v123_signal_overlap_count": int(len(v123_signal_overlap)),
        "v123_signal_overlap_tickers": v123_signal_overlap,
        "signal_quality_scope": "historical_quant_core_monthly_fixed_horizon_only",
        "signal_quality_scope_note": (
            "Historical fixed-horizon quality validates the quant core. Current external news/Reddit "
            "and v123 overlays are attribution inputs; daily archived catalysts now preserve the evidence needed "
            "for future full historical ablation once enough observations accumulate."
        ),
        "signal_quality": quality.to_dict(orient="records") if not quality.empty else [],
        "latest_signal_csv": str(ARTIFACT_DIR / f"{PREFIX}_latest_signal.csv"),
        "candidate_csv": str(ARTIFACT_DIR / f"{PREFIX}_candidates.csv"),
        "signal_events_csv": str(ARTIFACT_DIR / f"{PREFIX}_signal_events.csv"),
        "signal_quality_csv": str(ARTIFACT_DIR / f"{PREFIX}_signal_quality.csv"),
        "report_md": str(report),
        "live_order_allowed": False,
        "final_decision": "NO LIVE ORDER",
        "no_live_order_reason": NO_LIVE_ORDER_REASON,
    }
    (ARTIFACT_DIR / f"{PREFIX}_status.json").write_text(json.dumps(_jsonable(status), indent=2), encoding="utf-8")
    pd.DataFrame([status]).to_csv(ARTIFACT_DIR / f"{PREFIX}_status.csv", index=False)
    print(json.dumps(_jsonable(status), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
