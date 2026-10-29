from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from us_quant.config import QuantError

RAW_FIELDS = ("open", "high", "low", "close")
RTOL, ATOL = 1e-10, 1e-9


def verify_bars(csv: pd.DataFrame, raw: dict, market: dict[str, pd.DataFrame]) -> tuple:
    index = market["close"].index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or list(csv.columns) != ["open", "high", "low", "close", "volume", "adj_close", "adj_open"]
        or not isinstance(csv.index, pd.DatetimeIndex)
        or csv.index.tz is not None
        or not csv.index.is_unique
        or not csv.index.is_monotonic_increasing
        or not index.isin(csv.index).all()
        or any(not frame.index.equals(index) or "SPY" not in frame for frame in market.values())
    ):
        raise QuantError("Keep the complete original calendar and the native SPY CSV fields.")
    records = raw["chart"]["result"]
    if len(records) != 1 or raw["chart"]["error"] is not None:
        raise QuantError("Require the original successful single-symbol quote response.")
    record = records[0]
    meta = record["meta"]
    if any(
        meta.get(key) != value
        for key, value in (
            ("symbol", "SPY"),
            ("currency", "USD"),
            ("instrumentType", "ETF"),
            ("exchangeTimezoneName", "America/New_York"),
            ("dataGranularity", "1d"),
        )
    ):
        raise QuantError("The range observations must be the same daily USD SPY fund source.")
    quote = record["indicators"]["quote"][0]
    adjusted = record["indicators"]["adjclose"][0]["adjclose"]
    times = record["timestamp"]
    if any(len(quote[key]) != len(times) for key in (*RAW_FIELDS, "volume")) or len(
        adjusted
    ) != len(times):
        raise QuantError("Every raw observation needs its complete OHLC/volume/adjustment tuple.")
    native = {}
    for position, timestamp in enumerate(times):
        day = pd.Timestamp(
            datetime.fromtimestamp(int(timestamp), ZoneInfo("America/New_York")).date()
        )
        if day in native:
            raise QuantError("Duplicate native dates cannot be collapsed into one bar.")
        values = [quote[name][position] for name in (*RAW_FIELDS, "volume")] + [adjusted[position]]
        if any(value is None for value in values):
            raise QuantError("A native source gap cannot become a filled extremum.")
        numeric = [Decimal(str(value)) for value in values]
        if any(not value.is_finite() or value <= 0 for value in numeric):
            raise QuantError("Native bars, adjustments and volumes must be finite and positive.")
        open_, high, low, close = numeric[:4]
        if low > min(open_, close) or high < max(open_, close):
            raise QuantError(
                "The source high/low must contain its actual opening and closing prices."
            )
        native[day] = numeric
    if not all(day in native for day in index):
        raise QuantError("A required original market date has no actual raw range observation.")
    selected = csv.loc[index]
    if not np.isfinite(selected).all().all() or (selected <= 0).any().any():
        raise QuantError("The original context cannot contain invalid CSV prices or volumes.")
    if (selected["low"] > selected[["open", "close"]].min(axis=1)).any() or (
        selected["high"] < selected[["open", "close"]].max(axis=1)
    ).any():
        raise QuantError("The native CSV's OHLC containment is invalid.")
    source_columns = (*RAW_FIELDS, "volume", "adj_close")
    maximum = {key: 0.0 for key in source_columns}
    for day in index:
        for key, value in zip(source_columns, native[day], strict=True):
            observed = float(selected.loc[day, key])
            difference = abs(Decimal(str(observed)) - value)
            if difference > Decimal(str(ATOL)) + Decimal(str(RTOL)) * abs(value):
                raise QuantError("Independent raw numeric fields disagree with the saved CSV.")
            maximum[key] = max(maximum[key], float(difference))
    factor = selected["adj_close"] / selected["close"]
    if not np.allclose(selected["open"] * factor, selected["adj_open"], rtol=RTOL, atol=ATOL):
        raise QuantError("The existing adjusted-opening convention does not match its source.")
    matched = {}
    for name, column in (
        ("open", "adj_open"),
        ("close", "adj_close"),
        ("raw_close", "close"),
        ("volume", "volume"),
    ):
        left, right = selected[column].to_numpy(), market[name]["SPY"].to_numpy()
        if not np.allclose(left, right, rtol=RTOL, atol=ATOL):
            raise QuantError("Range-source identity differs from the immutable registered market.")
        matched[name] = float(np.max(np.abs(left - right)))
    bars = selected.loc[:, list(RAW_FIELDS)].copy()
    return bars, {
        "context_sessions": len(index),
        "independently_checked_native_fields": len(index) * len(source_columns),
        "independently_checked_raw_OHLC_fields": len(index) * 4,
        "maximum_raw_CSV_difference_by_field": maximum,
        "maximum_registered_field_difference": matched,
        "identity_rtol": RTOL,
        "identity_atol": ATOL,
        "same_original_price_identity_tolerance": True,
        "raw_OHLC_containment_verified": True,
        "common_adjustment_factor_verified": True,
        "output_basis": "raw intraday OHLC; common adjustment scale cancels in same-day ratios",
        "registered_open_close_Rf_or_volume_replaced": False,
        "source_gaps_filled_or_original_dates_dropped": False,
        "new_variance_targets_or_strategy_returns_computed": False,
    }
