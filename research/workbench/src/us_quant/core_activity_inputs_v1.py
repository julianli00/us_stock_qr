from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from us_quant.config import QuantError

CORE = ("MTUM", "VLUE", "QUAL", "USMV")
RTOL, ATOL = 1e-10, 1e-9


def verify_source(symbol: str, csv: pd.DataFrame, raw: dict, market: dict) -> dict:
    index = market["close"].index
    if (
        symbol not in CORE
        or not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not csv.index.equals(index)
        or not {"open", "close", "volume", "adj_open", "adj_close"} <= set(csv.columns)
        or any(not frame.index.equals(index) or symbol not in frame for frame in market.values())
    ):
        raise QuantError("Activity inputs must retain the original complete fund calendar.")
    records = raw["chart"]["result"]
    if len(records) != 1 or raw["chart"]["error"] is not None:
        raise QuantError("Require one successful preserved native fund response.")
    record = records[0]
    if any(
        record["meta"].get(name) != expected
        for name, expected in (
            ("symbol", symbol),
            ("currency", "USD"),
            ("instrumentType", "ETF"),
            ("exchangeTimezoneName", "America/New_York"),
            ("dataGranularity", "1d"),
        )
    ):
        raise QuantError("Native instrument, currency or volume observation frequency changed.")
    if record.get("events", {}).get("splits"):
        raise QuantError("This source contract does not infer price/volume units across a split.")
    quote = record["indicators"]["quote"][0]
    adjusted = record["indicators"]["adjclose"][0]["adjclose"]
    timestamps = record["timestamp"]
    if any(len(quote[name]) != len(timestamps) for name in ("open", "close", "volume")) or len(
        adjusted
    ) != len(timestamps):
        raise QuantError("Native price, share-volume and adjustment arrays must align.")
    native = {}
    for i, timestamp in enumerate(timestamps):
        day = pd.Timestamp(
            datetime.fromtimestamp(int(timestamp), ZoneInfo("America/New_York")).date()
        )
        if day in native:
            raise QuantError("Do not collapse duplicate raw fund dates.")
        values = (quote["open"][i], quote["close"][i], quote["volume"][i], adjusted[i])
        if any(value is None for value in values):
            raise QuantError("Do not replace missing prices or activity with valid defaults.")
        values = tuple(Decimal(str(value)) for value in values)
        if any(not value.is_finite() or value <= 0 for value in values):
            raise QuantError("Prices, adjusted prices and reported share volume must be positive.")
        if values[2] != values[2].to_integral_value():
            raise QuantError(
                "The archived daily share volume must retain its native integer units."
            )
        native[day] = values
    if list(native) != list(index):
        raise QuantError("Keep every actual native date, without gap filling or deletion.")
    columns = ("open", "close", "volume", "adj_close")
    if not np.isfinite(csv[list(columns)]).all().all() or (csv[list(columns)] <= 0).any().any():
        raise QuantError("Nonfinite or zero activity cannot enter the weighted score.")
    errors = {name: 0.0 for name in columns}
    for day in index:
        for name, value in zip(columns, native[day], strict=True):
            observed = Decimal(str(float(csv.loc[day, name])))
            difference = abs(observed - value)
            limit = (
                Decimal(0)
                if name == "volume"
                else Decimal(str(ATOL)) + Decimal(str(RTOL)) * abs(value)
            )
            if difference > limit:
                raise QuantError("Independent native values disagree with preserved CSV inputs.")
            errors[name] = max(errors[name], float(difference))
    if not np.allclose(
        csv["adj_open"], csv["open"] * csv["adj_close"] / csv["close"], rtol=RTOL, atol=ATOL
    ):
        raise QuantError("The old adjusted-opening convention changed.")
    matched = {}
    for name, column in (
        ("open", "adj_open"),
        ("close", "adj_close"),
        ("raw_close", "close"),
        ("volume", "volume"),
    ):
        left, right = csv[column].to_numpy(), market[name][symbol].to_numpy()
        if name == "volume":
            equal = np.array_equal(left, right)
        else:
            equal = np.allclose(left, right, rtol=RTOL, atol=ATOL)
        if not equal:
            raise QuantError("Activity observations differ from the registered fund price panel.")
        matched[name] = float(np.max(np.abs(left - right)))
    return {
        "symbol": symbol,
        "sessions": len(index),
        "independent_native_fields_checked": 4 * len(index),
        "reported_split_events": 0,
        "reported_share_volume_native_CSV_registered_exact": True,
        "maximum_raw_CSV_difference_by_field": errors,
        "maximum_registered_field_difference": matched,
        "original_price_identity_rtol": RTOL,
        "original_price_identity_atol": ATOL,
        "activity_proxy_is_not_actual_traded_dollars_or_fund_flows": True,
        "source_gaps_filled_or_dates_deleted": False,
        "new_activity_scores_or_targets_computed": False,
    }
