from __future__ import annotations

import copy

import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.spy_range_inputs_v1 import verify_bars


def source():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    csv = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [103.0, 104.0],
            "low": [99.0, 100.0],
            "close": [102.0, 101.0],
            "volume": [1000.0, 1000.0],
            "adj_close": [51.0, 50.5],
            "adj_open": [50.0, 51.0],
        },
        index=dates,
    )
    raw = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "SPY",
                        "currency": "USD",
                        "instrumentType": "ETF",
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": "1d",
                    },
                    "timestamp": [
                        int(day.tz_localize("America/New_York").timestamp()) for day in dates
                    ],
                    "indicators": {
                        "quote": [
                            {
                                name: csv[name].tolist()
                                for name in ("open", "high", "low", "close", "volume")
                            }
                        ],
                        "adjclose": [{"adjclose": csv["adj_close"].tolist()}],
                    },
                }
            ],
        }
    }
    market = {
        name: pd.DataFrame({"SPY": csv[column]})
        for name, column in (
            ("open", "adj_open"),
            ("close", "adj_close"),
            ("raw_close", "close"),
            ("volume", "volume"),
        )
    }
    return csv, raw, market


def test_original_prices_unchanged_and_raw_ranges_retained():
    csv, raw, market = source()
    before = copy.deepcopy(market)
    bars, proof = verify_bars(csv, raw, market)
    pd.testing.assert_frame_equal(bars, csv[["open", "high", "low", "close"]])
    assert proof["independently_checked_native_fields"] == 12
    assert proof["new_variance_targets_or_strategy_returns_computed"] is False
    for name in market:
        pd.testing.assert_frame_equal(market[name], before[name])


@pytest.mark.parametrize(
    "damage",
    [
        "missing_date",
        "duplicate_date",
        "wrong_symbol",
        "bad_high",
        "bad_low",
        "null_high",
        "wrong_adjustment",
        "market_mismatch",
    ],
)
def test_bad_source_is_not_repaired_or_substituted(damage):
    csv, raw, market = source()
    record = raw["chart"]["result"][0]
    if damage == "missing_date":
        csv = csv.iloc[1:]
    elif damage == "duplicate_date":
        record["timestamp"][1] = record["timestamp"][0]
    elif damage == "wrong_symbol":
        record["meta"]["symbol"] = "QQQ"
    elif damage == "bad_high":
        record["indicators"]["quote"][0]["high"][0] = 101.0
    elif damage == "bad_low":
        record["indicators"]["quote"][0]["low"][0] = 101.0
    elif damage == "null_high":
        record["indicators"]["quote"][0]["high"][0] = None
    elif damage == "wrong_adjustment":
        csv.iloc[0, csv.columns.get_loc("adj_open")] = 49
    else:
        market["close"].iloc[0, 0] = 52
    with pytest.raises(QuantError):
        verify_bars(csv, raw, market)
