from __future__ import annotations

import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.core_activity_inputs_v1 import verify_source


def fixture():
    index = pd.to_datetime(["2020-01-02", "2020-01-03"])
    csv = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [103.0, 103.0],
            "low": [99.0, 100.0],
            "close": [102.0, 101.0],
            "volume": [1000.0, 2000.0],
            "adj_close": [51.0, 50.5],
            "adj_open": [50.0, 51.0],
        },
        index=index,
    )
    raw = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "MTUM",
                        "currency": "USD",
                        "instrumentType": "ETF",
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": "1d",
                    },
                    "timestamp": [
                        int(day.tz_localize("America/New_York").timestamp()) for day in index
                    ],
                    "indicators": {
                        "quote": [
                            {name: csv[name].tolist() for name in ("open", "close", "volume")}
                        ],
                        "adjclose": [{"adjclose": csv["adj_close"].tolist()}],
                    },
                    "events": {"dividends": {}},
                }
            ],
        }
    }
    market = {
        name: pd.DataFrame({"MTUM": csv[column]})
        for name, column in (
            ("open", "adj_open"),
            ("close", "adj_close"),
            ("raw_close", "close"),
            ("volume", "volume"),
        )
    }
    return csv, raw, market


def test_independent_share_units_and_original_prices_agree():
    csv, raw, market = fixture()
    result = verify_source("MTUM", csv, raw, market)
    assert result["independent_native_fields_checked"] == 8
    assert result["reported_share_volume_native_CSV_registered_exact"] is True
    assert result["new_activity_scores_or_targets_computed"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "split",
        "duplicate",
        "missing",
        "zero_volume",
        "volume_mismatch",
        "symbol",
        "adjustment",
        "panel",
    ],
)
def test_unsupported_activity_input_is_not_repaired(damage):
    csv, raw, market = fixture()
    row = raw["chart"]["result"][0]
    if damage == "split":
        row["events"]["splits"] = {"date": {"numerator": 2, "denominator": 1}}
    elif damage == "duplicate":
        row["timestamp"][1] = row["timestamp"][0]
    elif damage == "missing":
        row["indicators"]["quote"][0]["volume"][0] = None
    elif damage == "zero_volume":
        row["indicators"]["quote"][0]["volume"][0] = 0
    elif damage == "volume_mismatch":
        row["indicators"]["quote"][0]["volume"][0] += 1
    elif damage == "symbol":
        row["meta"]["symbol"] = "SOXX"
    elif damage == "adjustment":
        csv.iloc[0, csv.columns.get_loc("adj_open")] = 49
    else:
        market["close"].iloc[0, 0] = 52
    with pytest.raises(QuantError):
        verify_source("MTUM", csv, raw, market)
