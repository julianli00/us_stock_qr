from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.free_quotes import parse_public_reference

NOW = pd.Timestamp("2026-09-29 13:36:30Z")


@pytest.fixture
def chart():
    times = [
        pd.Timestamp("2026-09-29 13:34:00Z"),
        pd.Timestamp("2026-09-29 13:35:00Z"),
        pd.Timestamp("2026-09-29 13:36:20Z"),
    ]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "SPY",
                        "currency": "USD",
                        "instrumentType": "ETF",
                        "dataGranularity": "1m",
                    },
                    "timestamp": [int(value.timestamp()) for value in times],
                    "indicators": {"quote": [{"close": [700.0, 701.0, 999.0]}]},
                }
            ],
        },
    }


def test_only_completed_one_minute_bars_are_used(chart):
    result = parse_public_reference(chart, "SPY", NOW, "a" * 64)
    assert result.price == 701.0
    assert result.bar_end == "2026-09-29T13:36:00+00:00"
    assert not result.consolidated_nbbo and not result.realtime_entitlement_verified


@pytest.mark.parametrize(
    "problem",
    [
        "stale",
        "symbol",
        "currency",
        "type",
        "interval",
        "duplicates",
        "missing",
        "nan",
        "no_completed",
        "provider_error",
    ],
)
def test_invalid_or_stale_free_quotes_are_not_execution_references(chart, problem):
    payload = deepcopy(chart)
    result = payload["chart"]["result"][0]
    if problem == "stale":
        result["timestamp"] = [value - 3600 for value in result["timestamp"]]
    elif problem in {"symbol", "currency", "type", "interval"}:
        key = {"type": "instrumentType", "interval": "dataGranularity"}.get(problem, problem)
        result["meta"][key] = "wrong"
    elif problem == "duplicates":
        result["timestamp"][1] = result["timestamp"][0]
    elif problem == "missing":
        result["indicators"]["quote"][0]["close"].pop()
    elif problem == "nan":
        result["indicators"]["quote"][0]["close"][1] = float("nan")
    elif problem == "no_completed":
        result["timestamp"] = [value + 3600 for value in result["timestamp"]]
    else:
        payload["chart"]["error"] = {"description": "limited"}
    with pytest.raises(QuantError):
        parse_public_reference(payload, "SPY", NOW, "a" * 64)


def test_captured_reference_expires_and_cannot_change_feed_identity(chart):
    reference = parse_public_reference(chart, "SPY", NOW, "a" * 64)
    with pytest.raises(QuantError, match="stale"):
        reference.validate_for_experiment(NOW + pd.Timedelta(seconds=31))
    with pytest.raises(QuantError, match="Unknown"):
        replace(reference, source="imaginary_realtime").validate_for_experiment(NOW)
