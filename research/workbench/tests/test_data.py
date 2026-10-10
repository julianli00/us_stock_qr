from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import completed_session, sessions
from us_quant.config import QuantError
from us_quant.data import parse_chart, risk_free_returns, verify_dataset
from us_quant.storage import digest_json, file_digest, write_json, write_text_atomic


def chart_fixture():
    index = sessions("2020-01-30", "2020-02-04")
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": "SPY", "currency": "USD", "instrumentType": "ETF"},
                    "timestamp": [
                        int((day.tz_localize("UTC") + pd.Timedelta(hours=15)).timestamp())
                        for day in index
                    ],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100, 100, 98, 98],
                                "high": [101, 101, 99, 99],
                                "low": [99, 99, 97, 97],
                                "close": [100, 100, 98, 98],
                                "volume": [1000, 1000, 1000, 1000],
                            }
                        ],
                        "adjclose": [{"adjclose": [98, 98, 98, 98]}],
                    },
                }
            ],
        },
    }


def test_adjusted_open_includes_distribution_consistently():
    frame = parse_chart(chart_fixture(), "SPY", "2020-01-30", "2020-02-04")
    assert (frame["adj_close"] == 98).all()
    assert (frame["adj_open"] == 98).all()
    assert frame["close"].pct_change().iloc[2] == pytest.approx(-0.02)
    assert frame["adj_close"].pct_change().iloc[2] == 0


@pytest.mark.parametrize(
    "problem", ["missing", "wrong_symbol", "missing_adjustment", "nan", "duplicate"]
)
def test_bad_chart_fails_instead_of_filling_prices(problem):
    payload = deepcopy(chart_fixture())
    result = payload["chart"]["result"][0]
    if problem == "missing":
        result["timestamp"].pop()
        for values in result["indicators"]["quote"][0].values():
            values.pop()
        result["indicators"]["adjclose"][0]["adjclose"].pop()
    elif problem == "wrong_symbol":
        result["meta"]["symbol"] = "QQQ"
    elif problem == "missing_adjustment":
        del result["indicators"]["adjclose"]
    elif problem == "nan":
        result["indicators"]["quote"][0]["open"][0] = None
    else:
        result["timestamp"][1] = result["timestamp"][0]
    with pytest.raises(QuantError):
        parse_chart(payload, "SPY", "2020-01-30", "2020-02-04")


def test_risk_free_lags_quotes_and_accrues_weekends():
    index = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04"])
    rates = pd.Series(
        [4.0, 5.0, 20.0, 1.0],
        index=pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]),
    )
    values = risk_free_returns(index, rates)
    assert values.iloc[1] == pytest.approx((1 - 0.05 * 91 / 360) ** (-3 / 91) - 1)
    assert values.iloc[2] == pytest.approx((1 - 0.20 * 91 / 360) ** (-1 / 91) - 1)
    changed = rates.copy()
    changed.iloc[-1] = 49
    pd.testing.assert_series_equal(values, risk_free_returns(index, changed))


def test_stale_risk_free_is_not_silently_zero():
    rates = pd.Series([5.0], index=pd.to_datetime(["2019-12-01"]))
    with pytest.raises(QuantError, match="recent"):
        risk_free_returns(pd.to_datetime(["2020-01-31"]), rates)


def test_incomplete_close_and_holidays():
    assert completed_session(pd.Timestamp("2026-09-29 10:00Z")) == pd.Timestamp("2026-09-28")
    assert completed_session(pd.Timestamp("2026-09-29 20:10Z")) == pd.Timestamp("2026-09-28")
    assert completed_session(pd.Timestamp("2026-09-29 20:31Z")) == pd.Timestamp("2026-09-29")
    assert completed_session(pd.Timestamp("2026-07-04 16:00Z")) == pd.Timestamp("2026-07-02")


def test_missing_or_nan_panel_is_rejected(market_factory):
    data = market_factory()
    broken = data.open.copy()
    broken.iloc[3, 0] = np.nan
    with pytest.raises(QuantError, match="open"):
        replace(data, open=broken).validate()
    with pytest.raises(QuantError, match="session"):
        replace(data, close=data.close.drop(data.close.index[3])).validate()


def test_csv_tampering_is_detected(tmp_path, config):
    files = {}
    for name in (*config.symbols, "IRX"):
        path = tmp_path / f"{name}.csv"
        write_text_atomic(path, "date,close\n2021-12-31,100\n")
        files[path.name] = file_digest(path)
    write_json(
        tmp_path / "manifest.json",
        {
            "phase": "development",
            "protocol_sha256": digest_json(config.to_dict()),
            "files": files,
        },
    )
    verify_dataset(tmp_path, config, "development")
    write_text_atomic(tmp_path / "SPY.csv", "date,close\n2021-12-31,999\n")
    with pytest.raises(QuantError, match="fingerprint"):
        verify_dataset(tmp_path, config, "development")
