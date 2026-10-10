from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from us_quant.calendar import completed_session, sessions
from us_quant.config import QuantError, ResearchConfig
from us_quant.storage import (
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame
    close: pd.DataFrame
    raw_close: pd.DataFrame
    volume: pd.DataFrame
    risk_free: pd.Series

    def validate(self) -> None:
        index = self.close.index
        if len(index) < 2 or not index.is_unique or not index.is_monotonic_increasing:
            raise QuantError(
                "Price dates must be unique, sorted, and contain at least two sessions."
            )
        expected = sessions(index[0], index[-1])
        if not index.equals(expected):
            raise QuantError("Price data has missing or unexpected exchange sessions.")
        for name, frame in (
            ("open", self.open),
            ("close", self.close),
            ("raw_close", self.raw_close),
            ("volume", self.volume),
        ):
            if not frame.index.equals(index) or not frame.columns.equals(self.close.columns):
                raise QuantError(f"Misaligned {name} panel.")
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise QuantError(f"Nonpositive or nonfinite {name} observations.")
        if not self.risk_free.index.equals(index) or not np.isfinite(self.risk_free).all():
            raise QuantError("The risk-free proxy is missing or misaligned.")
        if (self.close.pct_change(fill_method=None).iloc[1:].abs() > 0.60).any().any():
            raise QuantError("An ETF daily return exceeds 60%; audit corporate actions before use.")


def parse_chart(payload: dict, symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        chart = payload["chart"]
        if chart.get("error") or not chart.get("result"):
            raise QuantError(f"Yahoo returned an error for {symbol}: {chart.get('error')}")
        result = chart["result"][0]
        if result["meta"]["symbol"] != symbol:
            raise QuantError(f"Provider returned the wrong symbol for {symbol}.")
        if symbol != "^IRX" and (
            result["meta"].get("currency") != "USD" or result["meta"].get("instrumentType") != "ETF"
        ):
            raise QuantError(f"{symbol} is not identified as a USD ETF by the provider.")
        dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
        dates = dates.tz_convert("America/New_York").normalize().tz_localize(None)
        prices = result["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {name: prices[name] for name in ("open", "high", "low", "close", "volume")},
            index=dates,
            dtype=float,
        )
        frame.index.name = "date"
        if symbol == "^IRX":
            frame = frame[["close"]]
            valid = frame["close"].dropna()
            if not np.isfinite(valid).all() or ((valid < -1) | (valid > 50)).any():
                raise QuantError("Invalid ^IRX discount yield, expected percentage-point units.")
            frame = frame.dropna()
        else:
            frame["adj_close"] = result["indicators"]["adjclose"][0]["adjclose"]
            frame["adj_open"] = frame["open"] * frame["adj_close"] / frame["close"]
            if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
                raise QuantError(f"{symbol} contains missing, nonfinite, or nonpositive prices.")
            if (frame["low"] > frame[["open", "close"]].min(axis=1) + 1e-6).any() or (
                frame["high"] < frame[["open", "close"]].max(axis=1) - 1e-6
            ).any():
                raise QuantError(f"{symbol} has inconsistent OHLC bars.")
        frame = frame.loc[start:end]
        if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise QuantError(f"{symbol} has empty, duplicate, or unsorted data.")
        if symbol != "^IRX":
            expected = sessions(start, end)
            if not frame.index.equals(expected):
                missing = expected.difference(frame.index).strftime("%Y-%m-%d").tolist()
                extra = frame.index.difference(expected).strftime("%Y-%m-%d").tolist()
                raise QuantError(f"{symbol}: missing sessions {missing[:8]}, extra {extra[:8]}.")
        return frame
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise QuantError(f"Malformed chart response for {symbol}: {exc}") from exc


def fetch_dataset(config: ResearchConfig, phase: str, output: Path) -> dict:
    if phase not in {"development", "holdout", "forward"}:
        raise QuantError("Data phase must be development, holdout, or forward.")
    if output.exists():
        raise QuantError(f"Dataset already exists: {output}. Verify/reuse it; do not overwrite it.")
    start = config.data_start if phase == "development" else config.development_end
    end = config.development_end if phase == "development" else config.as_of
    if phase == "forward":
        end = completed_session().date().isoformat()
        start = (pd.Timestamp(end) - pd.DateOffset(years=3)).date().isoformat()
    if pd.Timestamp(end) > completed_session():
        raise QuantError("Requested data includes an incomplete or future exchange session.")
    fetched: dict[str, tuple[str, pd.DataFrame, str]] = {}
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-quant-research/0.1 (personal research)"
        for symbol in (*config.symbols, config.risk_free_symbol):
            fetch_start = (
                (pd.Timestamp(start) - pd.Timedelta(days=14)).date().isoformat()
                if symbol == config.risk_free_symbol
                else start
            )
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
            params = {
                "period1": int(pd.Timestamp(fetch_start, tz="UTC").timestamp()),
                "period2": int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp()),
                "interval": "1d",
                "events": "div,splits",
            }
            try:
                response = client.get(url, params=params, timeout=(15, 75))
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise QuantError(
                    f"Public data request failed for {symbol}; no synthetic fallback: {exc}"
                ) from exc
            frame = parse_chart(payload, symbol, fetch_start, end)
            fetched[symbol] = (response.text, frame, response.url)
    output.mkdir(parents=True)
    files: dict[str, str] = {}
    sources = {}
    for symbol, (raw, frame, url) in fetched.items():
        filename = "IRX" if symbol == "^IRX" else symbol
        raw_path = output / "raw" / f"{filename}.json"
        csv_path = output / f"{filename}.csv"
        write_text_atomic(raw_path, raw)
        write_text_atomic(csv_path, frame.to_csv(float_format="%.12g"))
        for path in (raw_path, csv_path):
            files[path.relative_to(output).as_posix()] = file_digest(path)
        events = json.loads(raw)["chart"]["result"][0].get("events", {})
        sources[symbol] = {
            "url": url,
            "rows": len(frame),
            "first_date": frame.index[0].date().isoformat(),
            "last_date": frame.index[-1].date().isoformat(),
            "dividend_events": len(events.get("dividends", {})),
            "split_events": len(events.get("splits", {})),
        }
    manifest = {
        "schema_version": 1,
        "phase": phase,
        "retrieved_at": utc_now(),
        "protocol_sha256": digest_json(config.to_dict()),
        "provider": "Yahoo Finance public chart endpoint (unofficial API, not execution data)",
        "start": start,
        "end": end,
        "files": files,
        "sources": sources,
        "adjustment": "adjusted close; adjusted open = raw open * adjusted close / raw close",
        "cash_interest": "zero; Treasury proxy is subtracted only when calculating Sharpe",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_dataset(path: Path, config: ResearchConfig, phase: str) -> dict:
    manifest = read_json(path / "manifest.json")
    if manifest.get("protocol_sha256") != digest_json(config.to_dict()):
        raise QuantError("Dataset does not match the registered protocol.")
    if manifest.get("phase") != phase:
        raise QuantError(f"Expected {phase} data, not {manifest.get('phase')}.")
    required = {f"{symbol}.csv" for symbol in config.symbols} | {"IRX.csv"}
    if not required <= set(manifest["files"]):
        raise QuantError("Dataset manifest omits required price files.")
    for name, expected in manifest["files"].items():
        resolved = (path / name).resolve()
        if not resolved.is_relative_to(path.resolve()) or not resolved.is_file():
            raise QuantError(f"Unsafe or missing dataset path: {name}")
        if file_digest(resolved) != expected:
            raise QuantError(f"Dataset fingerprint mismatch: {name}")
    return manifest


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def risk_free_returns(index: pd.DatetimeIndex, rates: pd.Series) -> pd.Series:
    if not rates.index.is_unique or not rates.index.is_monotonic_increasing:
        raise QuantError("Risk-free dates must be unique and sorted.")
    result = []
    for i, day in enumerate(index):
        prior = index[i - 1] if i else day - pd.Timedelta(days=1)
        location = rates.index.searchsorted(prior, side="right") - 1
        if location < 0 or (prior - rates.index[location]).days > 7:
            raise QuantError(f"No sufficiently recent lagged risk-free quote for {day.date()}.")
        discount = float(rates.iloc[location]) / 100.0
        if not np.isfinite(discount) or not -0.01 <= discount <= 0.5:
            raise QuantError("Invalid Treasury discount yield.")
        # ^IRX is a 13-week bank-discount yield, not a cash-account credit rate.
        bill_price = 1.0 - discount * 91.0 / 360.0
        days = (day - prior).days
        result.append(bill_price ** (-days / 91.0) - 1.0)
    return pd.Series(result, index=index, name="risk_free")


def load_market(
    config: ResearchConfig,
    development: Path,
    holdout: Path | None = None,
    *,
    phase: str = "development",
) -> MarketData:
    verify_dataset(development, config, phase)
    if holdout is not None:
        verify_dataset(holdout, config, "holdout")
    panels: dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        before = _read_csv(development / f"{symbol}.csv")
        if holdout is not None:
            after = _read_csv(holdout / f"{symbol}.csv")
            anchor = before.index[-1]
            if anchor not in after.index:
                raise QuantError("Holdout is missing the overlap session needed for adjustment.")
            if not np.isclose(
                before.at[anchor, "close"], after.at[anchor, "close"], rtol=1e-6, atol=1e-6
            ):
                raise QuantError(f"{symbol} raw overlap price changed; audit the source revision.")
            ratio = after.at[anchor, "adj_close"] / before.at[anchor, "adj_close"]
            before[["adj_open", "adj_close"]] *= ratio
            before = pd.concat([before, after.loc[after.index > anchor]])
        panels[symbol] = before
    rates = _read_csv(development / "IRX.csv")["close"]
    if holdout is not None:
        after_rates = _read_csv(holdout / "IRX.csv")["close"]
        rates = pd.concat([rates, after_rates.loc[after_rates.index > rates.index[-1]]])
    close = pd.DataFrame({key: value["adj_close"] for key, value in panels.items()})
    data = MarketData(
        open=pd.DataFrame({key: value["adj_open"] for key, value in panels.items()}),
        close=close,
        raw_close=pd.DataFrame({key: value["close"] for key, value in panels.items()}),
        volume=pd.DataFrame({key: value["volume"] for key, value in panels.items()}),
        risk_free=risk_free_returns(close.index, rates),
    )
    data.validate()
    return data
