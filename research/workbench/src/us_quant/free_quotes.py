from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from us_quant.config import QuantError
from us_quant.paper import APPROVED_ETFS, IBPaperBroker, load_paper_config, now_utc
from us_quant.storage import digest_bytes, write_json


@dataclass(frozen=True)
class ReferencePrice:
    symbol: str
    price: float
    bar_start: str
    bar_end: str
    received_at: str
    source: str
    source_sha256: str
    realtime_entitlement_verified: bool = False
    consolidated_nbbo: bool = False

    def validate_for_experiment(self, now: pd.Timestamp) -> None:
        if self.symbol not in APPROVED_ETFS or not math.isfinite(self.price) or self.price <= 0:
            raise QuantError("Invalid experimental reference price.")
        start, end, received = map(pd.Timestamp, (self.bar_start, self.bar_end, self.received_at))
        if any(value.tzinfo is None for value in (start, end, received, now)):
            raise QuantError("Reference-price times must include timezones.")
        if (
            end - start != pd.Timedelta(minutes=1)
            or end > received
            or received > now + pd.Timedelta(seconds=2)
        ):
            raise QuantError("Invalid or incomplete public one-minute bar.")
        if not 0 <= (now - end).total_seconds() <= 180 or (now - received).total_seconds() > 30:
            raise QuantError("Public reference is stale; do not submit an experimental order.")
        if self.source != "yahoo_public_completed_1m_bar" or not self.source_sha256:
            raise QuantError("Unknown reference source.")


def parse_public_reference(
    payload: dict, symbol: str, received: pd.Timestamp, source_hash: str
) -> ReferencePrice:
    if received.tzinfo is None or symbol not in APPROVED_ETFS:
        raise QuantError("A supported ETF and timezone-aware receipt time are required.")
    try:
        chart = payload["chart"]
        if chart.get("error") or not chart.get("result"):
            raise QuantError(f"Public chart request failed: {chart.get('error')}")
        result = chart["result"][0]
        metadata = result["meta"]
        if (
            metadata["symbol"] != symbol
            or metadata.get("currency") != "USD"
            or metadata.get("instrumentType") != "ETF"
            or metadata.get("dataGranularity") != "1m"
        ):
            raise QuantError(
                "Public reference has the wrong symbol, currency, asset type, or interval."
            )
        timestamps = result["timestamp"]
        prices = result["indicators"]["quote"][0]["close"]
        if len(timestamps) != len(prices) or timestamps != sorted(set(timestamps)):
            raise QuantError("Public intraday data is misaligned or unordered.")
        completed = []
        for timestamp, price in zip(timestamps, prices, strict=True):
            if price is None:
                continue
            start = pd.Timestamp(timestamp, unit="s", tz="UTC")
            if start.second != 0 or start.microsecond != 0:
                continue
            end = start + pd.Timedelta(minutes=1)
            if end > received:
                continue
            if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
                raise QuantError("Nonfinite or nonpositive intraday reference value.")
            completed.append((start, end, float(price)))
        if not completed:
            raise QuantError("No complete public one-minute bar is available.")
        start, end, price = completed[-1]
        reference = ReferencePrice(
            symbol,
            price,
            start.isoformat(),
            end.isoformat(),
            received.isoformat(),
            "yahoo_public_completed_1m_bar",
            source_hash,
        )
        reference.validate_for_experiment(received)
        return reference
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise QuantError(f"Malformed public quote response: {exc}") from exc


def fetch_public_reference(symbol: str) -> ReferencePrice:
    if symbol not in APPROVED_ETFS:
        raise QuantError(
            "Public execution references are restricted to the unleveraged ETF allowlist."
        )
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    try:
        response = requests.get(
            url,
            params={"range": "1d", "interval": "1m", "includePrePost": "true"},
            headers={"User-Agent": "us-quant-research/0.1 (personal research)"},
            timeout=(10, 20),
        )
        response.raise_for_status()
        return parse_public_reference(
            response.json(), symbol, now_utc(), digest_bytes(response.content)
        )
    except (requests.RequestException, ValueError) as exc:
        raise QuantError(f"Free public quote failed; no silent fallback: {exc}") from exc


def delayed_reference(symbol: str, paper_config: Path) -> dict:
    from ib_async import Stock
    from ib_async.wrapper import RequestError

    config = load_paper_config(paper_config)
    if symbol not in config.allowed_symbols:
        raise QuantError("The delayed-data symbol is not allowed.")
    try:
        with IBPaperBroker(config, readonly=True) as broker:
            contracts = broker.ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
            if len(contracts) != 1:
                raise QuantError("Delayed data requires an unambiguous contract.")
            contract = contracts[0]
            broker.ib.reqMarketDataType(3)
            ticker = broker.ib.reqMktData(contract, "", False, False)
            try:
                broker.ib.sleep(5)
                if (
                    not math.isfinite(ticker.bid)
                    or not math.isfinite(ticker.ask)
                    or not 0 < ticker.bid <= ticker.ask
                    or ticker.time is None
                    or ticker.marketDataType not in {1, 3}
                ):
                    raise QuantError("No usable delayed/reference bid and ask arrived.")
                return {
                    "symbol": symbol,
                    "source": "IBKR marketDataType=3 request",
                    "returned_market_data_type": ticker.marketDataType,
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "receipt_time": ticker.time.isoformat(),
                    "exchange_timestamp_known": False,
                    "nominal_delay_minutes": [10, 15] if ticker.marketDataType == 3 else [0, 0],
                    "qualified_execution_reference": False,
                    "warning": "Receipt time is not exchange time; no automatic trading fallback.",
                    "orders_sent": 0,
                }
            finally:
                broker.ib.cancelMktData(contract)
    except RequestError as exc:
        raise QuantError(f"Delayed data request failed ({exc.code}): {exc.message}") from exc


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "free-quote", help="Explicit free reference feeds; not certified NBBO."
    )
    command.add_argument("--symbol", required=True)
    command.add_argument("--source", choices=["yahoo", "ibkr-delayed"], required=True)
    command.add_argument("--paper-config", type=Path, default=Path("config/paper.json"))
    command.add_argument("--output", type=Path)


def dispatch_free_quote(args: argparse.Namespace) -> dict:
    if args.output and args.output.exists():
        raise QuantError("Refusing to overwrite a quote audit artifact.")
    result = (
        asdict(fetch_public_reference(args.symbol))
        if args.source == "yahoo"
        else delayed_reference(args.symbol, args.paper_config)
    )
    result.update({"mode": "free_reference_only", "order_authority": False})
    if args.output:
        write_json(args.output, result)
    return result
