from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
import requests

from us_quant.calendar import completed_session, market_calendar, next_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart, risk_free_returns
from us_quant.macro_factor_tilt import load_macro
from us_quant.storage import digest_json, file_digest, read_json, write_json, write_text_atomic
from us_quant.volatility_term_risk import load_terms

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("collector_id") != "prospective_public_market_data_v1"
        or policy.get("timezone") != "Asia/Shanghai"
        or policy.get("collection_hour") != 9
        or policy.get("symbols")
        != ["SPY", "MTUM", "VLUE", "QUAL", "USMV", "GLD", "IEF", "TLT", "BIL"]
        or policy.get("risk_free_symbol") != "^IRX"
        or policy.get("lookback_years") != 2
        or policy.get("macro_series") != ["T10Y3M", "DFII10"]
        or policy.get("option_indices") != ["VIX", "VIX3M"]
        or policy.get("record_before_next_open") is not True
        or policy.get("first_snapshot_is_baseline_only") is not True
        or any(
            policy.get(key) is not False
            for key in (
                "backfill_missed_observations",
                "compute_strategy_returns",
                "order_authority",
            )
        )
    ):
        raise QuantError("Prospective collection must preserve its data-only authority and timing.")


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    names = (
        "prospective_data.py",
        "calendar.py",
        "data.py",
        "macro_factor_tilt.py",
        "volatility_term_risk.py",
        "storage.py",
    )
    return digest_json({name: file_digest(folder / name) for name in names})


def curl_csv(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "10",
            "--max-time",
            "90",
            url,
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=105,
    )
    if result.returncode:
        raise QuantError(f"Public CSV acquisition failed with curl status {result.returncode}.")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise QuantError("The public CSV acquisition returned no data.")


def acquire(policy: dict, day: pd.Timestamp, output: Path) -> dict:
    start = (day - pd.DateOffset(years=policy["lookback_years"])).date().isoformat()
    end = str(day.date())
    frames, source_records = {}, []
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-stock-qr research (public market data archive)"
        for symbol in (*policy["symbols"], policy["risk_free_symbol"]):
            first = (
                str((pd.Timestamp(start) - pd.Timedelta(days=14)).date())
                if symbol == "^IRX"
                else start
            )
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
            try:
                response = client.get(
                    url,
                    params={
                        "period1": int(pd.Timestamp(first, tz="UTC").timestamp()),
                        "period2": int((day.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()),
                        "interval": "1d",
                        "events": "div,splits",
                    },
                    timeout=(10, 60),
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                response_status = getattr(getattr(exc, "response", None), "status_code", None)
                raise QuantError(
                    f"Public quote acquisition failed for {symbol}: "
                    f"{type(exc).__name__}, HTTP status {response_status}"
                ) from exc
            name = "IRX" if symbol == "^IRX" else symbol
            write_text_atomic(output / "raw" / f"{name}.json", response.text)
            frames[symbol] = parse_chart(payload, symbol, first, end)
            write_text_atomic(
                output / "quotes" / f"{name}.csv", frames[symbol].to_csv(float_format="%.17g")
            )
            source_records.append(
                {"source": symbol, "url": response.url, "retrieved_at": utc_now().isoformat()}
            )
    close = pd.DataFrame({symbol: frames[symbol]["adj_close"] for symbol in policy["symbols"]})
    data = MarketData(
        pd.DataFrame({symbol: frames[symbol]["adj_open"] for symbol in policy["symbols"]}),
        close,
        pd.DataFrame({symbol: frames[symbol]["close"] for symbol in policy["symbols"]}),
        pd.DataFrame({symbol: frames[symbol]["volume"] for symbol in policy["symbols"]}),
        risk_free_returns(close.index, frames["^IRX"]["close"]),
    )
    data.validate()
    if data.close.index[-1] != day:
        raise QuantError("The acquired ETF snapshot does not reach the intended completed session.")
    macro_records = []
    for name in policy["macro_series"]:
        macro_start = str((pd.Timestamp(start) - pd.Timedelta(days=14)).date())
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}&cosd={macro_start}&coed={end}"
        path = output / "macro" / f"{name}.csv"
        curl_csv(url, path)
        frame = pd.read_csv(path)
        if list(frame.columns) != ["observation_date", name]:
            raise QuantError("Unexpected official macro data schema.")
        macro_records.append(
            {
                "series": name,
                "url": url,
                "path": path.name,
                "sha256": file_digest(path),
                "retrieved_at": utc_now().isoformat(),
            }
        )
    write_json(output / "macro" / "verified-manifest.json", {"sources": macro_records})
    macro, availability = load_macro(output / "macro", close.index)
    write_text_atomic(output / "macro" / "aligned.csv", macro.to_csv(float_format="%.17g"))
    for name, audit in availability.items():
        write_text_atomic(output / "macro" / f"{name}-availability.csv", audit.to_csv())
    option_records = []
    for name in policy["option_indices"]:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
        path = output / "option_indices" / f"{name}.csv"
        curl_csv(url, path)
        option_records.append(
            {
                "symbol": name,
                "url": url,
                "sha256": file_digest(path),
                "retrieved_at": utc_now().isoformat(),
            }
        )
    write_json(output / "option_indices" / "manifest.json", {"sources": option_records})
    terms = load_terms(output / "option_indices", close.index)
    write_text_atomic(output / "option_indices" / "aligned.csv", terms.to_csv(float_format="%.17g"))
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    return {
        "session": end,
        "first_price_session": str(close.index[0].date()),
        "price_sessions": len(close),
        "symbols": policy["symbols"],
        "quote_sources": source_records,
        "macro_sources": macro_records,
        "option_index_sources": option_records,
        "all_required_sources_verified": True,
        "strategy_returns_calculated": False,
    }


class ProspectiveArchive:
    def __init__(self, directory: Path, policy: dict):
        validate_policy(policy)
        if directory.is_symlink():
            raise QuantError("Prospective archive cannot be a symlink.")
        self.directory, self.policy = directory, policy

    def initialize(self, now: datetime | None = None) -> dict:
        now = utc_now() if now is None else now
        if now.tzinfo is None:
            raise QuantError("Prospective registration needs a timezone-aware time.")
        if self.directory.exists():
            raise QuantError("Do not reset or replace an existing prospective archive.")
        self.directory.mkdir(parents=True, mode=0o700)
        record = {
            "schema_version": 1,
            "registered_at": now.isoformat(),
            "policy_sha256": digest_json(self.policy),
            "collector_sha256": fingerprint(),
            "mode": "data_only_prospective_archive",
            "order_authority": False,
            "old_forward_ledgers_modified": False,
        }
        write_json(self.directory / "registration.json", record)
        write_json(self.directory / "head.json", {"last_session": None, "receipt_sha256": "0" * 64})
        return record

    def verify(self) -> list[dict]:
        if any((self.directory / name).is_symlink() for name in ("registration.json", "head.json")):
            raise QuantError("Prospective archive control files cannot be symlinks.")
        registration = read_json(self.directory / "registration.json")
        if (
            registration.get("policy_sha256") != digest_json(self.policy)
            or registration.get("collector_sha256") != fingerprint()
            or registration.get("mode") != "data_only_prospective_archive"
            or registration.get("order_authority") is not False
        ):
            raise QuantError("Prospective collection policy/code changed; preserve the archive.")
        records, previous = [], "0" * 64
        for path in sorted((self.directory / "receipts").glob("*.json")):
            record = read_json(path)
            day = pd.Timestamp(record.get("session"))
            observed = pd.Timestamp(record.get("observed_at"))
            if (
                path.is_symlink()
                or record.get("previous_receipt_sha256") != previous
                or record.get("registration_sha256")
                != file_digest(self.directory / "registration.json")
                or record.get("session") != path.stem
                or record.get("order_authority") is not False
                or pd.isna(day)
                or day.tzinfo is not None
                or not market_calendar().is_session(day)
                or pd.isna(observed)
                or observed.tzinfo is None
                or observed < pd.Timestamp(registration["registered_at"])
                or observed < market_calendar().session_close(day) + pd.Timedelta(minutes=30)
                or observed >= market_calendar().session_open(next_session(day))
            ):
                raise QuantError("Prospective receipt history was changed or is incomplete.")
            expected_gaps = (
                [str(value.date()) for value in sessions(records[-1]["session"], day)[1:-1]]
                if records
                else []
            )
            if (
                record["baseline_only"] != (not records)
                or record["missed_sessions"] != expected_gaps
            ):
                raise QuantError("Prospective baseline or gap accounting was altered.")
            records.append(record)
            previous = file_digest(path)
        head = read_json(self.directory / "head.json")
        if head != {
            "last_session": records[-1]["session"] if records else None,
            "receipt_sha256": previous,
        }:
            raise QuantError("Prospective receipt head is missing, changed or partially committed.")
        if records:
            last = records[-1]
            snapshot = self.directory / last["snapshot_path"]
            if (
                snapshot.is_symlink()
                or not snapshot.resolve().is_relative_to(self.directory.resolve())
                or file_digest(snapshot / "manifest.json") != last["manifest_sha256"]
            ):
                raise QuantError("The latest prospective snapshot is missing or revised.")
            manifest = read_json(snapshot / "manifest.json")
            for relative, expected in manifest["files"].items():
                path = snapshot / relative
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(snapshot.resolve())
                    or file_digest(path) != expected
                ):
                    raise QuantError("A latest-snapshot input was revised after collection.")
        return records

    def status(self) -> dict:
        records = self.verify()
        gaps = [day for record in records for day in record["missed_sessions"]]
        return {
            "mode": "data_only_prospective_archive",
            "baseline_session": records[0]["session"] if records else None,
            "latest_session": records[-1]["session"] if records else None,
            "complete_snapshots": len(records),
            "observations_after_baseline": max(0, len(records) - 1),
            "missed_sessions": gaps,
            "contiguous_since_baseline": not gaps,
            "strategy_returns_calculated": False,
            "eligible_for_strategy_promotion": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "receipt_chain_head": read_json(self.directory / "head.json")["receipt_sha256"],
        }

    def collect(self, fetcher=acquire, clock=utc_now) -> dict:
        now = clock()
        if now.tzinfo is None:
            raise QuantError("Prospective collection needs a timezone-aware clock.")
        registration = read_json(self.directory / "registration.json")
        if pd.Timestamp(now) < pd.Timestamp(registration["registered_at"]):
            raise QuantError("Prospective observations cannot predate registration.")
        day = completed_session(pd.Timestamp(now))
        key = str(day.date())
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self.verify()
            if records and records[-1]["session"] == key:
                return {"action": "already_collected", **self.status()}
            if records and records[-1]["session"] > key:
                raise QuantError("Do not backdate a prospective collection.")
            if pd.Timestamp(now) >= market_calendar().session_open(next_session(day)):
                return {"action": "missed_preopen_deadline", **self.status()}
            attempt = (
                self.directory
                / "attempts"
                / (now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
            )
            attempt.mkdir(parents=True, mode=0o700)
            try:
                metadata = fetcher(self.policy, day, attempt)
                finished = clock()
                if (
                    metadata.get("session") != key
                    or metadata.get("all_required_sources_verified") is not True
                    or finished.tzinfo is None
                    or finished < now
                    or pd.Timestamp(finished) >= market_calendar().session_open(next_session(day))
                    or completed_session(pd.Timestamp(finished)) != day
                ):
                    raise QuantError(
                        "Snapshot was incomplete or acquired after its permissible deadline."
                    )
                files = {
                    path.relative_to(attempt).as_posix(): file_digest(path)
                    for path in sorted(attempt.rglob("*"))
                    if path.is_file()
                }
                if not files:
                    raise QuantError("A complete snapshot must retain actual source artifacts.")
                manifest = {
                    **metadata,
                    "collected_at": finished.isoformat(),
                    "files": files,
                    "historical_rows_are_not_past_observations": True,
                    "order_authority": False,
                }
                write_json(attempt / "manifest.json", manifest)
            except (QuantError, OSError, subprocess.SubprocessError, ValueError) as exc:
                write_json(
                    attempt / "failure.json",
                    {
                        "session": key,
                        "failed_at": clock().isoformat(),
                        "error_type": type(exc).__name__,
                        "complete_snapshot": False,
                        "observation_recorded": False,
                        "order_authority": False,
                    },
                )
                raise
            missed = []
            previous = "0" * 64
            if records:
                prior = pd.Timestamp(records[-1]["session"])
                missed = [str(date.date()) for date in sessions(prior, day)[1:-1]]
                previous = file_digest(self.directory / "receipts" / f"{prior.date()}.json")
            receipt = {
                "session": key,
                "observed_at": finished.isoformat(),
                "baseline_only": not records,
                "missed_sessions": missed,
                "snapshot_path": attempt.relative_to(self.directory).as_posix(),
                "manifest_sha256": file_digest(attempt / "manifest.json"),
                "registration_sha256": file_digest(self.directory / "registration.json"),
                "previous_receipt_sha256": previous,
                "strategy_returns_calculated": False,
                "order_authority": False,
            }
            write_json(self.directory / "receipts" / f"{key}.json", receipt)
            write_json(
                self.directory / "head.json",
                {
                    "last_session": key,
                    "receipt_sha256": file_digest(self.directory / "receipts" / f"{key}.json"),
                },
            )
            return {"action": "collected", **self.status()}


def main():
    parser = argparse.ArgumentParser(
        description="Archive observed public data; no orders or strategy returns."
    )
    parser.add_argument("action", choices=("init", "collect", "status"))
    parser.add_argument("--policy", type=Path, default=ROOT / "config/prospective-data.json")
    parser.add_argument("--directory", type=Path, default=ROOT / "data/prospective-market-v1")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    try:
        archive = ProspectiveArchive(args.directory, read_json(args.policy))
        if args.action == "init":
            result = archive.initialize()
        elif args.action == "collect":
            result = archive.collect()
        else:
            result = archive.status()
        if args.export:
            if args.export.exists() and read_json(args.export) != result:
                raise QuantError("Do not replace a prior collection-status checkpoint.")
            if not args.export.exists():
                write_json(args.export, result)
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(2, f"Prospective data collection blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
