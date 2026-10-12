from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests

from us_quant.calendar import market_calendar
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.macro_factor_tilt import align_observations
from us_quant.multifactor_stability import corporate_actions
from us_quant.prospective_data import (
    ProspectiveArchive,
    curl_csv,
    fingerprint as parent_fingerprint,
    utc_now,
)
from us_quant.research_program import safe_file
from us_quant.storage import digest_json, file_digest, read_json, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/prospective-factor-inputs.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("collector_id") != "prospective_factor_inputs_v2"
        or policy.get("timezone") != "Asia/Shanghai"
        or policy.get("collection_hour") != 9
        or policy.get("parent_policy") != "config/prospective-data.json"
        or policy.get("parent_archive") != "data/prospective-market-v1"
        or policy.get("additional_symbols") != ["IJR", "PKW"]
        or policy.get("additional_macro_series") != ["BAA10Y"]
        or policy.get("publication_delay_sessions") != 2
        or policy.get("maximum_observation_age_days") != 7
        or any(
            policy.get(name) is not True
            for name in (
                "inherit_only_matching_completed_session",
                "retain_parent_acquisition_times",
                "record_before_next_open",
                "first_snapshot_is_baseline_only",
            )
        )
        or any(
            policy.get(name) is not False
            for name in ("backfill_missed_observations", "compute_strategy_returns", "order_authority")
        )
    ):
        raise QuantError("Expanded acquisition must preserve its fixed inputs and data-only authority.")


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json(
        {
            "parent_collector": parent_fingerprint(),
            "extension": file_digest(Path(__file__)),
            "corporate_actions": file_digest(folder / "multifactor_stability.py"),
            "path_validation": digest_json(inspect.getsource(safe_file)),
        }
    )


class ExpandedArchive:
    def __init__(self, directory: Path, policy: dict, *, root: Path = ROOT):
        validate_policy(policy)
        directory = directory if directory.is_absolute() else root / directory
        parent_directory = root / policy["parent_archive"]
        if (
            directory.is_symlink()
            or not directory.resolve().is_relative_to(root.resolve())
            or directory.resolve().is_relative_to(parent_directory.resolve())
            or parent_directory.resolve().is_relative_to(directory.resolve())
        ):
            raise QuantError("The expanded archive must be a separate, in-workbench directory.")
        self.root, self.directory, self.policy = root, directory, policy
        parent_policy = read_json(safe_file(root, policy["parent_policy"]))
        self.parent = ProspectiveArchive(parent_directory, parent_policy)
        self.journal = ProspectiveArchive(directory, parent_policy)

    def initialize(self, now=None) -> dict:
        self.parent.verify()
        record = self.journal.initialize(now)
        record.update(
            {
                "extension_policy_sha256": digest_json(self.policy),
                "extension_collector_sha256": fingerprint(),
                "parent_archive": self.policy["parent_archive"],
                "parent_registration_sha256": file_digest(self.parent.directory / "registration.json"),
                "extension_mode": "data_only_expanded_factor_inputs",
            }
        )
        write_json(self.directory / "registration.json", record)
        self.verify()
        return record

    def parent_snapshot(self, day: pd.Timestamp) -> tuple[dict, dict, Path]:
        records = self.parent.verify()
        if not records or records[-1]["session"] != str(day.date()):
            raise QuantError("Capture the matching completed-session parent snapshot first.")
        receipt = records[-1]
        manifest_path = safe_file(
            self.parent.directory,
            f"{receipt['snapshot_path']}/manifest.json",
            receipt["manifest_sha256"],
        )
        manifest = read_json(manifest_path)
        return receipt, manifest, manifest_path.parent

    def parent_reference(self, day: pd.Timestamp) -> dict:
        receipt, _, _ = self.parent_snapshot(day)
        return {
            "archive": self.policy["parent_archive"],
            "session": receipt["session"],
            "receipt_sha256": file_digest(
                self.parent.directory / "receipts" / f"{receipt['session']}.json"
            ),
            "manifest_sha256": receipt["manifest_sha256"],
            "observed_at": receipt["observed_at"],
        }

    def verify(self) -> list[dict]:
        self.parent.verify()
        registration = read_json(safe_file(self.directory, "registration.json"))
        if (
            registration.get("extension_policy_sha256") != digest_json(self.policy)
            or registration.get("extension_collector_sha256") != fingerprint()
            or registration.get("parent_archive") != self.policy["parent_archive"]
            or registration.get("parent_registration_sha256")
            != file_digest(self.parent.directory / "registration.json")
            or registration.get("extension_mode") != "data_only_expanded_factor_inputs"
        ):
            raise QuantError("The expanded policy/code or frozen parent registration changed.")
        records = self.journal.verify()
        for receipt in records:
            manifest = read_json(
                safe_file(
                    self.directory,
                    f"{receipt['snapshot_path']}/manifest.json",
                    receipt["manifest_sha256"],
                )
            )
            reference = manifest.get("parent_reference", {})
            if not isinstance(reference, dict) or set(reference) != {
                "archive",
                "session",
                "receipt_sha256",
                "manifest_sha256",
                "observed_at",
            }:
                raise QuantError("The expanded snapshot is missing a complete parent reference.")
            parent_receipt = read_json(
                safe_file(
                    self.parent.directory,
                    f"receipts/{receipt['session']}.json",
                    reference.get("receipt_sha256"),
                )
            )
            if (
                reference.get("archive") != self.policy["parent_archive"]
                or reference["receipt_sha256"]
                != file_digest(self.parent.directory / "receipts" / f"{receipt['session']}.json")
                or reference.get("session") != receipt["session"]
                or reference.get("manifest_sha256") != parent_receipt["manifest_sha256"]
                or reference.get("observed_at") != parent_receipt["observed_at"]
                or pd.Timestamp(reference["observed_at"]) > pd.Timestamp(receipt["observed_at"])
                or manifest.get("additional_symbols") != self.policy["additional_symbols"]
                or manifest.get("additional_macro_series") != self.policy["additional_macro_series"]
                or manifest.get("strategy_returns_calculated") is not False
                or manifest.get("all_required_sources_verified") is not True
            ):
                raise QuantError("Expanded provenance does not match its actual same-session parent.")
            self.validate_extra_times(
                manifest, pd.Timestamp(receipt["session"]), pd.Timestamp(receipt["observed_at"])
            )
        return records

    def validate_extra_times(self, manifest: dict, day: pd.Timestamp, finished: pd.Timestamp) -> None:
        quotes = manifest.get("additional_quote_sources")
        macro = manifest.get("additional_macro_source")
        if (
            not isinstance(quotes, list)
            or len(quotes) != len(self.policy["additional_symbols"])
            or any(not isinstance(row, dict) for row in quotes)
            or {row.get("symbol") for row in quotes} != set(self.policy["additional_symbols"])
            or not isinstance(macro, dict)
            or macro.get("series") != "BAA10Y"
            or pd.isna(finished)
            or finished.tzinfo is None
        ):
            raise QuantError("The expanded snapshot must identify all three additional sources.")
        registration = read_json(safe_file(self.directory, "registration.json"))
        earliest = max(
            pd.Timestamp(registration["registered_at"]),
            market_calendar().session_close(day) + pd.Timedelta(minutes=30),
        )
        for source in [*quotes, macro]:
            acquired = pd.Timestamp(source.get("retrieved_at"))
            if (
                pd.isna(acquired)
                or acquired.tzinfo is None
                or acquired < earliest
                or acquired > finished
            ):
                raise QuantError("An extra input acquisition time is missing or outside its window.")

    def status(self) -> dict:
        self.verify()
        return {
            **self.journal.status(),
            "mode": "data_only_expanded_factor_inputs",
            "additional_symbols": self.policy["additional_symbols"],
            "additional_macro_series": self.policy["additional_macro_series"],
            "new_catalog_definitions_registered_by_collection": False,
            "parent_archive_modified": False,
            "expanded_baseline_is_independent_strategy_evidence": False,
        }

    def acquire(self, policy: dict, day: pd.Timestamp, output: Path) -> dict:
        receipt, inherited, snapshot = self.parent_snapshot(day)
        reference = self.parent_reference(day)
        for relative, digest in inherited["files"].items():
            source = safe_file(snapshot, relative, digest)
            destination = output / "parent" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(destination) != digest:
                raise QuantError("A parent input changed while the expanded snapshot copied it.")
        write_json(output / "parent-provenance.json", reference)
        frames = {}
        for name in ("open", "close", "raw_close", "volume", "risk_free"):
            frames[name] = pd.read_csv(
                output / "parent" / f"{name}.csv", index_col=0, parse_dates=True
            )
        index = frames["close"].index
        start, end = str(index[0].date()), str(day.date())
        extra_sources = []
        with requests.Session() as client:
            client.headers["User-Agent"] = "us-stock-qr research (expanded public data archive)"
            for symbol in policy["additional_symbols"]:
                try:
                    response = client.get(
                        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                        params={
                            "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
                            "period2": int((day.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()),
                            "interval": "1d",
                            "events": "div,splits",
                        },
                        timeout=(10, 60),
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    raise QuantError(
                        f"Expanded quote acquisition failed for {symbol}: "
                        f"{type(exc).__name__}, HTTP status {status}"
                    ) from exc
                write_text_atomic(output / "extra" / f"{symbol}.json", response.text)
                observed = parse_chart(payload, symbol, start, end)
                actions = corporate_actions(payload, observed, symbol)
                if not observed.index.equals(index):
                    raise QuantError("Extra fund history differs from the inherited session panel.")
                write_text_atomic(
                    output / "extra" / f"{symbol}.csv",
                    observed.to_csv(float_format="%.17g"),
                )
                for name, field in (
                    ("open", "adj_open"),
                    ("close", "adj_close"),
                    ("raw_close", "close"),
                    ("volume", "volume"),
                ):
                    frames[name][symbol] = observed[field]
                extra_sources.append(
                    {
                        "symbol": symbol,
                        "url": response.url,
                        "retrieved_at": utc_now().isoformat(),
                        "sessions": len(observed),
                        **actions,
                    }
                )
        macro_start = str((index[0] - pd.Timedelta(days=14)).date())
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y&cosd={macro_start}&coed={end}"
        macro_path = output / "extra" / "BAA10Y.csv"
        curl_csv(url, macro_path)
        raw = pd.read_csv(macro_path, na_values=["."])
        if list(raw.columns) != ["observation_date", "BAA10Y"]:
            raise QuantError("The expanded credit input is not the declared Baa/Treasury series.")
        credit = pd.Series(
            pd.to_numeric(raw["BAA10Y"], errors="raise").to_numpy(),
            index=pd.to_datetime(raw["observation_date"]),
            name="BAA10Y",
        )
        known, availability = align_observations(credit, index)
        write_text_atomic(
            output / "extra" / "BAA10Y-aligned.csv", known.to_csv(float_format="%.17g")
        )
        write_text_atomic(output / "extra" / "BAA10Y-availability.csv", availability.to_csv())
        data = MarketData(
            frames["open"],
            frames["close"],
            frames["raw_close"],
            frames["volume"],
            frames["risk_free"]["risk_free"],
        )
        data.validate()
        if data.close.index[-1] != day:
            raise QuantError("The expanded panel must reach the actual completed session.")
        for name, frame in frames.items():
            write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
        return {
            "session": end,
            "first_price_session": start,
            "price_sessions": len(index),
            "symbols": list(data.close.columns),
            "parent_reference": reference,
            "inherited_quote_sources": inherited["quote_sources"],
            "inherited_macro_sources": inherited["macro_sources"],
            "inherited_option_index_sources": inherited["option_index_sources"],
            "additional_quote_sources": extra_sources,
            "additional_symbols": policy["additional_symbols"],
            "additional_macro_series": policy["additional_macro_series"],
            "additional_macro_source": {
                "series": "BAA10Y",
                "url": url,
                "retrieved_at": utc_now().isoformat(),
                "maximum_observation_age_days": int(availability["age_calendar_days"].max()),
            },
            "parent_observed_at_not_relabelled": receipt["observed_at"],
            "all_required_sources_verified": True,
            "raw_proprietary_credit_inputs_not_for_publication": True,
            "strategy_returns_calculated": False,
        }

    def collect(self, fetcher=None, clock=utc_now) -> dict:
        acquire = self.acquire if fetcher is None else fetcher
        pending = []

        def checked_fetch(_parent_policy, day, output):
            self.verify()
            expected = self.parent_reference(day)
            result = acquire(self.policy, day, output)
            if (
                result.get("parent_reference") != expected
                or result.get("additional_symbols") != self.policy["additional_symbols"]
                or result.get("additional_macro_series") != self.policy["additional_macro_series"]
                or result.get("strategy_returns_calculated") is not False
            ):
                raise QuantError("Expanded inputs must bind the verified parent and data-only scope.")
            pending.append((day, result))
            return result

        def checked_clock():
            current = clock()
            if pending:
                day, metadata = pending.pop()
                self.validate_extra_times(metadata, day, pd.Timestamp(current))
            return current

        result = self.journal.collect(checked_fetch, checked_clock)
        return {**result, **self.status()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture expanded prospective inputs; no strategy returns."
    )
    parser.add_argument("action", choices=("init", "collect", "status"))
    parser.add_argument("--directory", type=Path, default=ROOT / "data/prospective-factor-inputs-v2")
    args = parser.parse_args()
    try:
        archive = ExpandedArchive(args.directory, read_json(POLICY))
        if args.action == "init":
            result = archive.initialize()
        elif args.action == "collect":
            result = archive.collect()
        else:
            result = archive.status()
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(2, f"Expanded data collection blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
