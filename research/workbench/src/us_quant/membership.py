from __future__ import annotations

import argparse
import csv
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, utc_now, write_json


def membership_date(value: str) -> date:
    try:
        result = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise QuantError("Membership dates must use YYYY-MM-DD.") from exc
    if result.isoformat() != value:
        raise QuantError("Membership dates must use YYYY-MM-DD.")
    return result


@dataclass(frozen=True)
class MembershipHistory:
    root: Path
    manifest: dict
    dates: tuple[date, ...]
    labels: tuple[frozenset[str], ...]

    def on(self, as_of: str) -> dict:
        day = membership_date(as_of)
        if day < self.dates[0]:
            raise QuantError("Requested date precedes the first historical membership snapshot.")
        if day > self.dates[-1]:
            raise QuantError(
                "Requested date exceeds the last source snapshot; do not extend stale history."
            )
        if day > date.today():
            raise QuantError("Future membership is not verified historical evidence.")
        position = bisect_right(self.dates, day) - 1
        return {
            "mode": "historical_membership_reference_only",
            "requested_as_of": as_of,
            "effective_source_snapshot": self.dates[position].isoformat(),
            "member_count": len(self.labels[position]),
            "source_labels": sorted(self.labels[position]),
            "source_repository": self.manifest["repository"],
            "source_commit_sha": self.manifest["commit_sha"],
            "source_sha256": self.manifest["files"]["membership.csv"]["sha256"],
            "source_completeness_independently_verified": False,
            "permanent_security_identity_verified": False,
            "delisted_price_coverage_verified": False,
            "stock_strategy_qualified": False,
            "order_authority": False,
            "warnings": [
                "This reconstructed membership reference is not an official audited universe.",
                "Labels may follow vendor naming conventions; do not treat them as broker symbols.",
                "A date-aware security master is required before joining filings or price series.",
                "A company missing from current prices may be renamed, acquired, or delisted.",
                "Membership counts alone do not establish completeness.",
                *self.manifest.get("source_caveats", []),
            ],
        }

    def summary(self) -> dict:
        counts = [len(labels) for labels in self.labels]
        return {
            "source_repository": self.manifest["repository"],
            "source_commit_sha": self.manifest["commit_sha"],
            "license": self.manifest["license"],
            "first_snapshot": self.dates[0].isoformat(),
            "last_snapshot": self.dates[-1].isoformat(),
            "snapshot_count": len(self.dates),
            "unique_source_labels": len(frozenset.union(*self.labels)),
            "minimum_member_count": min(counts),
            "maximum_member_count": max(counts),
            "source_completeness_independently_verified": False,
            "stock_strategy_qualified": False,
            "order_authority": False,
        }


def load_membership(root: Path) -> MembershipHistory:
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("repository") != "fja05680/sp500"
        or manifest.get("license") != "MIT"
        or not isinstance(manifest.get("commit_sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["commit_sha"])
    ):
        raise QuantError("The membership reference must match the reviewed repository/license.")
    required = {"membership.csv", "LICENSE.txt", "UPSTREAM_README.txt"}
    files = manifest.get("files")
    if not isinstance(files, dict) or not required <= set(files):
        raise QuantError("Membership provenance must retain the data, license, and source caveats.")
    for name, metadata in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise QuantError("Unsafe or missing membership source file.")
        if not isinstance(metadata, dict) or file_digest(path) != metadata.get("sha256"):
            raise QuantError(f"Membership source fingerprint mismatch: {name}")
    if "MIT License" not in (root / "LICENSE.txt").read_text():
        raise QuantError("The approved source license notice is missing.")
    dates, snapshots = [], []
    with (root / "membership.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", "tickers"]:
            raise QuantError(
                "Expected historical date/tickers snapshots, not a current ticker list."
            )
        for row in reader:
            if row.get("date") is None or row.get("tickers") is None or None in row:
                raise QuantError("Malformed historical membership row.")
            day = membership_date(row["date"])
            if dates and day <= dates[-1]:
                raise QuantError("Membership snapshots must have unique, increasing dates.")
            labels = row["tickers"].split(",")
            if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,19}", label) for label in labels):
                raise QuantError("Invalid or empty upstream security label.")
            if len(labels) != len(set(labels)):
                raise QuantError("Duplicate labels within a membership snapshot.")
            dates.append(day)
            snapshots.append(frozenset(labels))
    if not dates:
        raise QuantError("Historical membership source is empty.")
    return MembershipHistory(root, manifest, tuple(dates), tuple(snapshots))


def write_membership_snapshot(root: Path, as_of: str, output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite a historical membership audit artifact.")
    history = load_membership(root)
    result = history.on(as_of)
    result["created_at"] = utc_now()
    result["source_summary"] = history.summary()
    result["implementation_sha256"] = file_digest(Path(__file__))
    write_json(output, result)
    return result


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "membership",
        help="Query a fingerprinted historical universe reference; no order authority.",
    )
    command.add_argument(
        "--source",
        type=Path,
        default=Path("data/stock-universe/fja-sp500-a2430f2af0c7"),
    )
    command.add_argument("--as-of", required=True)
    command.add_argument("--output", type=Path, required=True)
