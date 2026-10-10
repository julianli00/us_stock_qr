from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, next_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.macro_factor_tilt import build_targets, load_macro
from us_quant.prospective_data import (
    ProspectiveArchive,
    fingerprint as parent_fingerprint,
    utc_now,
)
from us_quant.research_program import ResearchProgram, safe_file
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/prospective-target-observations.json"
CANDIDATE = "macro_real_yield_factor_tilt"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("observer_id") != "prospective_rejected_candidate_targets_v1"
        or policy.get("parent_policy") != "config/prospective-data.json"
        or policy.get("parent_archive") != "data/prospective-market-v1"
        or policy.get("source_registration")
        != {
            "path": "evidence/macro_tilt_20261010_registration.json",
            "sha256": "172fde23ff19e0a42c9c46f492d1230f88c2036e409f643148f862f2ff840e25",
        }
        or policy.get("source_candidate_spec_sha256")
        != "b9a581c1dac686b26fa780fb97fbea76521e7b1d9e9522408ac89396f3d549c2"
        or policy.get("source_candidate_id") != CANDIDATE
        or policy.get("required_original_review_status") != "rejected_historical"
        or policy.get("target_investment_budget") != 0.98
        or policy.get("hypothetical_base_delay_sessions") != 1
        or policy.get("hypothetical_stress_delay_sessions") != 2
        or any(
            policy.get(key) is not True
            for key in (
                "initial_target_from_latest_known_completed_month",
                "update_only_for_new_completed_month",
                "retain_recorded_target_when_no_new_month",
            )
        )
        or any(
            policy.get(key) is not False
            for key in (
                "compute_strategy_returns", "initialize_portfolio", "submit_orders",
                "update_research_champion", "order_authority",
            )
        )
    ):
        raise QuantError("Target observations must preserve the fixed rejected rule and no-order scope.")


def observed_target(data: MarketData, macro: pd.DataFrame, spec: dict, policy: dict) -> tuple[str, dict]:
    targets = build_targets(data, macro, spec["configuration"], policy).dropna(how="all")
    if targets.empty:
        raise QuantError("A complete warmup and known month-end target are required.")
    source_day = targets.index[-1]
    target = targets.iloc[-1]
    if (
        source_day > data.close.index[-1]
        or not is_month_end(source_day)
        or not np.isfinite(target).all()
        or (target < 0).any()
        or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Observed target violates its completed-month or cash-funded constraints.")
    return str(source_day.date()), {symbol: float(value) for symbol, value in target.items()}


class TargetJournal:
    def __init__(self, directory: Path, policy: dict, *, root: Path = ROOT):
        validate_policy(policy)
        directory = directory if directory.is_absolute() else root / directory
        parent = root / policy["parent_archive"]
        if (
            directory.is_symlink()
            or not directory.resolve().is_relative_to(root.resolve())
            or directory.resolve().is_relative_to(parent.resolve())
            or parent.resolve().is_relative_to(directory.resolve())
        ):
            raise QuantError("Target records need a separate, in-workbench journal.")
        self.root, self.directory, self.policy = root, directory, policy
        parent_policy = read_json(safe_file(root, policy["parent_policy"]))
        self.parent = ProspectiveArchive(parent, parent_policy)
        self.journal = ProspectiveArchive(directory, parent_policy)

    def fingerprint(self) -> str:
        reference = self.policy["source_registration"]
        return digest_json(
            {
                "observer": file_digest(Path(__file__)),
                "parent_collector": parent_fingerprint(),
                "source_registration": file_digest(safe_file(self.root, reference["path"], reference["sha256"])),
                "path_validation": digest_json(inspect.getsource(safe_file)),
            }
        )

    def source_reference(self) -> tuple[dict, dict]:
        reference = self.policy["source_registration"]
        registration = read_json(safe_file(self.root, reference["path"], reference["sha256"]))
        matches = [row for row in registration["candidates"] if row["spec"]["id"] == CANDIDATE]
        if len(matches) != 1:
            raise QuantError("One exact previously registered candidate is required.")
        spec = matches[0]["spec"]
        if digest_json(spec) != self.policy["source_candidate_spec_sha256"]:
            raise QuantError("The historical source candidate specification changed.")
        for path, digest in spec["frozen_files"].items():
            safe_file(self.root, path, digest)
        if file_digest(Path(build_targets.__code__.co_filename)) != spec["frozen_files"][
            "src/us_quant/macro_factor_tilt.py"
        ]:
            raise QuantError("The executed target rule is not the original frozen implementation.")
        program = ResearchProgram(
            self.root / "runtime/research-program.sqlite3",
            read_json(self.root / "config/research-program.json"),
            root=self.root,
        )
        try:
            state = program.status()
            current = next(
                (row for row in state["registered_candidates"] if row["id"] == CANDIDATE),
                None,
            )
            review = next(
                (row for row in state["candidate_reviews"] if row["candidate_id"] == CANDIDATE),
                None,
            )
            if (
                current is None
                or review is None
                or digest_json(current) != digest_json(spec)
                or review["status"] != "rejected_historical"
                or review["historical_gates_passed"] is not False
            ):
                raise QuantError("The fixed observation profile must remain distinct from qualification.")
        finally:
            program.close()
        return spec, read_json(self.root / "config/macro-factor-tilt.json")

    def validate_target(self, target: dict) -> None:
        weights = target.get("weights")
        if not isinstance(weights, dict) or set(weights) != {
            "SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV"
        }:
            raise QuantError("The target must specify every instrument in the original rule.")
        values = np.array(list(weights.values()), dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 0.98, rtol=0, atol=1e-12)
            or any(weights[symbol] <= 0 for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "GLD"))
            or weights["SPY"] != 0
            or weights["IEF"] != 0
        ):
            raise QuantError("The target must preserve the original factor mandate and cash funding.")

    def initialize(self, now=None) -> dict:
        self.source_reference()
        self.parent.verify()
        record = self.journal.initialize(now)
        record.update(
            {
                "observer_mode": "prospective_target_observations_only",
                "observer_policy_sha256": digest_json(self.policy),
                "observer_sha256": self.fingerprint(),
                "parent_registration_sha256": file_digest(self.parent.directory / "registration.json"),
                "source_candidate_spec_sha256": self.policy["source_candidate_spec_sha256"],
                "historical_gates_passed": False,
            }
        )
        write_json(self.directory / "registration.json", record)
        self.verify()
        return record

    def parent_snapshot(self, day: pd.Timestamp) -> tuple[dict, Path]:
        records = self.parent.verify()
        if not records or records[-1]["session"] != str(day.date()):
            raise QuantError("Target observation needs an actually captured matching-session input.")
        record = records[-1]
        manifest = safe_file(
            self.parent.directory,
            f"{record['snapshot_path']}/manifest.json",
            record["manifest_sha256"],
        )
        return record, manifest.parent

    def verify(self) -> list[dict]:
        self.source_reference()
        self.parent.verify()
        registration = read_json(safe_file(self.directory, "registration.json"))
        if (
            registration.get("observer_policy_sha256") != digest_json(self.policy)
            or registration.get("observer_mode") != "prospective_target_observations_only"
            or registration.get("observer_sha256") != self.fingerprint()
            or registration.get("parent_registration_sha256")
            != file_digest(self.parent.directory / "registration.json")
            or registration.get("source_candidate_spec_sha256") != self.policy["source_candidate_spec_sha256"]
            or registration.get("historical_gates_passed") is not False
        ):
            raise QuantError("The target-observation registration or original rule was changed.")
        records = self.journal.verify()
        previous = None
        previous_target_digest = None
        for record in records:
            manifest = read_json(
                safe_file(
                    self.directory,
                    f"{record['snapshot_path']}/manifest.json",
                    record["manifest_sha256"],
                )
            )
            source = manifest["parent_reference"]
            parent_path = safe_file(
                self.parent.directory,
                f"receipts/{record['session']}.json",
                source["receipt_sha256"],
            )
            parent = read_json(parent_path)
            if (
                source["observed_at"] != parent["observed_at"]
                or source["manifest_sha256"] != parent["manifest_sha256"]
                or pd.Timestamp(source["observed_at"]) > pd.Timestamp(record["observed_at"])
                or manifest["source_candidate_id"] != CANDIDATE
                or manifest["historical_gates_passed"] is not False
                or manifest["strategy_returns_calculated"] is not False
                or manifest["orders_submitted"] is not False
                or manifest["portfolio_initialized"] is not False
            ):
                raise QuantError("Target provenance or its explicitly non-trading scope changed.")
            source_month = pd.Timestamp(manifest["source_rule_signal_session"])
            session = pd.Timestamp(record["session"])
            if (
                pd.isna(source_month)
                or source_month.tzinfo is not None
                or str(source_month.date()) != manifest["source_rule_signal_session"]
                or source_month > session
                or not is_month_end(source_month)
            ):
                raise QuantError("An unobserved month cannot generate a prospective target.")
            target_path = safe_file(
                self.directory,
                f"{record['snapshot_path']}/target.json",
                manifest["files"]["target.json"],
            )
            target = read_json(
                target_path
            )
            self.validate_target(target)
            generated = previous is None or source_month > previous
            if (
                target.get("source_rule_signal_session") != manifest["source_rule_signal_session"]
                or manifest["new_target_generated"] != generated
                or (previous is not None and source_month < previous)
                or manifest["hypothetical_base_execution_session"]
                != (str(next_session(session).date()) if generated else None)
                or manifest["hypothetical_stress_execution_session"]
                != (str(next_session(next_session(session)).date()) if generated else None)
            ):
                raise QuantError("Target month, change cadence or hypothetical execution date changed.")
            previous = source_month
            target_digest = file_digest(target_path)
            if not generated and target_digest != previous_target_digest:
                raise QuantError("A completed old month cannot be silently retuned after new observations.")
            previous_target_digest = target_digest
        return records

    def status(self) -> dict:
        records = self.verify()
        manifests = [
            read_json(self.directory / row["snapshot_path"] / "manifest.json") for row in records
        ]
        return {
            "mode": "prospective_target_observations_only",
            "source_candidate_id": CANDIDATE,
            "historical_status": "rejected_historical",
            "source_sessions_recorded": len(records),
            "new_target_observations": sum(row["new_target_generated"] for row in manifests),
            "latest_source_session": records[-1]["session"] if records else None,
            "latest_rule_signal_session": manifests[-1]["source_rule_signal_session"] if manifests else None,
            "missed_observation_sessions": [day for row in records for day in row["missed_sessions"]],
            "receipt_chain_head": read_json(self.directory / "head.json")["receipt_sha256"],
            "new_market_data_acquired_by_observer": False,
            "strategy_returns_calculated": False,
            "portfolio_initialized": False,
            "orders_submitted": False,
            "research_champion_updated": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }

    def collect(self, clock=utc_now, generator=observed_target) -> dict:
        pending = []

        def collect_target(_parent_policy, day, output):
            records = self.verify()
            parent, snapshot = self.parent_snapshot(day)
            source = {
                "receipt_sha256": file_digest(self.parent.directory / "receipts" / f"{day.date()}.json"),
                "manifest_sha256": parent["manifest_sha256"],
                "observed_at": parent["observed_at"],
            }
            spec, policy = self.source_reference()
            panels = {
                name: pd.read_csv(snapshot / f"{name}.csv", index_col=0, parse_dates=True)
                for name in ("open", "close", "raw_close", "volume", "risk_free")
            }
            data = MarketData(
                panels["open"].drop(columns="TLT"),
                panels["close"].drop(columns="TLT"),
                panels["raw_close"].drop(columns="TLT"),
                panels["volume"].drop(columns="TLT"),
                panels["risk_free"]["risk_free"],
            )
            data.validate()
            months = [value for value in data.close.index if is_month_end(value)]
            if not months:
                raise QuantError("No completed source month is available.")
            latest = str(months[-1].date())
            previous = None
            if records:
                previous = read_json(self.directory / records[-1]["snapshot_path"] / "manifest.json")
                if latest < previous["source_rule_signal_session"]:
                    raise QuantError("Do not backdate a recorded target month.")
            generated = previous is None or latest > previous["source_rule_signal_session"]
            if generated:
                macro, _ = load_macro(snapshot / "macro", data.close.index)
                month, weights = generator(data, macro, spec, policy)
                if month != latest:
                    raise QuantError("The frozen rule did not produce the latest observed month.")
                target = {"source_rule_signal_session": month, "weights": weights}
            else:
                target = read_json(self.directory / records[-1]["snapshot_path"] / "target.json")
            self.validate_target(target)
            write_json(output / "target.json", target)
            pending.append(source["observed_at"])
            return {
                "session": str(day.date()),
                "parent_reference": source,
                "source_candidate_id": CANDIDATE,
                "source_rule_signal_session": latest,
                "new_target_generated": generated,
                "hypothetical_base_execution_session": str(next_session(day).date()) if generated else None,
                "hypothetical_stress_execution_session": (
                    str(next_session(next_session(day)).date()) if generated else None
                ),
                "historical_gates_passed": False,
                "all_required_sources_verified": True,
                "strategy_returns_calculated": False,
                "orders_submitted": False,
                "portfolio_initialized": False,
            }

        def checked_clock():
            current = clock()
            if pending:
                acquired = pd.Timestamp(pending.pop())
                finished = pd.Timestamp(current)
                if pd.isna(finished) or finished.tzinfo is None:
                    raise QuantError("Target observations require timezone-aware completion times.")
                if acquired > finished:
                    raise QuantError("The target cannot be recorded before its actual inputs existed.")
            return current

        result = self.journal.collect(collect_target, checked_clock)
        return {"action": result["action"], **self.status()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Observe frozen research targets; no portfolio or orders.")
    parser.add_argument("action", choices=("init", "collect", "status"))
    parser.add_argument("--directory", type=Path, default=ROOT / "data/prospective-target-observations-v1")
    args = parser.parse_args()
    try:
        journal = TargetJournal(args.directory, read_json(POLICY))
        if args.action == "init":
            result = journal.initialize()
        elif args.action == "collect":
            result = journal.collect()
        else:
            result = journal.status()
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Target observation blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
