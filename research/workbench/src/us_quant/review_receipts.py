from __future__ import annotations

import argparse
import fcntl
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from us_quant.candidate_receipts import (
    input_reference,
    local_path,
    validate_receipt,
    write_new_receipt,
)
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, safe_file, timestamp
from us_quant.storage import digest_json, file_digest, read_json

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {(years, scenario) for years in (10, 5) for scenario in ("base", "stress")}


def evidence_time(value) -> pd.Timestamp:
    try:
        if not isinstance(value, str):
            raise ValueError("not an explicit timestamp string")
        at = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise QuantError("Review evidence timestamps must be valid and timezone-aware.") from exc
    if pd.isna(at) or at.tzinfo is None:
        raise QuantError("Review evidence timestamps must be valid and timezone-aware.")
    return at


def complete_paths(record: dict) -> None:
    paths = record.get("paths")
    if (
        not isinstance(paths, list)
        or len(paths) != 4
        or any(not isinstance(row, dict) for row in paths)
        or any(
            type(row.get("years")) is not int or not isinstance(row.get("scenario"), str)
            for row in paths
        )
        or {(row.get("years"), row.get("scenario")) for row in paths} != PAIRS
    ):
        raise QuantError("Only all four complete frozen horizon/scenario paths can be reviewed.")


def verify_references(root: Path, value) -> None:
    if isinstance(value, dict):
        if "path" in value:
            if not isinstance(value.get("sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", value["sha256"]
            ):
                raise QuantError("Every submitted account or source file needs an explicit hash.")
            safe_file(root, value["path"], value["sha256"])
        for child in value.values():
            verify_references(root, child)
    elif isinstance(value, list):
        for child in value:
            verify_references(root, child)


def original_registration(program: ResearchProgram, candidate_id: str) -> tuple[dict, dict]:
    program.verify()
    candidate = program.db.execute(
        "SELECT spec_sha,registered_at,body FROM candidates WHERE id=?", (candidate_id,)
    ).fetchone()
    if candidate is None:
        raise QuantError("Register the actual frozen candidate before submitting review evidence.")
    spec = json.loads(candidate["body"])
    if not isinstance(spec.get("frozen_files"), dict) or not spec["frozen_files"]:
        raise QuantError("The original candidate lacks a complete frozen source manifest.")
    matches = [
        row
        for row in program.db.execute(
            "SELECT seq,at,body,previous_sha,sha FROM events WHERE kind='candidate'"
        )
        if json.loads(row["body"]).get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise QuantError("The review must remain attached to its original candidate event.")
    event = matches[0]
    registration = json.loads(event["body"])
    if (
        spec.get("id") != candidate_id
        or registration["spec_sha256"] != candidate["spec_sha"]
        or registration["registered_at"] != candidate["registered_at"]
        or candidate["registered_at"] != event["at"]
        or registration.get("order_authority") is not False
    ):
        raise QuantError("The review must remain attached to its original candidate event.")
    for path, digest in spec["frozen_files"].items():
        verify_references(program.root, {"path": path, "sha256": digest})
    return spec, {
        "registration": registration,
        "registration_event": {
            "seq": event["seq"],
            "at": event["at"],
            "previous_sha256": event["previous_sha"],
            "sha256": event["sha"],
        },
    }


def complete_sources(bundle: dict) -> None:
    market = bundle.get("market")
    if not isinstance(market, dict) or any(
        not isinstance(market.get(name), dict) or "path" not in market[name]
        for name in ("open", "close", "raw_close", "volume", "risk_free")
    ):
        raise QuantError("A complete frozen market is required; missing sources cannot be filled.")
    for row in bundle["paths"]:
        if any(
            not isinstance(row.get(name), dict) or "path" not in row[name]
            for name in ("strategy", "strategy_bt", "spy", "spy_bt", "targets", "weights")
        ):
            raise QuantError(
                "Each horizon needs original targets, weights and both independent accounts."
            )


def stored_review(program: ResearchProgram, candidate_id: str, bundle: dict) -> dict | None:
    program.verify()
    row = program.db.execute(
        "SELECT at,evidence_sha,body FROM reviews WHERE candidate_id=?", (candidate_id,)
    ).fetchone()
    if row is None:
        return None
    if row["evidence_sha"] != digest_json(bundle):
        raise QuantError(
            "The completed review belongs to different evidence; preserve its outcome."
        )
    review = json.loads(row["body"])
    matches = [
        event
        for event in program.db.execute(
            "SELECT seq,at,body,previous_sha,sha FROM events WHERE kind='review'"
        )
        if json.loads(event["body"]).get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise QuantError("Recover only a uniquely committed complete review event.")
    event = matches[0]
    if (
        review != json.loads(event["body"])
        or review["at"] != row["at"]
        or row["at"] != event["at"]
        or review["evidence_sha256"] != row["evidence_sha"]
        or review.get("order_authority") is not False
        or review.get("live_strategy_update") is not False
        or review.get("independent_forward_validation") is not False
        or not evidence_time(bundle.get("completed_at"))
        <= evidence_time(row["at"])
        <= pd.Timestamp(timestamp())
    ):
        raise QuantError("A review receipt cannot relabel a different event or trading authority.")
    complete_paths(review)
    return {
        "review": review,
        "review_event": {
            "seq": event["seq"],
            "at": event["at"],
            "previous_sha256": event["previous_sha"],
            "sha256": event["sha"],
        },
    }


def check_review_receipt(receipt: dict, expected: dict, now) -> None:
    validate_receipt(receipt, expected, now)
    if pd.Timestamp(receipt["recorded_at"]) < pd.Timestamp(expected["review"]["at"]):
        raise QuantError(
            "A receipt cannot claim completion before the actual review was committed."
        )


def record_or_recover(
    candidate_id: str,
    bundle_path: Path,
    receipt_path: Path,
    *,
    root: Path = ROOT,
    allow_new_review: bool = True,
    receipt_writer=write_new_receipt,
) -> dict:
    root = root.resolve()
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", candidate_id):
        raise QuantError("A stable registered candidate identifier is required.")
    bundle, source = input_reference(root, bundle_path)
    complete_paths(bundle)
    complete_sources(bundle)
    receipt_path = local_path(root, receipt_path)
    if (
        receipt_path.suffix != ".json"
        or receipt_path == root / source["path"]
        or not any(receipt_path.is_relative_to(root / name) for name in ("reports", "evidence"))
    ):
        raise QuantError("Choose a specific in-workbench JSON receipt, never the input bundle.")
    policy = read_json(safe_file(root, "config/research-program.json"))
    ledger = safe_file(root, "runtime/research-program.sqlite3")
    lock_path = local_path(root, Path("runtime/candidate-review-receipts.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        program = ResearchProgram(ledger, policy, root=root)
        try:
            spec, original = original_registration(program, candidate_id)
            if (
                bundle.get("candidate_spec_sha256") != digest_json(spec)
                or bundle.get("market") != spec["market"]
                or bundle.get("as_of") != spec["evaluation_as_of"]
                or bundle.get("order_authority") is not False
                or bundle.get("leveraged_products_allowed") is not False
            ):
                raise QuantError(
                    "Review evidence must match the original frozen candidate and market."
                )
            if not (
                evidence_time(original["registration"]["registered_at"])
                <= evidence_time(bundle.get("completed_at"))
                <= pd.Timestamp(timestamp())
            ):
                raise QuantError(
                    "Review completion must follow actual registration, not a fake clock."
                )
            verify_references(root, bundle)
            saved = stored_review(program, candidate_id, bundle)
            created = False
            if saved is None:
                if receipt_path.exists():
                    raise QuantError("An existing receipt has no actual committed review.")
                if not allow_new_review:
                    raise QuantError(
                        "Recovery cannot submit an unreviewed or incomplete evaluation."
                    )
                program.review(candidate_id, bundle)
                created = True
                saved = stored_review(program, candidate_id, bundle)
                if saved is None:
                    raise QuantError("No complete review was committed; do not publish a receipt.")
            else:
                program.review(candidate_id, bundle, audit_existing=True)
            safe_file(root, source["path"], source["sha256"])
            verify_references(root, bundle)
            for path, digest in spec["frozen_files"].items():
                verify_references(root, {"path": path, "sha256": digest})
            expected = {
                "schema_version": 1,
                **original,
                **saved,
                "evidence_bundle": source,
                "candidate_spec_sha256": digest_json(spec),
                "research_policy_sha256": digest_json(policy),
                "complete_source_bound_replay_verified": True,
                "partial_paths_completed_or_filled": False,
                "receipt_is_new_portfolio_evaluation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            if receipt_path.exists():
                receipt, _ = input_reference(root, receipt_path)
                check_review_receipt(receipt, expected, timestamp())
                action = "already_recorded"
            else:
                receipt = {
                    **expected,
                    "recorded_at": timestamp().isoformat(),
                    "receipt_writer_sha256": file_digest(Path(__file__)),
                }
                receipt_writer(receipt_path, receipt)
                written, _ = input_reference(root, receipt_path)
                check_review_receipt(written, expected, timestamp())
                action = "review_recorded" if created else "recovered_review"
            program.verify()
            return {
                "action": action,
                "candidate_id": candidate_id,
                "actual_reviewed_at": saved["review"]["at"],
                "historical_status": saved["review"]["status"],
                "historical_gates_passed": saved["review"]["historical_gates_passed"],
                "existing_review_read_only_reaudited": not created,
                "new_reviews_recorded": int(created),
                "new_strategy_evaluations": int(created),
                "receipt": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": file_digest(receipt_path),
                "review_event_sha256": saved["review_event"]["sha256"],
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
        finally:
            program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit only complete frozen reviews or recover receipts; never orders."
    )
    parser.add_argument("action", choices=("record", "recover"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                record_or_recover(
                    args.candidate_id,
                    args.bundle,
                    args.receipt,
                    allow_new_review=args.action == "record",
                ),
                indent=2,
            )
        )
    except (QuantError, sqlite3.Error, OSError) as exc:
        parser.exit(
            2, f"Review receipt blocked; preserve all existing accounts and review events: {exc}\n"
        )


if __name__ == "__main__":
    main()
