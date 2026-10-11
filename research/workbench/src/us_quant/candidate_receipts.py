from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, safe_file, timestamp
from us_quant.storage import digest_bytes, digest_json, file_digest, read_json, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).resolve().parents[2]


def local_path(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise QuantError("Candidate receipts and inputs must stay inside the isolated workbench.")
    return path.resolve()


def input_reference(root: Path, path: Path) -> tuple[dict, dict]:
    path = local_path(root, path)
    relative = path.relative_to(root).as_posix()
    safe_file(root, relative)
    raw = path.read_bytes()
    try:
        content = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise QuantError(
            "An incomplete input or receipt cannot be replaced with a valid-looking default."
        ) from exc
    if not isinstance(content, dict):
        raise QuantError("A candidate input or receipt must be an explicit JSON object.")
    try:
        body_sha256 = digest_json(content)
    except ValueError as exc:
        raise QuantError("Nonfinite JSON inputs cannot identify a frozen registration.") from exc
    return content, {
        "path": relative,
        "sha256": digest_bytes(raw),
        "body_sha256": body_sha256,
    }


def recorded_registration(program: ResearchProgram, spec: dict, readiness: dict) -> dict | None:
    program.verify()
    candidate = program.db.execute(
        "SELECT spec_sha,registered_at,body FROM candidates WHERE id=?", (spec.get("id"),)
    ).fetchone()
    if candidate is None:
        return None
    if candidate["spec_sha"] != digest_json(spec) or json.loads(candidate["body"]) != spec:
        raise QuantError("The candidate already exists with a different frozen specification.")
    matching = [
        event
        for event in program.db.execute(
            "SELECT seq,at,body,previous_sha,sha FROM events WHERE kind='candidate'"
        )
        if json.loads(event["body"]).get("candidate_id") == spec["id"]
    ]
    if len(matching) != 1:
        raise QuantError("Recover only a uniquely verified actual candidate registration event.")
    event = matching[0]
    body = json.loads(event["body"])
    if (
        body["spec_sha256"] != candidate["spec_sha"]
        or body["registered_at"] != candidate["registered_at"]
        or body["registered_at"] != event["at"]
        or body["readiness_sha256"] != digest_json(readiness)
        or body.get("order_authority") is not False
    ):
        raise QuantError(
            "Recovery requires the original registered readiness and event, not refreshed inputs."
        )
    return {
        "registration": body,
        "registration_event": {
            "seq": event["seq"],
            "at": event["at"],
            "previous_sha256": event["previous_sha"],
            "sha256": event["sha"],
        },
    }


def check_eligibility(spec: dict, root: Path, now) -> None:
    _, policy = registered_generator(spec, root)
    if "eligible_not_before" not in policy:
        return
    try:
        eligible = pd.Timestamp(policy["eligible_not_before"])
    except (ValueError, TypeError) as exc:
        raise QuantError(
            "Candidate eligibility must be an explicit timezone-aware instant."
        ) from exc
    if pd.isna(eligible) or eligible.tzinfo is None:
        raise QuantError("Candidate eligibility must be an explicit timezone-aware instant.")
    if pd.Timestamp(now) < eligible:
        raise QuantError(f"Candidate registration is not eligible before {eligible.isoformat()}.")


def validate_receipt(receipt: dict, expected: dict, now) -> None:
    if set(receipt) != {*expected, "recorded_at", "receipt_writer_sha256"} or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise QuantError("An existing receipt differs from the original registration; preserve it.")
    try:
        recorded = pd.Timestamp(receipt["recorded_at"])
    except (ValueError, TypeError) as exc:
        raise QuantError("Candidate receipt timestamps must be valid and timezone-aware.") from exc
    if (
        pd.isna(recorded)
        or recorded.tzinfo is None
        or not pd.Timestamp(expected["registration"]["registered_at"])
        <= recorded
        <= pd.Timestamp(now)
        or not isinstance(receipt["receipt_writer_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt["receipt_writer_sha256"])
    ):
        raise QuantError("A receipt cannot predate registration or invent a future recovery time.")


def write_new_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_json(temporary, receipt)
        # A complete, fsynced file appears atomically without replacing another receipt.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise QuantError("The receipt appeared concurrently; do not overwrite it.") from exc
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def register_or_recover(
    spec_path: Path,
    readiness_path: Path,
    receipt_path: Path,
    *,
    root: Path = ROOT,
    allow_new_registration: bool = True,
    receipt_writer=write_new_receipt,
) -> dict:
    root = root.resolve()
    spec, spec_reference = input_reference(root, spec_path)
    readiness, readiness_reference = input_reference(root, readiness_path)
    receipt_path = local_path(root, receipt_path)
    if (
        receipt_path.suffix != ".json"
        or not any(
            receipt_path.is_relative_to(root / name) and receipt_path != root / name
            for name in ("evidence", "reports")
        )
        or receipt_path
        in {
            root / spec_reference["path"],
            root / readiness_reference["path"],
        }
    ):
        raise QuantError("Choose a specific JSON receipt under reports or evidence, not an input.")
    if spec.get("order_authority") is not False or not isinstance(spec.get("frozen_files"), dict):
        raise QuantError("A no-order frozen candidate specification is required.")
    for path, digest in spec["frozen_files"].items():
        safe_file(root, path, digest)
    policy = read_json(safe_file(root, "config/research-program.json"))
    ledger = safe_file(root, "runtime/research-program.sqlite3")
    lock_path = local_path(root, Path("runtime/candidate-registration-receipts.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = timestamp()
        program = ResearchProgram(ledger, policy, root=root)
        try:
            saved = recorded_registration(program, spec, readiness)
            created = False
            if saved is None:
                if receipt_path.exists():
                    raise QuantError("An existing receipt has no actual candidate registration.")
                if not allow_new_registration:
                    raise QuantError("Recovery cannot register an unknown candidate.")
                check_eligibility(spec, root, now)
                program.register_candidate(spec, readiness, now=now)
                created = True
                saved = recorded_registration(program, spec, readiness)
                if saved is None:
                    raise QuantError("No committed registration exists; do not publish a receipt.")
            for reference in (spec_reference, readiness_reference):
                safe_file(root, reference["path"], reference["sha256"])
            expected = {
                "schema_version": 1,
                **saved,
                "candidate_spec": spec_reference,
                "readiness": readiness_reference,
                "research_policy_sha256": digest_json(policy),
                "new_strategy_evaluations": 0,
                "portfolio_returns_computed": False,
                "research_champion_updated": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            if receipt_path.exists():
                receipt, _ = input_reference(root, receipt_path)
                validate_receipt(receipt, expected, now)
                action = "already_recorded"
            else:
                receipt = {
                    **expected,
                    "recorded_at": timestamp().isoformat(),
                    "receipt_writer_sha256": file_digest(Path(__file__)),
                }
                receipt_writer(receipt_path, receipt)
                written, _ = input_reference(root, receipt_path)
                validate_receipt(written, expected, timestamp())
                action = "registered" if created else "recovered_registration"
            program.verify()
            return {
                "action": action,
                "candidate_id": spec["id"],
                "actual_registered_at": saved["registration"]["registered_at"],
                "new_candidate_registration": created,
                "receipt": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": file_digest(receipt_path),
                "registration_event_sha256": saved["registration_event"]["sha256"],
                "new_strategy_evaluations": 0,
                "portfolio_returns_computed": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
        finally:
            program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover verified candidate receipts; no evaluation or orders."
    )
    parser.add_argument("action", choices=("register", "recover"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                register_or_recover(
                    args.spec,
                    args.readiness,
                    args.receipt,
                    allow_new_registration=args.action == "register",
                ),
                indent=2,
            )
        )
    except (QuantError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f"Candidate receipt blocked; preserve any committed registration: {exc}\n")


if __name__ == "__main__":
    main()
