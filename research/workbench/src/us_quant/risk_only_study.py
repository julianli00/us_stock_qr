from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

from us_quant import equity_budget_attribution as publication_core
from us_quant import factor_covariance_risk as strategy
from us_quant.candidate_receipts import (
    input_reference, local_path, recorded_registration, register_or_recover, write_new_receipt,
)
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_json, file_digest, new_output_directory, read_json, utc_now, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "data/factor-covariance-risk-20261011"
PACKET = ROOT / "evidence/risk_only_covariance_20261011_registration.json"
TARGET_PROOF = ROOT / "evidence/risk_only_covariance_20261011_target_verification.json"


def program() -> ResearchProgram:
    return ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"), root=ROOT,
    )


def fingerprint() -> str:
    return digest_json({
        name: file_digest(Path(__file__).parent / name)
        for name in (
            "risk_only_study.py", "factor_covariance_risk.py", "research_program.py",
            "strategy_replay.py", "cash_funded_accounting_v2.py", "bt_audit.py",
            "candidate_receipts.py", "research_weekly.py", "equity_budget_attribution.py",
        )
    })


def register(packet_path: Path = PACKET) -> dict:
    packet_path = local_path(ROOT, packet_path)
    if packet_path.exists() or packet_path.suffix != ".json" or not packet_path.is_relative_to(ROOT / "evidence"):
        raise QuantError("Preserve the original metadata registration packet.")
    policy = read_json(ROOT / "config/factor-covariance-risk.json")
    strategy.validate(policy)
    proof, proof_reference = input_reference(ROOT, TARGET_PROOF)
    if (
        proof["checked_risk_targets"] != 244
        or proof["candidate_returns_or_metrics_computed"] is not False
        or proof["new_registered_candidates"] != 0
    ):
        raise QuantError("Independent full target/optimization proof must precede registration.")
    records = []
    for candidate in strategy.CANDIDATES:
        spec_path = PREPARED / f"{candidate['id']}-spec.json"
        if file_digest(spec_path) != proof["prepared_spec_sha256"][candidate["id"]]:
            raise QuantError("A prepared configuration changed after its independent target proof.")
        receipt_path = ROOT / "reports/risk-only-covariance-plan-20261011" / f"{candidate['id']}-registration.json"
        admission = register_or_recover(spec_path, PREPARED / "readiness.json", receipt_path, root=ROOT)
        receipt = read_json(safe_file(ROOT, admission["receipt"], admission["receipt_sha256"]))
        spec = read_json(spec_path)
        if spec["configuration"] != candidate:
            raise QuantError("Register only both fixed covariance methods, without substitutions.")
        records.append({
            "spec": spec, "registration": receipt["registration"],
            "readiness": receipt["readiness"], "candidate_spec": receipt["candidate_spec"],
        })
    packet = {
        "schema_version": 1, "study": policy, "recorded_at": utc_now(),
        "runner_sha256": fingerprint(), "engine_sha256": engine_fingerprint(),
        "target_verification": proof_reference, "candidates": records,
        "new_strategy_evaluations": 0, "registered_before_new_strategy_outcomes": True,
        "new_economic_factor_definitions": 0,
        "existing_W42_queue_or_forward_rules_changed": False,
        "investment_objective_verified": False, "order_authority": False,
    }
    write_new_receipt(packet_path, packet)
    return packet


def publication(reference: dict, entries: list[dict], commit: str) -> dict:
    inherited = publication_core.published_inputs(
        reference, entries, read_json(ROOT / "config/equity-budget-attribution.json"), commit,
    )
    required = {
        "src/us_quant/risk_only_study.py",
        "src/us_quant/factor_covariance_risk.py",
        "src/us_quant/research_program.py",
        "src/us_quant/strategy_replay.py",
        "config/factor-covariance-risk.json",
        TARGET_PROOF.relative_to(ROOT).as_posix(),
        "evidence/risk_only_covariance_20261011_source_context.json",
    }
    own = verify_publication(ROOT, reference, required, commit)
    return {**own, "published_imported_core": inherited}


def evaluate(packet_path: Path, output: Path, commit: str) -> dict:
    packet, reference = input_reference(ROOT, packet_path)
    policy = read_json(ROOT / "config/factor-covariance-risk.json")
    strategy.validate(policy)
    entries = packet.get("candidates")
    if (
        not isinstance(entries, list) or len(entries) != 2
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"spec", "registration", "readiness", "candidate_spec"}
            or any(not isinstance(entry[key], dict) for key in entry)
            for entry in entries
        )
    ):
        raise QuantError("Both original candidate specifications and receipts are required.")
    if (
        packet.get("study") != policy or packet.get("runner_sha256") != fingerprint()
        or packet.get("engine_sha256") != engine_fingerprint()
        or packet.get("registered_before_new_strategy_outcomes") is not True
        or packet.get("new_strategy_evaluations") != 0
        or packet.get("order_authority") is not False
        or [entry["spec"].get("configuration") for entry in entries] != list(strategy.CANDIDATES)
    ):
        raise QuantError("Keep the original complete covariance registration and runner.")
    safe_file(ROOT, packet["target_verification"]["path"], packet["target_verification"]["sha256"])
    for entry in entries:
        spec = entry["spec"]
        if (
            spec["id"] != entry["registration"]["candidate_id"]
            or entry["registration"]["spec_sha256"] != digest_json(spec)
        ):
            raise QuantError("Each outcome must identify its original actual candidate registration.")
        actual = read_json(safe_file(
            ROOT, entry["candidate_spec"]["path"], entry["candidate_spec"]["sha256"],
        ))
        if actual != spec:
            raise QuantError("The submitted metadata packet differs from its original prepared spec.")
        safe_file(ROOT, entry["readiness"]["path"], entry["readiness"]["sha256"])
        for name, digest in spec["frozen_files"].items():
            safe_file(ROOT, name, digest)
    published = publication(reference, entries, commit)
    output = local_path(ROOT, output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports"):
        raise QuantError("Use a new isolated private study output directory.")
    with_program = program()
    created = False
    results = []
    try:
        for entry in entries:
            actual = with_program.db.execute(
                "SELECT body FROM candidates WHERE id=?", (entry["spec"]["id"],),
            ).fetchone()
            if actual is None or json.loads(actual["body"]) != entry["spec"]:
                raise QuantError("A publication packet cannot register an unknown candidate.")
            readiness = read_json(safe_file(
                ROOT, entry["readiness"]["path"], entry["readiness"]["sha256"],
            ))
            saved = recorded_registration(with_program, entry["spec"], readiness)
            if saved is None or saved["registration"] != entry["registration"]:
                raise QuantError("The packet differs from its actual original registration event.")
            if with_program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (entry["spec"]["id"],),
            ).fetchone() is not None or entry["spec"]["id"] in with_program.technical_failures():
                raise QuantError("Do not repeat a reviewed or technically rejected strategy trial.")
        new_output_directory(output)
        created = True
        write_new_receipt(output / "publication.json", {
            **published, "verified_on_remote_at": utc_now(),
            "publication_fence_predates_new_strategy_accounts": True,
            "order_authority": False, "investment_objective_verified": False,
        })
        started = utc_now()
        for entry in entries:
            result = with_program.evaluate_registered(entry["spec"]["id"], output)
            if "evaluation_action" in result:
                raise QuantError("Do not duplicate a completed or technically rejected candidate trial.")
            results.append(result)
            write_json(output / "progress.json", {
                "started_at": started, "completed_candidate_reviews": results,
                "investment_objective_verified": False, "order_authority": False,
            })
        result = {
            "schema_version": 1, "started_at": started, "completed_at": utc_now(),
            "published_registration_commit": commit, "registration_sha256": reference["sha256"],
            "candidate_reviews": results, "new_strategy_evaluations": len(results),
            "new_economic_factor_definitions": 0,
            "investment_objective_verified": False, "order_authority": False,
        }
        write_json(output / "results.json", result)
        return result
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        if created:
            write_json(output / "failure.json", {
                "recorded_at": utc_now(), "reason": str(exc),
                "completed_candidate_reviews_retained": len(results),
                "partial_outputs_deleted_or_filled": False,
                "investment_objective_verified": False, "order_authority": False,
            })
        raise
    finally:
        with_program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish-before-performance covariance study; never orders.")
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--registration", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/risk-only-covariance-20261011")
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        if args.action == "register":
            result = register(args.registration)
            print(json.dumps({
                "registered_candidates": len(result["candidates"]),
                "recorded_at": result["recorded_at"], "new_strategy_evaluations": 0,
                "investment_objective_verified": False,
            }, indent=2))
        else:
            result = evaluate(args.registration, args.output, args.published_commit)
            print(json.dumps({
                "completed_candidate_reviews": len(result["candidate_reviews"]),
                "all_candidate_outcomes_retained": True, "investment_objective_verified": False,
            }, indent=2))
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Risk-only study blocked; preserve complete evidence: {exc}\n")


if __name__ == "__main__":
    main()
