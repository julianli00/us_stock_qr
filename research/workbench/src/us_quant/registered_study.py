from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path

from us_quant import equity_budget_attribution as publication_core
from us_quant.candidate_receipts import input_reference, local_path, recorded_registration, register_or_recover, write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.research_weekly import verify_publication
from us_quant.review_receipts import evidence_time
from us_quant.storage import digest_json, file_digest, new_output_directory, read_json, utc_now, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).resolve().parents[2]


def fingerprint() -> str:
    return digest_json({
        name: file_digest(Path(__file__).parent / name)
        for name in (
            "registered_study.py", "research_program.py", "strategy_replay.py",
            "candidate_receipts.py", "research_weekly.py", "equity_budget_attribution.py",
            "cash_funded_accounting_v2.py", "bt_audit.py", "factor_validation.py",
        )
    })


def program() -> ResearchProgram:
    return ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json"), root=ROOT,
    )


def policy_input(path: Path) -> tuple[dict, dict]:
    policy, reference = input_reference(ROOT, path)
    if (
        not reference["path"].startswith("config/") or not reference["path"].endswith(".json")
        or policy.get("schema_version") != 1 or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("new_configurations") != 2 or policy.get("new_economic_factor_definitions") != 0
        or not isinstance(policy.get("candidates"), list) or len(policy["candidates"]) != 2
        or any(
            not isinstance(row, dict) or not isinstance(row.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", row["id"]) for row in policy["candidates"]
        )
        or len({row["id"] for row in policy["candidates"]}) != 2
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("A registered study needs two fixed existing-factor configurations and no trading authority.")
    return policy, reference


def register(policy_path: Path, prepared: Path, proof_path: Path, packet_path: Path) -> dict:
    policy, policy_reference = policy_input(policy_path)
    prepared, packet_path = local_path(ROOT, prepared), local_path(ROOT, packet_path)
    if (
        not prepared.is_relative_to(ROOT / "data") or prepared == ROOT / "data"
        or not packet_path.is_relative_to(ROOT / "evidence") or packet_path.suffix != ".json"
        or packet_path.exists()
    ):
        raise QuantError("Preserve existing packets and use complete original private preparation.")
    proof, proof_reference = input_reference(ROOT, proof_path)
    if (
        proof.get("candidate_returns_or_metrics_computed") is not False
        or not isinstance(proof.get("prepared_spec_sha256"), dict)
    ):
        raise QuantError("A complete source and rule proof must precede portfolio outcome calculation.")
    records = []
    for configuration in policy["candidates"]:
        spec_path = prepared / f"{configuration['id']}-spec.json"
        spec, spec_reference = input_reference(ROOT, spec_path)
        if (
            not isinstance(spec.get("frozen_files"), dict)
            or spec.get("id") != configuration["id"] or spec.get("configuration") != configuration
            or spec["frozen_files"].get(policy_reference["path"]) != policy_reference["sha256"]
            or proof["prepared_spec_sha256"].get(configuration["id"]) != spec_reference["sha256"]
        ):
            raise QuantError("Each original specification must match its frozen policy and independent source proof.")
        registered_generator(spec, ROOT)
        receipt_path = ROOT / "reports/registered-study-receipts" / f"{configuration['id']}.json"
        admission = register_or_recover(spec_path, prepared / "readiness.json", receipt_path, root=ROOT)
        receipt, _ = input_reference(ROOT, ROOT / admission["receipt"])
        records.append({
            "spec": spec, "candidate_spec": receipt["candidate_spec"],
            "readiness": receipt["readiness"], "registration": receipt["registration"],
        })
    packet = {
        "schema_version": 1, "recorded_at": utc_now(), "policy": policy,
        "policy_reference": policy_reference, "pre_account_proof": proof_reference,
        "prepared_path": prepared.relative_to(ROOT).as_posix(),
        "runner_sha256": fingerprint(), "engine_sha256": engine_fingerprint(), "candidates": records,
        "registered_before_strategy_accounts": True, "new_strategy_evaluations": 0,
        "new_economic_factor_definitions": 0, "order_authority": False, "investment_objective_verified": False,
    }
    write_new_receipt(packet_path, packet)
    return packet


def publication(reference: dict, entries: list[dict], policy_reference: dict, proof_reference: dict, commit: str) -> dict:
    inherited = publication_core.published_inputs(
        reference, entries, read_json(ROOT / "config/equity-budget-attribution.json"), commit,
    )
    own = verify_publication(ROOT, reference, {
        "src/us_quant/registered_study.py", "src/us_quant/research_program.py",
        "src/us_quant/strategy_replay.py", policy_reference["path"], proof_reference["path"],
    }, commit)
    return {**own, "published_imported_core": inherited}


def evaluate(packet_path: Path, output: Path, commit: str) -> dict:
    packet, reference = input_reference(ROOT, packet_path)
    policy_reference, proof_reference = packet.get("policy_reference"), packet.get("pre_account_proof")
    if (
        not isinstance(policy_reference, dict) or not isinstance(proof_reference, dict)
        or not {"path", "sha256"}.issubset(policy_reference)
        or not {"path", "sha256"}.issubset(proof_reference)
        or any(not isinstance(ref[name], str) for ref in (policy_reference, proof_reference)
               for name in ("path", "sha256"))
    ):
        raise QuantError("Keep explicit original policy and pre-account proof references.")
    policy, actual_policy_reference = policy_input(ROOT / policy_reference["path"])
    proof = read_json(safe_file(ROOT, proof_reference["path"], proof_reference["sha256"]))
    entries = packet.get("candidates")
    if (
        policy_reference != actual_policy_reference or policy != packet.get("policy")
        or packet.get("runner_sha256") != fingerprint() or packet.get("engine_sha256") != engine_fingerprint()
        or packet.get("registered_before_strategy_accounts") is not True
        or packet.get("new_strategy_evaluations") != 0 or packet.get("order_authority") is not False
        or proof.get("candidate_returns_or_metrics_computed") is not False
        or not isinstance(proof.get("prepared_spec_sha256"), dict)
        or not isinstance(entries, list) or len(entries) != 2
        or any(
            not isinstance(row, dict)
            or set(row) != {"spec", "candidate_spec", "readiness", "registration"}
            or any(not isinstance(row[name], dict) for name in row)
            or not isinstance(row["spec"].get("id"), str)
            or not isinstance(row["spec"].get("frozen_files"), dict)
            or any(
                not {"path", "sha256"}.issubset(row[key])
                or any(not isinstance(row[key][name], str) for name in ("path", "sha256"))
                for key in ("candidate_spec", "readiness")
            ) for row in entries
        )
        or [row["spec"].get("configuration") for row in entries] != policy["candidates"]
    ):
        raise QuantError("The exact two-candidate registration, code, proof or authority changed.")
    for entry in entries:
        spec, source = entry["spec"], entry["candidate_spec"]
        original = read_json(safe_file(ROOT, source["path"], source["sha256"]))
        if (
            original != spec or entry["registration"].get("spec_sha256") != digest_json(spec)
            or proof.get("prepared_spec_sha256", {}).get(spec["id"]) != source["sha256"]
            or spec["frozen_files"].get(policy_reference["path"]) != policy_reference["sha256"]
            or evidence_time(entry["registration"].get("registered_at")) > evidence_time(packet.get("recorded_at"))
            or evidence_time(packet.get("recorded_at")) > evidence_time(utc_now())
        ):
            raise QuantError("The packet changed original admitted inputs, proof or registration timestamps.")
        registered_generator(spec, ROOT)
        for name, digest in spec["frozen_files"].items():
            safe_file(ROOT, name, digest)
    published = publication(reference, entries, policy_reference, proof_reference, commit)
    output = local_path(ROOT, output)
    if not output.is_relative_to(ROOT / "reports") or output == ROOT / "reports":
        raise QuantError("Use a new isolated private account output directory.")
    with_program = program()
    created, reviews = False, []
    try:
        for entry in entries:
            source = entry["readiness"]
            readiness = read_json(safe_file(ROOT, source["path"], source["sha256"]))
            saved = recorded_registration(with_program, entry["spec"], readiness)
            if saved is None or saved["registration"] != entry["registration"]:
                raise QuantError("Evaluation requires each actual original candidate registration event.")
            if with_program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (entry["spec"]["id"],),
            ).fetchone() is not None or entry["spec"]["id"] in with_program.technical_failures():
                raise QuantError("Do not repeat a reviewed or terminally invalid strategy trial.")
        new_output_directory(output)
        created = True
        write_new_receipt(output / "publication.json", {
            **published, "verified_on_remote_at": utc_now(),
            "publication_fence_predates_new_strategy_accounts": True,
            "investment_objective_verified": False, "order_authority": False,
        })
        started = utc_now()
        for entry in entries:
            review = with_program.evaluate_registered(entry["spec"]["id"], output)
            if "evaluation_action" in review:
                raise QuantError("An already completed or invalid strategy cannot become another trial.")
            reviews.append(review)
            write_json(output / "progress.json", {
                "started_at": started, "completed_reviews": reviews,
                "investment_objective_verified": False, "order_authority": False,
            })
        result = {
            "schema_version": 1, "started_at": started, "completed_at": utc_now(),
            "published_registration_commit": commit, "registration_sha256": reference["sha256"],
            "candidate_reviews": reviews, "new_strategy_evaluations": len(reviews),
            "new_economic_factor_definitions": 0, "investment_objective_verified": False, "order_authority": False,
        }
        write_json(output / "results.json", result)
        return result
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        if created:
            write_json(output / "failure.json", {
                "recorded_at": utc_now(), "reason": str(exc), "completed_reviews_retained": len(reviews),
                "partial_outputs_deleted_or_filled": False, "investment_objective_verified": False, "order_authority": False,
            })
        raise
    finally:
        with_program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Register and publish-before-evaluate two existing-factor strategies.")
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        if args.action == "register":
            if args.policy is None or args.prepared is None or args.proof is None:
                raise QuantError("Registration requires exact policy, prepared inputs and pre-account proof.")
            result = register(args.policy, args.prepared, args.proof, args.registration)
            print(json.dumps({"actual_candidate_registrations": 2, "new_strategy_evaluations": 0,
                              "recorded_at": result["recorded_at"], "investment_objective_verified": False}, indent=2))
        else:
            if args.output is None:
                raise QuantError("Evaluation requires a new isolated output directory.")
            result = evaluate(args.registration, args.output, args.published_commit)
            print(json.dumps({"completed_candidate_reviews": len(result["candidate_reviews"]),
                              "investment_objective_verified": False}, indent=2))
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Registered study blocked; preserve exact evidence: {exc}\n")


if __name__ == "__main__":
    main()
