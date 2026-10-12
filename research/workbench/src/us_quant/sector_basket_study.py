from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

from us_quant import equity_budget_attribution as publication_core
from us_quant import sector_basket_strategy as strategy
from us_quant.candidate_receipts import input_reference, local_path, recorded_registration, register_or_recover, write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.research_weekly import verify_publication
from us_quant.storage import digest_json, file_digest, new_output_directory, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "data/sector-basket-strategy-20261011-v3"
TARGET_PROOF = ROOT / "evidence/sector_basket_strategy_20261011_target_verification_v2.json"
PACKET = ROOT / "evidence/sector_basket_strategy_20261011_registration.json"


def program() -> ResearchProgram:
    return ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json"), root=ROOT,
    )


def fingerprint() -> str:
    return digest_json({
        name: file_digest(Path(__file__).parent / name)
        for name in (
            "sector_basket_study.py", "sector_basket_strategy.py", "sector_basket_sources.py",
            "research_program.py", "strategy_replay.py", "cash_funded_accounting_v2.py",
            "bt_audit.py", "candidate_receipts.py", "research_weekly.py", "equity_budget_attribution.py",
        )
    })


def register(packet_path: Path = PACKET) -> dict:
    packet_path = local_path(ROOT, packet_path)
    if packet_path.exists() or packet_path.suffix != ".json" or not packet_path.is_relative_to(ROOT / "evidence"):
        raise QuantError("Preserve the original complete sector-basket registration packet.")
    policy = read_json(ROOT / "config/sector-basket-strategy.json")
    strategy.validate(policy)
    proof, proof_reference = input_reference(ROOT, TARGET_PROOF)
    if proof["checked_defined_targets"] != 244 or proof["candidate_returns_or_metrics_computed"] is not False:
        raise QuantError("Independent complete target/source proof must precede strategy registration.")
    records = []
    for candidate in strategy.CANDIDATES:
        spec_path = PREPARED / f"{candidate['id']}-spec.json"
        if file_digest(spec_path) != proof["prepared_spec_sha256"][candidate["id"]]:
            raise QuantError("The independently verified prepared sector specification changed.")
        receipt_path = ROOT / "reports/sector-basket-sources-20261011" / f"{candidate['id']}-registration.json"
        result = register_or_recover(spec_path, PREPARED / "readiness.json", receipt_path, root=ROOT)
        receipt = read_json(safe_file(ROOT, result["receipt"], result["receipt_sha256"]))
        spec = read_json(spec_path)
        if spec["configuration"] != candidate:
            raise QuantError("Only both fixed basket constructions may enter the study.")
        records.append({
            "spec": spec, "registration": receipt["registration"],
            "readiness": receipt["readiness"], "candidate_spec": receipt["candidate_spec"],
        })
    packet = {
        "schema_version": 1, "recorded_at": utc_now(), "study": policy,
        "runner_sha256": fingerprint(), "engine_sha256": engine_fingerprint(),
        "target_verification": proof_reference, "candidates": records,
        "registered_before_strategy_returns": True, "new_strategy_evaluations": 0,
        "new_economic_factor_definitions": 0, "existing_W42_queue_or_forward_rules_changed": False,
        "investment_objective_verified": False, "order_authority": False,
    }
    write_new_receipt(packet_path, packet)
    return packet


def publication(reference: dict, entries: list[dict], commit: str) -> dict:
    inherited = publication_core.published_inputs(
        reference, entries, read_json(ROOT / "config/equity-budget-attribution.json"), commit,
    )
    own = verify_publication(ROOT, reference, {
        "src/us_quant/sector_basket_study.py", "src/us_quant/sector_basket_strategy.py",
        "src/us_quant/sector_basket_sources.py", "src/us_quant/research_program.py",
        "src/us_quant/strategy_replay.py", "config/sector-basket-strategy.json",
        "config/sector-basket-sources.json", TARGET_PROOF.relative_to(ROOT).as_posix(),
        "evidence/sector_basket_source_readiness_20261011.json",
    }, commit)
    return {**own, "published_imported_core": inherited}


def evaluate(packet_path: Path, output: Path, commit: str) -> dict:
    packet, reference = input_reference(ROOT, packet_path)
    policy = read_json(ROOT / "config/sector-basket-strategy.json")
    strategy.validate(policy)
    entries = packet.get("candidates")
    if (
        not isinstance(entries, list) or len(entries) != 2
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"spec", "registration", "readiness", "candidate_spec"}
            or any(not isinstance(entry[name], dict) for name in entry) for entry in entries
        )
        or packet.get("study") != policy or packet.get("runner_sha256") != fingerprint()
        or packet.get("engine_sha256") != engine_fingerprint()
        or packet.get("registered_before_strategy_returns") is not True
        or packet.get("new_strategy_evaluations") != 0 or packet.get("order_authority") is not False
        or [entry["spec"].get("configuration") for entry in entries] != list(strategy.CANDIDATES)
    ):
        raise QuantError("Preserve the exact original two-candidate sector-basket study.")
    safe_file(ROOT, packet["target_verification"]["path"], packet["target_verification"]["sha256"])
    for entry in entries:
        spec, ref = entry["spec"], entry["candidate_spec"]
        actual = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
        if actual != spec or entry["registration"]["spec_sha256"] != digest_json(spec):
            raise QuantError("The published packet differs from its originally admitted specification.")
        for name, digest in spec["frozen_files"].items():
            safe_file(ROOT, name, digest)
    published = publication(reference, entries, commit)
    output = local_path(ROOT, output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports"):
        raise QuantError("Use a new isolated private sector study output directory.")
    with_program = program()
    created, results = False, []
    try:
        for entry in entries:
            ref = entry["readiness"]
            readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
            saved = recorded_registration(with_program, entry["spec"], readiness)
            if saved is None or saved["registration"] != entry["registration"]:
                raise QuantError("Only actual original candidate events can support the published study.")
            if with_program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (entry["spec"]["id"],),
            ).fetchone() is not None or entry["spec"]["id"] in with_program.technical_failures():
                raise QuantError("Never duplicate a completed or invalid strategy trial.")
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
                raise QuantError("An existing reviewed or terminal trial cannot become a new result.")
            results.append(review)
            write_json(output / "progress.json", {"started_at": started, "completed_reviews": results})
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
                "completed_reviews_retained": len(results), "partial_outputs_deleted_or_filled": False,
                "investment_objective_verified": False, "order_authority": False,
            })
        raise
    finally:
        with_program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish-before-performance fixed sector basket; never orders.")
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--registration", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sector-basket-strategy-20261011")
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        result = register(args.registration) if args.action == "register" else evaluate(
            args.registration, args.output, args.published_commit,
        )
        print(json.dumps({
            "registered_candidates" if args.action == "register" else "completed_reviews": 2,
            "new_strategy_evaluations": result["new_strategy_evaluations"],
            "investment_objective_verified": False,
        }, indent=2))
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Sector-basket study blocked; preserve exact evidence: {exc}\n")


if __name__ == "__main__":
    main()
