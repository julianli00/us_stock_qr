from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import equity_budget_attribution as publication_core
from us_quant.candidate_receipts import input_reference, local_path, write_new_receipt
from us_quant.cash_funded_accounting_v2 import engine_reference
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.research_weekly import verify_publication
from us_quant.review_receipts import evidence_time
from us_quant.selection_validation import fingerprint as original_fingerprint
from us_quant.selection_validation import joint_maximum, original_bundle, verified_accounts
from us_quant.selection_validation import validate_policy as validate_original_policy
from us_quant.storage import digest_json, file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/selection-accounting-cohorts.json"
COHORTS = ("legacy_original", "cash_funded_v2")
REFERENCES = ("source_snapshot", "source_bound_audit", "original_policy", "previous_policy",
              "previous_registration", "previous_results", "previous_reproducibility", "previous_preservation")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version":1, "audit_id":"current_recurring_selection_accounting_cohorts_20261011",
        "scope":"all76_completed_recurring_reviews_split_by_original_accounting", "as_of":"2026-10-05",
        "included_candidate_count":76, "total_disclosed_configurations":160,
        "earlier_configurations_not_in_joint_inference":84, "terminal_invalid_code_versions_not_in_inference":2,
        "cohort_candidate_counts":{"legacy_original":19, "cash_funded_v2":57},
        "cohort_comparison_counts_per_horizon":{"legacy_original":57, "cash_funded_v2":171},
        "horizons_years":[10, 5], "cost_comparisons":[
            {"strategy_scenario":"base", "spy_scenario":"base"},
            {"strategy_scenario":"stress", "spy_scenario":"stress"},
            {"strategy_scenario":"stress", "spy_scenario":"base"},
        ],
        "capital_usd":10000.0, "bootstrap_samples":4000, "block_sessions":21, "seed":20261011,
        "familywise_alpha":0.05, "familywise_cell_count":4, "alpha_per_cohort_horizon_cell":0.0125,
        "annualization_sessions":252,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(policy.get(key) is not False for key in (
            "order_authority", "change_research_qualification_policy", "independent_forward_validation",
            "full_search_selection_adjusted_alpha_verified"))
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)):
        raise QuantError("Freeze all original accounting cohorts,cost comparisons and inherited resampling settings.")


def accounting_cohort(review: dict) -> str:
    if "accounting_engine" not in review:
        return "legacy_original"
    if review["accounting_engine"] != engine_reference():
        raise QuantError("An unknown or changed accounting version cannot be silently assigned to an inference cohort.")
    return "cash_funded_v2"


def cohort_members(state: dict, audit: dict, previous_ids: list[str], policy: dict) -> dict[str, list[str]]:
    ids = [row["candidate_id"] for row in state["candidate_reviews"]]
    if (len(ids) != 76 or len(set(ids)) != 76 or state["total_evaluated_configurations"] != 160
        or len(state["registered_candidates"]) != 78 or state["pending_candidate_ids"]
        or len(previous_ids) != 36 or ids[:36] != previous_ids
        or audit["reviewed_candidates"] != 76 or audit["regenerated_strategy_paths"] != 304
        or audit["ledger_unchanged"] is not True or audit["event_chain_sha256"] != state["event_chain_sha256"]
        or len(audit["reviews"]) != 76 or {row["candidate_id"] for row in audit["reviews"]} != set(ids)):
        raise QuantError("Keep every current reviewed outcome and original36prefix without dropping failures or paths.")
    result = {name:[] for name in COHORTS}
    for review in state["candidate_reviews"]:
        result[accounting_cohort(review)].append(review["candidate_id"])
    if any(len(result[name]) != policy["cohort_candidate_counts"][name] for name in COHORTS):
        raise QuantError("The actual19legacy/57funded-v2 taxonomy must match its frozen accounting identities.")
    return result


def source_state(policy: dict) -> tuple[dict, dict]:
    validate_policy(policy)
    values = {name:read_json(safe_file(ROOT, policy[name]["path"], policy[name]["sha256"])) for name in REFERENCES}
    validate_original_policy(values["original_policy"])
    for name in ("as_of", "horizons_years", "cost_comparisons", "capital_usd", "bootstrap_samples",
                 "block_sessions", "seed", "familywise_alpha", "annualization_sessions"):
        if policy[name] != values["original_policy"][name]:
            raise QuantError("Current inference cannot refit any inherited statistic or resampling choice.")
    state, audit = values["source_snapshot"], values["source_bound_audit"]
    cohorts = cohort_members(state, audit, values["previous_registration"]["included_candidate_ids"], policy)
    if audit["engine_sha256"] != engine_fingerprint():
        raise QuantError("Use the actual current complete source-bound audit,not a stale-engine snapshot.")
    return state, cohorts


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({
        "cohort_adapter":file_digest(Path(__file__)), "unchanged_statistic_and_source_reader":original_fingerprint(),
        **{name:file_digest(directory / name) for name in (
            "cash_funded_accounting_v2.py", "research_weekly.py", "candidate_receipts.py", "review_receipts.py",
            "equity_budget_attribution.py")},
    })


def source_bindings(state: dict) -> dict:
    result = {}
    for review in state["candidate_reviews"]:
        bundle, reference = original_bundle(review)
        if (bundle.get("accounting_engine") != review.get("accounting_engine")
            or digest_json(bundle) != review["evidence_sha256"]):
            raise QuantError("Every cohort source must retain its original actual review accounting and evidence digest.")
        result[review["candidate_id"]] = {"bundle":reference, "review_evidence_sha256":review["evidence_sha256"]}
    return result


def register(output: Path, packet_path: Path) -> dict:
    policy, policy_reference = input_reference(ROOT, POLICY)
    state, cohorts = source_state(policy)
    output, packet_path = local_path(ROOT, output), local_path(ROOT, packet_path)
    if (output == ROOT / "data" or not output.is_relative_to(ROOT / "data")
        or not packet_path.is_relative_to(ROOT / "evidence") or packet_path.suffix != ".json"
        or packet_path.exists()):
        raise QuantError("Use new isolated source-only registration artifacts and preserve previous packets.")
    program = ResearchProgram(ROOT / "runtime/research-program.sqlite3",
                              read_json(ROOT / "config/research-program.json"), root=ROOT)
    try:
        if program.status() != state:
            raise QuantError("The actual cohort changed before registration;do not reuse old statistical scope.")
        bindings = source_bindings(state)
        new_output_directory(output)
        packet = {
            "schema_version":1, "registered_at":utc_now(), "policy":policy, "policy_reference":policy_reference,
            "auditor_sha256":fingerprint(), "program_engine_sha256":engine_fingerprint(),
            "event_chain_sha256":state["event_chain_sha256"], "cohort_candidate_ids":cohorts,
            "original_bundle_bindings":bindings, "prepared_path":output.relative_to(ROOT).as_posix(),
            "included_candidate_count":76, "included_comparisons_per_horizon":228, "familywise_cells":4,
            "settings_frozen_before_joint_resampling":True, "joint_statistical_outcomes_computed":False,
            "candidate_history_already_exposed":True, "new_strategy_evaluations":0, "order_authority":False,
        }
        write_new_receipt(output / "registration.json", packet)
        write_new_receipt(packet_path, packet)
        if program.status() != state:
            raise QuantError("Read-only inference registration must not race or alter the actual research cohort.")
        return packet
    finally:
        program.close()


def matrices(state: dict, cohorts: dict, policy: dict, bindings: dict) -> dict:
    result = {name:{} for name in COHORTS}
    for years in policy["horizons_years"]:
        columns = {name:{} for name in COHORTS}
        for review in state["candidate_reviews"]:
            name = review["candidate_id"]
            reference = bindings[name]["bundle"]
            bundle = read_json(safe_file(ROOT, reference["path"], reference["sha256"]))
            if (digest_json(bundle) != review["evidence_sha256"]
                or review["evidence_sha256"] != bindings[name]["review_evidence_sha256"]
                or bundle.get("accounting_engine") != review.get("accounting_engine")):
                raise QuantError("Do not replace an original accounting-specific reviewed bundle.")
            accounts = verified_accounts(bundle, review, years, policy)
            for comparison in policy["cost_comparisons"]:
                own, spy = comparison["strategy_scenario"], comparison["spy_scenario"]
                strategy, benchmark = accounts[own]["strategy"], accounts[spy]["spy"]
                if (strategy["return"] <= -1).any() or (benchmark["return"] <= -1).any():
                    raise QuantError("Actual insolvency cannot be hidden inside log-growth inference.")
                columns[accounting_cohort(review)][f"{name}::{own}_vs_{spy}"] = (
                    np.log1p(strategy["return"])-np.log1p(benchmark["return"])
                )
        for cohort in COHORTS:
            frame = pd.DataFrame(columns[cohort])
            expected = [f"{name}::{row['strategy_scenario']}_vs_{row['spy_scenario']}"
                        for name in cohorts[cohort] for row in policy["cost_comparisons"]]
            if (list(frame.columns) != expected or frame.shape != (
                    2512 if years == 10 else 1254, policy["cohort_comparison_counts_per_horizon"][cohort])
                or not np.isfinite(frame.to_numpy()).all()):
                raise QuantError("Every fixed candidate/scenario column and genuine date must enter its accounting cohort.")
            result[cohort][str(years)] = frame
    return result


def analyze(inputs: dict, policy: dict) -> dict:
    if (set(inputs) != set(COHORTS) or any(set(group) != {"10", "5"} for group in inputs.values())
        or any(not inputs[COHORTS[0]][str(years)].index.equals(inputs[COHORTS[1]][str(years)].index)
               for years in policy["horizons_years"])):
        raise QuantError("Both accounting cohorts must share exact dated inputs within each horizon.")
    analyses = {}
    for cohort in COHORTS:
        analyses[cohort] = {}
        for horizon_index, years in enumerate(policy["horizons_years"]):
            summary = joint_maximum(
                inputs[cohort][str(years)], samples=policy["bootstrap_samples"], block=policy["block_sessions"],
                seed=policy["seed"]+horizon_index, alpha=policy["alpha_per_cohort_horizon_cell"],
            )
            for row in summary["comparison_results"]:
                candidate, comparison = row["comparison_id"].split("::", 1)
                own, spy = comparison.split("_vs_", 1)
                row.update({"candidate_id":candidate, "strategy_scenario":own, "spy_scenario":spy})
                row["annualized_252_log_growth_advantage"] = row["observed_mean_daily_log_advantage"]*252
                row["simultaneous_lower_annualized_252_log_growth_advantage"] = (
                    row["simultaneous_lower_mean_daily_log_advantage"]*252
                )
            analyses[cohort][str(years)] = summary
    return analyses


def published_inputs(reference: dict, policy: dict, commit: str) -> dict:
    core = publication_core.published_inputs(
        reference, [], read_json(ROOT / "config/equity-budget-attribution.json"), commit,
    )
    required = {Path(__file__).relative_to(ROOT).as_posix(), POLICY.relative_to(ROOT).as_posix(),
                "src/us_quant/selection_validation.py", "src/us_quant/factor_validation.py",
                "src/us_quant/research_weekly.py", "src/us_quant/candidate_receipts.py",
                "src/us_quant/review_receipts.py", "src/us_quant/cash_funded_accounting_v2.py",
                "src/us_quant/equity_budget_attribution.py", "config/equity-budget-attribution.json",
                *(policy[name]["path"] for name in REFERENCES)}
    explicit = verify_publication(ROOT, reference, required, commit)
    return {**explicit, "published_frozen_imported_core":core}


def run(packet_path: Path, output: Path, commit: str) -> dict:
    packet, reference = input_reference(ROOT, packet_path)
    policy = read_json(POLICY)
    state, cohorts = source_state(policy)
    if (packet.get("policy") != policy or packet.get("auditor_sha256") != fingerprint()
        or packet.get("program_engine_sha256") != engine_fingerprint()
        or packet.get("event_chain_sha256") != state["event_chain_sha256"]
        or packet.get("cohort_candidate_ids") != cohorts
        or packet.get("original_bundle_bindings") != source_bindings(state)
        or packet.get("policy_reference") != {"path":POLICY.relative_to(ROOT).as_posix(), "sha256":file_digest(POLICY),
                                              "body_sha256":digest_json(policy)}
        or packet.get("settings_frozen_before_joint_resampling") is not True
        or packet.get("joint_statistical_outcomes_computed") is not False
        or evidence_time(packet.get("registered_at")) > evidence_time(utc_now())):
        raise QuantError("Use the exact actually registered cohort/settings/code/source bindings before inference.")
    publication = published_inputs(reference, policy, commit)
    output = local_path(ROOT, output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports"):
        raise QuantError("Use a new isolated private source-only inference directory.")
    program = ResearchProgram(ROOT / "runtime/research-program.sqlite3",
                              read_json(ROOT / "config/research-program.json"), root=ROOT)
    created = False
    try:
        if program.status() != state:
            raise QuantError("The actual cohort changed after publication;do not reinterpret its frozen scope.")
        new_output_directory(output)
        created = True
        write_new_receipt(output / "publication.json", {
            **publication, "verified_on_remote_at":utc_now(),
            "publication_fence_predates_joint_resampling":True, "investment_objective_verified":False, "order_authority":False,
        })
        started = utc_now()
        inputs = matrices(state, cohorts, policy, packet["original_bundle_bindings"])
        input_references = {}
        for cohort in COHORTS:
            input_references[cohort] = {}
            for years in policy["horizons_years"]:
                path = output / f"{cohort}-{years}y-inputs.csv"
                write_text_atomic(path, inputs[cohort][str(years)].to_csv(float_format="%.17g"))
                input_references[cohort][str(years)] = {"path":path.relative_to(ROOT).as_posix(), "sha256":file_digest(path)}
        analyses = analyze(inputs, policy)
        if program.status() != state:
            raise QuantError("The read-only diagnostic cannot alter or race changed research state.")
        cells = [value for group in analyses.values() for value in group.values()]
        result = {
            "schema_version":1, "started_at":started, "completed_at":utc_now(), "audit_id":policy["audit_id"],
            "registration":reference, "auditor_sha256":fingerprint(), "published_registration_commit":commit,
            "event_chain_sha256":state["event_chain_sha256"], "program_engine_sha256":engine_fingerprint(),
            "analyses":analyses, "input_matrices":input_references, "cohort_candidate_ids":cohorts,
            "source_bundles":packet["original_bundle_bindings"], "included_candidate_count":76,
            "included_comparisons_per_horizon":228, "familywise_cells":4,
            "bonferroni_omnibus_p_value_for_scoped_family":min(1.0, 4*min(
                row["scope_limited_omnibus_p_value"] for row in cells)),
            "overlapping_horizons_or_accounting_cohorts_assumed_independent":False,
            "earlier_configurations_excluded_from_joint_inference":84,
            "terminal_invalid_code_versions_excluded":2, "total_disclosed_configurations":160,
            "old_selection_artifacts_or_qualification_policy_changed":False, "ledger_unchanged":True,
            "legacy_original_cell_claimed_current_funded_performance":False,
            "full_search_selection_adjusted_alpha_verified":False, "independent_forward_validation":False,
            "investment_objective_verified":False, "new_strategy_evaluations":0,
            "new_accounts_controls_or_factor_definitions":0, "order_authority":False,
        }
        write_json(output / "results.json", result)
        return result
    except (QuantError, OSError, ValueError) as exc:
        if created:
            write_json(output / "failure.json", {"at":utc_now(), "reason":str(exc),
                       "partial_results_preserved":True, "cohort_or_statistical_settings_refitted":False,
                       "investment_objective_verified":False, "order_authority":False})
        raise
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only fixed current accounting-cohort selection inference.")
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        result = register(args.output, args.registration) if args.action == "register" else run(
            args.registration, args.output, args.published_commit,
        )
        print(json.dumps({"included_original_reviews":76, "new_strategy_evaluations":0,
                          "event_chain_sha256":result["event_chain_sha256"], "investment_objective_verified":False}, indent=2))
        if args.action == "run":
            print(json.dumps({cohort:{years:{"comparisons":row["comparisons"],
                "scoped_p_value":row["scope_limited_omnibus_p_value"],
                "positive_simultaneous_lower_bounds":sum(value["strictly_positive_simultaneous_lower_bound"]
                                                          for value in row["comparison_results"])}
                for years, row in group.items()} for cohort, group in result["analyses"].items()}, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Accounting-cohort selection analysis blocked: {exc}\n")


if __name__ == "__main__":
    main()
