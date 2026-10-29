from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from us_quant import ssd_expert_meta_research_v1 as source
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import digest_json, read_json, utc_now

base, common, current = source.base, source.common, source.current
ROOT = base.ROOT
POLICY = ROOT / "config/strategy-expert-meta-execution-v2.json"
FOLDER = base.REPORTS / "strategy-expert-meta-execution-20261029-v2"
PACKET = ROOT / "evidence/strategy_expert_meta_execution_20261029_registration_v2.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/strategy_expert_meta_execution_20261029_results_v2.json"
PRIOR_HEAD = "a174773bc066618d507e206613375c57027d4bc64191ea171f540abf192f3725"
CANDIDATES = (
    {
        "id": "expert_pool_equal_weight_execution_v2",
        "source_candidate_id": "expert_pool_equal_weight_v1",
    },
    {
        "id": "expert_pool_rolling_pair_execution_v2",
        "source_candidate_id": "expert_pool_rolling_pair_v1",
    },
)
IDS = tuple(c["id"] for c in CANDIDATES)
EXTENSIONS = (
    "config/strategy-expert-meta-execution-v2.json",
    "src/us_quant/ssd_expert_meta_execution_v2.py",
    "tests/test_ssd_expert_meta_execution_v2.py",
)
check_frozen = source.check_frozen


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_strategy_expert_meta_execution_20261029_v2",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 351,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 207,
        "source_failure_stage": "expert_meta_publication_interface",
        "source_first_target_registration_preserved": True,
        "reuse_exact_source_targets_and_audits": True,
        "financial_rule_parameter_or_gate_changes": False,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "new_economic_hypotheses": 0,
        "technical_execution_releases": 2,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()):
        raise QuantError("This release fixes only publication integration, not the financial rule.")
    if any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A technical execution release is not qualification or trading authority.")


def inputs():
    policy = read_json(POLICY)
    validate(policy)
    events = base.journal_events(base.JOURNAL)
    if len(events) < 351 or events[350]["sha256"] != PRIOR_HEAD:
        raise QuantError("Preserve both actual v1 publication Interface failures.")
    data, _, parent, original, benchmarks = source.inputs()
    packet = source.checked_packet(parent)
    if (
        policy["source_registration"] != base.reference(source.PACKET)
        or policy["source_financial_policy"] != parent["policy"]
        or policy["source_terminal_evidence"] != base.reference(source.PUBLIC_RESULTS)
    ):
        raise QuantError("The original pool, financial policy or failed release cannot change.")
    failure = read_json(base.checked_reference(policy["source_failure"]))
    if (
        failure["funded_meta_paths_started"] != 0
        or failure["source_targets_and_scores_valid_and_unchanged"] is not True
    ):
        raise QuantError(
            "Only this pre-account publication Interface failure can be corrected here."
        )
    bindings = {}
    for candidate in CANDIDATES:
        identifier = candidate["source_candidate_id"]
        failed = base.matching_events("technical_failure", identifier)
        if (
            len(failed) != 1
            or failed[0]["body"]["stage"] != policy["source_failure_stage"]
            or base.matching_events("path_started", identifier)
            or base.matching_events("review", identifier)
        ):
            raise QuantError(
                "Do not reuse a signal/data/funding failure or replay a funded source."
            )
        entry = next(r for r in packet["entries"] if r["candidate_id"] == identifier)
        proof = read_json(base.checked_reference(entry["target_proof"]["proof"]))
        base.checked_reference(proof["information_audits"])
        bindings[candidate["id"]] = {
            "source_candidate_id": identifier,
            "source_registered_at": entry["registered_at"],
            "source_registered_event_sha256": entry["registered_event_sha256"],
            "source_target_proof_at": entry["target_proof_at"],
            "targets": entry["target_proof"]["targets"],
            "proof": entry["target_proof"]["proof"],
            "information_audits": proof["information_audits"],
        }
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(POLICY),
        "code": source.unique_refs(
            [*parent["code"], *[base.reference(ROOT / p) for p in EXTENSIONS[1:]]]
        ),
        "policies": source.unique_refs([*parent["policies"], base.reference(POLICY)]),
        "source_financial_policy": policy["source_financial_policy"],
        "source_meta_registration": policy["source_registration"],
        "source_target_bindings": bindings,
        "prior_completed_configurations": 207,
        "prior_journal_sequence": 351,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "construction": "same_expert_meta_targets_versioned_publication_Interface_fix",
        "prior_evidence": source.unique_refs(
            [
                *parent["prior_evidence"],
                policy["source_registration"],
                policy["source_terminal_evidence"],
                policy["source_failure"],
                *[
                    ref
                    for b in bindings.values()
                    for ref in (b["targets"], b["proof"], b["information_audits"])
                ],
                base.reference(FOLDER / "technical-release-intent.json"),
            ]
        ),
        "validation_evidence": [
            *parent["validation_evidence"],
            *[
                base.reference(FOLDER / p)
                for p in ("offline-tests-checked-v2.xml", "offline-tests-checked-v2.log")
            ],
        ],
        "new_economic_hypotheses": 0,
        "new_meta_scores_or_targets_computed": False,
    }
    check_frozen(frozen)
    return data, frozen, original, benchmarks


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or packet["new_economic_hypotheses"] != 0
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"])
            for name in IDS
            for e in base.matching_events("path_started", name)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "order_authority",
                "independent_forward_validation",
            )
        )
    ):
        raise QuantError("Keep the actual pre-account technical-release registration.")
    return packet


def failure(stage, identifiers, error):
    source.failure(stage, identifiers, error)


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, frozen, _, _ = inputs()
        prior = common.counts(
            [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] not in IDS]
        )
        if (
            prior["new_registered_candidates"] != 32
            or prior["new_completed_strategy_evaluations"] != 29
            or prior["new_technical_failures"] != 3
            or prior["new_pending_candidates"] != 0
        ):
            raise QuantError("Do not erase the two pre-account v1 failures or the earlier failure.")
        for candidate in CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            found = base.matching_events("registered", candidate["id"])
            if found and (len(found) != 1 or found[0]["body"] != specification):
                raise QuantError("A registered execution correction cannot change.")
            if not found:
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal execution release cannot be silently retried.")
        for name in IDS:
            if base.matching_events("target_proof", name):
                continue
            runtime_guard()
            binding = frozen["source_target_bindings"][name]
            old_proof = read_json(base.checked_reference(binding["proof"]))
            base.checked_reference(binding["targets"])
            base.checked_reference(binding["information_audits"])
            proof_path = FOLDER / name / "proof.json"
            proof = {
                **old_proof,
                "checked_at": utc_now(),
                "source_preparation": binding,
                "verification_values_scope": (
                    "unchanged_original_preparation_certificate_hashes_rechecked_not_recomputed"
                ),
                "source_targets_reused_byte_for_byte": True,
                "new_meta_scoring_or_target_generation": False,
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
            }
            common.retain_receipt(proof_path, proof, ("checked_at",))
            base.append_event(
                base.JOURNAL,
                "target_proof",
                name,
                {
                    "registered_event_sha256": base.matching_events("registered", name)[0][
                        "sha256"
                    ],
                    "targets": binding["targets"],
                    "proof": base.reference(proof_path),
                },
            )
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [common.packet_entry(c, frozen) for c in CANDIDATES],
                "new_registered_candidates": 2,
                "new_economic_hypotheses": 0,
                "candidate_accounts_or_metrics_computed": False,
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        current.progress(
            "expert_meta_execution_v2_registered_waiting_publication",
            active_compute_job=None,
            expert_meta_execution_registration=base.reference(PACKET),
            pending_steps=["publish_execution_v2_registration", "eight_meta_funded_paths"],
        )
        return base.reference(PACKET)


def publication(frozen, commit):
    # This verifies the complete closure, including unchanged manifest-pinned imports.
    dependencies = base.published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, base.reference(PACKET), explicit, commit)
    found = [
        e
        for e in base.matching_events("publication")
        if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with this release's actual original publication commit.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews = base.matching_events("review", name)
        failures = base.matching_events("technical_failure", name)
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in base.matching_events("path_completed", name)
        }
        if reviews:
            if len(reviews) != 1 or failures or reviews[0]["body"]["paths"] != paths:
                raise QuantError("Completed execution-version evidence disagrees.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failures else "pending",
                "paths": paths,
                "failure_event_sha256": failures[0]["sha256"] if failures else None,
                "failure_stage": failures[0]["body"]["stage"] if failures else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, _, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Preserve completed accounts without replay.")
            return base.reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_meta_execution_v2",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            protocol = read_json(base.PROTOCOL)
            for name in IDS:
                active = (name,)
                if base.matching_events("review", name) or base.matching_events(
                    "technical_failure", name
                ):
                    continue
                binding = frozen["source_target_bindings"][name]
                targets = source.execution.frame(binding["targets"])
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_meta_execution_v2",
                            active_candidate=name,
                            active_path=key,
                        )
                        paths[key] = source.execution.evaluate_path(
                            data, targets, frozen, name, window, scenario, benchmarks
                        )
                base.append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed"
                        if all(p["all_point_gates_pass"] for p in paths.values())
                        else "rejected",
                        "paths": paths,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            common.retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "new_shadow_account_runs": 0,
                    "new_economic_hypotheses": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "expert_meta_execution_v2_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_meta_execution_results=base.reference(RESULTS),
                pending_steps=["publish_all_expert_meta_execution_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_meta_execution_v2", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_meta_execution_v2_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result():
    base.storage_guard()
    _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Only terminal actual execution releases can be reported.")
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"]["cagr"]
        for r in benchmarks["benchmarks"]
    }
    reviews = []
    for name, row in actual.items():
        paths = []
        for key, path in row["paths"].items():
            year, scenario = key.split("-")
            years = int(year[:-1])
            paths.append(
                {
                    **{
                        k: path[k]
                        for k in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        symbol: lookup[years, scenario, symbol] for symbol in ("SPY", "QQQ")
                    },
                }
            )
        reviews.append(
            {
                "candidate_id": name,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": base.reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": base.reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "new_economic_hypotheses": 0,
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return base.reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Versioned publication Adapter for unchanged expert meta rules."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("No live-source request in the technical execution release.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual v2 publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Expert meta execution release blocked: {error}\n")


if __name__ == "__main__":
    main()
