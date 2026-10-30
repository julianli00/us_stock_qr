from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from xml.etree import ElementTree

import pandas as pd

from us_quant import expert_five_net_wealth_mixture_v1 as rules
from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_five_expert_meta_v1 as previous
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.ssd_research_v1 import append_event, matching_events, published_inputs, reference
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = previous.base, previous.common, previous.current, previous.ROOT
FOLDER = base.REPORTS / "expert-five-net-mixture-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-five-net-mixture-20261030-v1"
PACKET = ROOT / "evidence/expert_five_net_mixture_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_five_net_mixture_20261030_results_v1.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/expert-five-net-wealth-mixture-v1.json",
    "src/us_quant/expert_five_net_wealth_mixture_v1.py",
    "src/us_quant/ssd_expert_five_net_mixture_v1.py",
    "tests/test_expert_five_net_wealth_mixture_v1.py",
    "tests/test_ssd_expert_five_net_mixture_v1.py",
)
PARENT_SPEC = "557aa3f10626b80768cd99ff799d5d682badc97cdc2070ccaba8060a73cc75d1"
ORIGINAL_KERNEL = ROOT / "src/us_quant/expert_net_wealth_mixture_v1.py"


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or policy["source_specification_sha256"] != PARENT_SPEC
        or parent["policy"] != policy["source_five_policy"]
    ):
        raise QuantError("Use the unchanged fixed five-policy source registration.")
    previous.previous.check_frozen(parent)
    previous.checked_packet(parent)
    data, experts, source_policy, _, source_refs, clocks, _, benchmarks = previous.source_inputs()
    if (
        source_policy["expert_ids"] != policy["expert_ids"]
        or source_policy["mechanism_groups"] != policy["mechanism_groups"]
        or clocks != parent["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("The fixed five cached experts or their actual source clocks changed.")
    old_packet = read_json(base.checked_reference(policy["prior_four_net_registration"]))
    old = old_packet["entries"][0]["specification"]["frozen"]
    old_policy = read_json(base.checked_reference(policy["source_four_net_policy"]))
    rules.original.validate(old_policy)
    if (
        old["policy"] != policy["source_four_net_policy"]
        or old["market"] != parent["market"]
        or old["protocol"] != parent["protocol"]
        or old["execution_policy"] != parent["execution_policy"]
        or any(
            policy[k] != old_policy[k]
            for k in (
                "initial_decision_session",
                "shadow_reference_start",
                "observation_horizon_sessions",
                "maximum_complete_learning_intervals",
                "learning_clock",
                "training_memory",
                "validation_clock",
                "learning_state_resets_for_five_year_account",
                "temperature_or_forgetting_parameter",
                "switch_cost_proxy_affects_allocation",
                "reference_capital_usd",
                "switch_cost_bps_per_side",
                "switch_commission_per_changed_asset_usd",
                "invested_target_budget",
                "cash_reserve",
                "actual_weight_trade_band",
            )
        )
    ):
        raise QuantError("Pool inclusion must not retune the existing complete-NET mixture law.")
    old_results = read_json(base.checked_reference(policy["prior_four_net_results"]))
    five_results = read_json(base.checked_reference(policy["previous_five_results"]))
    if (
        old_results["registration_packet"] != policy["prior_four_net_registration"]
        or five_results["registration_packet"] != policy["source_registration"]
        or old_results["new_completed_strategy_evaluations"] != 1
        or five_results["new_completed_strategy_evaluations"] != 2
        or old_results["order_authority"] is not False
        or five_results["order_authority"] is not False
    ):
        raise QuantError(
            "Preserve the actual complete controls rather than rerunning or replacing them."
        )
    diagnosis = read_json(base.checked_reference(policy["saved_five_diagnosis"]))
    if diagnosis["new_strategy_evaluations"] != 0 or diagnosis["new_model_parameters_selected"]:
        raise QuantError("A saved diagnosis cannot become an unregistered new financial rule.")
    refs = previous.previous.source.unique_refs(
        [
            *source_refs,
            *[
                policy[k]
                for k in (
                    "source_registration",
                    "source_five_policy",
                    "previous_five_results",
                    "prior_four_net_registration",
                    "prior_four_net_results",
                    "source_four_net_policy",
                    "saved_five_diagnosis",
                )
            ],
            old_policy["previous_master_results"],
            reference(ORIGINAL_KERNEL),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, policy, parent, refs, benchmarks


def validation_references():
    xmls = [
        FOLDER / "offline-study-tests-v1.xml",
        FOLDER / "kernel-full-bound-tests-attempt-01.xml",
    ]
    suites = [
        suite
        for path in xmls
        for suite in ElementTree.parse(path).getroot().findall(".//testsuite")
    ]
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_packed_compositions_and_ranks_cover_exactly_the_valid_simplex",
        "test_one_period_matches_analytic_five_policy_dirichlet_integral",
        "test_packed_recursion_matches_independent_streamed_integration_at_every_prefix",
        "test_identical_streams_keep_uniform_prior_at_full_sparse_interval_and_memory_bound",
        "test_true_initial_anchor_and_prequential_validation_precede_assimilation",
        "test_every_allocation_matches_independent_integration_and_capital_interface",
        "test_prefix_and_future_perturbations_leave_past_targets_identical",
        "test_cached_five_source_loader_does_not_run_completed_financial_entrypoints",
        "test_publication_checks_full_dependencies_before_packet_and_fence",
        "test_failed_publication_cannot_start_a_funded_path",
        "test_canonical_export_preserves_original_public_flags_and_every_scalar",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
    }
    runtime = read_json(FOLDER / "kernel-full-bound-runtime-v1.json")
    if (
        not required.issubset(names)
        or sum(int(s.get("tests", 0)) for s in suites) < len(required)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        or runtime["pytest_exit_code"] != 0
        or runtime["OS_ru_maxrss_unit"] != "macOS_bytes"
        or type(runtime["observed_process_maximum_resident_set_bytes"]) is not int
        or runtime["observed_process_maximum_resident_set_bytes"] <= 0
        or runtime["single_mass_bytes_asserted_by_test"] != rules.MAX_MASS_BYTES
        or runtime["old_plus_new_coefficient_bytes_asserted_by_test"] != rules.MAX_COEFFICIENT_BYTES
    ):
        raise QuantError(
            "Require completed causal, sparse full-bound and publication contract tests."
        )
    return [
        reference(path)
        for path in (
            *xmls,
            FOLDER / "offline-study-tests-v1.log",
            FOLDER / "kernel-full-bound-tests-attempt-01.log",
            FOLDER / "kernel-full-bound-runtime-v1.json",
        )
    ]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Read the existing complete/partial implementation freeze, not repeat it.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, policy, parent, refs, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        if (
            len(events) != 455
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 41
            or counts["new_completed_strategy_evaluations"] != 38
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
        ):
            raise QuantError("Freeze against the actual unchanged post-diagnosis journal.")
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            design["actual_continuation_event"] != 40745
            or design["actual_invocation"] != reference(INVOCATION)
            or design["expert_ids"] != list(rules.EXPERT_IDS)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("The real design and scheduled invocation origin changed.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 40745,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "question": policy["methodology"]["question"],
                "real_new_learner_scores_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_accounts_or_admissions_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "five_net_mixture_rule_frozen",
            active_compute_job=None,
            five_net_mixture_intent=reference(INTENT),
            pending_steps=["register_five_net_mixture_before_real_learning_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_learner_scores_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome implementation freeze changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": previous.previous.source.unique_refs(
            [
                *parent["code"],
                reference(ORIGINAL_KERNEL),
                *[reference(ROOT / p) for p in OWN[1:]],
            ]
        ),
        "policies": previous.previous.source.unique_refs(
            [
                *parent["policies"],
                policy["source_four_net_policy"],
                reference(rules.POLICY),
            ]
        ),
        "prior_completed_configurations": 216,
        "prior_journal_sequence": 455,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "source_expert_pool_policy": parent["policy"],
        "expert_pool_definition_role": "unchanged_five_complete_policies_not_old_pair_ranking",
        "construction": "packed_exact_uniform_five_complete_NET_policy_score_integration",
        "construction_inputs": [reference(INTENT), *refs],
        "execution_adapter": reference(
            ROOT / "src/us_quant/expert_initial_monthly_execution_v1.py"
        ),
        "execution_adapter_role": "unchanged_initial_plus_monthly_actual_holdings_025_band",
        "prior_evidence": previous.previous.source.unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(DESIGN),
                reference(INVOCATION),
                reference(INTENT),
            ]
        ),
        "validation_evidence": previous.previous.source.unique_refs(
            [
                *parent["validation_evidence"],
                *validation_references(),
            ]
        ),
        "actual_factor_participation_is_measured_in_target_proof": False,
        "actual_factor_participation_is_measured_in_funded_account": True,
        "shadow_net_scores_are_actual_main_wealth": False,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
    }
    previous.previous.check_frozen(frozen)
    return data, experts, frozen, benchmarks


def failure(stage, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in IDS:
            if (
                matching_events("registered", name)
                and not matching_events("review", name)
                and not matching_events("technical_failure", name)
            ):
                append_event(
                    base.JOURNAL,
                    "technical_failure",
                    name,
                    {
                        "stage": stage,
                        "reason": str(error),
                        "partial_outputs_retained": True,
                        "candidate_accounts_started": bool(matching_events("path_started", name)),
                    },
                )
    current.progress(
        "five_net_mixture_expired" if expired else "five_net_mixture_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_failure=str(error),
    )


def prepare(data, experts, frozen):
    name = IDS[0]
    directory = TARGETS / name
    proof_path = directory / "proof.json"
    if proof_path.exists():
        proof = read_json(proof_path)
        if proof["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation, not regenerate targets.")
        for ref in (proof["targets"], proof["information_audits"]):
            base.checked_reference(ref)
        return {"targets": proof["targets"], "proof": reference(proof_path)}
    new_output_directory(directory)
    policy = read_json(base.checked_reference(frozen["policy"]))
    target, audits, periods = rules.build_targets(
        experts, data.close.index, tuple(data.close.columns), policy, before_decision=runtime_guard
    )
    verification = rules.verify_targets(
        experts, data.close.index, tuple(data.close.columns), policy, target, audits, periods
    )
    prefixes = []
    for position in (0, 1, 2, 40, 80):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        earlier, aa, pp = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            policy,
            before_decision=runtime_guard,
        )
        if (
            not earlier.equals(target.loc[:day])
            or aa != audits[: position + 1]
            or pp != [r for r in periods if r["end"] <= str(day.date())]
        ):
            raise QuantError("A future observation changed the actual five-policy prefix.")
        prefixes.append(str(day.date()))
    if len(audits) != 121 or len(periods) != 119 or sys.platform != "darwin":
        raise QuantError("Keep the full original decision lattice and recorded memory units.")
    write_text_atomic(directory / "targets.csv", target.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "information-audits.json",
        {
            "expert_vector_order": list(rules.EXPERT_IDS),
            "decisions": audits,
            "complete_periods": periods,
        },
    )
    write_new_receipt(
        proof_path,
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "targets": reference(directory / "targets.csv"),
            "information_audits": reference(directory / "information-audits.json"),
            **verification,
            "prefix_truncations": prefixes,
            "maximum_mass_array_bytes": max(a["learner_mass_array_bytes"] for a in audits),
            "maximum_coefficient_buffer_bytes": max(
                a["learner_maximum_coefficient_buffer_bytes"] for a in audits
            ),
            "observed_process_maximum_resident_set_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "OS_ru_maxrss_unit": "macOS_bytes",
            "coefficient_bytes_are_total_process_RAM": False,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "gross_or_shadow_score_is_actual_net_main_performance": False,
        },
    )
    return {"targets": reference(directory / "targets.csv"), "proof": reference(proof_path)}


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 1
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"]) for n in IDS for e in matching_events("path_started", n)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The actual pre-account registration identity or clock changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        old_events = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] not in IDS]
        if len(old_events) != 455 or old_events[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve the actual pre-study financial history.")
        if any(matching_events("technical_failure", n) for n in IDS):
            raise QuantError("A failed registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("Registered five-policy mixture parameters cannot change.")
            if not found:
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        try:
            if not matching_events("target_proof", IDS[0]):
                current.progress(
                    "preparing_registered_five_net_mixture",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                proof = prepare(data, experts, frozen)
                append_event(
                    base.JOURNAL,
                    "target_proof",
                    IDS[0],
                    {
                        "registered_event_sha256": matching_events("registered", IDS[0])[0][
                            "sha256"
                        ],
                        **proof,
                    },
                )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [common.packet_entry(c, frozen) for c in rules.CANDIDATES],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 1,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": (
                        "formal main paths; numeric states are not economic trials"
                    ),
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("five_net_mixture_preparation", error)
            raise
        except KeyboardInterrupt:
            current.progress("five_net_mixture_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "five_net_mixture_registered_waiting_publication",
            active_compute_job=None,
            five_net_mixture_registration=reference(PACKET),
            pending_steps=["publish_exact_five_net_mixture_packet_before_four_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence, not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed, completed = (
            matching_events(k, name) for k in ("review", "technical_failure", "path_completed")
        )
        if (
            len({e["body"]["path_key"] for e in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError("Duplicate completion or terminal events cannot select a winner.")
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in completed
        }
        if reviews:
            if (
                failed
                or set(paths) != base.PATH_KEYS
                or reviews[0]["body"]["paths"] != paths
                or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"]
                is not True
            ):
                raise QuantError("A complete review requires every actual formal path.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["candidates"] != states(frozen) or saved[
                "frozen_specification_sha256"
            ] != digest_json(frozen):
                raise QuantError("Preserve actual completed outcomes without replay.")
            return reference(RESULTS)
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_five_net_mixture",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            target = frame(matching_events("target_proof", IDS[0])[0]["body"]["targets"])
            protocol = read_json(base.PROTOCOL)
            if not matching_events("review", IDS[0]) and not matching_events(
                "technical_failure", IDS[0]
            ):
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        previous.previous.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_five_net_mixture", active_candidate=IDS[0], active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, IDS[0], window, scenario, benchmarks
                        )
                participation = previous.previous.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    IDS[0],
                    {
                        "candidate_id": IDS[0],
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": participation,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            write_new_receipt(
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
                    "new_source_requests": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "five_net_mixture_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                five_net_mixture_results=reference(RESULTS),
                pending_steps=["publish_all_four_five_net_mixture_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("five_net_mixture_evaluation", error)
            raise
        except KeyboardInterrupt:
            current.progress("five_net_mixture_interrupted", active_compute_job=None)
            raise


def public_result():
    _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = states(frozen), read_json(RESULTS)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Publish actual complete outcomes or retained technical failures only.")
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
                    "actual_factor_participation": row.get("actual_factor_participation", {}).get(
                        key
                    ),
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
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Bounded exact five-NET-policy mixture; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Five-policy NET mixture cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Five-policy NET mixture blocked: {error}\n")


if __name__ == "__main__":
    main()
