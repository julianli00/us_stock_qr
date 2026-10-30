from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from xml.etree import ElementTree

import pandas as pd

from us_quant import expert_headroom_stress_v1 as rules
from us_quant import ssd_expert_meta_research_v1 as source
from us_quant import ssd_expert_path_baskets_v1 as previous
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.research_weekly import verify_publication
from us_quant.ssd_research_v1 import (
    append_event,
    matching_events,
    published_inputs,
    reference,
)
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = previous.base, previous.common, previous.current, previous.ROOT
FOLDER = base.REPORTS / "expert-headroom-stress-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
PACKET = ROOT / "evidence/expert_headroom_stress_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_headroom_stress_20261030_results_v1.json"
PROOFS = ROOT / "data/ssd-expert-headroom-stress-20261030-v1"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/expert-headroom-stress-v1.json",
    "src/us_quant/expert_headroom_stress_v1.py",
    "src/us_quant/ssd_expert_headroom_stress_v1.py",
    "tests/test_expert_headroom_stress_v1.py",
    "tests/test_ssd_expert_headroom_stress_v1.py",
)
PROOF_ROLE = "reused_nominal_expert_mix_and_frozen_closed_loop_transition_not_issued_targets"
PARENT_SPEC = "ed749875042c06b5ab343cb9e06feaf240f83a1251bced92e0bf954ae2f79e2c"
SOURCE_SPECS = {
    rules.CANDIDATES[0][
        "source_candidate_id"
    ]: "3c9ba6f21342237d0d5e26a88e3c1c4f1cab25491498e07023d5a1fe0e466edb",
    rules.CANDIDATES[1]["source_candidate_id"]: PARENT_SPEC,
}
SYNTHETIC_STATES = (
    (10000.0, 10000.0, 0.0),
    (10000.0, 10000.0, 0.98),
    (9000.0, 10000.0, 0.98),
    (8490.0, 10000.0, 0.98),
)
unique_refs, check_frozen = source.unique_refs, previous.check_frozen


def dependency_inputs():
    policy = read_json(rules.POLICY)
    parent = read_json(base.checked_reference(policy["dependency_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("Keep the exact post-basket dependency specification.")
    return (
        unique_refs(
            [
                *parent["code"],
                reference(ROOT / "src/us_quant/expert_master_protection_v1.py"),
            ]
        ),
        unique_refs(
            [
                *parent["policies"],
                reference(ROOT / "config/expert-master-protection-v1.json"),
            ]
        ),
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    design = read_json(DESIGN)
    packet = read_json(base.checked_reference(policy["dependency_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or parent["python_version"] != list(sys.version_info[:3])
        or any(version(name) != expected for name, expected in parent["package_versions"].items())
        or design["dependency_parent_registration"] != policy["dependency_registration"]
        or design["dependency_parent_specification_sha256"] != PARENT_SPEC
        or design["candidates"] != list(rules.CANDIDATES)
        or design["nominal_sources"] != policy["nominal_sources"]
    ):
        raise QuantError("Keep the recorded source roles, accepted environment and dependencies.")
    check_frozen(parent)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The immutable original research ledger changed.")
    refs = [
        *design["source_references"],
        reference(DESIGN),
        reference(INVOCATION),
        policy["dependency_registration"],
        policy["motivation_saved_diagnosis"],
    ]
    entries, targets = {}, {}
    data = review_market({"market": parent["market"]}, ROOT)
    for candidate in rules.CANDIDATES:
        name, identifier = candidate["id"], candidate["source_candidate_id"]
        registration = policy["source_registrations"][identifier]
        original = read_json(base.checked_reference(registration))
        matches = [e for e in original["entries"] if e["candidate_id"] == identifier]
        if len(matches) != 1:
            raise QuantError("Each fixed source role needs its single original registration.")
        entry = matches[0]
        specification = entry["specification"]["frozen"]
        if (
            digest_json(specification) != SOURCE_SPECS[identifier]
            or common.packet_entry(entry["specification"]["candidate"], specification) != entry
            or policy["nominal_sources"][identifier] != entry["target_proof"]["targets"]
        ):
            raise QuantError("Read exact archived nominal targets without old source preparation.")
        saved = read_json(base.checked_reference(policy["source_results"][identifier]))
        reviews = [r for r in saved["reviews"] if r["candidate_id"] == identifier]
        if (
            saved["registration_packet"] != registration
            or len(reviews) != 1
            or reviews[0]["status"] != "rejected"
            or {p["path_key"] for p in reviews[0]["paths"]} != set(base.PATH_KEYS)
        ):
            raise QuantError("Preserve all four original rejected source outcomes.")
        frame = pd.read_csv(
            base.checked_reference(policy["nominal_sources"][identifier]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        for window in read_json(base.PROTOCOL)["windows"]:
            rules.nominal_inputs(data, frame, window["start"], window["end"])
        targets[name], entries[identifier] = frame, entry
        refs.extend(
            (
                registration,
                policy["source_results"][identifier],
                policy["nominal_sources"][identifier],
                entry["target_proof"]["proof"],
            )
        )
    code, policies = dependency_inputs()
    refs = unique_refs([*refs, *code, *policies])
    for ref in refs:
        base.checked_reference(ref)
    benchmarks = read_json(base.BENCHMARKS / "results.json")
    if (
        benchmarks["protocol_sha256"] != parent["protocol"]["sha256"]
        or benchmarks["benchmark_accounts"] != 8
        or benchmarks["new_strategy_evaluations"] != 0
    ):
        raise QuantError("Keep the eight source-verified matched-cost SPY/QQQ accounts.")
    return data, targets, policy, parent, entries, refs, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    required = {
        "test_affine_interval_handles_both_slope_signs_and_contradictions",
        "test_review_horizon_uses_next_real_month_end_and_original_delay",
        "test_independent_scalar_stress_scenarios_fraction_and_funding_match",
        "test_signed_headroom_and_current_close_allowance_are_not_charged_fees",
        "test_short_history_is_explicitly_unscored_and_bad_prices_are_errors",
        "test_actual_monthly_feedback_keeps_continuous_main_peak_and_full_calendar",
        "test_initial_pending_request_is_retained_without_overwrite",
        "test_fresh_later_account_never_inherits_source_or_long_main_peak",
        "test_prefix_and_future_perturbations_preserve_known_monthly_feedback",
        "test_holding_band_does_not_force_below_band_reference_orders",
        "test_delayed_gap_can_breach_reference_without_a_drawdown_guarantee",
        "test_funded_replay_and_independent_bt_match_the_new_closed_loop",
        "test_final_delayed_request_does_not_force_terminal_liquidation",
        "test_tampered_stress_state_scenario_or_certificate_is_rejected",
        "test_cached_loader_never_invokes_closed_entrypoints",
        "test_complete_local_import_closure_is_bound",
        "test_closed_loop_proof_binds_only_old_nominal_targets",
        "test_real_probe_preparation_starts_after_both_registrations",
        "test_complete_dependencies_precede_the_new_publication_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_original_paths_without_private_rows",
    }
    if not required.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed stress, feedback, source, funding and publication tests."
        )
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError(
            "Passing validation must bind the exact current five implementation files."
        )
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    common.output_guard(INTENT)
    if INTENT.exists():
        raise QuantError("Preserve any complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, policy, parent, _, refs, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            len(events) != 552
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 50
            or counts["new_completed_strategy_evaluations"] != 46
            or counts["new_technical_failures"] != 4
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 51584
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual unchanged post-diagnosis journal.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 51584,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "question": policy["methodology"]["question"],
                "proof_role": PROOF_ROLE,
                "candidate_accounts_or_metrics_computed": False,
                "real_meta_scores_or_nominal_targets_recomputed": False,
                "real_new_stress_probes_computed": False,
                "state_contingent_issued_targets_precomputed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_headroom_stress_frozen",
            active_compute_job=None,
            expert_headroom_stress_intent=reference(INTENT),
            pending_steps=["register_both_rules_before_real_synthetic_capital_stress_probes"],
        )
        return reference(INTENT)


def inputs():
    data, targets, policy, parent, entries, refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["proof_role"] != PROOF_ROLE
        or any(
            intent.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "real_meta_scores_or_nominal_targets_recomputed",
                "real_new_stress_probes_computed",
                "state_contingent_issued_targets_precomputed",
            )
        )
    ):
        raise QuantError("The new causal transition, tests or admitted nominal sources changed.")
    code, policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs([*code, *[reference(ROOT / p) for p in OWN[1:]]]),
        "policies": unique_refs([*policies, reference(rules.POLICY)]),
        "selection_policy": reference(rules.POLICY),
        "prior_completed_configurations": 224,
        "prior_journal_sequence": 552,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "source_expert_pool_policy": parent["policy"],
        "expert_pool_definition_role": "unchanged_six_experts_in_two_fixed_allocator_source_roles",
        "construction": "monthly_actual_main_headroom_observed_asset_stress",
        "construction_inputs": [reference(INTENT), policy["dependency_registration"]],
        "execution_adapter": reference(ROOT / OWN[1]),
        "execution_adapter_role": "monthly_actual_main_headroom_single_pending_original_band",
        "source_nominal_target_bindings": {
            c["id"]: {
                "source_candidate_id": c["source_candidate_id"],
                **entries[c["source_candidate_id"]]["target_proof"],
            }
            for c in rules.CANDIDATES
        },
        "target_proof_role": PROOF_ROLE,
        "state_contingent_issued_targets_precomputed": False,
        "actual_factor_participation_is_measured_in_target_proof": False,
        "actual_factor_participation_is_measured_in_funded_account": True,
        "archive_import_dependencies_are_additional_strategy_experts": False,
        "public_archive_policy_pins_are_additional_active_allocators": False,
        "prior_evidence": unique_refs([*parent["prior_evidence"], *refs, reference(INTENT)]),
        "validation_evidence": unique_refs(
            [
                *parent["validation_evidence"],
                *validation_references(),
            ]
        ),
    }
    check_frozen(frozen)
    return data, targets, frozen, entries, benchmarks


def transition_binding(candidate, frozen, source_entry, registered):
    nominal = source_entry["target_proof"]["targets"]
    binding = frozen["source_nominal_target_bindings"][candidate["id"]]
    if (
        binding["source_candidate_id"] != candidate["source_candidate_id"]
        or binding["targets"] != nominal
        or binding["proof"] != source_entry["target_proof"]["proof"]
        or frozen["target_proof_role"] != PROOF_ROLE
        or frozen["state_contingent_issued_targets_precomputed"] is not False
    ):
        raise QuantError("A nominal source proof cannot become precomputed final issued targets.")
    base.checked_reference(nominal)
    base.checked_reference(binding["proof"])
    return {
        "checked_at": utc_now(),
        "candidate_id": candidate["id"],
        "registered_event_sha256": registered["sha256"],
        "source_candidate_id": candidate["source_candidate_id"],
        "source_registered_event_sha256": source_entry["registered_event_sha256"],
        "source_nominal_targets": nominal,
        "source_nominal_target_proof": binding["proof"],
        "controller_transition": frozen["execution_adapter"],
        "validation_evidence": frozen["validation_evidence"],
        "frozen_specification_sha256": digest_json(frozen),
        "proof_role": PROOF_ROLE,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "state_contingent_issued_targets_precomputed": False,
        "main_drawdown_guarantee": False,
    }


def transition_proof(data, baseline, candidate, frozen, source_entry, registered, directory):
    proof = transition_binding(candidate, frozen, source_entry, registered)
    active = baseline.dropna(how="all")
    path = directory / "synthetic-capital-stress-probes.jsonl"
    maximum_target = maximum_fraction = maximum_money = maximum_scenario = 0.0
    count = feasible = infeasible = unscored = 0
    base.storage_guard()
    with path.open("x", encoding="utf-8") as output:
        for day, nominal in active.iterrows():
            runtime_guard()
            history = data.close.loc[:day]
            for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                cost, delay = scenario["cost_bps_per_side"], scenario["delay_sessions"]
                for nav, peak, gross in SYNTHETIC_STATES:
                    current_weights = pd.Series(0.0, index=baseline.columns)
                    current_weights["BIL"] = gross
                    state = rules.MainCloseState(day, nav, peak, current_weights)
                    desired, audit = rules.resolve(nominal, state, history, cost, delay)
                    checked = rules.verify_resolution(
                        nominal, state, history, cost, delay, desired, audit
                    )
                    maximum_target = max(maximum_target, checked["maximum_target_difference"])
                    maximum_fraction = max(maximum_fraction, checked["maximum_fraction_difference"])
                    maximum_money = max(
                        maximum_money, checked["maximum_money_allowance_difference_USD"]
                    )
                    maximum_scenario = max(
                        maximum_scenario, checked["maximum_scenario_coefficient_difference"]
                    )
                    count += 1
                    feasible += audit["feasible"] is True
                    infeasible += audit["feasible"] is False
                    unscored += audit["feasible"] is None
                    output.write(
                        json.dumps(
                            {
                                "signal_session": str(day.date()),
                                "scenario": scenario["id"],
                                "synthetic_capital_NAV_USD": nav,
                                "synthetic_peak_USD": peak,
                                "synthetic_actual_gross": gross,
                                "stress": audit,
                                "desired_reference_target": desired.tolist(),
                                "independent_scalar_verification": checked,
                                "actual_main_account": False,
                            },
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    prefix_checks = 0
    for i in sorted(
        {0, min(1, len(active) - 1), min(6, len(active) - 1), len(active) // 2, len(active) - 1}
    ):
        day, nominal = active.index[i], active.iloc[i]
        current_weights = pd.Series(0.0, index=baseline.columns)
        current_weights["BIL"] = 0.98
        state = rules.MainCloseState(day, 9000.0, 10000.0, current_weights)
        for cost, delay in ((5.0, 1), (20.0, 2)):
            history = data.close.loc[:day]
            desired, audit = rules.resolve(nominal, state, history, cost, delay)
            prefix, prefix_audit = rules.resolve(nominal, state, history.tail(127), cost, delay)
            if not desired.equals(prefix) or audit != prefix_audit:
                raise QuantError("A known stress transition depends on unavailable history.")
            prefix_checks += 1
    return {
        **proof,
        "synthetic_capital_probes": reference(path),
        "source_decision_sessions": len(active),
        "synthetic_stress_cases": count,
        "feasible_cases": feasible,
        "infeasible_funding_standby_cases": infeasible,
        "unscored_short_history_cases": unscored,
        "synthetic_capital_state_definitions": [list(s) for s in SYNTHETIC_STATES],
        "synthetic_known_price_prefix_checks": prefix_checks,
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_fraction_difference": maximum_fraction,
        "maximum_independent_cost_allowance_difference_USD": maximum_money,
        "maximum_independent_scenario_coefficient_difference": maximum_scenario,
        "real_source_stress_observations_used_only_with_synthetic_capital": True,
        "actual_main_NAV_peak_or_issued_path_computed": False,
    }


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, targets, frozen, source_entries, _ = inputs()
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A failed registered stress version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("The registered identity or frozen stress rule cannot change.")
            if not found:
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        try:
            for candidate in rules.CANDIDATES:
                name = candidate["id"]
                if matching_events("target_proof", name):
                    common.packet_entry(candidate, frozen)
                    continue
                runtime_guard()
                registration = matching_events("registered", name)[0]
                source_entry = source_entries[candidate["source_candidate_id"]]
                directory, path = PROOFS / name, PROOFS / name / "proof.json"
                if path.exists():
                    proof = read_json(path)
                    if (
                        proof["registered_event_sha256"] != registration["sha256"]
                        or proof["frozen_specification_sha256"] != digest_json(frozen)
                        or proof["proof_role"] != PROOF_ROLE
                    ):
                        raise QuantError(
                            "Retain conflicting partial stress preparation as a failure."
                        )
                    base.checked_reference(proof["synthetic_capital_probes"])
                else:
                    new_output_directory(directory)
                    write_new_receipt(
                        path,
                        transition_proof(
                            data,
                            targets[name],
                            candidate,
                            frozen,
                            source_entry,
                            registration,
                            directory,
                        ),
                    )
                append_event(
                    base.JOURNAL,
                    "target_proof",
                    name,
                    {
                        "registered_event_sha256": registration["sha256"],
                        "targets": source_entry["target_proof"]["targets"],
                        "proof": reference(path),
                        "proof_role": PROOF_ROLE,
                        "candidate_accounts_or_metrics_computed": False,
                        "state_contingent_issued_targets_precomputed": False,
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
                    "new_registered_candidates": 2,
                    "new_economic_hypotheses": 2,
                    "new_controller_families": 1,
                    "new_independent_economic_factors": 0,
                    "target_proof_role": PROOF_ROLE,
                    "candidate_accounts_or_metrics_computed": False,
                    "state_contingent_issued_targets_precomputed": False,
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_headroom_stress_transition_preparation", IDS, error)
            raise
        current.progress(
            "expert_headroom_stress_registered_waiting_publication",
            active_compute_job=None,
            expert_headroom_stress_registration=reference(PACKET),
            pending_steps=["publish_exact_generative_packet_before_eight_funded_paths"],
        )
        return reference(PACKET)


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or packet["target_proof_role"] != PROOF_ROLE
        or packet["raw_inputs_and_target_matrices_remain_private"] is not True
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"])
            for name in IDS
            for e in matching_events("path_started", name)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "state_contingent_issued_targets_precomputed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError(
            "The actual closed-loop preregistration packet or causal boundary changed."
        )
    for entry in entries:
        body = entry["target_proof"]
        proof = read_json(base.checked_reference(body["proof"]))
        nominal = frozen["source_nominal_target_bindings"][entry["candidate_id"]]["targets"]
        if (
            body["targets"] != nominal
            or body["proof_role"] != PROOF_ROLE
            or body["candidate_accounts_or_metrics_computed"] is not False
            or body["state_contingent_issued_targets_precomputed"] is not False
            or proof["source_nominal_targets"] != nominal
            or proof["proof_role"] != PROOF_ROLE
            or proof["frozen_specification_sha256"] != digest_json(frozen)
            or proof["controller_transition"] != frozen["execution_adapter"]
            or proof["state_contingent_issued_targets_precomputed"] is not False
            or proof["real_source_stress_observations_used_only_with_synthetic_capital"] is not True
            or proof["actual_main_NAV_peak_or_issued_path_computed"] is not False
            or proof["synthetic_capital_state_definitions"] != [list(s) for s in SYNTHETIC_STATES]
            or proof["synthetic_stress_cases"]
            != proof["source_decision_sessions"] * 2 * len(SYNTHETIC_STATES)
            or proof["synthetic_stress_cases"]
            != sum(
                proof[k]
                for k in (
                    "feasible_cases",
                    "infeasible_funding_standby_cases",
                    "unscored_short_history_cases",
                )
            )
        ):
            raise QuantError("Preserve the honest nominal binding and frozen transition proof.")
        base.checked_reference(proof["synthetic_capital_probes"])
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in identifiers:
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
        "expert_headroom_stress_expired" if expired else "expert_headroom_stress_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_failure=str(error),
    )


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
            raise QuantError("Resume the original publication fence, not a new result-time commit.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL,
        "publication",
        None,
        {"study_id": frozen["study_id"], "publication": receipt},
    )
    return receipt


def evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    completed = [
        e for e in matching_events("path_completed", identifier) if e["body"]["path_key"] == key
    ]
    if completed:
        if len(completed) != 1:
            raise QuantError("Never choose among duplicate completed financial paths.")
        return base.verify_path(completed[0]["body"]["receipt"], frozen, identifier, key)
    starts = [
        e for e in matching_events("path_started", identifier) if e["body"]["path_key"] == key
    ]
    if starts:
        last = starts[-1]
        receipt_path = ROOT / last["body"]["directory"] / "complete.json"
        if receipt_path.exists():
            receipt = base.verify_path(reference(receipt_path), frozen, identifier, key)
            append_event(
                base.JOURNAL,
                "path_completed",
                identifier,
                {
                    "path_key": key,
                    "receipt": reference(receipt_path),
                    "recovered_actual_complete_receipt": True,
                },
            )
            return receipt
        if not any(
            e["body"].get("started_event_sha256") == last["sha256"]
            for e in matching_events("path_interrupted", identifier)
        ):
            append_event(
                base.JOURNAL,
                "path_interrupted",
                identifier,
                {
                    "path_key": key,
                    "started_event_sha256": last["sha256"],
                    "partial_output_preserved": True,
                    "reason": "Exclusive lock reacquired without a complete path receipt.",
                },
            )
    runtime_guard()
    directory = base.ACCOUNTS / identifier / f"{key}-attempt-{len(starts) + 1:03d}"
    append_event(
        base.JOURNAL,
        "path_started",
        identifier,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    new_output_directory(directory)
    policy = read_json(base.checked_reference(frozen["policy"]))
    candidate = next(c for c in rules.CANDIDATES if c["id"] == identifier)
    result, issued, decisions = rules.run(
        data,
        targets,
        candidate,
        policy,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback = rules.verify_feedback(
        data,
        targets,
        result,
        issued,
        decisions,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    independent = base.independent_equity(
        data,
        issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(result.frame["return"], result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    criteria = base.gates(
        own,
        lookup[window["years"], scenario["id"], "SPY"],
        lookup[window["years"], scenario["id"], "QQQ"],
        lookup[window["years"], "base", "SPY"],
        scenario["id"],
    )
    execution = base.execution_statistics(result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": result.frame,
        "weights.csv": result.weights,
        "issued.csv": issued,
        "independent-bt.csv": independent,
    }
    base.storage_guard()
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": identifier,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": execution,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"]
        - lookup[window["years"], scenario["id"], "SPY"]["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"]
        - lookup[window["years"], scenario["id"], "QQQ"]["cagr"],
        "stress_vs_base_cost_qqq_diagnostic": own["cagr"]
        - lookup[window["years"], "base", "QQQ"]["cagr"]
        if scenario["id"] == "stress"
        else None,
        "independent_bt_max_equity_difference_usd": difference,
        "independent_main_wealth_feedback": feedback,
        "endogenous_issued_targets_generated_with_this_actual_funded_path": True,
        "nominal_source_targets_reused_without_regeneration": True,
        "artifacts": [reference(directory / name) for name in (*frames, "decisions.json")],
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    append_event(
        base.JOURNAL,
        "path_completed",
        identifier,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed = (
            matching_events("review", name),
            matching_events("technical_failure", name),
        )
        completed = matching_events("path_completed", name)
        if (
            len({e["body"]["path_key"] for e in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError(
                "Duplicate actual completion or terminal events cannot select a winner."
            )
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in completed
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("A closed-loop review differs from its actual completed paths.")
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
        data, targets, frozen, _, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["candidates"] != states(frozen):
                raise QuantError("Read completed results without changing or replaying them.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_headroom_stress",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            protocol = read_json(base.PROTOCOL)
            for name in IDS:
                active = (name,)
                if matching_events("review", name) or matching_events("technical_failure", name):
                    continue
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        previous.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_headroom_stress",
                            active_candidate=name,
                            active_path=key,
                        )
                        paths[key] = evaluate_path(
                            data, targets[name], frozen, name, window, scenario, benchmarks
                        )
                actual_participation = participation.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"]
                    for p in actual_participation.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": actual_participation,
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
                    "new_source_requests": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
                ("finished_at",),
            )
            current.progress(
                "expert_headroom_stress_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_headroom_stress_results=reference(RESULTS),
                pending_steps=["publish_all_eight_master_protection_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_headroom_stress_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_headroom_stress_interrupted", active_compute_job=None)
            raise


def public_result():
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Publish actual completed outcomes or retained technical failures only.")
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
                            "independent_main_wealth_feedback",
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {s: lookup[years, scenario, s] for s in ("SPY", "QQQ")},
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
        description="Monthly actual-main headroom and observed asset stress; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Closed-loop expert research cannot contact live market sources.")

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
        parser.exit(2, f"Expert master protection blocked: {error}\n")


if __name__ == "__main__":
    main()
