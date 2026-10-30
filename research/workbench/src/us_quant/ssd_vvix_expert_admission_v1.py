from __future__ import annotations

import argparse
import ast
import json
import os
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_vvix_admission_v1 as rules
from us_quant import ssd_anfci_expert_eligibility_v1 as import_template
from us_quant import ssd_expert_payoff_persistence_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.storage import digest_json, read_json, utc_now
from us_quant.strategy_experts_v1 import load_expert

ROOT, base = prior.ROOT, prior.base
FOLDER = base.REPORTS / "vvix-expert-admission-20261030-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "admission-intent-v1.json"
OUTPUT = FOLDER / "admission-v1.json"
PUBLIC = ROOT / "evidence/vvix_uncertainty_expert_20261030_admission_v1.json"
ADMISSION = "ssd_vvix_expert_admission_20261030_v1"
DESIGN_SHA = "0b0d27e93ebdcd8f4eff863c849c30451ec962348cf85f2cefd95cec38421171"
OWN = (
    "src/us_quant/expert_vvix_admission_v1.py",
    "src/us_quant/ssd_vvix_expert_admission_v1.py",
    "tests/test_expert_vvix_admission_v1.py",
    "tests/test_ssd_vvix_expert_admission_v1.py",
)


def dependencies():
    return FunctionType(import_template.dependencies.__code__, {**globals(), "ast": ast})()


def common_market(design):
    packet_path = prior.source.PACKET.relative_to(ROOT).as_posix()
    references = [ref for ref in design["source_references"] if ref["path"] == packet_path]
    if len(references) != 1:
        raise QuantError("Bind the exact existing common market's original registration.")
    frozen = read_json(base.checked_reference(references[0]))["entries"][0]["specification"][
        "frozen"
    ]
    if (
        digest_json(frozen) != prior.PARENT_SPEC
        or frozen["market"]["risk_free"] != design["active_common_risk_free"]
        or design["active_common_risk_free"] != design["source_native_risk_free"]
    ):
        raise QuantError("Do not replace the active common market or RF interpretation.")
    return frozen["market"]


def common_risk_free(design):
    return review_market({"market": common_market(design)}, ROOT).risk_free


def source_references():
    if prior.reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the pre-admission role,clock and no-new-view decision.")
    design = read_json(DESIGN)
    if (
        design["admission_id"] != ADMISSION
        or design["actual_continuation_event"] != 57172
        or design["candidate_record"]["id"] != rules.SOURCE
        or design["candidate_record"]["mechanism_group"] != rules.GROUP
        or design["actual_cached_expert_loading_or_admission_checks_started"] is not False
        or design["required_information_clock"] != rules.CLOCK
        or design["required_source_scope"] != rules.SCOPE
        or design["current_six_expert_pool_changed"] is not False
    ):
        raise QuantError("Require the actual unchanged bounded option-expert admission identity.")
    artifacts = []
    for ref in design["candidate_record"]["completed_paths"].values():
        artifacts.extend(read_json(base.checked_reference(ref))["artifacts"])
    refs = prior.unique_refs(
        [
            prior.reference(DESIGN),
            design["actual_invocation"],
            prior.reference(FOLDER / "source-public-progress-baseline-v1.json"),
            *design["source_references"],
            *common_market(design).values(),
            *artifacts,
            *dependencies(),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return refs


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve an existing expert-admission freeze instead of refreezing.")
    xml = FOLDER / "offline-admission-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_archival_scope_and_after_index_close_clock_are_not_widened",
        "test_cached_observation_availability_and_maturity_are_distinct",
        "test_native_RF_mismatch_is_not_replaced",
        "test_native_monthly_gap_is_not_filled",
        "test_loader_requires_explicit_after_index_close_clock",
        "test_closed_source_generators_accounts_and_E_review_are_not_called",
        "test_dependency_capture_does_not_mutate_template",
        "test_recorded_scope_survives_unrelated_future_configuration_additions",
        "test_public_admission_excludes_raw_targets_and_return_paths",
        "test_second_admission_attempt_is_terminal",
        "test_admission_interface_uses_only_cached_observations_and_preserves_clocks",
        "test_failed_cached_admission_retains_attempt_without_source_rescue",
        "test_admitted_loader_preserves_real_clock_scope_and_exact_common_RF",
        "test_active_common_RF_reuses_the_existing_frozen_reader",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require clock,scope,causal-cache,RF and no-replay admission checks.")
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    own = [prior.reference(ROOT / path) for path in OWN]
    outputs = [prior.reference(xml), prior.reference(FOLDER / "offline-admission-tests-v1.log")]
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError("Freeze the actual tested current implementation and validation bytes.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "admission_id": ADMISSION,
            "actual_continuation_event": 57172,
            "design": prior.reference(DESIGN),
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                prior.reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "passed_offline_checks": count,
            "source_references": refs,
            "actual_cached_admission_started": False,
            "maximum_actual_admission_passes": 1,
            "new_financial_trials": 0,
            "new_source_requests": 0,
            "current_six_expert_pool_changed": False,
            "active_common_RF_reader": "us_quant.research_program.review_market",
            "active_common_RF_reader_changed": False,
            "investment_objective_verified": False,
        },
    )
    return {"intent": prior.reference(INTENT), "source_references": len(refs), "tests": count}


def checked_intent():
    intent = read_json(INTENT)
    declared = read_json(DESIGN)["source_references"]
    retained = {(ref["path"], ref["sha256"]) for ref in intent["source_references"]}
    if (
        intent["admission_id"] != ADMISSION
        or intent["actual_cached_admission_started"] is not False
        or intent["maximum_actual_admission_passes"] != 1
        or intent["design"] != prior.reference(DESIGN)
        or prior.reference(DESIGN)["sha256"] != DESIGN_SHA
        or intent["implementation_and_test_inputs"]
        != [prior.reference(ROOT / path) for path in OWN]
        or not {(ref["path"], ref["sha256"]) for ref in declared}.issubset(retained)
    ):
        raise QuantError("The recorded admission source scope or tested implementation changed.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    if OUTPUT.exists() and read_json(OUTPUT)["intent"] != prior.reference(INTENT):
        raise QuantError("An existing admission binds the exact original pre-admission intent.")
    return intent


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def cached_expert(design):
    record = design["candidate_record"]
    packet = read_json(base.checked_reference(record["registration"]))
    entry = next(row for row in packet["entries"] if row["candidate_id"] == rules.SOURCE)
    frozen = entry["specification"]["frozen"]
    if (
        digest_json(frozen) != design["source_frozen_specification_sha256"]
        or frozen["market"]["risk_free"] != design["source_native_risk_free"]
        or design["source_native_risk_free"] != design["active_common_risk_free"]
    ):
        raise QuantError("Use the complete frozen source and unchanged active common RF identity.")
    policy = read_json(base.checked_reference(design["source_policy"]))
    source_scope = rules.validate_scope(
        policy,
        read_json(base.checked_reference(policy["vvix_readiness"])),
        read_json(base.checked_reference(policy["vvix_manifest"])),
        read_json(base.checked_reference(design["source_target_proof"])),
    )
    expert, technical = load_expert(record, rules.ASSETS)
    if technical["standalone_financial_verdict_preserved"] != "rejected":
        raise QuantError("Preserve the original source verdict,not a post-outcome replacement.")
    targets, shadow = frame(technical["target_reference"]), frame(technical["shadow_reference"])
    return expert, technical, source_scope, targets, shadow


def require_clock(value):
    if value != rules.CLOCK:
        raise QuantError(
            "A VVIX expert decision requires the declared after16:15ET information clock."
        )


def load_admitted(universe, common_rf, *, decision_information_clock):
    require_clock(decision_information_clock)
    if universe != rules.ASSETS:
        raise QuantError("Use only the original admitted fourteen-asset observation lattice.")
    intent = checked_intent()
    admission = read_json(OUTPUT)
    design = read_json(DESIGN)
    if (
        admission["admission_id"] != ADMISSION
        or admission["expert"]["record"] != design["candidate_record"]
        or admission["new_strategy_evaluations"] != 0
        or admission["current_six_expert_pool_changed"] is not False
        or admission["source_original_registered_at"] != design["candidate_record"]["introduced_at"]
        or pd.Timestamp(admission["admitted_at"]).tzinfo is None
        or pd.Timestamp(admission["admitted_at"]) < pd.Timestamp(intent["created_at"])
        or pd.Timestamp(admission["admitted_at"])
        <= pd.Timestamp(admission["source_original_registered_at"])
    ):
        raise QuantError("An admitted representative needs its real unchanged evidence and clocks.")
    expert, technical, source_scope, _, shadow = cached_expert(design)
    if (
        not isinstance(common_rf, pd.Series)
        or not shadow.index.isin(common_rf.index).all()
        or not np.array_equal(
            shadow["risk_free"].to_numpy(), common_rf.loc[shadow.index].to_numpy()
        )
        or admission["source_scope"] != source_scope
        or admission["expert"]["technical"] != technical
    ):
        raise QuantError("A future caller cannot replace native NET observations or the common RF.")
    return expert, admission


def public_report(admission):
    return {
        "schema_version": 1,
        "admission_id": ADMISSION,
        "admitted_at": admission["admitted_at"],
        "private_admission": prior.reference(OUTPUT),
        "candidate_id": admission["expert"]["record"]["id"],
        "mechanism_group": rules.GROUP,
        "source_original_registered_at": admission["source_original_registered_at"],
        "source_original_registration_sequence": 217,
        "source_original_target_proof_sequence": 218,
        "source_original_review_sequence": 228,
        "source_original_standalone_financial_verdict": "rejected",
        "financial_failure_is_not_expert_admission_failure": True,
        "source_scope": admission["source_scope"],
        "observation_checks": admission["observation_checks"],
        "four_cached_native_paths_checked": 4,
        "original_source_technical_failures": 0,
        "source_frozen_specification_sha256": admission["expert"]["technical"][
            "source_frozen_specification_sha256"
        ],
        "source_inputs_checked": admission["source_inputs_checked"],
        "analysis_import_and_configuration_dependencies": admission[
            "analysis_import_and_configuration_dependencies"
        ],
        "passed_offline_checks": admission["passed_offline_checks"],
        "active_common_RF_reader": admission["active_common_RF_reader"],
        "active_common_RF_reader_changed": False,
        "new_recommendation_or_assessment_view_artifacts": 0,
        "source_generator_account_or_verifier_pipeline_replayed": False,
        "new_strategy_evaluations": 0,
        "new_funded_paths": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "technically_admitted_representatives_now_available": 7,
        "existing_six_expert_pool_changed": False,
        "expanded_pool_or_next_meta_rule_frozen": False,
        "new_independent_economic_risk_source_claimed": False,
        "all_current_financial_counts_unchanged": admission[
            "all_current_financial_counts_unchanged"
        ],
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def admit():
    runtime_guard()
    attempt = FOLDER / "admission-attempt-v1.json"
    if attempt.exists() or OUTPUT.exists():
        raise QuantError("Preserve the first actual admission attempt;no silent second pass.")
    intent = checked_intent()
    counts = prior.current.current_counts(base.journal_events(base.JOURNAL))
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "admission_intent": prior.reference(INTENT),
            },
        )
        try:
            design = read_json(DESIGN)
            expert, technical, source_scope, targets, shadow = cached_expert(design)
            observations_path = FOLDER / "cached-observation-checks-v1.jsonl"
            with observations_path.open("x") as stream:

                def retain(row):
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()

                observations = rules.check_cached_observations(
                    expert, targets, shadow, common_risk_free(design), retain=retain
                )
            checked_intent()
            if counts != prior.current.current_counts(base.journal_events(base.JOURNAL)):
                raise QuantError("Technical expert admission cannot alter financial trial counts.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(exc),
                    "partial_inputs_preserved": True,
                    "admission_intent": prior.reference(INTENT),
                    "source_original_verdict_or_current_pool_changed": False,
                    "new_strategy_evaluations": 0,
                },
            )
            prior.current.progress(
                "vvix_expert_admission_blocked",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pending_steps=["preserve_first_admission_failure_without_repair_or_retry"],
            )
            raise
        admitted_at = utc_now()
        admission = {
            "schema_version": 1,
            "admission_id": ADMISSION,
            "admitted_at": admitted_at,
            "intent": prior.reference(INTENT),
            "source_original_registered_at": design["candidate_record"]["introduced_at"],
            "expert": {"record": design["candidate_record"], "technical": technical},
            "source_scope": source_scope,
            "observation_checks": observations,
            "observation_check_records": prior.reference(observations_path),
            "source_inputs_checked": len(intent["source_references"]),
            "analysis_import_and_configuration_dependencies": len(dependencies()),
            "passed_offline_checks": intent["passed_offline_checks"],
            "active_common_RF_reader": "us_quant.research_program.review_market",
            "active_common_RF_reader_changed": False,
            "all_current_financial_counts_unchanged": counts,
            "new_strategy_evaluations": 0,
            "new_source_requests": 0,
            "new_recommendation_or_assessment_view_artifacts": 0,
            "current_six_expert_pool_changed": False,
            "expanded_pool_or_next_meta_rule_frozen": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
        }
        write_new_receipt(OUTPUT, admission)
        write_new_receipt(PUBLIC, public_report(admission))
    prior.current.progress(
        "vvix_expert_admitted_pending_publication",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        vvix_expert_admission=prior.reference(OUTPUT),
        pending_steps=["publish_complete_cached_expert_admission_without_changing_six_expert_pool"],
    )
    return {"admission": prior.reference(OUTPUT), "public_summary": prior.reference(PUBLIC)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clock-bound cached VVIX strategy-expert admission."
    )
    parser.add_argument("action", choices=("freeze", "admit"))
    args = parser.parse_args(argv)
    print(json.dumps(freeze() if args.action == "freeze" else admit(), indent=2))


if __name__ == "__main__":
    main()
