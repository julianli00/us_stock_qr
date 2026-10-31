from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_pair_menu_extension_v1 as rules
from us_quant import ssd_anfci_expert_eligibility_v1 as import_template
from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import file_digest, read_json, utc_now

ROOT = base.ROOT
FOLDER = base.REPORTS / "expert-pair-menu-extension-20261031-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "diagnostic-intent-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/expert_pair_menu_extension_20261031_v1.json"
DIAGNOSIS = "ssd_expert_pair_menu_extension_20261031_v1"
DESIGN_SHA = "5eb616eaccecfffad3e4699e290243af63e2b14126eb1faf30ba1b289e84eb06"
OWN = (
    "src/us_quant/expert_pair_menu_extension_v1.py",
    "src/us_quant/ssd_expert_pair_menu_extension_v1.py",
    "tests/test_expert_pair_menu_extension_v1.py",
    "tests/test_ssd_expert_pair_menu_extension_v1.py",
)
reference = base.reference


def dependencies():
    return FunctionType(
        import_template.dependencies.__code__, {**globals(), "ast": ast, "prior": base}
    )()


def source_references():
    if reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError(
            "Preserve the actual pre-comparison menu interpretation and stopping rule."
        )
    design = read_json(DESIGN)
    if (
        design["diagnosis_id"] != DIAGNOSIS
        or design["actual_continuation_event"] != 59199
        or design["expert_ids_six"] != list(rules.SIX)
        or design["expert_ids_seven"] != list(rules.SEVEN)
        or design["actual_pair_matching_restricted_ranking_or_diagnostic_result_computed"]
        is not False
        or design["new_financial_pool_or_policy_selected"] is not False
        or design["maximum_actual_comparison_passes"] != 1
        or design["decision_information_clock"] != rules.CLOCK
        or design["original_fixed_numeric_limits"]
        != {
            "unit_weight_sum": 1e-12,
            "shared_expert_weights": 1e-10,
            "shared_training_correlation": 1e-10,
            "shared_marginal_variance_contributions": 1e-12,
        }
    ):
        raise QuantError("Require the real fixed zero-trial stored-menu diagnostic identity.")
    refs = {}
    for ref in [
        reference(DESIGN),
        design["actual_invocation"],
        reference(FOLDER / "source-public-progress-baseline-v1.json"),
        *design["source_references"],
        *dependencies(),
    ]:
        if ref["path"] in refs and refs[ref["path"]] != ref:
            raise QuantError("Conflicting source identities cannot become a stored comparison.")
        refs[ref["path"]] = ref
    result = [refs[key] for key in sorted(refs)]
    for ref in result:
        base.checked_reference(ref)
    return result


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve an existing diagnostic freeze instead of refreezing.")
    xml = FOLDER / "offline-diagnostic-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_direct_and_shared_choice_categories_are_distinct",
        "test_restricted_recorded_ranking_is_not_original_six_replay",
        "test_unmatured_states_do_not_receive_successful_ranks",
        "test_all_shared_and_added_pairs_and_clocks_are_required",
        "test_original_shared_risk_limits_cannot_be_waived",
        "test_failure_preserves_matched_inputs_without_rescue",
        "test_original_source_score_target_and_account_entrypoints_are_never_called",
        "test_complete_dependency_capture_keeps_original_globals",
        "test_public_report_has_no_raw_pairs_weights_or_individual_scores",
        "test_second_actual_comparison_attempt_is_terminal",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require matched-clock,menu,maturity,risk and no-replay diagnostic tests.")
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    own = [reference(ROOT / path) for path in OWN]
    outputs = [reference(xml), reference(FOLDER / "offline-diagnostic-tests-v1.log")]
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError("Freeze only the actual current tested diagnostic and output bytes.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "diagnosis_id": DIAGNOSIS,
            "actual_continuation_event": 59199,
            "design": reference(DESIGN),
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "passed_offline_checks": count,
            "source_references": refs,
            "actual_stored_comparison_started": False,
            "maximum_actual_comparison_passes": 1,
            "new_strategy_evaluations": 0,
            "new_source_requests": 0,
            "new_financial_configuration_selected": False,
            "investment_objective_verified": False,
        },
    )
    return {"intent": reference(INTENT), "source_references": len(refs), "tests": count}


def checked_intent():
    intent = read_json(INTENT)
    if (
        intent["diagnosis_id"] != DIAGNOSIS
        or intent["design"] != reference(DESIGN)
        or intent["actual_stored_comparison_started"] is not False
        or intent["maximum_actual_comparison_passes"] != 1
        or intent["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or intent["source_references"] != source_references()
    ):
        raise QuantError(
            "Stored diagnostic interpretation,source scope or code changed after freeze."
        )
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    return intent


def read_inputs(design):
    return (
        read_json(base.checked_reference(design["source_six_audits"]))["decisions"],
        read_json(base.checked_reference(design["source_seven_audits"]))["decisions"],
    )


def public_report(result):
    return {
        "schema_version": 1,
        "diagnosis_id": DIAGNOSIS,
        "completed_at": result["completed_at"],
        "private_diagnostic": reference(RESULT),
        "precomparison_intent": result["intent"],
        "summary": result["summary"],
        "source_inputs_checked": result["source_inputs_checked"],
        "analysis_import_and_configuration_dependencies": result[
            "analysis_import_and_configuration_dependencies"
        ],
        "passed_offline_checks": result["passed_offline_checks"],
        "restricted_menu_is_diagnostic_of_stored_seven_scores_not_six_replay": True,
        "source_scores_risks_targets_or_accounts_recomputed": False,
        "individual_pairs_weights_scores_or_raw_values_published": False,
        "new_strategy_evaluations": 0,
        "new_funded_paths": 0,
        "new_source_requests": 0,
        "new_financial_configuration_selected": False,
        "all_financial_counts_unchanged": result["all_financial_counts_unchanged"],
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def diagnose():
    runtime_guard()
    attempt = FOLDER / "comparison-attempt-v1.json"
    if attempt.exists() or RESULT.exists():
        raise QuantError("Preserve the first actual comparison instead of a silent second pass.")
    intent = checked_intent()
    counts = current.current_counts(base.journal_events(base.JOURNAL))
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "diagnostic_intent": reference(INTENT),
            },
        )
        try:
            design = read_json(DESIGN)
            before, after = read_inputs(design)
            log = FOLDER / "actual-matched-inputs-resolved-v1.jsonl"
            with log.open("x") as stream:

                def retain(row):
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()

                summary = rules.compare(before, after, retain=retain)
            checked_intent()
            if (
                counts != current.current_counts(base.journal_events(base.JOURNAL))
                or file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256
            ):
                raise QuantError("A zero-trial stored diagnosis cannot alter financial history.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "stage": "stored_menu_comparison",
                    "reason": str(error),
                    "diagnostic_intent": reference(INTENT),
                    "partial_matched_records_preserved": True,
                    "new_strategy_evaluations": 0,
                    "no_solver_parameter_or_tolerance_rescue": True,
                },
            )
            current.progress(
                "pair_menu_extension_diagnosis_blocked",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pending_steps=["preserve_first_fixed_diagnostic_failure_without_rescue"],
            )
            raise
        result = {
            "schema_version": 1,
            "diagnosis_id": DIAGNOSIS,
            "completed_at": utc_now(),
            "intent": reference(INTENT),
            "summary": summary,
            "actual_matched_input_resolved_records": reference(log),
            "source_inputs_checked": len(intent["source_references"]),
            "analysis_import_and_configuration_dependencies": len(dependencies()),
            "passed_offline_checks": intent["passed_offline_checks"],
            "all_financial_counts_unchanged": counts,
            "new_strategy_evaluations": 0,
            "new_source_requests": 0,
            "new_financial_configuration_selected": False,
            "investment_objective_verified": False,
        }
        write_new_receipt(RESULT, result)
        write_new_receipt(PUBLIC, public_report(result))
    current.progress(
        "pair_menu_extension_diagnosis_complete_pending_publication",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        pair_menu_extension_result=reference(RESULT),
        pending_steps=["publish_bounded_menu_evidence_without_new_financial_rule"],
    )
    return {"diagnostic_result": reference(RESULT), "public_summary": reference(PUBLIC)}


def main():
    parser = argparse.ArgumentParser(
        description="Read-only recorded pair-menu extension diagnosis."
    )
    parser.add_argument("action", choices=("freeze", "diagnose"))
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("A stored diagnostic cannot request live sources.")

    sys.addaudithook(offline)
    print(json.dumps(freeze() if args.action == "freeze" else diagnose(), indent=2))


if __name__ == "__main__":
    main()
