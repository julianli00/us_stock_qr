from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_pair_allocation_decomposition_v1 as rules
from us_quant import ssd_anfci_expert_eligibility_v1 as import_template
from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import file_digest, read_json, utc_now

ROOT = base.ROOT
FOLDER = base.REPORTS / "expert-pair-allocation-decomposition-20261031-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "diagnostic-intent-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/expert_pair_allocation_decomposition_20261031_v1.json"
DIAGNOSIS = "ssd_expert_pair_allocation_decomposition_20261031_v1"
DESIGN_SHA = "cc08787ab116fe5ab9036b0d1c13e700fbd109c77f1d0feb8ab68d9adee40c0e"
OWN = (
    "src/us_quant/expert_pair_allocation_decomposition_v1.py",
    "src/us_quant/ssd_expert_pair_allocation_decomposition_v1.py",
    "tests/test_expert_pair_allocation_decomposition_v1.py",
    "tests/test_ssd_expert_pair_allocation_decomposition_v1.py",
)
reference = base.reference


def dependencies():
    return FunctionType(
        import_template.dependencies.__code__, {**globals(), "ast": ast, "prior": base}
    )()


def source_references():
    if reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the actual pre-diagnostic source interpretation and stop rule.")
    design = read_json(DESIGN)
    if (
        design["diagnosis_id"] != DIAGNOSIS
        or design["actual_continuation_event"] != 61475
        or design["actual_invocation"] != reference(INVOCATION)
        or design["expert_ids"] != list(rules.EXPERT_IDS)
        or design["decision_information_clock"] != rules.CLOCK
        or design["source_uniform_candidate_id"] != rules.UNIFORM
        or design["source_ordinal_candidate_id"] != rules.ORDINAL
        or design["maximum_actual_comparison_passes"] != 1
        or design["actual_new_allocation_matching_decomposition_or_concentration_results_computed"]
        is not False
        or design["new_financial_configuration_selected"] is not False
        or design["original_numeric_limits"]
        != {
            "unit_weight_or_pairmass_sum": 1e-12,
            "paired_expert_weights": 1e-10,
            "paired_training_correlation": 1e-10,
            "paired_marginal_variance": 1e-12,
            "allocation_identity_residual": 1e-10,
        }
    ):
        raise QuantError("Require the actual fixed zero-trial stored allocation diagnosis.")
    refs = {}
    for ref in [
        reference(DESIGN),
        reference(INVOCATION),
        reference(FOLDER / "source-public-progress-baseline-v1.json"),
        *design["source_references"],
        *dependencies(),
    ]:
        if ref["path"] in refs and refs[ref["path"]] != ref:
            raise QuantError("Conflicting immutable source identities cannot be merged.")
        refs[ref["path"]] = ref
    result = [refs[name] for name in sorted(refs)]
    for ref in result:
        base.checked_reference(ref)
    return result


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve an existing diagnostic freeze instead of repeating it.")
    xml = FOLDER / "offline-diagnostic-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_complete_uniform_anchored_identity_and_concentration",
        "test_neutral_initial_states_are_not_mature_risk_results",
        "test_recorded_pair_risk_limits_cannot_be_waived",
        "test_recorded_mass_state_clock_or_capital_tampering_is_rejected",
        "test_failure_preserves_matched_input_before_decomposition",
        "test_source_pair_and_log_inputs_are_read_without_original_entrypoints",
        "test_complete_dependency_capture_keeps_original_globals",
        "test_public_report_contains_only_aggregate_diagnostic_not_individual_source_rows",
        "test_second_actual_diagnostic_attempt_is_terminal",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require complete identity,clock,maturity,source and no-retry tests.")
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
        raise QuantError("Freeze only current tested code and actual successful output bytes.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "diagnosis_id": DIAGNOSIS,
            "actual_continuation_event": 61475,
            "design": reference(DESIGN),
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "source_references": refs,
            "passed_offline_checks": count,
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
        raise QuantError("The actual fixed diagnostic interpretation,source scope or code changed.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    return intent


def read_inputs(design):
    recorded = read_json(base.checked_reference(design["source_information_audits"]))["candidates"]
    ids = (rules.UNIFORM, rules.ORDINAL)
    if set(recorded) != set(ids):
        raise QuantError("Use exactly the frozen two complete source allocation identities.")
    records = 0
    with base.checked_reference(design["source_actual_input_resolved_log"]).open() as stream:
        for name in ids:
            for position, audit in enumerate(recorded[name]):
                before, after = json.loads(stream.readline()), json.loads(stream.readline())
                if (
                    before["record"] != "decision_input"
                    or after["record"] != "decision_resolved"
                    or before["candidate_id"] != name
                    or after["candidate_id"] != name
                    or before["decision_index"] != position
                    or after["decision_index"] != position
                    or before["as_of"] != audit["decision_session"]
                    or before["information_clock"] != rules.CLOCK
                    or before["previous_weights"] != audit["previous_expert_weights"]
                    or before["previous_weights_binary64_hex"]
                    != [float(value).hex() for value in audit["previous_expert_weights"]]
                    or before["binary64_values_truncated_to_fixed_decimals"] is not False
                    or after["audit"] != audit
                    or [item["id"] for item in before["observations"]] != list(rules.EXPERT_IDS)
                    or [item["target_known_at"] for item in before["observations"]]
                    != audit["expert_target_known_at"]
                ):
                    raise QuantError(
                        "The immutable source input/resolved log disagrees with audits."
                    )
                records += 2
        if stream.readline() or records != 484:
            raise QuantError("Retain every original484source input/resolved records,not a subset.")
    return recorded[rules.UNIFORM], recorded[rules.ORDINAL]


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
        "source484input_resolved_records_verified_without_pipeline_replay": True,
        "new_diagnostic_arithmetic_not_original_source_memory": True,
        "original_scores_risks_ranks_targets_or_accounts_regenerated": False,
        "individual_pairs_weights_scores_or_raw_source_records_published": False,
        "main_account_PNL_fees_CAGR_or_drawdown_counterfactual_computed": False,
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
        raise QuantError("Preserve the first actual diagnostic instead of a second comparison.")
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
        current.progress(
            "running_allocation_mass_decomposition",
            active_compute_job={"pid": os.getpid(), "diagnosis_id": DIAGNOSIS},
        )
        try:
            u, o = read_inputs(read_json(DESIGN))
            log = FOLDER / "actual-matched-inputs-resolved-v1.jsonl"
            with log.open("x") as stream:

                def retain(row):
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()

                summary = rules.compare(u, o, retain=retain)
            record_count = sum(1 for _ in log.open())
            if record_count != 242:
                raise QuantError("Retain exactly242new matched input/resolved diagnostic records.")
            checked_intent()
            if (
                counts != current.current_counts(base.journal_events(base.JOURNAL))
                or file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256
            ):
                raise QuantError("A zero-trial stored diagnosis cannot change financial history.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "stage": "recorded_allocation_mass_decomposition",
                    "reason": str(error),
                    "diagnostic_intent": reference(INTENT),
                    "partial_source_and_matched_records_preserved": True,
                    "new_strategy_evaluations": 0,
                    "no_solver_parameter_or_tolerance_rescue": True,
                },
            )
            current.progress(
                "allocation_mass_decomposition_blocked",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pending_steps=["preserve_first_fixed_diagnostic_failure_without_rescue"],
            )
            raise
        except KeyboardInterrupt:
            write_new_receipt(
                FOLDER / "interrupted-v1.json",
                {
                    "interrupted_at": utc_now(),
                    "diagnostic_intent": reference(INTENT),
                    "partial_source_and_matched_records_preserved": True,
                    "new_strategy_evaluations": 0,
                    "second_attempt_allowed": False,
                },
            )
            current.progress(
                "allocation_mass_decomposition_interrupted",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pending_steps=["preserve_interrupted_fixed_attempt_without_retry"],
            )
            raise
        result = {
            "schema_version": 1,
            "diagnosis_id": DIAGNOSIS,
            "completed_at": utc_now(),
            "intent": reference(INTENT),
            "summary": summary,
            "actual_matched_input_resolved_records": reference(log),
            "actual_matched_record_count": record_count,
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
        "allocation_mass_decomposition_complete_pending_publication",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        allocation_decomposition_result=reference(RESULT),
        pending_steps=["publish_bounded_allocation_evidence_without_new_financial_rule"],
    )
    return {"diagnostic_result": reference(RESULT), "public_summary": reference(PUBLIC)}


def main():
    parser = argparse.ArgumentParser(description="Read-only recorded allocation-mass diagnosis.")
    parser.add_argument("action", choices=("freeze", "diagnose"))
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("A stored allocation diagnosis cannot request live sources.")

    sys.addaudithook(offline)
    print(json.dumps(freeze() if args.action == "freeze" else diagnose(), indent=2))


if __name__ == "__main__":
    main()
