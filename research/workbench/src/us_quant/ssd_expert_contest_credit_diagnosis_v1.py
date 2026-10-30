from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree

from us_quant import expert_contest_credit_diagnosis_v1 as rules
from us_quant import ssd_expert_contextual_contest_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import digest_json, read_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "reports/ssd-continuous-research-v1/expert-contest-credit-diagnosis-20261030-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/expert_contest_credit_diagnosis_20261030_v1.json"
TRACE = source.TARGETS / "learning-input-output-records.jsonl"
DIAGNOSIS = "ssd_expert_contest_credit_diagnosis_20261030_v1"
DESIGN_SHA = "3b8280a94326b6baa46ef71190aa05da5629892bd651bfdf36a514016b930886"
OWN = (
    "src/us_quant/expert_contest_credit_diagnosis_v1.py",
    "src/us_quant/ssd_expert_contest_credit_diagnosis_v1.py",
    "tests/test_expert_contest_credit_diagnosis_v1.py",
    "tests/test_ssd_expert_contest_credit_diagnosis_v1.py",
)


def source_references():
    if source.reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the preanalysis credit question,formula and stopping condition.")
    design = read_json(DESIGN)
    if (
        design["diagnosis_id"] != DIAGNOSIS
        or design["actual_continuation_event"] != 55393
        or design["new_diagnostic_scores_or_support_summaries_computed"] is not False
        or design["new_financial_trials"] != 0
    ):
        raise QuantError("Require the real fixed read-only diagnostic identity.")
    packet = read_json(source.PACKET)
    frozen = packet["entries"][0]["specification"]["frozen"]
    if digest_json(frozen) != "c6c039b228be7d24163ba74a0343ab1424ad294c627c0a2a52c70ae16c4caaab":
        raise QuantError("The completed contextual study is not an editable diagnostic input.")
    old_intent = read_json(source.INTENT)
    refs = source.unique_refs(
        [
            source.reference(DESIGN),
            design["actual_invocation"],
            design["source_public_progress_baseline"],
            *design["source_references"],
            *old_intent["source_references"],
            *old_intent["model_and_test_files"],
            *frozen["code"],
            *frozen["policies"],
        ]
    )
    for ref in refs:
        source.base.checked_reference(ref)
    return refs


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("An existing diagnostic intent must be resumed,not refrozen.")
    xml = FOLDER / "offline-diagnostic-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_exact_prequential_loss_uses_prior_not_assimilated_posterior",
        "test_tied_credit_uses_expected_multiclass_loss_not_zero_distribution_loss",
        "test_unmatured_main_assessment_does_not_erase_recorded_credit_forecast",
        "test_unseen_context_retains_declared_prior_and_actual_support",
        "test_declared_initial_partial_is_explicitly_unscored",
        "test_credit_clock_count_posterior_and_maturity_tampering_fails",
        "test_following_context_and_applied_risk_weights_do_not_leak_into_current_score",
        "test_record_adapter_preserves_exact_four_record_groups",
        "test_record_adapter_rejects_truncation_duplicate_or_altered_allocation",
        "test_source_parsing_does_not_call_closed_entrypoints",
        "test_public_report_does_not_export_monthly_vectors_or_raw_records",
        "test_second_actual_analysis_attempt_is_terminal",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require complete exact loss,clock,stream and closed-entrypoint checks.")
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    own = [source.reference(ROOT / path) for path in OWN]
    outputs = [source.reference(xml), source.reference(FOLDER / "offline-diagnostic-tests-v1.log")]
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError("Bind the current tested implementation before reading real assessments.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "schema_version": 1,
            "diagnosis_id": DIAGNOSIS,
            "created_at": utc_now(),
            "design": source.reference(DESIGN),
            "actual_continuation_event": 55393,
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                source.reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "passed_offline_tests": count,
            "source_references": refs,
            "actual_analysis_started": False,
            "maximum_actual_stream_passes": 1,
            "new_financial_trials": 0,
            "new_source_requests": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
        },
    )
    return {"intent": source.reference(INTENT), "source_references": len(refs), "tests": count}


def read_decisions(path: Path):
    group = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuantError(f"Malformed saved contest trace at line{line_number}.") from exc
            if not isinstance(row, dict):
                raise QuantError(f"Saved contest trace line{line_number} is not a record.")
            group.append(row)
            if len(group) != 4:
                continue
            before, first, second, whole = group
            if (
                [row.get("stage") for row in group]
                != ["input", "allocation_resolved", "allocation_resolved", "resolved"]
                or [row.get("candidate_id") for row in (first, second)] != list(rules.IDS)
                or any(row.get("session") != before.get("session") for row in group)
                or not isinstance(whole.get("audit"), dict)
                or whole["audit"].get("decision_session") != before.get("session")
                or not isinstance(whole["audit"].get("allocations"), dict)
                or any(
                    row.get("allocation") != whole["audit"]["allocations"].get(name)
                    for row, name in zip((first, second), rules.IDS, strict=True)
                )
            ):
                raise QuantError(f"Changed input/resolved alignment at saved line{line_number}.")
            yield before, whole["audit"]
            group = []
    if group:
        raise QuantError("Retain a truncated trace as failure,not a shorter successful assessment.")


def checked_intent():
    intent = read_json(INTENT)
    if (
        intent["diagnosis_id"] != DIAGNOSIS
        or intent["actual_analysis_started"] is not False
        or intent["maximum_actual_stream_passes"] != 1
        or intent["source_references"] != source_references()
        or intent["implementation_and_test_inputs"]
        != [source.reference(ROOT / path) for path in OWN]
    ):
        raise QuantError("Diagnostic implementation,validation and source inputs cannot change.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        source.base.checked_reference(ref)
    return intent


def public_report(result: dict):
    return {
        "schema_version": 1,
        "diagnosis_id": DIAGNOSIS,
        "completed_at": result["completed_at"],
        "analysis_intent_sha256": result["analysis_intent"]["sha256"],
        "source_completed_contextual_registration": source.reference(source.PACKET),
        "source_complete_rejected_results": source.reference(source.PUBLIC_RESULTS),
        "source_decisions_preserved": result["source_decisions_preserved"],
        "source_trace_records_preserved": result["source_trace_records_preserved"],
        "complete_next_month_pairs": result["complete_next_month_pairs"],
        "initial_partial_excluded": result["initial_partial_excluded"],
        "final_forecast_without_full_outcome": result["final_forecast_without_full_outcome"],
        "aggregates": result["aggregates"],
        "exact_integer_and_rational_operands_preserved": True,
        "forecast_taken_before_next_credit_assimilation": True,
        "credit_forecast_interpretation_introduced_by_this_diagnostic": True,
        "applied_risk_tilt_weights_scored_as_probabilities": False,
        "posterior_availability_is_main_risk_score_maturity": False,
        "new_source_return_context_credit_risk_target_or_account_pipeline": False,
        "alternative_priors_contexts_scores_or_parameters_selected": False,
        "statistical_significance_or_independent_forecast_calibration_claimed": False,
        "financial_cost_return_or_qualification_improvement_established": False,
        "journal_derived_counts_unchanged": result["journal_derived_counts_unchanged"],
        "new_financial_trials": 0,
        "new_funded_paths": 0,
        "new_source_requests": 0,
        "order_authority": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }


def diagnose():
    runtime_guard()
    attempt = FOLDER / "analysis-attempt-v1.json"
    if attempt.exists() or RESULT.exists():
        raise QuantError("Preserve the first actual diagnostic attempt;do not silently retry.")
    checked_intent()
    old_counts = source.current.current_counts(source.base.journal_events(source.base.JOURNAL))
    expected_audits = read_json(source.TARGETS / "information-audits.json")["decisions"]
    rows, excluded, previous, pending = [], [], None, None
    decisions = 0
    output = FOLDER / "prequential-rows-v1.jsonl"
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": source.reference(INTENT),
            },
        )
        try:
            with output.open("x") as stream:
                for inputs, audit in read_decisions(TRACE):
                    runtime_guard()
                    pending = {
                        "previous_audit": previous,
                        "current_input": inputs,
                        "current_audit": audit,
                    }
                    if decisions >= len(expected_audits) or audit != expected_audits[decisions]:
                        raise QuantError(
                            "The stream and saved complete information audit disagree."
                        )
                    if previous is None:
                        rules.snapshot(audit)
                        if (
                            audit["decision_session"] != "2016-10-05"
                            or audit["complete_learning_months"] != 0
                            or inputs.get("previous_decision") is not None
                        ):
                            raise QuantError(
                                "Start at the real original initial decision,not warmup."
                            )
                    else:
                        result = rules.assess_transition(previous, inputs, audit)
                        stream.write(
                            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
                        )
                        stream.flush()
                        if result["status"] == "scored_complete_month":
                            rows.append(result)
                        else:
                            excluded.append(result)
                    previous = audit
                    decisions += 1
            if (
                decisions != 121
                or len(rows) != 119
                or len(excluded) != 1
                or previous["decision_session"] != "2026-09-30"
                or previous["complete_learning_months"] != 119
            ):
                raise QuantError(
                    "Preserve all121decisions,119fulloutcomes and the final unscored month."
                )
            checked_intent()
            counts = source.current.current_counts(source.base.journal_events(source.base.JOURNAL))
            if counts != old_counts:
                raise QuantError(
                    "A read-only credit diagnosis cannot change financial trial counts."
                )
        except QuantError as exc:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(exc),
                    "pending_input": pending,
                    "completed_pairs": len(rows),
                    "partial_outputs_retained": True,
                    "source_entrypoints_called": False,
                    "new_financial_trials": 0,
                },
            )
            raise
        aggregates = {
            "all_complete_pairs": rules.summarize(rows),
            "main_assessment_mature": rules.summarize(
                [row for row in rows if row["forecast_main_assessment_mature"]]
            ),
            "main_assessment_unmatured": rules.summarize(
                [row for row in rows if not row["forecast_main_assessment_mature"]]
            ),
            "by_forecast_context": {
                str(context): rules.summarize(
                    [row for row in rows if row["forecast_context"] == context]
                )
                for context in range(4)
            },
        }
        result = {
            "schema_version": 1,
            "diagnosis_id": DIAGNOSIS,
            "completed_at": utc_now(),
            "analysis_intent": source.reference(INTENT),
            "source_decisions_preserved": decisions,
            "source_trace_records_preserved": 4 * decisions,
            "complete_next_month_pairs": len(rows),
            "initial_partial_excluded": excluded,
            "final_forecast_without_full_outcome": {
                "decision_session": previous["decision_session"],
                "complete_following_month_available": False,
                "included_in_loss_means": False,
            },
            "aggregates": aggregates,
            "private_per_month_rows": source.reference(output),
            "journal_derived_counts_unchanged": counts,
            "source_entrypoints_called": False,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
        }
        write_new_receipt(RESULT, result)
        write_new_receipt(PUBLIC, public_report(result))
    source.current.progress(
        "expert_contest_credit_diagnostic_complete",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        expert_contest_credit_diagnostic_result=source.reference(RESULT),
        pending_steps=["publish_sanitized_credit_diagnosis_without_new_financial_configuration"],
    )
    return {"result": source.reference(RESULT), "public_summary": source.reference(PUBLIC)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen read-only next-month credit assessment.")
    parser.add_argument("action", choices=("freeze", "diagnose"))
    args = parser.parse_args(argv)
    print(json.dumps(freeze() if args.action == "freeze" else diagnose(), indent=2))


if __name__ == "__main__":
    main()
