from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np

from us_quant import expert_score_deletion_influence_v1 as law
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_expert_opinion_geometry_audit_v1 as prior
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import read_json, utc_now, write_json

ROOT = law.ROOT
FOLDER = base.REPORTS / "expert-score-deletion-influence-20261105-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_score_deletion_influence_20261105_results_v1.json"
OWN = (
    "config/expert-score-deletion-influence-v1.json",
    "src/us_quant/expert_score_deletion_influence_v1.py",
    "src/us_quant/ssd_expert_score_deletion_influence_v1.py",
    "tests/test_expert_score_deletion_influence_v1.py",
    "EXPERT_SCORE_DELETION_INFLUENCE_V1.md",
)


def dependencies():
    return FunctionType(
        prior.dependencies.__code__, {**prior.dependencies.__globals__, **globals()}
    )()


def source_inputs():
    policy = read_json(law.POLICY)
    law.validate(policy)
    closed = read_json(base.checked_reference(policy["source_completion"]))
    if (
        closed["status"] != "all_four_original_paths_rejected_and_cycle_closed"
        or closed["actual_authoritative_counts"]["current_journal_event_count"] != 1074
        or closed["actual_authoritative_counts"]["total_legacy_plus_ssd_completed_configurations"]
        != 268
        or policy["source_trace"] not in closed["exact_evidence_references"]
    ):
        raise QuantError("Use the exact closed exposed-history source, not a substitute.")
    trace = base.checked_reference(policy["source_trace"])
    return policy, closed, trace


def progress(phase, **fields):
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "score_deletion_influence_" + phase,
            "study_id": law.ID,
            "active_compute_job": None,
            **current_counts(base.journal_events(base.JOURNAL)),
            **fields,
        },
    )


def initialize():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
            raise QuantError("Preserve a complete or partial first diagnostic freeze.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        policy, closed, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v2.json")
        own = [base.reference(ROOT / p) for p in OWN]
        suites = (
            ElementTree.parse(FOLDER / "offline-tests-v2.xml").getroot().findall(".//testsuite")
        )
        if (
            validation["implementation_and_test_inputs"] != own
            or validation["exit_code"] != 0
            or validation["ruff_returncode"] != 0
            or validation["synthetic_only"] is not True
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Freeze the exact tested retained/synthetic Interface and moment oracle."
            )
        for ref in closed["exact_evidence_references"]:
            base.checked_reference(ref)
        refs = [
            policy["source_completion"],
            policy["source_trace"],
            *[
                base.reference(FOLDER / p)
                for p in (
                    "baseline-v1.json",
                    "design-decision-v1.json",
                    "actual-invocation-v1.json",
                    "validation-source-v1.json",
                    "offline-tests-v1.xml",
                    "offline-tests-v1.log",
                    "offline-ruff-v1.log",
                    "validation-source-v2.json",
                    "offline-tests-v2.xml",
                    "offline-tests-v2.log",
                    "offline-ruff-v2.log",
                    "preflight-failure-explanation-v1.json",
                )
            ],
        ]
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "study_id": law.ID,
                "criteria": policy,
                "implementation_and_test_inputs": own,
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_new_deletion_calculation_executed": False,
                "new_financial_evaluations": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="one_actual_read_only_conditional_diagnosis")
    return base.reference(INTENT)


def _case(result):
    baseline = result["conditional_full_sample_winner"]
    decisions = result["deletions"]
    usable = baseline is not None
    return {
        "decision_session": result["decision_session"],
        "state": result["state"],
        "source_selected_pair": result["source_selected_pair"],
        "conditional_full_sample_winner": baseline,
        "source_vs_kernel_baseline_winner_differs": result[
            "source_vs_kernel_baseline_winner_differs"
        ],
        "source_key_bucket_differences": result["source_key_bucket_differences"],
        "deletion_experiments": len(decisions),
        "changed_vs_kernel_baseline": (
            sum(row["conditional_winner"] != baseline for row in decisions) if usable else None
        ),
        "changed_vs_recorded_source_winner": (
            sum(row["conditional_winner"] != result["source_selected_pair"] for row in decisions)
            if result["source_selected_pair"] is not None
            else None
        ),
        "training_changes_vs_kernel_baseline": (
            sum(row["conditional_winner"] != baseline for row in decisions[:84]) if usable else None
        ),
        "validation_changes_vs_kernel_baseline": (
            sum(row["conditional_winner"] != baseline for row in decisions[84:]) if usable else None
        ),
        "deletions_with_no_scorable_conditional_pair": sum(
            row["no_scorable_conditional_pair"] for row in decisions
        ),
        "deletion_pair_degenerate_sample_cases": sum(
            len(row["eligible_pairs_with_degenerate_deleted_sample"]) for row in decisions
        ),
        "maximum_independent_score_gap": result["maximum_independent_score_gap"],
        "maximum_source_baseline_score_gap": result["maximum_source_baseline_score_gap"],
        "oracle_sample_score_checks": result["oracle_sample_score_checks"],
    }


def _summary(cases):
    scored = [c for c in cases if c["conditional_full_sample_winner"] is not None]
    if not scored:
        raise QuantError("The fixed source must retain actual scored catalogue cases.")
    changes = [c["changed_vs_kernel_baseline"] for c in scored]
    return {
        "all_source_decisions": len(cases),
        "source_unscored_origins": sum(c["state"] == "source_unscored" for c in cases),
        "mature_cases_with_full_sample_kernel_winner": len(scored),
        "conditional_deletion_experiments": sum(c["deletion_experiments"] for c in cases),
        "decisions_with_any_single_return_winner_change": sum(v > 0 for v in changes),
        "changed_vs_kernel_baseline_total": sum(changes),
        "changed_vs_recorded_source_winner_total": sum(
            c["changed_vs_recorded_source_winner"] for c in scored
        ),
        "training_single_return_winner_changes": sum(
            c["training_changes_vs_kernel_baseline"] for c in scored
        ),
        "validation_single_return_winner_changes": sum(
            c["validation_changes_vs_kernel_baseline"] for c in scored
        ),
        "changes_per_scored_decision": {
            "minimum": min(changes),
            "median": float(np.median(changes)),
            "maximum": max(changes),
        },
        "full_sample_source_vs_kernel_winner_disagreement_cases": sum(
            c["source_vs_kernel_baseline_winner_differs"] for c in scored
        ),
        "full_sample_pair_key_bucket_differences": sum(
            c["source_key_bucket_differences"] for c in scored
        ),
        "deletions_with_no_scorable_conditional_pair": sum(
            c["deletions_with_no_scorable_conditional_pair"] for c in cases
        ),
        "deletion_pair_degenerate_sample_cases": sum(
            c["deletion_pair_degenerate_sample_cases"] for c in cases
        ),
        "maximum_independent_score_gap": max(c["maximum_independent_score_gap"] for c in scored),
        "maximum_source_baseline_score_gap": max(
            c["maximum_source_baseline_score_gap"] for c in scored
        ),
        "independent_sample_score_checks": sum(c["oracle_sample_score_checks"] for c in cases),
        "deletion_cases_are_independent_trials_or_probabilities": False,
        "conditional_ranking_changes_are_actual_turnover": False,
        "covariance_refitted_or_source_policy_replayed": False,
        "new_trading_policy_membership_or_financial_acceptance": False,
        "new_financial_evaluations": 0,
    }


def run():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        attempt = FOLDER / "actual-calculation-started-v1.json"
        if attempt.exists() or RESULTS.exists():
            raise QuantError("Preserve a completed or partial diagnosis, never replay it.")
        frozen = read_json(INTENT)
        for ref in (
            frozen["implementation_and_test_inputs"]
            + frozen["complete_dependency_inputs"]
            + frozen["source_references"]
        ):
            base.checked_reference(ref)
        if frozen["package_versions"] != [
            list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
        ]:
            raise QuantError("The exact accepted environment changed after freezing.")
        policy, closed, trace = source_inputs()
        before_counts = current_counts(base.journal_events(base.JOURNAL))
        if before_counts != closed["actual_authoritative_counts"]:
            raise QuantError("Use the actual unchanged J1074/268 baseline.")
        calendar = sessions("2016-10-05", "2026-10-05")
        dates = [str(d.date()) for i, d in enumerate(calendar) if i == 0 or is_month_end(d)]
        if len(dates) != policy["source_decisions"]:
            raise QuantError("Retain the complete original decision calendar.")
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "PID": os.getpid(),
                "diagnostic_freeze": base.reference(INTENT),
            },
        )
        output = FOLDER / "actual-diagnostic-inputs-resolved-v1.jsonl"
        cases = []
        try:
            with trace.open() as old, output.open("x") as retained:
                for index, day in enumerate(dates):
                    lifecycle.runtime_guard()
                    before, after = json.loads(old.readline()), json.loads(old.readline())
                    if before["decision_index"] != index or before["as_of"] != day:
                        raise QuantError("Retain source ordering and every unscored origin.")
                    retained.write(
                        json.dumps(
                            {
                                "record": "diagnostic_input",
                                "source_before": before,
                                "source_after": after,
                            },
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    retained.flush()
                    os.fsync(retained.fileno())
                    result = law.diagnose(before, after, policy)
                    retained.write(
                        json.dumps(
                            {
                                "record": "diagnostic_resolved",
                                "decision_index": index,
                                "result": result,
                            },
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    retained.flush()
                    os.fsync(retained.fileno())
                    cases.append(_case(result))
                if old.readline():
                    raise QuantError("Unexpected trailing retained source records.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "actual-diagnostic-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "partial_records_preserved": True,
                    "actual_source": policy["source_completion"],
                },
            )
            raise
        if current_counts(base.journal_events(base.JOURNAL)) != before_counts:
            raise QuantError("A read-only diagnosis cannot add financial journal entries.")
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "study_id": law.ID,
                "diagnostic_freeze": base.reference(INTENT),
                "source_completion": policy["source_completion"],
                "source_trace": policy["source_trace"],
                "actual_input_resolved_records": base.reference(output),
                "cases": cases,
                "summary": _summary(cases),
                "actual_authoritative_counts": before_counts,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress("diagnosis_complete", next_step="publish_descriptive_evidence_and_close")
    return base.reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        if PUBLIC.exists():
            raise QuantError("Preserve the complete first public projection without rewriting.")
        saved = read_json(RESULTS)
        base.checked_reference(saved["actual_input_resolved_records"])
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "retained_conditional_return_deletion_diagnosis",
                "study_id": law.ID,
                "completed_at": saved["completed_at"],
                "source_completion": saved["source_completion"],
                "source_trace_sha256": saved["source_trace"]["sha256"],
                "private_results_sha256": base.reference(RESULTS)["sha256"],
                "diagnostic_freeze_sha256": saved["diagnostic_freeze"]["sha256"],
                "summary": saved["summary"],
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "raw_daily_inputs_pair_profiles_or_per_date_results_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return base.reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only fixed-catalogue score influence, no accounts."
    )
    parser.add_argument("action", choices=("freeze", "run", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": initialize, "run": run, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
