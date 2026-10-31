from __future__ import annotations

import argparse
import ast
import json
import os
import sqlite3
import sys
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import legacy_term_risk_eligibility_v1 as rules
from us_quant import ssd_anfci_expert_eligibility_v1 as import_template
from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import file_digest, read_json, utc_now

ROOT = base.ROOT
FOLDER = base.REPORTS / "legacy-term-risk-expert-eligibility-20261031-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
RESULT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/legacy_term_risk_expert_eligibility_20261031_v1.json"
REVIEW = "ssd_legacy_term_risk_expert_eligibility_20261031_v1"
DESIGN_SHA = "f719b381c9a390498b5e1e6094c9f401575d82f5f119a513dcf70614c70263ec"
OWN = (
    "src/us_quant/legacy_term_risk_eligibility_v1.py",
    "src/us_quant/ssd_legacy_term_risk_eligibility_v1.py",
    "tests/test_legacy_term_risk_eligibility_v1.py",
    "tests/test_ssd_legacy_term_risk_eligibility_v1.py",
)
reference = base.reference


def dependencies():
    return FunctionType(
        import_template.dependencies.__code__, {**globals(), "ast": ast, "prior": base}
    )()


def source_references():
    if reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the actual pre-review eligibility scope and no-rescue rule.")
    design = read_json(DESIGN)
    if (
        design["review_id"] != REVIEW
        or design["candidate_id"] != rules.CANDIDATE
        or design["actual_continuation_event"] != 62019
        or design["actual_invocation"] != reference(INVOCATION)
        or design["maximum_actual_reviews"] != 1
        or design["actual_new_eligibility_RF_calendar_funding_checks_computed"] is not False
        or design["new_expert_admitted"] is not False
        or design["active_pool_changed"] is not False
        or design["fixed_numeric_checks"]
        != {
            "target_budget": 0.98,
            "target_sum_tolerance": 1e-12,
            "nonnegative_cash_usd": 0,
            "gross_exposure_max": 1.0,
            "book_return_NAV_reconciliation_tolerance": 1e-12,
            "existing_BT_max_error_usd": 1e-6,
        }
    ):
        raise QuantError("Keep fixed original funding limits and the actual review-only identity.")
    refs = {}
    for ref in [
        reference(DESIGN),
        reference(INVOCATION),
        reference(FOLDER / "source-public-progress-baseline-v1.json"),
        *design["source_references"],
        design["source_identity_snapshot"],
        *dependencies(),
    ]:
        if ref["path"] in refs and refs[ref["path"]] != ref:
            raise QuantError("Conflicting frozen source identities cannot become eligibility.")
        refs[ref["path"]] = ref
    result = [refs[name] for name in sorted(refs)]
    for ref in result:
        base.checked_reference(ref)
    return result


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve an existing eligibility freeze,not a new version in place.")
    xml = FOLDER / "offline-eligibility-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_cached_funding_is_separate_from_calendar_eligibility",
        "test_RF_exact_mismatch_is_explicit_without_replacement",
        "test_source_shape_budget_or_funding_failures_are_not_silent_success",
        "test_original_identity_is_preserved_without_return_rank_selection",
        "test_readers_never_replay_original_source_or_initialize_ledger",
        "test_dependency_capture_preserves_original_globals",
        "test_second_actual_review_is_terminal",
        "test_public_review_reports_blockers_not_admission_or_raw_daily_data",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed identity,calendar,RF,funding and no-replay tests.")
    own = [reference(ROOT / path) for path in OWN]
    outputs = [reference(xml), reference(FOLDER / "offline-eligibility-tests-v1.log")]
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError("Freeze only current tested eligibility code and successful output bytes.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "review_id": REVIEW,
            "actual_continuation_event": 62019,
            "design": reference(DESIGN),
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "source_references": refs,
            "passed_offline_checks": count,
            "actual_review_started": False,
            "maximum_actual_reviews": 1,
            "new_expert_admitted": False,
            "active_pool_changed": False,
            "new_strategy_evaluations": 0,
            "investment_objective_verified": False,
        },
    )
    current.progress(
        "legacy_term_expert_eligibility_rule_frozen",
        active_compute_job=None,
        term_expert_eligibility_intent=reference(INTENT),
        pending_steps=["one_cached_term_expert_eligibility_review"],
    )
    return {"intent": reference(INTENT), "source_references": len(refs), "tests": count}


def checked_intent():
    intent = read_json(INTENT)
    if (
        intent["review_id"] != REVIEW
        or intent["design"] != reference(DESIGN)
        or intent["actual_review_started"] is not False
        or intent["maximum_actual_reviews"] != 1
        or intent["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or intent["source_references"] != source_references()
    ):
        raise QuantError("The original review scope,code or frozen input identities changed.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    return intent


def legacy_snapshot(design):
    path = ROOT / "runtime/research-program.sqlite3"
    if file_digest(path) != base.LEGACY_SHA256:
        raise QuantError("Preserve the original immutable ledger.")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT c.spec_sha,c.semantic_sha,c.registered_at,c.body,r.at,r.evidence_sha,r.body "
            "FROM candidates c JOIN reviews r ON r.candidate_id=c.id WHERE c.id=?",
            (rules.CANDIDATE,),
        ).fetchone()
        events = db.execute(
            "SELECT seq,at,kind,body,sha FROM events WHERE kind IN ('candidate','review') "
            "AND json_extract(body,'$.candidate_id')=? ORDER BY seq",
            (rules.CANDIDATE,),
        ).fetchall()
        technical = db.execute(
            "SELECT count(*) FROM technical_failures WHERE candidate_id=?", (rules.CANDIDATE,)
        ).fetchone()[0]
    snapshot = read_json(base.checked_reference(design["source_identity_snapshot"]))
    if (
        row is None
        or technical
        or [e[0] for e in events] != [17, 19]
        or row[0] != snapshot["spec_sha256"]
        or row[1] != snapshot["semantic_sha256"]
        or row[2] != snapshot["registered_at"]
        or json.loads(row[3]) != snapshot["specification"]
        or row[4] != snapshot["reviewed_at"]
        or row[5] != snapshot["evidence_sha256"]
        or json.loads(row[6]) != snapshot["review"]
        or [e[4] for e in events] != [e["sha256"] for e in snapshot["events"]]
    ):
        raise QuantError("Preserve the actual original candidate17/review19 identity snapshot.")
    return snapshot


def read_inputs(design):
    snapshot = legacy_snapshot(design)
    registration = read_json(base.checked_reference(design["source_registration"]))
    results = read_json(base.checked_reference(design["source_results"]))
    bundle = read_json(base.checked_reference(design["source_bundle"]))
    policy = read_json(base.checked_reference(design["source_policy"]))
    identity = rules.check_identity(snapshot, registration, results, bundle, policy)
    frames = {}
    for row in bundle["paths"]:
        frames[row["years"], row["scenario"]] = {
            key: pd.read_csv(
                base.checked_reference(row[key]),
                index_col=0,
                parse_dates=True,
                float_precision="round_trip",
            )
            for key in ("strategy", "strategy_bt", "weights", "targets")
        }
        for ref in row.values():
            if isinstance(ref, dict) and set(ref) == {"path", "sha256"}:
                base.checked_reference(ref)
    common_rf = pd.read_csv(
        base.checked_reference(design["active_common_RF"]), index_col=0, parse_dates=True
    )
    if list(common_rf.columns) != ["risk_free"]:
        raise QuantError("Keep the existing active common RF column and reader convention.")
    return snapshot, bundle, frames, common_rf["risk_free"], identity


def public_report(result):
    return {
        "schema_version": 1,
        "review_id": REVIEW,
        "completed_at": result["completed_at"],
        "private_eligibility": reference(RESULT),
        "prereview_intent": result["intent"],
        "source_candidate_id": rules.CANDIDATE,
        "summary": result["summary"],
        "source_inputs_checked": result["source_inputs_checked"],
        "analysis_dependencies": result["analysis_dependencies"],
        "passed_offline_checks": result["passed_offline_checks"],
        "source_identity": result["source_identity"],
        "raw_native_targets_or_daily_accounts_published": False,
        "new_expert_admitted": False,
        "active_pool_changed": False,
        "new_recommendation_or_RF_view_created": False,
        "original_source_target_or_account_pipeline_replayed": False,
        "new_strategy_evaluations": 0,
        "new_source_requests": 0,
        "all_financial_counts_unchanged": result["all_financial_counts_unchanged"],
        "new_financial_configuration_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def review():
    runtime_guard()
    attempt = FOLDER / "review-attempt-v1.json"
    if attempt.exists() or RESULT.exists():
        raise QuantError("Preserve the first fixed eligibility review instead of a second pass.")
    intent = checked_intent()
    counts = current.current_counts(base.journal_events(base.JOURNAL))
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt, {"started_at": utc_now(), "pid": os.getpid(), "intent": reference(INTENT)}
        )
        current.progress(
            "running_legacy_term_expert_eligibility",
            active_compute_job={"pid": os.getpid(), "review_id": REVIEW},
        )
        try:
            snapshot, bundle, frames, rf, identity = read_inputs(read_json(DESIGN))
            summary = rules.check(snapshot, bundle, frames, rf)
            checked_intent()
            if counts != current.current_counts(base.journal_events(base.JOURNAL)):
                raise QuantError(
                    "A source-only eligibility review cannot change financial history."
                )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "reason": str(error),
                    "intent": reference(INTENT),
                    "no_source_RFFill_Adapter_or_numerical_rescue": True,
                    "new_strategy_evaluations": 0,
                },
            )
            current.progress(
                "legacy_term_expert_eligibility_blocked",
                active_compute_job=None,
                pending_steps=["preserve_first_fixed_review_failure_without_rescue"],
            )
            raise
        except KeyboardInterrupt:
            current.progress("legacy_term_expert_eligibility_interrupted", active_compute_job=None)
            raise
        result = {
            "schema_version": 1,
            "review_id": REVIEW,
            "completed_at": utc_now(),
            "intent": reference(INTENT),
            "summary": summary,
            "source_identity": identity,
            "source_inputs_checked": len(intent["source_references"]),
            "analysis_dependencies": len(dependencies()),
            "passed_offline_checks": intent["passed_offline_checks"],
            "all_financial_counts_unchanged": counts,
            "new_strategy_evaluations": 0,
            "new_expert_admitted": False,
            "active_pool_changed": False,
            "investment_objective_verified": False,
        }
        write_new_receipt(RESULT, result)
        write_new_receipt(PUBLIC, public_report(result))
    current.progress(
        "legacy_term_expert_eligibility_complete_pending_publication",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        term_expert_eligibility_result=reference(RESULT),
        pending_steps=["publish_complete_review_without_Adapter_or_pool_change"],
    )
    return {"private_eligibility": reference(RESULT), "public_summary": reference(PUBLIC)}


def main():
    parser = argparse.ArgumentParser(
        description="Read-only original term expert eligibility;no trading."
    )
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Cached term eligibility cannot request live sources.")

    sys.addaudithook(offline)
    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))


if __name__ == "__main__":
    main()
