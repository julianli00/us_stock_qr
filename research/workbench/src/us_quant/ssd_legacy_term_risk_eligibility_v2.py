from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from types import FunctionType
from xml.etree import ElementTree

from us_quant import legacy_term_risk_eligibility_v2 as rules
from us_quant import ssd_anfci_expert_eligibility_v1 as import_template
from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_legacy_term_risk_eligibility_v1 as original
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import read_json, utc_now

ROOT = base.ROOT
FOLDER = base.REPORTS / "legacy-term-risk-expert-eligibility-20261031-v2"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
RESULT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/legacy_term_risk_expert_eligibility_20261031_v2.json"
REVIEW = "ssd_legacy_term_risk_expert_eligibility_20261031_v2"
DESIGN_SHA = "392cac99e78c861e1105cca16289cd3bd093a948a4ad66094b43d70e7019487d"
OWN = (
    "src/us_quant/legacy_term_risk_eligibility_v2.py",
    "src/us_quant/ssd_legacy_term_risk_eligibility_v2.py",
    "tests/test_legacy_term_risk_eligibility_v2.py",
    "tests/test_ssd_legacy_term_risk_eligibility_v2.py",
)
reference = base.reference


def dependencies():
    return FunctionType(
        import_template.dependencies.__code__, {**globals(), "ast": ast, "prior": base}
    )()


def source_references():
    if reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the independent V2 time-range contract and source scope.")
    d = read_json(DESIGN)
    if (
        d["review_id"] != REVIEW
        or d["candidate_id"] != rules.CANDIDATE
        or d["actual_continuation_event"] != 62856
        or d["creator_directive_native_seq"] != 62625
        or d["actual_invocation"] != reference(INVOCATION)
        or d["original_V1_remains_terminal"] is not True
        or d["maximum_actual_reviews"] != 1
        or d["actual_new_V2_RF_calendar_funding_checks_computed"] is not False
        or d["new_expert_admitted"] is not False
        or d["active_pool_changed"] is not False
        or d["fixed_numeric_checks"]
        != {
            "target_budget": 0.98,
            "target_sum_tolerance": 1e-12,
            "nonnegative_cash_usd": 0,
            "gross_exposure_max": 1.0,
            "book_return_NAV_reconciliation_tolerance": 1e-12,
            "existing_BT_max_error_usd": 1e-6,
        }
    ):
        raise QuantError("Keep the actual independent review identity and original numeric limits.")
    refs = {}
    for ref in [
        reference(DESIGN),
        reference(INVOCATION),
        d["creator_directive"],
        reference(FOLDER / "source-public-progress-baseline-v1.json"),
        *d["source_references"],
        *dependencies(),
    ]:
        if ref["path"] in refs and refs[ref["path"]] != ref:
            raise QuantError("Conflicting frozen source identities cannot become V2 eligibility.")
        refs[ref["path"]] = ref
    result = [refs[name] for name in sorted(refs)]
    for ref in result:
        base.checked_reference(ref)
    return result


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve an existing V2 freeze instead of refreezing it.")
    xml = FOLDER / "offline-eligibility-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_full_prehistory_and_defined_formal_anchor_are_separate",
        "test_missing_or_partial_formal_anchor_is_not_filled",
        "test_formal_capital_or_anchor_budget_cannot_change",
        "test_RF_funding_and_native_Adapter_checks_remain_complete",
        "test_native_calendar_is_not_filtered_to_formal_window",
        "test_readers_keep_full_source_without_original_V1_check_or_review",
        "test_dependency_capture_preserves_original_globals",
        "test_second_actual_V2_review_is_terminal",
        "test_public_conclusion_does_not_claim_admission_or_create_views",
    }
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    if not required.issubset(names) or any(
        int(suite.get(k, 0)) for suite in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require complete prehistory,formal-anchor,RF,funding and no-replay tests."
        )
    own = [reference(ROOT / path) for path in OWN]
    outputs = [reference(xml), reference(FOLDER / "offline-eligibility-tests-v1.log")]
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError("Freeze actual current V2 code and successful test output bytes.")
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "review_id": REVIEW,
            "actual_continuation_event": 62856,
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
            "old_V1_remains_terminal": True,
            "new_expert_admitted": False,
            "active_pool_changed": False,
            "new_strategy_evaluations": 0,
            "investment_objective_verified": False,
        },
    )
    current.progress(
        "legacy_term_expert_v2_eligibility_rule_frozen",
        active_compute_job=None,
        term_expert_v2_intent=reference(INTENT),
        pending_steps=["one_independent_V2_cached_eligibility_review"],
    )
    return {"intent": reference(INTENT), "sources": len(refs), "tests": count}


def checked_intent():
    intent = read_json(INTENT)
    if (
        intent["review_id"] != REVIEW
        or intent["design"] != reference(DESIGN)
        or intent["actual_review_started"] is not False
        or intent["maximum_actual_reviews"] != 1
        or intent["old_V1_remains_terminal"] is not True
        or intent["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or intent["source_references"] != source_references()
    ):
        raise QuantError("The independently frozen V2 code or contract changed.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    return intent


def read_inputs(design):
    # This helper only reads immutable identity/artifacts; it never invokes the V1 numeric review.
    return original.read_inputs(design)


def public_report(result):
    return {
        "schema_version": 2,
        "review_id": REVIEW,
        "completed_at": result["completed_at"],
        "private_eligibility": reference(RESULT),
        "prereview_intent": result["intent"],
        "source_candidate_id": rules.CANDIDATE,
        "summary": result["summary"],
        "source_identity": result["source_identity"],
        "source_inputs_checked": result["source_inputs_checked"],
        "analysis_dependencies": result["analysis_dependencies"],
        "passed_offline_checks": result["passed_offline_checks"],
        "full_source_prehistory_and_all25early_requests_preserved": True,
        "formal_anchor_not_global_first_active_contract": True,
        "old_V1_failure_remains_terminal": True,
        "raw_native_targets_or_daily_accounts_published": False,
        "new_expert_admitted": False,
        "active_pool_changed": False,
        "new_recommendation_RF_view_Adapter_or_source_waiver_created": False,
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
        raise QuantError("Preserve the first independent V2 review instead of a second pass.")
    intent = checked_intent()
    counts = current.current_counts(base.journal_events(base.JOURNAL))
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt, {"started_at": utc_now(), "pid": os.getpid(), "intent": reference(INTENT)}
        )
        current.progress(
            "running_legacy_term_expert_v2_eligibility", active_compute_job={"pid": os.getpid()}
        )
        try:
            snapshot, bundle, frames, rf, identity = read_inputs(read_json(DESIGN))
            summary = rules.check(snapshot, bundle, frames, rf)
            for row in summary["paths"]:
                if (
                    row["path_key"].startswith("10y")
                    and row["native_requests_before_formal_anchor_preserved"] != 25
                ):
                    raise QuantError("The complete original25pre-anchor requests must remain.")
            checked_intent()
            if counts != current.current_counts(base.journal_events(base.JOURNAL)):
                raise QuantError("A technical-only V2 review cannot change financial history.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "reason": str(error),
                    "intent": reference(INTENT),
                    "old_V1_or_source_code_changed": False,
                    "no_Adapter_RFview_or_numerical_rescue": True,
                    "new_strategy_evaluations": 0,
                },
            )
            current.progress(
                "legacy_term_expert_v2_eligibility_blocked",
                active_compute_job=None,
                pending_steps=["preserve_first_independent_V2_failure_without_retry"],
            )
            raise
        except KeyboardInterrupt:
            current.progress(
                "legacy_term_expert_v2_eligibility_interrupted", active_compute_job=None
            )
            raise
        result = {
            "schema_version": 2,
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
        "legacy_term_expert_v2_eligibility_complete_pending_publication",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        term_expert_v2_result=reference(RESULT),
        pending_steps=["publish_full_V2_conclusion_without_views_Adapters_or_finance"],
    )
    return {"private_eligibility": reference(RESULT), "public_summary": reference(PUBLIC)}


def main():
    parser = argparse.ArgumentParser(description="Independent V2 technical-only term eligibility.")
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Cached V2 eligibility cannot request live sources.")

    sys.addaudithook(offline)
    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))


if __name__ == "__main__":
    main()
