from __future__ import annotations

import argparse
import ast
import json
import os
from xml.etree import ElementTree

import pandas as pd

from us_quant import expert_anfci_scope_v1 as scope
from us_quant import ssd_expert_payoff_persistence_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import read_json, utc_now

ROOT, base = prior.ROOT, prior.base
FOLDER = base.REPORTS / "expert-anfci-eligibility-review-20261030-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
RESULT = FOLDER / "review-result-v1.json"
PUBLIC = ROOT / "evidence/anfci_expert_eligibility_20261030_v1.json"
REVIEW = "ssd_expert_anfci_eligibility_review_20261030_v1"
DESIGN_SHA = "79059b39646a29bb186253b4a7481bdf9cd0ce4e855140935554bd6f91373e9e"
OWN = (
    "src/us_quant/expert_anfci_scope_v1.py",
    "src/us_quant/ssd_anfci_expert_eligibility_v1.py",
    "tests/test_expert_anfci_scope_v1.py",
    "tests/test_ssd_anfci_expert_eligibility_v1.py",
)


def dependencies():
    pending = [*OWN, "src/us_quant/__init__.py"]
    required = set()
    while pending:
        path = pending.pop()
        if path in required:
            continue
        required.add(path)
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (
                    [f"us_quant.{name.name}" for name in node.names]
                    if node.module == "us_quant"
                    else [node.module]
                )
            for name in names:
                local = "src/" + name.replace(".", "/") + ".py"
                if name.startswith("us_quant.") and (ROOT / local).is_file():
                    pending.append(local)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    manifest = read_json(
        base.checked_reference(
            {
                "path": "IMPORT_MANIFEST.json",
                "sha256": base.BUNDLE_SHA256,
            }
        )
    )
    prefix = "research/workbench/"
    policies = {
        path[len(prefix) :]
        for path in contract["files"]
        if path.startswith(prefix + "config/") and path.endswith(".json")
    } | {
        path
        for path in manifest["file_sha256"]
        if path.startswith("config/") and path.endswith(".json")
    }
    refs = [prior.reference(ROOT / path) for path in sorted(required | policies)]
    for ref in refs:
        original = manifest["file_sha256"].get(ref["path"])
        if original is not None and ref["sha256"] != original:
            raise QuantError("An original imported dependency cannot change during source review.")
    return refs


def source_references():
    if prior.reference(DESIGN)["sha256"] != DESIGN_SHA:
        raise QuantError("Preserve the actual pre-review source scope and no-fill decision.")
    design = read_json(DESIGN)
    if (
        design["review_id"] != REVIEW
        or design["candidate_id"] != scope.CANDIDATE
        or design["actual_continuation_event"] != 56650
        or design["actual_cached_query_or_target_review_computed"] is not False
        or design["source_full_225_admission_may_be_overridden"] is not False
        or design["original_age_limit_or_abstention_semantics_may_change"] is not False
        or design["new_expert_admitted"] is not False
    ):
        raise QuantError("Require the real bounded read-only expert eligibility identity.")
    refs = prior.unique_refs(
        [
            prior.reference(DESIGN),
            design["actual_invocation"],
            prior.reference(FOLDER / "source-public-progress-baseline-v1.json"),
            *design["source_references"],
            *dependencies(),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return refs


def freeze():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Resume an existing review intent;do not refreeze it.")
    xml = FOLDER / "offline-review-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_original_scope_and_sparse_abstention_are_preserved",
        "test_source_false_flags_and_age_limit_cannot_be_waived",
        "test_gap_cannot_be_carried_zero_filled_or_retimed",
        "test_current_original_monthly_adapter_rejects_the_same_gap",
        "test_review_preserves_all_original_inputs",
        "test_closed_financial_and_source_entrypoints_are_never_called",
        "test_public_report_has_no_source_values_targets_or_financial_metrics",
        "test_second_actual_review_attempt_is_terminal",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require source-scope,abstention,Interface and read-only review checks.")
    count = sum(int(suite.get("tests", 0)) for suite in suites)
    validation = read_json(FOLDER / "validation-source-receipt-v1.json")
    own = [prior.reference(ROOT / path) for path in OWN]
    outputs = [prior.reference(xml), prior.reference(FOLDER / "offline-review-tests-v1.log")]
    if (
        validation["implementation_and_test_inputs"] != own
        or validation["canonical_test_outputs"] != outputs
        or validation["passed_test_count"] != count
        or validation["exit_code"] != 0
    ):
        raise QuantError(
            "Bind the current tested review implementation before actual cached analysis."
        )
    refs = source_references()
    write_new_receipt(
        INTENT,
        {
            "created_at": utc_now(),
            "review_id": REVIEW,
            "design": prior.reference(DESIGN),
            "actual_continuation_event": 56650,
            "implementation_and_test_inputs": own,
            "validation_evidence": [
                *outputs,
                prior.reference(FOLDER / "validation-source-receipt-v1.json"),
            ],
            "passed_offline_checks": count,
            "source_references": refs,
            "actual_cached_review_started": False,
            "maximum_actual_review_passes": 1,
            "new_expert_admitted": False,
            "new_financial_trials": 0,
            "new_source_requests": 0,
            "investment_objective_verified": False,
        },
    )
    return {"intent": prior.reference(INTENT), "source_references": len(refs), "tests": count}


def checked_intent():
    intent = read_json(INTENT)
    if (
        intent["review_id"] != REVIEW
        or intent["actual_cached_review_started"] is not False
        or intent["maximum_actual_review_passes"] != 1
        or intent["source_references"] != source_references()
        or intent["implementation_and_test_inputs"]
        != [prior.reference(ROOT / path) for path in OWN]
    ):
        raise QuantError("Review source scope,code and validation cannot change after freeze.")
    for ref in intent["source_references"] + intent["validation_evidence"]:
        base.checked_reference(ref)
    return intent


def read_cached(design):
    source = {
        key: read_json(base.checked_reference(design[name]))
        for key, name in (
            ("readiness", "source_readiness"),
            ("manifest", "source_manifest"),
            ("validation", "source_validation"),
        )
    }
    target = pd.read_csv(
        base.checked_reference(design["source_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    decisions = {
        "targets": target,
        "proof": read_json(base.checked_reference(design["source_target_proof"])),
        "audits": read_json(base.checked_reference(design["source_information_audits"]))[
            "decisions"
        ],
    }
    return read_json(base.checked_reference(design["source_policy"])), source, decisions


def public_report(result):
    allowed = {
        "candidate_id",
        "status",
        "source_admitted_flags_preserved",
        "original_full_scope_queries",
        "original_fresh_queries",
        "original_stale_queries",
        "native_current_level_only_scope_is_blanket_source_approval",
        "native_monthly_opportunities",
        "native_fresh_target_rows",
        "native_source_abstentions",
        "first_native_target",
        "last_native_target",
        "current_dense_monthly_adapter_requires_all_post_warmup_month_ends",
        "current_dense_monthly_adapter_supports_native_target_calendar",
        "current_dense_monthly_adapter_missing_dates",
        "reason_codes",
        "no_new_request_is_rebalance_to_carried_target",
        "target_forward_filled_zero_filled_or_retimed",
        "new_availability_adapter_or_assessment_view_created",
        "original_standalone_verdict_reclassified_as_technical_invalidity",
        "new_expert_admitted",
        "current_expert_pool_changed",
        "new_financial_trials",
        "new_source_requests",
        "independent_forward_validation",
        "investment_objective_verified",
    }
    assessment = {key: result["assessment"][key] for key in sorted(allowed)}
    assessment["native_source_abstentions"] = [
        {
            key: row[key]
            for key in ("decision_session", "observation_date", "age_calendar_days", "meaning")
        }
        for row in assessment["native_source_abstentions"]
    ]
    return {
        "schema_version": 1,
        "review_id": REVIEW,
        "completed_at": result["completed_at"],
        "analysis_intent_sha256": result["analysis_intent"]["sha256"],
        "assessment": assessment,
        "original_standalone_review_preserved": result["original_standalone_review_preserved"],
        "original_technical_failures_added": 0,
        "all_current_financial_counts_unchanged": result["all_current_financial_counts_unchanged"],
        "source_generator_account_metric_or_admission_pipeline_replayed": False,
        "new_expert_admitted": False,
        "current_expert_pool_changed": False,
        "new_assessment_or_recommendation_view_artifacts": 0,
        "new_financial_trials": 0,
        "new_funded_paths": 0,
        "new_source_requests": 0,
        "original_full_scope_nonadmission_overridden": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def review():
    runtime_guard()
    attempt = FOLDER / "review-attempt-v1.json"
    if attempt.exists() or RESULT.exists():
        raise QuantError("Preserve the first actual review;no silent second cached pass.")
    checked_intent()
    counts = prior.current.current_counts(base.journal_events(base.JOURNAL))
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": prior.reference(INTENT),
            },
        )
        try:
            design = read_json(DESIGN)
            policy, source, decisions = read_cached(design)
            assessment = scope.assess_saved(policy, source, decisions)
            checked_intent()
            if counts != prior.current.current_counts(base.journal_events(base.JOURNAL)):
                raise QuantError("An eligibility review cannot change financial trial counts.")
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(exc),
                    "partial_inputs_preserved": True,
                    "source_references": prior.reference(INTENT),
                    "new_expert_admitted": False,
                    "new_financial_trials": 0,
                },
            )
            raise
        result = {
            "schema_version": 1,
            "review_id": REVIEW,
            "completed_at": utc_now(),
            "analysis_intent": prior.reference(INTENT),
            "assessment": assessment,
            "original_standalone_review_preserved": design["source_original_review"],
            "all_current_financial_counts_unchanged": counts,
            "source_refs": prior.reference(INTENT),
            "original_policy_and_targets_modified": False,
            "new_expert_admitted": False,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
        }
        write_new_receipt(RESULT, result)
        write_new_receipt(PUBLIC, public_report(result))
    prior.current.progress(
        "expert_anfci_eligibility_review_complete_not_admitted",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        expert_anfci_eligibility_review_result=prior.reference(RESULT),
        pending_steps=["publish_cached_eligibility_findings_without_source_or_pool_change"],
    )
    return {"result": prior.reference(RESULT), "public_summary": prior.reference(PUBLIC)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only cached financing-expert eligibility.")
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args(argv)
    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))


if __name__ == "__main__":
    main()
