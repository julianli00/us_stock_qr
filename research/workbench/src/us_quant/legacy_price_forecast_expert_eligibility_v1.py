from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import legacy_real_yield_expert_eligibility_v1 as native
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_macro_expert_inclusion_v1 as saved
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity

ROOT, BASE = saved.ROOT, saved.base
FOLDER = BASE.REPORTS / "legacy-price-forecast-expert-eligibility-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
SNAPSHOT = FOLDER / "source-sqlite-identity-snapshot-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
RESULT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/legacy_price_forecast_expert_eligibility_20261104_v1.json"
REVIEW = "ssd_legacy_price_forecast_expert_eligibility_20261104_v1"
CANDIDATE = "conditional_price_factor_model_v2"
GROUP = "supervised_price_conditional_budget"
ASSETS, PATHS = native.ASSETS, native.PATHS
REGISTRATION = ROOT / "evidence/conditional_factor_model_20261011_registration_v2.json"
SOURCE_RESULTS = ROOT / "evidence/conditional_factor_model_20261011_results_v2.json"
BUNDLE = ROOT / "reports/program-evaluations" / CANDIDATE / "bundle.json"
POLICY = ROOT / "config/conditional-factor-model.json"
AUDIT = ROOT / "evidence/conditional_model_training_audit_v2.json"
COMMON_RF = ROOT / "data/sector-growth-prepared-20261011/risk_free.csv"
OWN = (
    "src/us_quant/legacy_price_forecast_expert_eligibility_v1.py",
    "tests/test_legacy_price_forecast_expert_eligibility_v1.py",
    "tests/test_legacy_real_yield_expert_eligibility_v1.py",
    "tests/conftest.py",
    "LEGACY_PRICE_FORECAST_EXPERT_ELIGIBILITY_V1.md",
)


def check_native(*args):
    checker = FunctionType(
        native.check.__code__,
        {**native.check.__globals__, "CANDIDATE": CANDIDATE, "GROUP": GROUP},
    )
    return checker(*args)


def check_identity(snapshot, registration, results, bundle, policy):
    entries = [r for r in registration["candidates"] if r["spec"]["id"] == CANDIDATE]
    correction = policy["technical_correction"]
    if (
        snapshot["candidate_id"] != CANDIDATE
        or snapshot["technical_failures"] != 0
        or [e["sequence"] for e in snapshot["events"]] != [42, 44]
        or len(entries) != 1
        or entries[0]["spec"] != snapshot["specification"]
        or entries[0]["registration"]["registered_at"] != snapshot["registered_at"]
        or entries[0]["registration"]["spec_sha256"] != snapshot["spec_sha256"]
        or digest_json(snapshot["specification"]) != snapshot["spec_sha256"]
        or digest_json(bundle) != snapshot["evidence_sha256"]
        or results["candidates"][CANDIDATE] != snapshot["review"]
        or snapshot["review"]["status"] != "rejected_historical"
        or snapshot["specification"]["configuration"]
        != {"id": CANDIDATE, "augmented_information": False}
        or bundle["candidate_spec_sha256"] != snapshot["spec_sha256"]
        or bundle["market"] != snapshot["specification"]["market"]
        or set(snapshot["specification"]["asset_leverage"]) != set(ASSETS)
        or any(v != 1.0 for v in snapshot["specification"]["asset_leverage"].values())
        or policy["implementation_revision"] != 2
        or policy["training_months"] != 36
        or policy["minimum_training_months"] != 9
        or policy["model_parameters"] != {"ridge": {"alpha": 10.0, "solver": "svd"}}
        or policy["clip_current_features_to_training_range"] is not True
        or policy["minimum_predicted_excess_return"] != 0.005
        or policy["ranked_factor_shares"] != [0.35, 0.30, 0.20, 0.15]
        or policy["maximum_single_risk_sleeve"] != 0.70
        or policy["cash_reserve"] != 0.02
        or correction["economic_parameters_changed"] is not False
        or correction["strategy_return_outcomes_seen_before_correction"] is not False
        or registration["original_invalid_code_versions_retained"] is not True
        or registration["outcomes_computed"] is not False
        or bundle["order_authority"] is not False
        or bundle["leveraged_products_allowed"] is not False
        or snapshot["specification"]["leveraged_products_allowed"] is not False
        or snapshot["specification"]["order_authority"] is not False
        or not pd.Timestamp(snapshot["registered_at"])
        <= pd.Timestamp(bundle["completed_at"])
        <= pd.Timestamp(snapshot["reviewed_at"])
        or len(bundle["paths"]) != 4
        or {(p["years"], p["scenario"]) for p in bundle["paths"]} != set(PATHS)
    ):
        raise QuantError("Preserve original candidate42/review44, price-only law and V2 history.")
    return {
        "source_events": [42, 44],
        "source_registered_at": snapshot["registered_at"],
        "source_reviewed_at": snapshot["reviewed_at"],
        "source_standalone_financial_rejection_preserved": True,
        "source_V1_invalid_and_original_V2_correction_preserved": True,
        "augmented_sibling_reviewed_or_admitted": False,
    }


def check_training(audit, source_sha):
    if (
        audit["source_sha256"] != source_sha
        or audit["future_labels_used"] is not False
        or audit["source_bound_targets_verified"] is not True
        or audit["independent_forward_validation"] is not False
    ):
        raise QuantError("The original saved model audit is not a new successful fit.")
    rows = audit["models"][CANDIDATE]
    if not rows:
        raise QuantError("Missing stored training evidence cannot become a causal-model pass.")
    dates = pd.DatetimeIndex([r["decision_date"] for r in rows])
    expected = pd.DatetimeIndex([d for d in sessions(dates[0], dates[-1]) if is_month_end(d)])
    if not dates.equals(expected):
        raise QuantError("Preserve every original monthly training receipt in order.")
    for row in rows:
        first, last, known, day = [
            pd.Timestamp(row[key])
            for key in (
                "first_training_decision",
                "last_training_decision",
                "latest_label_available",
                "decision_date",
            )
        ]
        if (
            not first <= last < day
            or not last < known <= day
            or not 9 <= row["unique_training_months"] <= 36
            or row["training_rows"] != 5 * row["unique_training_months"]
            or row["hyperparameters_changed"] is not False
            or row["clipped_current_feature_values"] < 0
            or any(
                not isinstance(row[key], str)
                or len(row[key]) != 64
                or any(c not in "0123456789abcdef" for c in row[key])
                for key in (
                    "training_inputs_sha256",
                    "training_labels_sha256",
                    "current_features_sha256",
                )
            )
        ):
            raise QuantError(
                "Stored price-model labels, maturity or fixed hyperparameters changed."
            )
    return {
        "stored_training_receipts": len(rows),
        "first_stored_training_decision": str(dates[0].date()),
        "last_stored_training_decision": str(dates[-1].date()),
        "minimum_stored_mature_months": min(r["unique_training_months"] for r in rows),
        "maximum_stored_mature_months": max(r["unique_training_months"] for r in rows),
        "all_recorded_labels_available_by_decision": True,
        "hash_receipts_are_saved_fitted_coefficients": False,
        "new_fit_feature_label_or_forecast_executed": False,
        "separate_prefix_replay_executed": False,
        "audit_is_independent_forward_proof": False,
    }


def _references():
    snapshot, bundle = read_json(SNAPSHOT), read_json(BUNDLE)
    paths = [
        DESIGN,
        SNAPSHOT,
        FOLDER / "actual-invocation-v1.json",
        REGISTRATION,
        SOURCE_RESULTS,
        BUNDLE,
        POLICY,
        AUDIT,
        COMMON_RF,
        ROOT / "runtime/research-program.sqlite3",
        BUNDLE.parent / "review.json",
        ROOT / "evidence/conditional_model_numeric_diagnostic_v1.json",
        ROOT / "evidence/conditional_model_numeric_fix_v2.json",
        ROOT / "config/legacy-defensive-rotation-expert-v1.json",
        ROOT / "config/conditional-hedge-expert-v1.json",
        ROOT / "config/macro-expert-inclusion-v1.json",
    ]
    paths.extend(
        BASE.safe_file(ROOT, name, sha)
        for name, sha in snapshot["specification"]["frozen_files"].items()
    )
    paths.extend(BASE.checked_reference(r) for r in bundle["market"].values())
    paths.extend(
        BASE.checked_reference(path[key])
        for path in bundle["paths"]
        for key in ("strategy", "strategy_bt", "weights", "targets", "spy", "spy_bt")
    )
    source = saved.unique_refs([BASE.reference(p) for p in paths])
    old = read_json(
        BASE.REPORTS / "macro-extension-state-diagnosis-20261104-v1/analysis-intent-v1.json"
    )
    dependencies = saved.unique_refs(
        [
            *old["implementation_dependencies"],
            BASE.reference(ROOT / "src/us_quant/legacy_real_yield_expert_eligibility_v1.py"),
            *(BASE.reference(ROOT / name) for name in OWN),
        ]
    )
    return source, dependencies


def freeze():
    with lifecycle.research_job():
        if INTENT.exists():
            raise QuantError("Do not refreeze the complete or partial review identity.")
        os.umask(0o077)
        design = read_json(DESIGN)
        if (
            design["review_id"] != REVIEW
            or design["actual_continuation_event"] != 149729
            or design["actual_cached_numeric_review_executed"] is not False
        ):
            raise QuantError("Keep the mechanism-selected independent no-view review contract.")
        sources, dependencies = _references()
        xml, log = FOLDER / "offline-tests-v1.xml", FOLDER / "offline-tests-v1.log"
        proof = read_json(FOLDER / "validation-source-v1.json")
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            not suites
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["implementation_and_test_inputs"] != [BASE.reference(ROOT / p) for p in OWN]
            or proof["outputs"] != [BASE.reference(xml), BASE.reference(log)]
            or proof["exit_code"] != 0
        ):
            raise QuantError(
                "Bind the exact no-view source, model-clock and native funding checks."
            )
        for ref in sources + dependencies:
            BASE.checked_reference(ref)
        counts = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if counts["current_journal_event_count"] != 852:
            raise QuantError("The bounded source review starts from the actual closed852 state.")
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "review_id": REVIEW,
                "source_references": sources,
                "implementation_dependencies": dependencies,
                "validation_evidence": [
                    BASE.reference(xml),
                    BASE.reference(log),
                    BASE.reference(FOLDER / "validation-source-v1.json"),
                ],
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "financial_counts": counts,
                "actual_cached_numeric_review_executed": False,
                "maximum_actual_reviews": 1,
                "new_financial_trials": 0,
            },
        )
    return BASE.reference(INTENT)


def review():
    lifecycle.runtime_guard()
    if RESULT.exists() or (FOLDER / "review-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first review, never repair and retry.")
    intent = read_json(INTENT)
    if (
        intent["review_id"] != REVIEW
        or intent["actual_cached_numeric_review_executed"] is not False
        or intent["maximum_actual_reviews"] != 1
        or intent["package_versions"]
        != [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
    ):
        raise QuantError("The frozen reviewer or environment changed.")
    for ref in (
        intent["source_references"]
        + intent["implementation_dependencies"]
        + intent["validation_evidence"]
    ):
        BASE.checked_reference(ref)
    with lifecycle.research_job():
        os.umask(0o077)
        before = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if before != intent["financial_counts"]:
            raise QuantError("The source review cannot hide a new financial event.")
        write_new_receipt(
            FOLDER / "review-attempt-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "intent": BASE.reference(INTENT),
            },
        )
        try:
            snapshot, bundle = read_json(SNAPSHOT), read_json(BUNDLE)
            identity = check_identity(
                snapshot,
                read_json(REGISTRATION),
                read_json(SOURCE_RESULTS),
                bundle,
                read_json(POLICY),
            )
            training = check_training(
                read_json(AUDIT),
                snapshot["specification"]["frozen_files"][
                    "src/us_quant/conditional_factor_model.py"
                ],
            )
            frames = {
                (r["years"], r["scenario"]): {
                    key: pd.read_csv(
                        BASE.checked_reference(r[key]),
                        index_col=0,
                        parse_dates=True,
                        float_precision="round_trip",
                    )
                    for key in ("strategy", "strategy_bt", "weights", "targets")
                }
                for r in bundle["paths"]
            }
            common = pd.read_csv(COMMON_RF, index_col=0, parse_dates=True).iloc[:, 0]
            result = check_native(snapshot, bundle, frames, common)
            consumers = []
            for path in result["paths"]:
                key = (int(path["path_key"].split("y-")[0]), path["path_key"].split("y-")[1])
                frame = frames[key]
                if not path["unchanged_Artifact_Adapter_constructor_checks"]["monthly"]["passed"]:
                    consumers.append(
                        {
                            "path_key": path["path_key"],
                            "status": "not_observed_because_original_monthly_constructor_failed",
                            "unavailable_Interface_is_successful_score": False,
                        }
                    )
                    continue
                expert = ArchivedMonthlyExpert(
                    ExpertIdentity(CANDIDATE, GROUP, snapshot["registered_at"]),
                    frame["targets"],
                    frame["strategy"],
                    (
                        *ASSETS,
                        "TLT",
                        "QQQ",
                        "XLK",
                        "SOXX",
                        "IJR",
                        "PKW",
                    ),
                )
                dates = frame["strategy"].index
                observations = [expert.observe(day, 126) for day in (dates[124], dates[125])]
                if observations[0].score_mature or not observations[1].score_mature:
                    raise QuantError("Original125/126 maturity must not be filled or retimed.")
                consumers.append(
                    {
                        "path_key": path["path_key"],
                        "status": "unchanged_monthly_consumer_observed",
                        "observations": [len(o.net_returns) for o in observations],
                        "score_mature": [o.score_mature for o in observations],
                        "common_RF_gate_separate_from_constructor": True,
                    }
                )
            if saved.current_counts(BASE.journal_events(BASE.JOURNAL)) != before:
                raise QuantError("Expert eligibility cannot change financial trial counts.")
            for ref in intent["source_references"] + intent["implementation_dependencies"]:
                BASE.checked_reference(ref)
            packet = {
                "reviewed_at": utc_now(),
                "review_id": REVIEW,
                "intent": BASE.reference(INTENT),
                "identity": identity,
                "stored_training_evidence": training,
                "actual_consumer_checks": consumers,
                **result,
                "technical_review_is_pool_membership_or_financial_qualification": False,
                "source_model_and_augmented_sibling_not_rerun": True,
                "actual_financial_counts_unchanged": before,
                "order_authority": False,
            }
            write_new_receipt(RESULT, packet)
            write_new_receipt(PUBLIC, packet)
        except (QuantError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
            write_new_receipt(
                FOLDER / "review-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "partial_evidence_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(RESULT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One cached price-model expert eligibility review."
    )
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))
