from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import hmm_session_expert_checks_v1 as checks
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_research_v1 as base
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json
from us_quant.strategy_experts_v1 import load_expert

ROOT = base.ROOT
FOLDER = base.REPORTS / "archived-hmm-session-expert-eligibility-20261105-v1"
POLICY = ROOT / "config/archived-hmm-session-eligibility-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
RESULT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/archived_hmm_session_expert_eligibility_20261105_v1.json"
IDS = (
    "hmm_core_close_momentum_control_v1",
    "hmm_core_overnight_momentum_v1",
    "hmm_core_intraday_momentum_v1",
)
METHODS = ("close_to_close", "overnight", "intraday")
UNIVERSE = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)
OWN = (
    "config/archived-hmm-session-eligibility-v1.json",
    "src/us_quant/hmm_session_expert_checks_v1.py",
    "src/us_quant/archived_hmm_session_eligibility_v1.py",
    "tests/test_hmm_session_expert_checks_v1.py",
    "ARCHIVED_HMM_SESSION_ELIGIBILITY_V1.md",
    "tests/conftest.py",
)


def unique_refs(rows):
    found = {}
    for row in rows:
        if row["path"] in found and found[row["path"]] != row:
            raise QuantError("No conflicting original source reference may be selected.")
        found[row["path"]] = row
    return [found[key] for key in sorted(found)]


reference = base.reference


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_archived_hmm_session_expert_eligibility_20261105_v1",
        "prior_journal_sequence": 1144,
        "prior_completed_configurations": 274,
        "input_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "archived-hmm-session-expert-eligibility-20261105-v1/source-input-catalogue-v1.json"
            ),
            "sha256": "07edf662ad161e27c32d1031dd301bcd0e6e490abe4b31f739729105f9e5c70b",
        },
        "candidate_ids": list(IDS),
        "mechanism_group": "conditional_variance_growth",
        "native_monthly_opinions": 122,
        "original_calendar_opportunities": 131,
        "undefined_prehistory_months": 9,
        "first_native_opinion": "2016-08-31",
        "native_source_price_width": 12,
        "consumer_universe": list(UNIVERSE),
        "source_core_symbols": list(checks.CORE),
        "source_score_methods": list(METHODS),
        "minimum_actual_core_styles_per_path": 3,
        "original_independent_BT_tolerance_USD": 1e-6,
        "original_target_proof_tolerance": 1e-12,
        "source_input_return_sessions": 105,
        "source_input_recent_sessions_skipped": 21,
        "native_RF_comparison": (
            "exact_current_canonical_review_market_default_reader_all_four_books"
        ),
        "RF_rounding_normalization_cropping_or_new_assessment_view": False,
        "unrepresented_IJR_PKW_exposures": (
            "existing_ArchivedMonthlyExpert_structural_zeros_not_new_source_targets"
        ),
        "HMM_fits_states_session_scores_ranks_or_targets_recomputed": False,
        "new_shadow_or_funded_accounts": 0,
        "financial_return_performance_is_technical_admission_gate": False,
        "admitted_variants_are_independent_risk_families": False,
        "representative_or_pool_member_selected": False,
        "new_financial_evaluations": 0,
        "new_source_requests": 0,
        "source_financial_verdicts_or_global_rosters_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Preserve all original source/RF/technical standards, not a financial rescue."
        )


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def inputs():
    policy = read_json(POLICY)
    validate(policy)
    catalogue = read_json(base.checked_reference(policy["input_catalogue"]))
    packet = read_json(base.checked_reference(catalogue["source_registration"]))
    refs = [policy["input_catalogue"], *catalogue["source_references"]]
    if set(catalogue["descriptors"]) != set(IDS) or {
        e["candidate_id"] for e in packet["entries"]
    } != set(IDS):
        raise QuantError("Review all three fixed original sibling identities, not a winner.")
    for entry in packet["entries"]:
        proof = read_json(base.checked_reference(entry["target_proof"]["proof"]))
        refs.extend([proof["score_audits"], entry["specification"]["frozen"]["source_targets"]])
    refs = unique_refs(refs)
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, packet, refs


def progress(phase, **extra):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in extra and extra[k] != v for k, v in counts.items()):
        raise QuantError("Eligibility cannot override actual financial journal totals.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "study_id": read_json(POLICY)["study_id"],
            "phase": "hmm_session_eligibility_" + phase,
            "active_compute_job": None,
            **counts,
            **extra,
            "new_financial_evaluations_in_this_study": 0,
            "investment_goal_met": False,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first session-source review freeze.")
    with lifecycle.research_job():
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 171975
            or design["numeric_eligibility_review_performed"]
        ):
            raise QuantError("Declare all three source roles before numerical review.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, refs = inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Bind the exact successful source/funding/RF/consumer checks.")
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts["current_journal_event_count"] != 1144 or counts["new_pending_candidates"] != 0:
            raise QuantError("Freeze at the real J1144/274 closed financial baseline.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "criteria": read_json(POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "actual_counts": counts,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "numeric_eligibility_review_performed": False,
                "new_financial_evaluations": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="one_read_only_review_of_all_three_saved_siblings")
    return reference(INTENT)


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def inspect_source(identifier, method, catalogue, packet, canonical_rf):
    original_descriptor = catalogue["descriptors"][identifier]
    record = {**original_descriptor, "results": catalogue["source_public_results"]}
    entry = next(e for e in packet["entries"] if e["candidate_id"] == identifier)
    original = catalogue["original_source_events"][identifier]
    registered = [e for e in original if e["kind"] == "registered"]
    reviewed = [e for e in original if e["kind"] == "review"]
    proved = [e for e in original if e["kind"] == "target_proof"]
    public = read_json(base.checked_reference(catalogue["source_public_results"]))
    fences = catalogue["original_publication_fences"]
    if (
        len(registered) != 1
        or len(reviewed) != 1
        or len(proved) != 1
        or len(fences) != 1
        or entry["registered_event_sha256"] != registered[0]["sha256"]
        or entry["registered_at"] != registered[0]["at"]
        or registered[0]["body"] != entry["specification"]
        or proved[0]["body"] != entry["target_proof"]
        or fences[0]["body"]["publication"] != public["actual_publication_before_accounts"]
        or not registered[0]["sequence"]
        < proved[0]["sequence"]
        < fences[0]["sequence"]
        < reviewed[0]["sequence"]
        or reviewed[0]["body"]["status"] not in {"rejected", "historical_point_gates_passed"}
    ):
        raise QuantError(
            "Require unique unchanged original registration/proof/publication chronology."
        )
    frozen = entry["specification"]["frozen"]
    proof = read_json(base.checked_reference(entry["target_proof"]["proof"]))
    parent, child = frame(frozen["source_targets"]), frame(entry["target_proof"]["targets"])
    composition = checks.check_composition(
        parent,
        child,
        read_json(base.checked_reference(proof["score_audits"]))["decisions"],
        proof,
        method,
    )
    starts = [e for e in original if e["kind"] == "path_started"]
    ends = [e for e in original if e["kind"] == "path_completed"]
    if len(starts) != 4 or len(ends) != 4:
        raise QuantError("Keep exactly four original complete funded source paths.")
    rf_reports = {}
    for key, ref in record["completed_paths"].items():
        begin = next(e for e in starts if e["body"]["path_key"] == key)
        end = next(e for e in ends if e["body"]["path_key"] == key)
        receipt = base.verify_path(ref, frozen, identifier, key)
        if (
            end["body"]["receipt"] != ref
            or not fences[0]["sequence"]
            < begin["sequence"]
            < end["sequence"]
            < reviewed[0]["sequence"]
            or receipt["broker_orders"] != 0
            or not 0 <= receipt["independent_bt_max_equity_difference_usd"] <= 1e-6
        ):
            raise QuantError("An invalid account, delayed publication or BT cannot be admitted.")
        artifacts = {r["path"].rsplit("/", 1)[-1]: r for r in receipt["artifacts"]}
        account, weights = frame(artifacts["account.csv"]), frame(artifacts["weights.csv"])
        if not weights.columns.equals(child.columns):
            raise QuantError("Keep the full original twelve source coordinates.")
        expected = sessions("2016-10-06" if key.startswith("10") else "2021-10-06", "2026-10-05")
        rf_reports[key] = checks.check_book(account, weights, expected, canonical_rf)
    evidence = {
        "candidate_id": identifier,
        "method": method,
        "source_descriptor": record,
        "original_private_financial_result_preserved": catalogue["source_results"],
        "existing_Interface_uses_original_complete_public_review_summary": True,
        "mechanism_group": "conditional_variance_growth",
        "introduced_at": entry["registered_at"],
        "financial_reviewed_at": reviewed[0]["at"],
        "source_financial_verdict_preserved": reviewed[0]["body"]["status"],
        "source_composition": composition,
        "source_books": rf_reports,
        "source_native_universe_width": 12,
        "consumer_universe_width": 14,
        "new_RF_or_recommendation_view": False,
        "original_HMM_scores_targets_or_accounts_replayed": False,
    }
    if any(row["native_RF_exact_common"] is not True for row in rf_reports.values()):
        return None, {
            **evidence,
            "status": "blocked_native_RF_mismatch_no_adapter_or_view",
            "technical_expert_admitted": False,
        }
    expert, technical = load_expert(record, UNIVERSE)
    for day in (
        child.index[0],
        pd.Timestamp("2016-10-05"),
        pd.Timestamp("2017-04-05"),
        pd.Timestamp("2017-04-06"),
        pd.Timestamp("2026-10-05"),
    ):
        observed = expert.observe(day, 126)
        if not observed.risk_free.equals(canonical_rf.loc[observed.risk_free.index]):
            raise QuantError("Keep exact canonical native RF at every archived observation.")
        if observed.target is not None and (
            tuple(observed.target.index) != UNIVERSE
            or not (observed.target.loc[["IJR", "PKW"]] == 0).all()
        ):
            raise QuantError("Absent source coordinates are structural zeros, not new opinions.")
    boundary = [
        expert.observe(pd.Timestamp(day), 126) for day in ("2016-10-05", "2017-04-05", "2017-04-06")
    ]
    if (
        expert.observe(child.index[0], 126).target is not None
        or [len(o.net_returns) for o in boundary] != [0, 125, 126]
        or [o.score_mature for o in boundary] != [False, False, True]
    ):
        raise QuantError("Unavailable prehistory or price warmup cannot become mature NET history.")
    return expert, {
        **evidence,
        "status": "technical_eligibility_passed_admissible_nonselected",
        "technical_expert_admitted": True,
        "saved_source_metadata": technical,
        "initial_NET_observations": 0,
        "score_maturity_boundaries": [0, 125, 126],
        "absent_consumer_coordinates_are_exact_structural_zeros": True,
    }


def review():
    lifecycle.runtime_guard()
    if RESULT.exists() or (FOLDER / "review-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first review without retry.")
    with lifecycle.research_job():
        intent = read_json(INTENT)
        for ref in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        if (
            intent["package_versions"]
            != [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
            or read_json(POLICY) != intent["criteria"]
        ):
            raise QuantError("The exact reviewer or accepted pinned environment changed.")
        catalogue, packet, refs = inputs()
        if refs != intent["source_references"]:
            raise QuantError("The original archival source inventory changed.")
        write_new_receipt(
            FOLDER / "review-attempt-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "intent": reference(INTENT)},
        )
        data = review_market({"market": catalogue["canonical_current_market"]}, ROOT)
        if tuple(data.close.columns) != UNIVERSE:
            raise QuantError("Use the unchanged full current canonical fourteen-asset RF context.")
        candidates = {}
        for identifier, method in zip(IDS, METHODS, strict=True):
            lifecycle.runtime_guard()
            write_new_receipt(
                FOLDER / f"{identifier}-review-input-v1.json",
                {
                    "captured_at": utc_now(),
                    "descriptor": catalogue["descriptors"][identifier],
                    "canonical_current_market": catalogue["canonical_current_market"],
                },
            )
            try:
                _, evidence = inspect_source(identifier, method, catalogue, packet, data.risk_free)
                row = {
                    **evidence,
                    "admitted_at": utc_now() if evidence["technical_expert_admitted"] else None,
                }
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                row = {
                    "candidate_id": identifier,
                    "method": method,
                    "status": "blocked_original_technical_source_or_Interface_check",
                    "technical_expert_admitted": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_or_RF_repaired": False,
                }
            write_new_receipt(FOLDER / f"{identifier}-result-v1.json", row)
            candidates[identifier] = row
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts != intent["actual_counts"]:
            raise QuantError("A technical source review cannot hide a new financial event.")
        result = {
            "completed_at": utc_now(),
            "study_id": read_json(POLICY)["study_id"],
            "eligibility_intent": reference(INTENT),
            "candidates": candidates,
            "actual_authoritative_counts": counts,
            "global_pool_changed": False,
            "source_financial_verdicts_preserved": True,
            "admitted_variants_are_independent_risk_families": False,
            "new_financial_evaluations": 0,
            "new_source_requests": 0,
            "HMM_scores_targets_or_accounts_replayed": False,
            "RF_rounding_normalization_or_new_view": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_new_receipt(RESULT, result)
        write_new_receipt(PUBLIC, result)
        progress(
            "review_complete_awaiting_consumer_or_closure",
            complete_public_review=reference(PUBLIC),
            next_step="only_passed_sources_may_use_saved_consumer_no_RF_or_source_repair",
        )
    return reference(PUBLIC)


def load_admitted(admission_reference, identifier, universe, risk_free):
    if (
        identifier not in IDS
        or universe != UNIVERSE
        or not isinstance(risk_free, pd.Series)
        or not isinstance(risk_free.index, pd.DatetimeIndex)
        or not risk_free.index.is_unique
        or not risk_free.index.is_monotonic_increasing
        or risk_free.index.tz is not None
    ):
        raise QuantError("Do not select unknown sources, remap assets or replace the RF Interface.")
    result = read_json(base.checked_reference(admission_reference))
    row = result["candidates"][identifier]
    if (
        result["global_pool_changed"] is not False
        or type(result["new_financial_evaluations"]) is not int
        or result["new_financial_evaluations"] != 0
        or result["order_authority"] is not False
        or row["technical_expert_admitted"] is not True
        or row["status"] != "technical_eligibility_passed_admissible_nonselected"
        or result["RF_rounding_normalization_or_new_view"] is not False
    ):
        raise QuantError("A blocked source cannot receive a consumer, bridge or repaired RF.")
    intent = read_json(base.checked_reference(result["eligibility_intent"]))
    if not set(OWN).issubset(
        {r["path"] for r in intent["implementation_and_test_inputs"]}
    ) or not set(OWN).issubset({r["path"] for r in intent["complete_dependency_inputs"]}):
        raise QuantError("Require the actual complete frozen reviewer and dependency closure.")
    for ref in (
        intent["source_references"]
        + intent["complete_dependency_inputs"]
        + intent["implementation_and_test_inputs"]
    ):
        base.checked_reference(ref)
    admitted = pd.Timestamp(row["admitted_at"])
    if (
        admitted.tzinfo is None
        or pd.isna(admitted)
        or admitted
        < max(pd.Timestamp(intent["created_at"]), pd.Timestamp(row["financial_reviewed_at"]))
    ):
        raise QuantError("Actual technical admission cannot be backdated.")
    expert, _ = load_expert(row["source_descriptor"], universe)
    full = expert.observe(pd.Timestamp("2026-10-05"), 2512)
    if (
        len(full.risk_free) != 2512
        or not full.risk_free.index.isin(risk_free.index).all()
        or not full.risk_free.equals(risk_free.loc[full.risk_free.index])
    ):
        raise QuantError("The admitted source native RF must remain exact at the consumer.")
    return expert, row


def main():
    parser = argparse.ArgumentParser(description="Strict saved HMM-session expert eligibility.")
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "review": review}[args.action]()))


if __name__ == "__main__":
    main()
