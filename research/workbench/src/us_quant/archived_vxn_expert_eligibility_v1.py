from __future__ import annotations

import argparse
import json
import math
import os
from importlib.metadata import distributions
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_macro_expert_inclusion_v1 as saved
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import load_expert
from us_quant.vxn_relative_uncertainty_v1 import validate as validate_policy

ROOT, BASE = saved.ROOT, saved.base
FOLDER = BASE.REPORTS / "archived-vxn-expert-eligibility-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
DESCRIPTOR = FOLDER / "archived-expert-descriptor-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
RESULT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/archived_vxn_expert_eligibility_20261104_v1.json"
REVIEW = "ssd_archived_vxn_expert_eligibility_20261104_v1"
CANDIDATE = "hmm_vxn_relative_uncertainty_v1"
STOCKS = ("SOXX", "MTUM", "VLUE", "QUAL", "USMV")
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
COMMON_RF = ROOT / "data/sector-growth-prepared-20261011/risk_free.csv"
OWN = (
    "src/us_quant/archived_vxn_expert_eligibility_v1.py",
    "tests/test_archived_vxn_expert_eligibility_v1.py",
    "tests/conftest.py",
    "ARCHIVED_VXN_EXPERT_ELIGIBILITY_V1.md",
)


def frame(reference):
    return pd.read_csv(
        BASE.checked_reference(reference),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def check_scope(readiness, manifest):
    if any(
        row["source_admitted"] is not True
        or row["admission_scope"] != "archival_daily_closes_and_original_query_coverage_only"
        or row["coverage"]["full_daily_coverage"] is not True
        or row["coverage"]["all_original_monthly_queries_covered"] is not True
        or row["ratios_strategy_states_targets_or_returns_computed"] is not False
        for row in (readiness, manifest)
    ):
        raise QuantError("Keep the original complete archival-close source scope without a waiver.")
    if readiness["full_intraday_or_historical_methodology_chain_verified"] is not False:
        raise QuantError("A daily-close source is not an independently proven intraday archive.")
    return {
        "source_scope": readiness["admission_scope"],
        "source_scope_upgraded_or_new_query_executed": False,
    }


def check_composition(parent, child, audits, proof):
    if (
        not parent.index.equals(child.index)
        or not parent.columns.equals(child.columns)
        or not parent.isna().equals(child.isna())
        or not set(STOCKS).issubset(parent.columns)
        or proof["monthly_decisions"] != 122
        or proof["original_calendar_opportunities"] != 131
        or proof["undefined_prehistory_months"] != 9
        or proof["additional_source_months_dropped"] != 0
        or proof["source_model_refitting"] is not False
        or proof["original_source_signal_recomputed"] is not False
        or proof["candidate_accounts_or_metrics_computed"] is not False
        or proof["exact_nonstock_and_total_stock_budget_preserved"] is not True
        or len(proof["prefix_truncations"]) != 5
    ):
        raise QuantError(
            "Preserve full HMM/child calendars, old model evidence and source prehistory."
        )
    active = child.dropna(how="all")
    if (
        len(active) != 122
        or not all(is_month_end(d) for d in active.index)
        or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or len(audits) != 122
        or [r["decision_session"] for r in audits] != active.index.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep all122 original genuine opinions, not a cropped or filled source.")
    nonstock = child.columns.difference(STOCKS)
    if not child[nonstock].equals(parent[nonstock]):
        raise QuantError("A composition variant cannot change nonstock budget cells.")
    gaps = []
    for day, audit in zip(active.index, audits, strict=True):
        source_equity = math.fsum(float(parent.loc[day, symbol]) for symbol in STOCKS)
        child_equity = math.fsum(float(child.loc[day, symbol]) for symbol in STOCKS)
        gaps.append(abs(source_equity - child_equity))
        if (
            gaps[-1] > 1e-12
            or abs(source_equity - audit["source_equity_budget_preserved"]) > 1e-12
            or audit["VXN_quote_session"] != str(day.date())
            or audit["VIX_quote_session"] != str(day.date())
            or audit["information_clock_after_16_15_ET"] is not True
            or audit["hmm_refitted_or_states_recomputed"] is not False
            or audit["exact_portfolio_or_SOXX_implied_variance_claimed"] is not False
            or audit["true_variance_risk_premium_claimed"] is not False
            or audit["BIL_target_unchanged"] != float(child.loc[day, "BIL"])
            or audit["GLD_target_unchanged"] != float(child.loc[day, "GLD"])
            or not 0 < audit["SOXX_share_of_stock_budget"] < 1
            or any(
                row["last_observation"] != str(day.date()) or row["return_sessions"] != 21
                for row in audit["observed_variation"].values()
            )
        ):
            raise QuantError(
                "Keep original stock capital and stored option/price information clocks."
            )
    return {
        "genuine_native_monthly_opinions": 122,
        "undefined_prehistory_months_retained": 9,
        "maximum_stock_budget_conservation_difference": max(gaps),
        "all_nonstock_cells_exact": True,
        "old_prefix_certificates_read": 5,
        "stored_SOXX_share_minimum": min(r["SOXX_share_of_stock_budget"] for r in audits),
        "stored_SOXX_share_maximum": max(r["SOXX_share_of_stock_budget"] for r in audits),
        "HMM_variation_ratio_target_pipeline_replayed": False,
    }


def _inputs():
    record = read_json(DESCRIPTOR)
    entries = [
        e
        for e in read_json(BASE.checked_reference(record["registration"]))["entries"]
        if e["candidate_id"] == CANDIDATE
    ]
    if len(entries) != 1 or record["id"] != CANDIDATE or record["adapter"] != "monthly":
        raise QuantError("Require one original complete VXN source identity, not a pool promotion.")
    entry = entries[0]
    frozen = entry["specification"]["frozen"]
    proof = read_json(BASE.checked_reference(entry["target_proof"]["proof"]))
    return record, entry, frozen, proof


def _references():
    record, entry, frozen, proof = _inputs()
    sources = [
        BASE.reference(p)
        for p in (
            DESIGN,
            DESCRIPTOR,
            FOLDER / "actual-invocation-v1.json",
            FOLDER / "original-source-events-v1.json",
            COMMON_RF,
        )
    ]
    sources += [
        record["registration"],
        record["results"],
        entry["target_proof"]["proof"],
        entry["target_proof"]["targets"],
        proof["information_audits"],
    ]
    for key in (
        "code",
        "policies",
        "prior_evidence",
        "validation_evidence",
        "benchmark_artifacts",
        "volatility_sources",
    ):
        sources.extend(frozen[key])
    sources.extend(frozen["market"].values())
    for ref in record["completed_paths"].values():
        sources.append(ref)
        sources.extend(read_json(BASE.checked_reference(ref))["artifacts"])
    old = read_json(
        BASE.REPORTS / "pair-score-persistence-diagnosis-20261104-v1/analysis-intent-v1.json"
    )
    dependencies = saved.unique_refs(
        [
            *old["implementation_dependencies"],
            *(BASE.reference(ROOT / p) for p in OWN),
        ]
    )
    return saved.unique_refs(sources), dependencies


def freeze():
    with lifecycle.research_job():
        if INTENT.exists():
            raise QuantError("Keep the original complete or partial review identity.")
        os.umask(0o077)
        design = read_json(DESIGN)
        if (
            design["review_id"] != REVIEW
            or design["actual_continuation_event"] != 150633
            or design["actual_cached_numeric_review_executed"] is not False
        ):
            raise QuantError("Keep the fixed no-refit/no-promotion qualification contract.")
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
            raise QuantError("Bind the exact successful source-scope and conservation checks.")
        for ref in sources + dependencies:
            BASE.checked_reference(ref)
        counts = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if counts["current_journal_event_count"] != 852:
            raise QuantError("The fixed review begins from the actual closed852-event state.")
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
        raise QuantError("Keep the first actual review and any partial evidence; never retry.")
    intent = read_json(INTENT)
    if (
        intent["review_id"] != REVIEW
        or intent["maximum_actual_reviews"] != 1
        or intent["actual_cached_numeric_review_executed"] is not False
        or intent["package_versions"]
        != [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
    ):
        raise QuantError("The frozen reviewer identity or environment changed.")
    for ref in (
        intent["source_references"]
        + intent["implementation_dependencies"]
        + intent["validation_evidence"]
    ):
        BASE.checked_reference(ref)
    with lifecycle.research_job():
        os.umask(0o077)
        counts = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if counts != intent["financial_counts"]:
            raise QuantError("Technical eligibility cannot hide a new financial event.")
        write_new_receipt(
            FOLDER / "review-attempt-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "intent": BASE.reference(INTENT)},
        )
        try:
            record, entry, frozen, proof = _inputs()
            scope = check_scope(
                read_json(BASE.checked_reference(frozen["vxn_source_readiness"])),
                read_json(BASE.checked_reference(frozen["vxn_source_manifest"])),
            )
            policy = read_json(BASE.checked_reference(frozen["policy"]))
            validate_policy(policy)
            parent = frame(frozen["source_targets"])
            child = frame(entry["target_proof"]["targets"])
            composition = check_composition(
                parent,
                child,
                read_json(BASE.checked_reference(proof["information_audits"]))["decisions"],
                proof,
            )
            parent_entry = read_json(BASE.checked_reference(frozen["source_registration"]))[
                "entries"
            ][0]
            parent_proof = read_json(BASE.checked_reference(parent_entry["target_proof"]["proof"]))
            if len(parent_proof["retained_model_fit_receipts"]) != 122:
                raise QuantError("Keep every original saved HMM fit/filter receipt.")
            expert, loader = load_expert(record, UNIVERSE)
            common = pd.read_csv(COMMON_RF, index_col=0, parse_dates=True).iloc[:, 0]
            paths, blockers = [], []
            shadow = None
            for key, ref in record["completed_paths"].items():
                complete = read_json(BASE.checked_reference(ref))
                refs = [r for r in complete["artifacts"] if r["path"].endswith("/account.csv")]
                if len(refs) != 1:
                    raise QuantError("A completed source path needs one actual cached account.")
                book = frame(refs[0])
                if key == "10y-base":
                    shadow = book
                if not book.index.isin(common.index).all():
                    raise QuantError("Do not drop native RF dates to match the common calendar.")
                mismatches = book["risk_free"].to_numpy() != common.loc[book.index].to_numpy()
                if mismatches.any():
                    blockers.append({"path_key": key, "reason": "native_RF_not_exact_common"})
                paths.append(
                    {
                        "path_key": key,
                        "formal_return_sessions": len(book),
                        "native_RF_mismatch_count": int(mismatches.sum()),
                        "minimum_cash_usd": complete["execution"]["minimum_cash_usd"],
                        "maximum_gross_exposure": complete["execution"]["maximum_gross_exposure"],
                        "cached_independent_BT_NAV_difference_usd": complete[
                            "independent_bt_max_equity_difference_usd"
                        ],
                        "broker_orders": complete["broker_orders"],
                    }
                )
            if shadow is None:
                raise QuantError("The original ten-year-base scoring book is required.")
            days = [
                child.index[0],
                child.dropna(how="all").index[0],
                pd.Timestamp("2016-10-05"),
                shadow.index[124],
                shadow.index[125],
                pd.Timestamp("2026-10-05"),
            ]
            observations = []
            for day in days:
                value = expert.observe(day, 126)
                known = child.loc[:day].dropna(how="all")
                expected = None if known.empty else known.index[-1]
                count = min(126, len(shadow.loc[:day]))
                if (
                    value.target_known_at != expected
                    or len(value.net_returns) != count
                    or value.score_mature != (count == 126)
                    or (
                        expected is not None
                        and not value.target.equals(
                            known.iloc[-1].reindex(UNIVERSE, fill_value=0.0)
                        )
                    )
                    or (expected is None and value.target is not None)
                ):
                    raise QuantError("Actual native availability or125/126 maturity changed.")
                observations.append(
                    {
                        "as_of": str(day.date()),
                        "target_known_at": str(expected.date()) if expected is not None else None,
                        "target_available": value.target is not None,
                        "shadow_observations": count,
                        "score_mature": value.score_mature,
                        "carried_opinion_is_new_source_request": False,
                    }
                )
            if saved.current_counts(BASE.journal_events(BASE.JOURNAL)) != counts:
                raise QuantError("This review must not alter financial trial counts.")
            for ref in intent["source_references"] + intent["implementation_dependencies"]:
                BASE.checked_reference(ref)
            result = {
                "reviewed_at": utc_now(),
                "review_id": REVIEW,
                "candidate_id": CANDIDATE,
                "intent": BASE.reference(INTENT),
                "source_registered_at": record["introduced_at"],
                "status": (
                    "TECHNICALLY_USABLE_RELATED_HMM_OPTION_VARIANT_NOT_SELECTED"
                    if not blockers
                    else "NOT_ADMITTED_CURRENT_INTERFACE"
                ),
                "source_scope": scope,
                "composition": composition,
                "cached_loader_validation": loader,
                "actual_Interface_observations": observations,
                "native_paths": paths,
                "blockers": blockers,
                "original_HMM_fit_filter_receipts_retained": 122,
                "source_financial_rejection_preserved": True,
                "new_source_requests_or_financial_trials": 0,
                "new_model_filter_variation_ratio_target_or_account_pipeline": False,
                "new_RF_recommendation_assessment_view_or_source_Adapter": False,
                "selected_representative_or_pool_member": False,
                "original_active_rosters_changed": False,
                "actual_financial_counts_unchanged": counts,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_new_receipt(RESULT, result)
            write_new_receipt(PUBLIC, result)
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
    parser = argparse.ArgumentParser(description="One cached VXN-policy expert eligibility review.")
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))
