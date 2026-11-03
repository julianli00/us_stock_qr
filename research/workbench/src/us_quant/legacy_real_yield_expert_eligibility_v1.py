from __future__ import annotations

import argparse
import ast
from importlib.metadata import distributions
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_seven_expert_meta_v1 as helpers
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ArchivedMonthlyExpert, ExpertIdentity

ROOT = Path(__file__).resolve().parents[2]
BASE = helpers.base
FOLDER = BASE.REPORTS / "legacy-real-yield-expert-eligibility-20261103-v1"
DESIGN = FOLDER / "design-decision-v1.json"
SNAPSHOT = FOLDER / "source-sqlite-identity-snapshot-v1.json"
INTENT = FOLDER / "eligibility-intent-v1.json"
OUTPUT = FOLDER / "eligibility-result-v1.json"
PUBLIC = ROOT / "evidence/legacy_real_yield_expert_eligibility_20261103_v1.json"
CANDIDATE = "macro_real_yield_factor_tilt"
GROUP = "real_discount_rate_factor_composition"
ASSETS = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")
PATHS = ((10, "base"), (10, "stress"), (5, "base"), (5, "stress"))
OWN = (
    "src/us_quant/legacy_real_yield_expert_eligibility_v1.py",
    "tests/test_legacy_real_yield_expert_eligibility_v1.py",
    "tests/conftest.py",
)


def check_identity(snapshot, registration, results, bundle, policy):
    rows = [r for r in registration["candidates"] if r["spec"]["id"] == CANDIDATE]
    if (
        snapshot["candidate_id"] != CANDIDATE
        or snapshot["technical_failures"] != 0
        or [e["sequence"] for e in snapshot["events"]] != [21, 23]
        or len(rows) != 1
        or rows[0]["spec"] != snapshot["specification"]
        or rows[0]["registration"]["registered_at"] != snapshot["registered_at"]
        or rows[0]["registration"]["spec_sha256"] != snapshot["spec_sha256"]
        or digest_json(snapshot["specification"]) != snapshot["spec_sha256"]
        or digest_json(bundle) != snapshot["evidence_sha256"]
        or results["candidates"][CANDIDATE] != snapshot["review"]
        or snapshot["review"]["status"] != "rejected_historical"
        or bundle["candidate_spec_sha256"] != snapshot["spec_sha256"]
        or bundle["market"] != snapshot["specification"]["market"]
        or snapshot["specification"]["configuration"] != policy["candidates"][0]
        or snapshot["specification"]["configuration"] != {"id": CANDIDATE, "curve_defense": False}
        or set(snapshot["specification"]["asset_leverage"]) != set(ASSETS)
        or any(v != 1.0 for v in snapshot["specification"]["asset_leverage"].values())
        or policy["publication_delay_sessions"] != 2
        or policy["maximum_observation_age_days"] != 7
        or policy["real_yield_change_sessions"] != 63
        or policy["rising_real_yield_factor_shares"] != [0.15, 0.35, 0.15, 0.35]
        or policy["nonrising_real_yield_factor_shares"] != [0.35, 0.15, 0.35, 0.15]
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
        raise QuantError(
            "The original candidate21/review23,macro law or cached four-path source changed."
        )
    return {
        "actual_source_events": [21, 23],
        "source_identity_verified": True,
        "original_standalone_rejection_preserved": True,
        "original_macro_scope_not_upgraded_to_full_daily_PIT": True,
    }


def check(snapshot, bundle, frames, common_rf):
    if (
        set(frames) != set(PATHS)
        or not isinstance(common_rf, pd.Series)
        or common_rf.empty
        or not isinstance(common_rf.index, pd.DatetimeIndex)
        or not common_rf.index.is_unique
        or not common_rf.index.is_monotonic_increasing
        or not common_rf.index.equals(sessions(common_rf.index[0], common_rf.index[-1]))
        or not np.isfinite(common_rf).all()
    ):
        raise QuantError("Reference all four native paths and the unchanged complete common RF.")
    native = {(r["years"], r["scenario"]): r for r in bundle["paths"]}
    cached = {(r["years"], r["scenario"]): r for r in snapshot["review"]["paths"]}
    rows, blockers = [], []
    for years, scenario in PATHS:
        key = f"{years}y-{scenario}"
        record, source = native[years, scenario], frames[years, scenario]
        book, independent, weights, targets = [
            source[k] for k in ("strategy", "strategy_bt", "weights", "targets")
        ]
        first = "2016-10-06" if years == 10 else "2021-10-06"
        anchor, expected = previous_session(first), sessions(first, "2026-10-05")
        required = (
            "equity",
            "return",
            "cash",
            "gross_exposure",
            "turnover",
            "cost",
            "orders",
            "risk_free",
        )
        if (
            record["capital_usd"] != 10000
            or record["commission_per_order"] != 1
            or record["cost_bps"] != (5.0 if scenario == "base" else 20.0)
            or record["delay_sessions"] != (1 if scenario == "base" else 2)
            or not book.index.equals(expected)
            or not independent.index.equals(expected)
            or not weights.index.equals(expected)
            or tuple(weights.columns) != ASSETS
            or not set(required).issubset(book.columns)
            or not np.isfinite(book.loc[:, list(required)].to_numpy()).all()
            or not np.isfinite(independent["equity"]).all()
            or not np.isfinite(weights.to_numpy()).all()
            or (weights < 0).any().any()
            or (book["equity"] <= 0).any()
            or (book["return"] <= -1).any()
            or targets.empty
            or tuple(targets.columns) != ASSETS
            or not targets.index.is_unique
            or not targets.index.is_monotonic_increasing
            or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
            or not book.index.isin(targets.index).all()
            or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
            or not expected.isin(common_rf.index).all()
        ):
            raise QuantError(
                "The complete native source/return calendar or accounting schema is invalid."
            )
        active = targets.dropna(how="all")
        if (
            active.empty
            or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
            or active.loc[:anchor].empty
        ):
            raise QuantError(
                "The formal funding anchor needs an actual defined prior/current native target."
            )
        known = active.loc[:anchor].tail(1)
        nav_prior = book["equity"].shift(1)
        nav_prior.iloc[0] = 10000.0
        nav_gap = float(max(abs(book["equity"] / nav_prior - 1 - book["return"])))
        initial_gap = abs(float(book["equity"].iloc[0]) / 10000 - 1 - float(book["return"].iloc[0]))
        weight_gap = float(max(abs(weights.sum(axis=1) - book["gross_exposure"])))
        bt_gap = float(max(abs(book["equity"] - independent["equity"])))
        source_check = cached[years, scenario]
        if (
            source_check["market_and_target_replay_passed"] is not True
            or source_check["independent_bt_reexecuted"] is not True
            or source_check["max_independent_equity_error_usd"] > 1e-6
        ):
            raise QuantError("The original stored independent source evidence is incomplete.")
        funding = (
            float(book["cash"].min()) >= 0
            and 0 <= float(book["gross_exposure"].min())
            and float(book["gross_exposure"].max()) <= 1
            and (book["cost"] >= 0).all()
            and (book["orders"] >= 0).all()
            and nav_gap <= 1e-12
            and initial_gap <= 1e-12
            and weight_gap <= 1e-12
            and bt_gap <= 1e-6
        )
        rf = common_rf.loc[book.index]
        mismatch = book["risk_free"].to_numpy() != rf.to_numpy()
        dense = active.index.equals(targets.loc[active.index[0] :].index)
        monthends = pd.DatetimeIndex(
            [d for d in targets.loc[active.index[0] :].index if is_month_end(d)]
        )
        monthly = active.index.equals(monthends)
        constructor = {}
        identity = ExpertIdentity(CANDIDATE, GROUP, snapshot["registered_at"])
        for label, cls in (("monthly", ArchivedMonthlyExpert), ("dense", ArchivedDenseExpert)):
            try:
                cls(identity, targets, book, (*ASSETS, "TLT", "QQQ", "XLK", "SOXX", "IJR", "PKW"))
            except QuantError as error:
                constructor[label] = {"passed": False, "reason": str(error)}
            else:
                constructor[label] = {"passed": True}
        for passed, reason in (
            (funding, "native_cash_funding_or_accounting"),
            (not mismatch.any(), "native_RF_not_exact_unchanged_active_common"),
            (
                dense or monthly,
                "native_request_calendar_not_direct_existing_Monthly_or_Dense_Interface",
            ),
        ):
            if not passed:
                blockers.append({"path_key": key, "reason": reason})
        nonmonthly = active.index[~active.index.map(is_month_end)]
        rows.append(
            {
                "path_key": key,
                "formal_return_dates": len(book),
                "formal_anchor": str(anchor.date()),
                "formal_initial_capital_usd": 10000,
                "first_return_initial_capital_gap": initial_gap,
                "known_original_target_at_formal_anchor": str(known.index[0].date()),
                "formal_anchor_literal_active_row": anchor in active.index,
                "full_context_first_active_equals_anchor_required": False,
                "native_context_start": str(targets.index[0].date()),
                "native_context_end": str(targets.index[-1].date()),
                "native_first_active": str(active.index[0].date()),
                "native_request_rows": len(active),
                "native_requests_before_formal_anchor_preserved": int(
                    (active.index < anchor).sum()
                ),
                "full_native_nonmonthend_dates": nonmonthly.strftime("%Y-%m-%d").tolist(),
                "direct_monthly_calendar_compatible": monthly,
                "direct_dense_calendar_compatible": dense,
                "unchanged_Artifact_Adapter_constructor_checks": constructor,
                "native_cash_funding_checks_pass": bool(funding),
                "min_native_cash": float(book["cash"].min()),
                "max_native_gross": float(book["gross_exposure"].max()),
                "max_nav_return_gap": nav_gap,
                "max_weight_gross_gap": weight_gap,
                "max_cached_BT_NAV_gap_usd": bt_gap,
                "native_RF_exact_common": not bool(mismatch.any()),
                "RF_mismatch_count": int(mismatch.sum()),
                "first_RF_mismatch_date": str(book.index[np.flatnonzero(mismatch)[0]].date())
                if mismatch.any()
                else None,
                "max_RF_absolute_difference": float(max(abs(book["risk_free"] - rf))),
                "native_source_targets_or_RF_cropped_filled_redefined_or_rebuilt": False,
            }
        )
    return {
        "candidate_id": CANDIDATE,
        "mechanism_group": GROUP,
        "four_cached_native_paths_checked": 4,
        "status": "eligible_not_admitted" if not blockers else "not_admitted_current_Interface",
        "paths": rows,
        "blockers": blockers,
        "full_source_prehistory_retained": True,
        "native_funding_seed_rows_not_deleted_to_manufacture_monthly_compatibility": True,
        "new_RF_recommendation_assessment_view_or_Adapter": False,
        "source_pipeline_or_independent_accounts_replayed": False,
        "old_standalone_financial_rejection_preserved": True,
        "new_expert_admitted": False,
        "active_pool_changed": False,
        "new_financial_trials": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }


def references():
    snapshot = read_json(SNAPSHOT)
    bundle = read_json(
        ROOT / "reports/macro-tilt-20261010/macro_real_yield_factor_tilt/bundle.json"
    )
    paths = [
        DESIGN,
        SNAPSHOT,
        FOLDER / "actual-invocation-v1.json",
        ROOT / "evidence/macro_tilt_20261010_registration.json",
        ROOT / "evidence/macro_tilt_20261010_results.json",
        ROOT / "reports/macro-tilt-20261010/macro_real_yield_factor_tilt/bundle.json",
        ROOT / "config/macro-factor-tilt.json",
        ROOT / "data/sector-growth-prepared-20261011/risk_free.csv",
        ROOT / "runtime/research-program.sqlite3",
        ROOT / "evidence/real_yield_vintage_source_20261011_admission.json",
        ROOT / "evidence/real_yield_vintage_20261011_results_v2.json",
        ROOT / "evidence/real_yield_vintage_20261011_reproducibility_v2.json",
        ROOT / "evidence/real_yield_vintage_20261011_preservation_v2.json",
    ]
    paths.extend(
        BASE.safe_file(ROOT, name, digest)
        for name, digest in snapshot["specification"]["frozen_files"].items()
    )
    paths.extend(BASE.checked_reference(row) for row in bundle["market"].values())
    paths.extend(
        BASE.checked_reference(record[key])
        for record in bundle["paths"]
        for key in ("strategy", "strategy_bt", "weights", "targets")
    )
    return helpers.unique_refs([BASE.reference(p) for p in paths])


def dependencies():
    return FunctionType(
        helpers.admitted.import_template.dependencies.__code__,
        {**globals(), "ast": ast, "prior": helpers, "base": BASE},
    )()


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve the complete independent eligibility freeze.")
    with lifecycle.research_job():
        history = BASE.journal_events(BASE.JOURNAL)
        if (
            len(history) != 722
            or history[-1]["sha256"] != read_json(DESIGN)["current_journal_head"]
        ):
            raise QuantError(
                "Freeze this technical identity against its actual722-event source baseline."
            )
        xml = FOLDER / "offline-tests-v5.xml"
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if not suites or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise QuantError(
                "Complete the no-view monthly-anchor/RF/funding synthetic checks first."
            )
        proof = read_json(FOLDER / "validation-source-v2.json")
        expected_inputs = [BASE.reference(ROOT / p) for p in OWN]
        expected_inputs.append(BASE.reference(ROOT / "LEGACY_REAL_YIELD_EXPERT_ELIGIBILITY_V1.md"))
        outputs = [BASE.reference(xml), BASE.reference(FOLDER / "offline-tests-v5.log")]
        if (
            proof["implementation_and_test_inputs"] != expected_inputs
            or proof["canonical_test_outputs"] != outputs
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["exit_code"] != 0
        ):
            raise QuantError(
                "Successful canonical checks must bind current no-view implementation bytes."
            )
        deps = dependencies()
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "design": BASE.reference(DESIGN),
                "source_references": references(),
                "implementation_and_test_inputs": [
                    *deps,
                    BASE.reference(ROOT / "LEGACY_REAL_YIELD_EXPERT_ELIGIBILITY_V1.md"),
                ],
                "validation_evidence": [
                    *outputs,
                    BASE.reference(FOLDER / "validation-source-v2.json"),
                ],
                "passed_offline_checks": sum(int(s.get("tests", 0)) for s in suites),
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "source_numeric_review_executed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
    return BASE.reference(INTENT)


def review():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "review-attempt-v1.json").exists():
        raise QuantError(
            "Do not retry or repair the completed/partial independent review identity."
        )
    intent = read_json(INTENT)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen technical-review package environment changed.")
    for row in (
        intent["source_references"]
        + intent["implementation_and_test_inputs"]
        + intent["validation_evidence"]
    ):
        BASE.checked_reference(row)
    with lifecycle.research_job():
        before = helpers.current.current_counts(BASE.journal_events(BASE.JOURNAL))
        write_new_receipt(
            FOLDER / "review-attempt-v1.json",
            {"started_at": utc_now(), "intent": BASE.reference(INTENT)},
        )
        try:
            snapshot = read_json(SNAPSHOT)
            registration = read_json(ROOT / "evidence/macro_tilt_20261010_registration.json")
            results = read_json(ROOT / "evidence/macro_tilt_20261010_results.json")
            bundle = read_json(
                ROOT / "reports/macro-tilt-20261010/macro_real_yield_factor_tilt/bundle.json"
            )
            identity = check_identity(
                snapshot,
                registration,
                results,
                bundle,
                read_json(ROOT / "config/macro-factor-tilt.json"),
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
            common = pd.read_csv(
                ROOT / "data/sector-growth-prepared-20261011/risk_free.csv",
                index_col=0,
                parse_dates=True,
            ).iloc[:, 0]
            result = check(snapshot, bundle, frames, common)
            vintage = read_json(ROOT / "evidence/real_yield_vintage_20261011_results_v2.json")
            packet = {
                "completed_at": utc_now(),
                "intent": BASE.reference(INTENT),
                "identity": identity,
                **result,
                "existing_vintage_validation_scope": {
                    "monthly_comparisons": vintage["monthly_comparisons"],
                    "changed_state_queries": vintage["changed_rising_state_query_sessions"],
                    "full_daily_PIT_not_proven": True,
                    "existing_validation_not_replayed": True,
                    "formal_target_parity_reference": vintage[
                        "formal_target_proofs_including_explicit_seed"
                    ],
                },
                "actual_financial_counts_unchanged": before,
            }
            after = helpers.current.current_counts(BASE.journal_events(BASE.JOURNAL))
            if before != after:
                raise QuantError("Eligibility cannot change financial trial counts.")
            for row in intent["source_references"]:
                BASE.checked_reference(row)
            write_new_receipt(OUTPUT, packet)
            write_new_receipt(PUBLIC, packet)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "review-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_evidence_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One no-view read-only legacy macro expert eligibility identity."
    )
    parser.add_argument("action", choices=("freeze", "review"))
    args = parser.parse_args()
    import json

    print(json.dumps(freeze() if args.action == "freeze" else review(), indent=2))
