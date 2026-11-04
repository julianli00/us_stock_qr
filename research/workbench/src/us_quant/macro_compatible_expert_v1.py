from __future__ import annotations

import json
import sqlite3

import numpy as np

from us_quant import legacy_real_yield_expert_eligibility_v1 as original
from us_quant import macro_policy_compatibility_v1 as compatible
from us_quant import ssd_research_v1 as base
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json
from us_quant.strategy_experts_v1 import ExpertIdentity

ROOT = base.ROOT
CONTRACT_SHA = "0fd76e2964f7242ab0274cf535fc44d39602c1024614d1e579b662eb4b2095c8"
ADMISSION_ID = "macro_compatible_policy_opinion_expert_admission_20261104_v1"
ADMISSION_STATUS = "technically_admitted_scoped_original_macro_monthly_not_selected"
UNIVERSE = (*original.ASSETS, "TLT", "QQQ", "XLK", "SOXX", "IJR", "PKW")
SCOPE = {
    "source_candidate_id": original.CANDIDATE,
    "curve_defensive_sibling_admitted": False,
    "publication_delay_NYSE_sessions": 2,
    "maximum_observation_age_calendar_days": 7,
    "admitted_information_start": "2016-10-05",
    "admitted_information_end": "2026-10-05",
    "earlier_native_context_retained_not_new_daily_macro_admission": True,
    "existing_monthly_vintage_and_formal_target_parity_only": True,
    "full_daily_PIT_or_revision_free_source_admitted": False,
    "source_permission_or_coverage_upgraded_by_assessment_view": False,
}


def require(condition, message):
    if not condition:
        raise QuantError(message)


def check_scope(policy, evidence):
    require(
        policy["publication_delay_sessions"] == 2
        and policy["maximum_observation_age_days"] == 7
        and policy["real_yield_change_sessions"] == 63
        and policy["candidates"][0] == {"id": original.CANDIDATE, "curve_defense": False}
        and evidence["existing_validation_not_replayed"] is True,
        "Keep original monthly source admission, lag, age and composition-only policy.",
    )
    rows = [
        row
        for row in evidence["formal_target_parity_reference"]
        if row["candidate_id"] == original.CANDIDATE
    ]
    require(
        len(rows) == 2 and {row["years"] for row in rows} == {10, 5},
        "Both original formal-window vintage parity certificates are required.",
    )
    for row in rows:
        start = "2016-10-06" if row["years"] == 10 else "2021-10-06"
        count = 2512 if row["years"] == 10 else 1254
        require(
            row["formal_return_start"] == start
            and row["formal_return_end"] == "2026-10-05"
            and row["formal_return_sessions"] == count
            and row["seeded_original_and_source_replaced_targets_byte_identical"] is True
            and row["queried_DFII10_cells_only_replaced_other_macro_data_unchanged"] is True
            and row["this_is_not_a_fully_PIT_daily_feature_dataset"] is True
            and all(
                day < "2016-10-05"
                for day in row["changed_source_target_sessions_before_formal_window"]
            ),
            "Limited stored monthly parity cannot become full daily PIT or a source waiver.",
        )
    return dict(SCOPE)


def verify_existing_assessment(native, view, native_default_rf, common_default_rf):
    require(
        native.index.equals(view.index) and native.columns.equals(view.columns),
        "Do not crop, remap or fill the native book to verify an assessment view.",
    )
    columns = [name for name in native if name != "risk_free"]
    require(
        native[columns].equals(view[columns])
        and native["risk_free"].equals(native_default_rf.loc[native.index])
        and view["risk_free"].equals(common_default_rf.loc[view.index]),
        "Preserve exact native NET/money and the separately pinned common assessment RF.",
    )
    different = native["risk_free"].to_numpy() != view["risk_free"].to_numpy()
    require(bool(different.any()), "Do not claim the old unequal native RF became exact.")
    return {
        "native_nonRF_columns_preserved_exactly": columns,
        "native_RF_still_not_exact_common": True,
        "native_RF_vs_assessment_different_cells": int(different.sum()),
        "common_assessment_RF_exact_frozen_default_reader": True,
        "future_score_tie_or_financial_effect_absent_claimed": False,
    }


def _sqlite_identity(snapshot):
    path = ROOT / "runtime/research-program.sqlite3"
    require(file_digest(path) == base.LEGACY_SHA256, "The original immutable ledger changed.")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT c.spec_sha,c.semantic_sha,c.registered_at,c.body,r.at,r.evidence_sha,r.body "
            "FROM candidates c JOIN reviews r ON r.candidate_id=c.id WHERE c.id=?",
            (original.CANDIDATE,),
        ).fetchone()
        events = db.execute(
            "SELECT seq,at,kind,body,sha FROM events WHERE seq IN (21,23) ORDER BY seq"
        ).fetchall()
        failures = db.execute(
            "SELECT count(*) FROM technical_failures WHERE candidate_id=?",
            (original.CANDIDATE,),
        ).fetchone()[0]
    require(
        row is not None
        and row[0] == snapshot["spec_sha256"]
        and row[1] == snapshot["semantic_sha256"]
        and row[2] == snapshot["registered_at"]
        and json.loads(row[3]) == snapshot["specification"]
        and row[4] == snapshot["reviewed_at"]
        and row[5] == snapshot["evidence_sha256"]
        and json.loads(row[6]) == snapshot["review"]
        and failures == snapshot["technical_failures"] == 0
        and [
            {"sequence": e[0], "at": e[1], "kind": e[2], "body": json.loads(e[3]), "sha256": e[4]}
            for e in events
        ]
        == snapshot["events"],
        "The original candidate21/review23 identity differs from its pinned source snapshot.",
    )


def load_candidate(contract_reference, universe):
    """Load existing scoped source opinions; queries must stay within the returned scope."""
    require(
        tuple(universe) == UNIVERSE and contract_reference["sha256"] == CONTRACT_SHA,
        "Use this exact macro consumer contract and the admitted fourteen-asset lattice.",
    )
    contract = read_json(base.checked_reference(contract_reference))
    require(
        contract["admission_id"] == ADMISSION_ID
        and contract["source_policy_id"] == original.CANDIDATE
        and contract["original_direct_no_view_identity_still_blocked"] is True,
        "A new scoped consumer cannot rewrite the original blocked direct identity.",
    )
    for ref in contract["source_references"]:
        base.checked_reference(ref)
    snapshot = read_json(original.SNAPSHOT)
    bundle = read_json(
        ROOT / "reports/macro-tilt-20261010/macro_real_yield_factor_tilt/bundle.json"
    )
    policy = read_json(ROOT / "config/macro-factor-tilt.json")
    old = read_json(original.OUTPUT)
    _sqlite_identity(snapshot)
    original.check_identity(
        snapshot,
        read_json(ROOT / "evidence/macro_tilt_20261010_registration.json"),
        read_json(ROOT / "evidence/macro_tilt_20261010_results.json"),
        bundle,
        policy,
    )
    scope = check_scope(policy, old["existing_vintage_validation_scope"])
    for row in old["existing_vintage_validation_scope"]["formal_target_parity_reference"]:
        if row["candidate_id"] == original.CANDIDATE:
            base.checked_reference({"path": row["private_target_matrix"], "sha256": row["sha256"]})
    prior = read_json(base.checked_reference(contract["existing_compatibility_result"]))
    provenance = read_json(base.checked_reference(contract["existing_assessment_provenance"]))
    require(
        prior["status"] == "compatible_scoped_policy_opinions_with_explicit_assessment_view"
        and prior["explicit_new_assessment_view"] == contract["existing_assessment"]
        and provenance["assessment_view_reference"] == contract["existing_assessment"]
        and provenance["NET_NAV_cash_costs_orders_gross_turnover_dates_preserved_exactly"] is True
        and provenance["source_native_RF_still_unequal_common"] is True
        and provenance["future_scoring_or_financial_effect_absent_claimed"] is False
        and provenance["nominal_seed_stripped_target_view_created"] is False,
        "Reuse the completed exact assessment provenance, not another repaired view.",
    )
    native_rf = compatible.frame(
        snapshot["specification"]["market"]["risk_free"], round_trip=False
    ).iloc[:, 0]
    common_rf = compatible.frame(prior["common_RF_reference"], round_trip=False).iloc[:, 0]
    targets = {
        f"{row['years']}y-{row['scenario']}": compatible.frame(row["targets"])
        for row in bundle["paths"]
    }
    identity = ExpertIdentity(original.CANDIDATE, original.GROUP, contract["defined_at"])
    view = compatible.frame(contract["existing_assessment"])
    cached = {row["path_key"]: row for row in old["paths"]}
    funding = {}
    native10 = None
    for row in bundle["paths"]:
        key = f"{row['years']}y-{row['scenario']}"
        book = compatible.frame(row["strategy"])
        weights = compatible.frame(row["weights"])
        bt = compatible.frame(row["strategy_bt"])
        first = "2016-10-06" if row["years"] == 10 else "2021-10-06"
        dates = sessions(first, "2026-10-05")
        require(
            row["capital_usd"] == 10000
            and row["commission_per_order"] == 1
            and row["cost_bps"] == (5.0 if row["scenario"] == "base" else 20.0)
            and row["delay_sessions"] == (1 if row["scenario"] == "base" else 2)
            and book.index.equals(dates)
            and weights.index.equals(dates)
            and bt.index.equals(dates)
            and tuple(weights.columns) == original.ASSETS
            and cached[key]["native_cash_funding_checks_pass"] is True
            and cached[key]["max_cached_BT_NAV_gap_usd"] <= 1e-6
            and np.isfinite(weights.to_numpy()).all()
            and (weights >= 0).all().all()
            and np.max(abs(weights.sum(axis=1) - book["gross_exposure"])) <= 1e-12
            and np.max(abs(book["equity"] - bt["equity"])) <= 1e-6
            and book["risk_free"].equals(native_rf.loc[dates]),
            "All four original funded paths and cached independent accounting must remain valid.",
        )
        compatible.MacroSeedAwarePolicyOpinion(identity, targets, book, tuple(universe))
        mismatch = int((book["risk_free"] != common_rf.loc[dates]).sum())
        require(
            mismatch == cached[key]["RF_mismatch_count"] and mismatch > 0,
            "The original native/common RF mismatch cannot be hidden or relabelled exact.",
        )
        funding[key] = {
            "return_dates": len(dates),
            "native_RF_still_unequal_common_different_cells": mismatch,
            "min_cash": float(book["cash"].min()),
            "max_gross": float(book["gross_exposure"].max()),
            "original_cached_BT_gap_usd": cached[key]["max_cached_BT_NAV_gap_usd"],
            "source_book_not_replayed_or_metrics_recomputed": True,
        }
        if key == "10y-base":
            native10 = book
    require(native10 is not None, "A complete source requires its original ten-year base book.")
    assessed = verify_existing_assessment(native10, view, native_rf, common_rf)
    expert = compatible.MacroSeedAwarePolicyOpinion(identity, targets, view, tuple(universe))
    require(
        expert.preservation()["genuine_policy_rows"] == 131
        and set(expert.preservation()["native_full_active_rows_by_path"].values()) == {132},
        "Keep all original genuine monthly opinions and every native capital seed.",
    )
    return expert, {
        "source_policy_id": original.CANDIDATE,
        "scope": scope,
        "source_policy_registered_at": snapshot["registered_at"],
        "new_consumer_defined_at": contract["defined_at"],
        "counterfactual_historical_availability": True,
        "original_source_financial_status": snapshot["review"]["status"],
        "native_funding": funding,
        "assessment": assessed,
        "native_role_preservation": expert.preservation(),
        "source_native_RF_exact_common": False,
        "assessment_RF_exact_common": True,
        "old_direct_no_view_identity_still_blocked": True,
        "new_source_view_or_Adapter_created": False,
        "new_source_strategy_shadow_account_BT_or_benchmark_runs": 0,
        "new_expert_selected_or_ninth_or_pool_change": False,
        "investment_objective_verified": False,
    }


def load_admitted(admission_reference, universe):
    record = read_json(base.checked_reference(admission_reference))
    require(
        record["admission_id"] == ADMISSION_ID
        and record["status"] == ADMISSION_STATUS
        and record["scope"] == SCOPE
        and record["selected_representative_or_pool_member"] is False,
        "Use an actual scoped admission record, not automatic promotion or a source waiver.",
    )
    expert, checked = load_candidate(record["contract"], universe)
    require(
        checked == record["cached_source_checks"],
        "The admitted source identity, opinion roles or assessment convention changed.",
    )
    return expert
