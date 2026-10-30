from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.defensive_factor_rotation import validate_policy as validate_source_policy
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.research_weekly import verify_publication
from us_quant.ssd_expert_meta_research_v1 import unique_refs
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity

ROOT = base.ROOT
POLICY = ROOT / "config/legacy-defensive-rotation-expert-v1.json"
TEST = ROOT / "tests/test_legacy_defensive_rotation_expert_v1.py"
FOLDER = base.REPORTS / "legacy-defensive-rotation-expert-admission-20261030-v1"
INTENT = FOLDER / "admission-intent-v1.json"
OUTPUT = FOLDER / "admission-v1.json"
PUBLIC = ROOT / "evidence/legacy_defensive_rotation_expert_20261030_admission_v1.json"
VIEWS = ROOT / "data/ssd-defensive-rotation-expert-assessment-20261030-v1"
INVOCATION = FOLDER / "resumption-invocation-v1.json"
SOURCE = "four_factor_defense_diversified"
GROUP = "cross_asset_defensive_relative_strength"
FACTORS = ("MTUM", "VLUE", "QUAL", "USMV")
ASSETS = ("SPY", "IEF", "GLD", "BIL", *FACTORS, "TLT")
MANIFEST_SHA = "3b2ef6789348ac88b8fdf82805415f518da85bb72c1d652735e7aeaaa63666b6"
SPEC_SHA = "1cd67a5f371a3a4d7fd6a87cd1c633082b00a3af051262d2a0d14655186d0153"
SEMANTIC_SHA = "4badb35fa77aa16b59adf54841070d29d2e563a242a8d113987a56b546002dd1"
BUNDLE_SHA = "bd00cae1a685f2ce4cb1d8dba411f33071ff71b2173d262cc82e152668ac05ce"
JOURNAL_SHA = "4dd16707904ec6b8b2fc1aa3442ffd31af3372fadfc63b512ed11deff42b81d6"
NATIVE_ACCOUNTING = "legacy_default_without_accounting_engine_field"
CODE_DEPENDENCIES = [
    "src/us_quant/defensive_factor_rotation.py",
    "config/defensive-factor-rotation.json",
    "src/us_quant/research_program.py",
    "src/us_quant/strategy_replay.py",
    "src/us_quant/backtest.py",
    "src/us_quant/bt_audit.py",
    "src/us_quant/calendar.py",
    "src/us_quant/data.py",
    "src/us_quant/dual_horizon.py",
    "src/us_quant/multifactor_stability.py",
    "src/us_quant/storage.py",
]
ADMISSION_DEPENDENCIES = [
    "src/us_quant/ssd_current_progress_v1.py",
    "src/us_quant/ssd_research_v1.py",
    "src/us_quant/candidate_receipts.py",
    "src/us_quant/config.py",
    "src/us_quant/dual_benchmark_v1.py",
    "src/us_quant/research_weekly.py",
    "src/us_quant/ssd_expert_meta_research_v1.py",
    "src/us_quant/strategy_experts_v1.py",
    "src/us_quant/ssd_commitment_research_v1.py",
    "src/us_quant/daily_variance_disagreement_v1.py",
]


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def validate(policy):
    fixed = {
        "schema_version": 1,
        "admission_id": "ssd_legacy_defensive_rotation_expert_admission_20261030_v1",
        "source_candidate_id": SOURCE,
        "mechanism_group": GROUP,
        "source_registered_at": "2026-10-10T16:25:34.835292+00:00",
        "source_specification_sha256": SPEC_SHA,
        "source_semantic_sha256": SEMANTIC_SHA,
        "source_bundle_semantic_sha256": BUNDLE_SHA,
        "source_candidate_event_sequence": 28,
        "source_review_event_sequence": 30,
        "pinned_import_manifest_sha256": MANIFEST_SHA,
        "source_published_dependency_commit": "040d273acc1722ae06463a0f2b0d495059b3a8c9",
        "retained_code_dependency_paths": CODE_DEPENDENCIES,
        "admission_helper_dependency_paths": ADMISSION_DEPENDENCIES,
        "native_asset_count": 9,
        "genuine_month_end_recommendations": 128,
        "first_genuine_recommendation": "2016-02-29",
        "shadow_reference_path": "10y-base",
        "source_initial_capital_usd": 10000,
        "invested_target_budget": 0.98,
        "source_equity_share_min": 0.30,
        "source_equity_share_max": 0.70,
        "new_recommendation_view_artifacts": 1,
        "new_assessment_view_artifacts": 1,
        "new_strategy_evaluations": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "original_ledger_sha256": base.LEGACY_SHA256,
        "prior_ssd_journal_sequence": 467,
        "prior_ssd_journal_head_sha256": JOURNAL_SHA,
    }
    if not isinstance(policy, dict) or any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()
    ):
        raise QuantError("Keep the selected defensive policy, source identity and explicit views.")
    if not isinstance(policy.get("methodology"), dict):
        raise QuantError("Admission requires an explicit frozen methodology.")
    if any(
        policy["methodology"].get(k) is not False
        for k in (
            "active_common_reader_RF_definition_or_frozen_files_changed",
            "source_native_RF_claimed_equal_to_active_common_native_RF",
            "source_or_strategy_pipeline_replayed",
            "expanded_pool_or_next_meta_rule_frozen",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError(
            "Technical admission cannot rewrite native evidence or imply qualification."
        )


def legacy_snapshot(policy):
    path = ROOT / "runtime/research-program.sqlite3"
    if file_digest(path) != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT c.spec_sha,c.semantic_sha,c.registered_at,c.body,r.at,r.evidence_sha,r.body "
            "FROM candidates c JOIN reviews r ON r.candidate_id=c.id WHERE c.id=?",
            (SOURCE,),
        ).fetchone()
        events = db.execute(
            "SELECT seq,at,kind,body,sha FROM events "
            "WHERE kind IN ('candidate','review') AND json_extract(body,'$.candidate_id')=? "
            "ORDER BY seq",
            (SOURCE,),
        ).fetchall()
        technical = db.execute(
            "SELECT count(*) FROM technical_failures WHERE candidate_id=?", (SOURCE,)
        ).fetchone()[0]
    if row is None or [e[0] for e in events] != [28, 30] or technical:
        raise QuantError("Require a unique original policy/review without technical failures.")
    snap = read_json(base.checked_reference(policy["source_identity_snapshot"]))
    if (
        row[0] != snap["spec_sha256"]
        or row[1] != snap["semantic_sha256"]
        or row[2] != snap["registered_at"]
        or json.loads(row[3]) != snap["specification"]
        or row[4] != snap["reviewed_at"]
        or row[5] != snap["evidence_sha256"]
        or json.loads(row[6]) != snap["review"]
        or events[0][4] != policy["source_candidate_event_sha256"]
        or events[1][4] != policy["source_review_event_sha256"]
    ):
        raise QuantError("The immutable source identity differs from its preserved snapshot.")
    return snap


def check_identity(policy, snapshot, registration, results, bundle):
    spec, review = snapshot["specification"], snapshot["review"]
    found = [e for e in registration["candidates"] if e["spec"]["id"] == SOURCE]
    source_policy = read_json(
        base.checked_reference(
            {
                "path": "config/defensive-factor-rotation.json",
                "sha256": spec["frozen_files"]["config/defensive-factor-rotation.json"],
            }
        )
    )
    validate_source_policy(source_policy)
    if (
        len(found) != 1
        or found[0]["spec"] != spec
        or found[0]["registration"]["registered_at"] != policy["source_registered_at"]
        or found[0]["registration"]["spec_sha256"] != policy["source_specification_sha256"]
        or digest_json(spec) != policy["source_specification_sha256"]
        or snapshot["semantic_sha256"] != policy["source_semantic_sha256"]
        or snapshot["registered_at"] != policy["source_registered_at"]
        or spec["configuration"] != source_policy["candidates"][1]
        or spec["factor_ids"] != source_policy["factor_ids"]
        or spec["leveraged_products_allowed"] is not False
        or set(spec["asset_leverage"]) != set(ASSETS)
        or any(v != 1.0 for v in spec["asset_leverage"].values())
        or spec["order_authority"] is not False
        or registration["outcomes_computed"] is not False
        or digest_json(registration["readiness"]) != found[0]["registration"]["readiness_sha256"]
        or results["candidates"][SOURCE] != review
        or review["status"] != "rejected_historical"
        or review["order_authority"] is not False
        or review["independent_forward_validation"] is not False
        or bundle["candidate_spec_sha256"] != policy["source_specification_sha256"]
        or digest_json(bundle) != policy["source_bundle_semantic_sha256"]
        or snapshot["evidence_sha256"] != policy["source_bundle_semantic_sha256"]
        or bundle["market"] != spec["market"]
        or bundle["leveraged_products_allowed"] is not False
        or bundle["order_authority"] is not False
        or "accounting_engine" in bundle
        or not pd.Timestamp(snapshot["registered_at"])
        <= pd.Timestamp(bundle["completed_at"])
        <= pd.Timestamp(snapshot["reviewed_at"])
    ):
        raise QuantError("The source legacy identity, funding contract or original clocks changed.")
    for path, digest in spec["frozen_files"].items():
        if file_digest(ROOT / path) != digest:
            raise QuantError("Do not change the two originally frozen source model/policy files.")
    reviews = {(r["years"], r["scenario"]): r for r in review["paths"]}
    if set(reviews) != {(10, "base"), (10, "stress"), (5, "base"), (5, "stress")}:
        raise QuantError("A complete source must retain every original evidence path.")
    for path in bundle["paths"]:
        r = reviews[path["years"], path["scenario"]]
        if any(
            r[k] is not True
            for k in (
                "frozen_market_risk_free_used",
                "independent_bt_reexecuted",
                "market_and_target_replay_passed",
                "registered_strategy_targets_regenerated",
            )
        ):
            raise QuantError("Reuse actual prior source reproducibility proofs, not new replays.")
    return spec


def check_cached_funding(proof, bundle_ref, bundle):
    rows = proof["rows"]
    by_key = {f"{p['years']}y-{p['scenario']}": p for p in bundle["paths"]}
    if (
        proof["source_bundle"] != bundle_ref
        or len(rows) != 4
        or {r["path_key"] for r in rows} != base.PATH_KEYS
        or proof["ledger_unchanged"] is not True
        or proof["new_account_or_BT_runs"] != 0
        or proof["new_source_target_artifact_written"] is not False
        or proof["all_four_seed_removed_views_identical"] is not True
    ):
        raise QuantError("Require the unchanged complete cached funding and seed audit.")
    for row in rows:
        path = by_key[row["path_key"]]
        if (
            path["capital_usd"] != 10000
            or path["commission_per_order"] != 1
            or (path["cost_bps"], path["delay_sessions"])
            != ((5, 1) if path["scenario"] == "base" else (20, 2))
            or row["minimum_cash_usd"] < 0
            or not 0 <= row["maximum_gross_exposure"] <= 1
            or not np.isfinite(
                [
                    row["minimum_cash_usd"],
                    row["maximum_gross_exposure"],
                    row["cached_BT_maximum_NAV_difference_usd"],
                ]
            ).all()
            or row["cached_BT_maximum_NAV_difference_usd"] > 1e-6
            or row["native_target_reference"] != path["targets"]
            or row["genuine_month_end_rows"] != 128
            or row["source_strategy_target_or_BT_pipeline_run"] is not False
            or row["seed_equals_prior_genuine_month_end"] is not True
            or set(row["cached_accounting_errors"])
            != {
                "nav_usd",
                "cash_usd",
                "holdings_cash_usd",
                "return",
                "gross_weight",
                "cost_usd",
                "turnover",
            }
            or any(
                not np.isfinite(v) or not 0 <= v <= (1e-6 if k.endswith("_usd") else 1e-10)
                for k, v in row["cached_accounting_errors"].items()
            )
        ):
            raise QuantError(
                "A source with invalid cash, leverage, clocks or cached money cannot enter."
            )


def check_lineage(source, common_stored, common_active):
    observations = {}
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        native = source[name]
        stored = common_stored[name].loc[native.index, native.columns]
        active = common_active[name].loc[native.index, native.columns]
        if (
            not np.isfinite(native).all().all()
            or not np.isfinite(stored).all().all()
            or not np.isfinite(active).all().all()
            or not native.equals(stored)
        ):
            raise QuantError(
                "Require exact stored serialization lineage, not approximate alignment."
            )
        observations[name] = {
            "source_native_equals_common_stored_roundtrip_exactly": True,
            "source_vs_active_common_native_exact": native.equals(active),
            "source_vs_active_common_native_different_cells": int(
                (native != active).to_numpy().sum()
            ),
            "source_vs_active_common_native_maximum_difference": float(
                abs(native - active).to_numpy().max()
            ),
        }
    return observations


def nominal_view(targets, initial_seed):
    active = targets.dropna(how="all")
    if (
        active.empty
        or not np.isfinite(active).all().all()
        or (active < 0).any().any()
        or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
        or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("The native target lattice must be complete, unleveraged and budgeted.")
    initial_seed = pd.Timestamp(initial_seed)
    seeds = active.loc[[d for d in active.index if not is_month_end(d)]]
    prior = active.loc[[d for d in active.index if d < initial_seed and is_month_end(d)]]
    if (
        len(seeds) != 1
        or seeds.index[0] != initial_seed
        or prior.empty
        or not seeds.iloc[0].equals(prior.iloc[-1])
    ):
        raise QuantError(
            "Remove only an explicit synthetic seed equal to its strictly prior month end."
        )
    result = targets.copy()
    result.loc[initial_seed] = np.nan
    genuine = result.dropna(how="all")
    expected = pd.DatetimeIndex(
        [d for d in result.loc[genuine.index[0] :].index if is_month_end(d)]
    )
    if not genuine.index.equals(expected):
        raise QuantError("A missing genuine month end cannot be disguised as a removed seed.")
    if not result.loc[genuine.index].equals(targets.loc[genuine.index]):
        raise QuantError("Every genuine source recommendation must remain exactly unchanged.")
    return result


def assessment_view(native_shadow, native_source_rf, active_common_rf):
    index = native_shadow.index
    if (
        native_shadow.empty
        or not index.equals(sessions(index[0], index[-1]))
        or not np.isfinite(native_shadow).all().all()
        or not index.isin(native_source_rf.index).all()
        or not index.isin(active_common_rf.index).all()
        or not native_shadow["risk_free"].equals(native_source_rf.loc[index])
        or not np.isfinite(active_common_rf.loc[index]).all()
    ):
        raise QuantError(
            "A declared assessment RF view requires complete native and common clocks."
        )
    result = native_shadow.copy()
    result["risk_free"] = active_common_rf.loc[index].to_numpy()
    unchanged = [c for c in native_shadow if c != "risk_free"]
    if not result[unchanged].equals(native_shadow[unchanged]):
        raise QuantError("A scoring RF view cannot change NET money, returns or source fees.")
    return result


def published_source_dependencies(refs, registration_ref, commit):
    manifest_path = ROOT / "IMPORT_MANIFEST.json"
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    if (
        file_digest(manifest_path) != MANIFEST_SHA
        or contract["bundle_manifest_sha256"] != MANIFEST_SHA
    ):
        raise QuantError("The fixed original import manifest changed.")
    manifest = read_json(manifest_path)
    explicit, bundled = set(), []
    for ref in refs:
        base.checked_reference(ref)
        if "research/workbench/" + ref["path"] in contract["files"]:
            explicit.add(ref["path"])
        elif manifest["file_sha256"].get(ref["path"]) == ref["sha256"]:
            bundled.append(ref)
        else:
            raise QuantError("A required source dependency is neither explicit nor pinned.")
    published = verify_publication(ROOT, registration_ref, explicit, commit)
    for ref in [base.reference(manifest_path), *bundled]:
        blob = subprocess.run(
            ["git", "show", f"{commit}:research/workbench/{ref['path']}"],
            cwd=base.REPO,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
            raise QuantError("An actual Git blob differs from a required pinned source dependency.")
    return {
        **published,
        "pinned_import_dependencies_checked": len(bundled),
        "original_manifest_sha256": file_digest(manifest_path),
    }


def declared_inputs(policy):
    snapshot = legacy_snapshot(policy)
    bundle = read_json(base.checked_reference(policy["source_bundle"]))
    refs = [
        policy[k]
        for k in (
            "source_registration",
            "source_results",
            "source_bundle",
            "source_identity_snapshot",
            "selection",
            "bridge_plan",
            "cached_native_funding_and_seed_proof",
            "common_market_registration",
        )
    ]
    refs.extend(snapshot["specification"]["market"].values())
    refs.extend(base.reference(ROOT / p) for p in policy["retained_code_dependency_paths"])
    refs.extend(base.reference(ROOT / p) for p in policy["admission_helper_dependency_paths"])
    refs.append(base.reference(ROOT / "IMPORT_MANIFEST.json"))
    refs.append(base.reference(INVOCATION))
    common = read_json(base.checked_reference(policy["common_market_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    refs.extend(common["market"].values())
    for path in bundle["paths"]:
        refs.extend(
            path[k] for k in ("strategy", "strategy_bt", "weights", "targets", "spy", "spy_bt")
        )
    refs = unique_refs(refs)
    for ref in refs:
        base.checked_reference(ref)
    return snapshot, bundle, common, refs


def validations():
    xml = FOLDER / "offline-tests-v2.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    required = {
        "test_exact_serialization_lineage_is_not_a_tolerance",
        "test_only_duplicate_initial_seed_is_removed",
        "test_assessment_RF_view_changes_only_the_declared_reference",
        "test_existing_monthly_interface_has_known_target_but_unmatured_initial_score",
        "test_invalid_cached_cash_or_leverage_is_not_laundered",
        "test_publication_checks_every_explicit_and_pinned_dependency",
        "test_future_targets_and_RF_cannot_change_observed_prefix",
        "test_actual_identity_does_not_require_financial_pass",
        "test_changed_source_identity_is_not_an_admissible_expert",
        "test_immutable_original_identity_has_no_technical_failure",
    }
    if (
        not required.issubset(names)
        or sum(int(s.get("tests", 0)) for s in suites) < len(required)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Require completed exact-view, funding, causality and publication tests.")
    results = [base.reference(xml), base.reference(FOLDER / "offline-tests-v2.log")]
    receipt = FOLDER / "offline-validation-v2.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"]
        != [
            base.reference(p)
            for p in (POLICY, Path(__file__), TEST, TEST.parent / "test_strategy_experts_v1.py")
        ]
        or proof["canonical_test_outputs"] != results
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Offline validation must bind the exact current implementation bytes.")
    return [*results, base.reference(receipt)]


def admit():
    runtime_guard()
    if INTENT.exists() or OUTPUT.exists() or PUBLIC.exists() or VIEWS.exists():
        raise QuantError(
            "Preserve and read existing complete/partial admission instead of repeating it."
        )
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        policy = read_json(POLICY)
        validate(policy)
        snapshot, bundle, common, refs = declared_inputs(policy)
        own = [base.reference(p) for p in (POLICY, Path(__file__), TEST)] + validations()
        invocation = read_json(INVOCATION)
        event = invocation["native_event"]
        if (
            invocation["native_sequence"] != 41984
            or event["data"]["source"] != "system"
            or event["data"]["delivery"] != "idle"
            or hashlib.sha256(event["data"]["content"].encode()).hexdigest()
            != "46d88b147d139df4fd238fbdbbbc40fbfc543a5250b05d0bfb7c0a73bb937b92"
        ):
            raise QuantError("Record the actual current scheduled origin, not a maintenance turn.")
        before = base.journal_events(base.JOURNAL)
        if len(before) != 467 or before[-1]["sha256"] != policy["prior_ssd_journal_head_sha256"]:
            raise QuantError("Admit against the actual unchanged financial-research history.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "source_selection_continuation_event": 41392,
                "actual_admission_continuation_event": invocation["native_sequence"],
                "actual_invocation": base.reference(INVOCATION),
                "actual_delivery": "system_idle",
                "implementation_and_test_inputs": own,
                "source_inputs": refs,
                "new_strategy_evaluations": 0,
                "new_shadow_account_runs": 0,
                "planned_derived_recommendation_views": 1,
                "planned_derived_assessment_views": 1,
                "source_or_strategy_entrypoints_replayed": False,
                "order_authority": False,
            },
        )
        current.progress(
            "admitting_legacy_defensive_rotation_expert",
            active_compute_job={"pid": os.getpid(), "kind": "cached_expert_admission"},
        )
        try:
            spec = check_identity(
                policy,
                snapshot,
                read_json(base.checked_reference(policy["source_registration"])),
                read_json(base.checked_reference(policy["source_results"])),
                bundle,
            )
            funding = read_json(
                base.checked_reference(policy["cached_native_funding_and_seed_proof"])
            )
            check_cached_funding(funding, policy["source_bundle"], bundle)
            code_refs = [
                base.reference(ROOT / p)
                for p in (
                    policy["retained_code_dependency_paths"]
                    + policy["admission_helper_dependency_paths"]
                )
            ]
            source_publication = published_source_dependencies(
                code_refs,
                policy["source_registration"],
                policy["source_published_dependency_commit"],
            )
            native = review_market(spec, ROOT)
            active = review_market({"market": common["market"]}, ROOT)
            source_frames = {
                k: getattr(native, k) for k in ("open", "close", "raw_close", "volume")
            }
            source_frames["risk_free"] = native.risk_free.to_frame()
            stored = {k: frame(ref) for k, ref in common["market"].items()}
            active_frames = {
                k: getattr(active, k) for k in ("open", "close", "raw_close", "volume")
            }
            active_frames["risk_free"] = active.risk_free.to_frame()
            lineage = check_lineage(source_frames, stored, active_frames)
            nominal, native_shadow_ref = None, None
            for path in bundle["paths"]:
                key = f"{path['years']}y-{path['scenario']}"
                row = next(r for r in funding["rows"] if r["path_key"] == key)
                transformed = nominal_view(
                    frame(path["targets"]), row["synthetic_initial_seed_session"]
                )
                if nominal is None:
                    nominal = transformed
                elif not transformed.equals(nominal):
                    raise QuantError(
                        "All four native windows must preserve the same genuine intentions."
                    )
                if key == "10y-base":
                    native_shadow_ref = path["strategy"]
            if (
                nominal is None
                or native_shadow_ref is None
                or len(nominal.dropna(how="all")) != 128
                or str(nominal.dropna(how="all").index[0].date()) != "2016-02-29"
            ):
                raise QuantError(
                    "The selected source needs every actual genuine monthly recommendation."
                )
            native_shadow = frame(native_shadow_ref)
            shadow = assessment_view(native_shadow, native.risk_free, active.risk_free)
            if (
                len(shadow) != 2512
                or str(shadow.index[0].date()) != "2016-10-06"
                or str(shadow.index[-1].date()) != "2026-10-05"
            ):
                raise QuantError("Keep every original ten-year-base NET shadow observation.")
            new_output_directory(VIEWS)
            write_text_atomic(
                VIEWS / "genuine-monthly-targets.csv", nominal.to_csv(float_format="%.17g")
            )
            write_text_atomic(
                VIEWS / "native-NET-common-assessment-RF.csv", shadow.to_csv(float_format="%.17g")
            )
            target_ref = base.reference(VIEWS / "genuine-monthly-targets.csv")
            shadow_ref = base.reference(VIEWS / "native-NET-common-assessment-RF.csv")
            if not frame(target_ref).equals(nominal) or not frame(shadow_ref).equals(shadow):
                raise QuantError(
                    "Derived views must round-trip exactly, not silently change source values."
                )
            identity = ExpertIdentity(SOURCE, GROUP, utc_now())
            expert = ArchivedMonthlyExpert(identity, nominal, shadow, tuple(active.close.columns))
            initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
            final = expert.observe(pd.Timestamp("2026-10-05"), 2512)
            if (
                initial.target is None
                or str(initial.target_known_at.date()) != "2016-09-30"
                or len(initial.net_returns)
                or initial.score_mature
                or not final.risk_free.equals(active.risk_free.loc[final.risk_free.index])
            ):
                raise QuantError(
                    "Recommendation availability and common-RF scoring maturity stay distinct."
                )
            write_new_receipt(
                VIEWS / "view-provenance-v1.json",
                {
                    "created_at": utc_now(),
                    "admission_intent": base.reference(INTENT),
                    "native_target_reference": bundle["paths"][0]["targets"],
                    "native_NET_shadow_reference": native_shadow_ref,
                    "native_source_RF_reference": spec["market"]["risk_free"],
                    "common_assessment_RF_reference": common["market"]["risk_free"],
                    "target_reference": target_ref,
                    "assessment_shadow_reference": shadow_ref,
                    "exact_serialization_lineage": lineage,
                    "genuine_monthly_rows_unchanged": 128,
                    "assessment_only_RF_changed_cells": int(
                        (native_shadow["risk_free"] != shadow["risk_free"]).sum()
                    ),
                    "NET_NAV_returns_cash_costs_orders_unchanged": True,
                    "original_source_RF_and_standalone_metrics_unchanged": True,
                    "active_common_reader_RF_definition_unchanged": True,
                    "source_strategy_or_account_replayed": False,
                },
            )
            for ref in [*refs, *own]:
                base.checked_reference(ref)
            if (
                base.journal_events(base.JOURNAL) != before
                or file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256
            ):
                raise QuantError("Technical admission changed protected financial history.")
            admitted_at = utc_now()
            report = {
                "id": SOURCE,
                "mechanism_group": GROUP,
                "adapter": "monthly",
                "introduced_at": admitted_at,
                "actual_expert_admitted_at": admitted_at,
                "source_policy_registered_at": snapshot["registered_at"],
                "source_candidate_event_sequence": 28,
                "source_review_event_sequence": 30,
                "source_specification_sha256": policy["source_specification_sha256"],
                "source_bundle_semantic_sha256": policy["source_bundle_semantic_sha256"],
                "original_source_frozen_file_count": len(spec["frozen_files"]),
                "original_source_technical_failure_count": 0,
                "source_native_accounting_contract": NATIVE_ACCOUNTING,
                "source_legacy_engine_relabelled_v2": False,
                "native_target_reference": bundle["paths"][0]["targets"],
                "native_shadow_reference": native_shadow_ref,
                "target_reference": target_ref,
                "shadow_reference": shadow_ref,
                "view_provenance": base.reference(VIEWS / "view-provenance-v1.json"),
                "common_assessment_RF_reference": common["market"]["risk_free"],
                "native_source_RF_equals_active_common_native_RF": lineage["risk_free"][
                    "source_vs_active_common_native_exact"
                ],
                "exact_source_native_to_common_stored_serialization_lineage": True,
                "assessment_view_uses_unchanged_active_common_native_RF": True,
                "native_NET_money_and_original_metrics_unchanged": True,
                "source_native_cached_funding_and_seed_proof": policy[
                    "cached_native_funding_and_seed_proof"
                ],
                "four_native_paths_checked": 4,
                "genuine_monthly_recommendations": 128,
                "first_genuine_recommendation": "2016-02-29",
                "source_standalone_financial_verdict_preserved": snapshot["review"]["status"],
                "counterfactual_historical_availability": True,
                "source_shadow_state_is_not_main_capital_or_drawdown_floor": True,
            }
            body = {
                "admitted_at": admitted_at,
                "admission_policy": base.reference(POLICY),
                "intent": base.reference(INTENT),
                "expert": report,
                "source_publication_dependency_proof": source_publication,
                "source_inputs_checked": len(refs),
                "retained_source_code_dependencies_checked": len(CODE_DEPENDENCIES),
                "current_admission_helper_dependencies_checked": len(ADMISSION_DEPENDENCIES),
                "old_ledger_sha256_unchanged": base.LEGACY_SHA256,
                "journal_events_unchanged": len(before),
                "journal_head_sha256_unchanged": before[-1]["sha256"],
                "new_strategy_evaluations": 0,
                "new_shadow_account_runs": 0,
                "new_recommendation_view_artifacts": 1,
                "new_assessment_view_artifacts": 1,
                "new_source_requests": 0,
                "expanded_pool_or_next_meta_rule_frozen": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_new_receipt(OUTPUT, body)
            write_new_receipt(
                PUBLIC,
                {
                    "schema_version": 1,
                    "admission_id": policy["admission_id"],
                    "admitted_at": admitted_at,
                    "source_candidate_id": SOURCE,
                    "mechanism_group": GROUP,
                    "source_policy_registered_at": snapshot["registered_at"],
                    "source_registration": policy["source_registration"],
                    "source_results": policy["source_results"],
                    "private_admission": base.reference(OUTPUT),
                    "source_candidate_event_sequence": 28,
                    "source_review_event_sequence": 30,
                    "four_native_paths_checked": 4,
                    "genuine_monthly_recommendations": 128,
                    "first_genuine_recommendation": "2016-02-29",
                    "source_native_accounting_contract": report[
                        "source_native_accounting_contract"
                    ],
                    "native_source_RF_equals_active_common_native_RF": report[
                        "native_source_RF_equals_active_common_native_RF"
                    ],
                    "exact_source_native_to_common_stored_serialization_lineage": True,
                    "explicit_versioned_assessment_RF_view": True,
                    "native_NET_money_and_original_metrics_unchanged": True,
                    "active_common_reader_RF_definition_unchanged": True,
                    "new_recommendation_view_artifacts": 1,
                    "new_assessment_view_artifacts": 1,
                    "new_strategy_evaluations": 0,
                    "new_shadow_account_runs": 0,
                    "new_source_requests": 0,
                    "source_or_strategy_pipeline_replayed": False,
                    "expanded_pool_or_next_meta_rule_frozen": False,
                    "original_records_reclassified": False,
                    "raw_prices_documents_targets_or_daily_accounts_exported": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                    "order_authority": False,
                    "automatic_live_deployment": False,
                },
            )
            current.progress(
                "legacy_defensive_rotation_expert_admitted",
                active_compute_job=None,
                defensive_rotation_expert_admission=base.reference(OUTPUT),
                defensive_rotation_expert_public_admission=base.reference(PUBLIC),
                pending_steps=["publish_actual_defensive_rotation_expert_availability"],
            )
            print(
                json.dumps(
                    {
                        "admission": base.reference(OUTPUT),
                        "public": base.reference(PUBLIC),
                        "new_trials": 0,
                    }
                )
            )
        except (
            QuantError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.CalledProcessError,
        ) as error:
            write_new_receipt(
                FOLDER / "admission-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "reason": str(error),
                    "partial_outputs_retained": True,
                    "new_strategy_evaluations": 0,
                },
            )
            current.progress(
                "legacy_defensive_rotation_expert_blocked",
                active_compute_job=None,
                current_failure=str(error),
                partial_outputs_preserved=True,
            )
            raise
        except KeyboardInterrupt:
            current.progress(
                "legacy_defensive_rotation_expert_interrupted",
                active_compute_job=None,
                partial_outputs_preserved=True,
            )
            raise


if __name__ == "__main__":

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Cached expert admission cannot request market sources.")

    sys.addaudithook(offline)
    admit()
