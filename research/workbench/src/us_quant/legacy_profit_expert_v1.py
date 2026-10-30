from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.storage import digest_json, file_digest, read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity

ROOT = base.ROOT
POLICY = ROOT / "config/legacy-profit-expert-v1.json"
FOLDER = base.REPORTS / "profit-expert-admission-20261030-v1"
INTENT = FOLDER / "admission-intent-v1.json"
OUTPUT = FOLDER / "admission-v1.json"
PUBLIC = ROOT / "evidence/legacy_profit_expert_20261030_admission_v1.json"
SOURCE = "semiconductor_option_dollar_profit_share_confirmation"
GROUP = "common_quarter_profit_dollar_confirmation"
TEST = ROOT / "tests/test_legacy_profit_expert_v1.py"
REFERENCE_KEYS = (
    "source_registration",
    "source_results",
    "source_pre_account_proof",
    "source_reproducibility",
    "source_publication",
    "source_bundle",
    "source_nominal_targets",
    "saved_common_queries",
    "saved_comparisons",
    "saved_original_targets",
    "admitted_market_registration",
)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "admission_id": "ssd_legacy_profit_expert_admission_20261030_v1",
        "source_candidate_id": SOURCE,
        "mechanism_group": GROUP,
        "profit_measure": "output_share",
        "source_registered_at": "2026-10-12T04:06:48.358537+00:00",
        "source_specification_semantic_sha256": (
            "ec0a62259849d4d8a02df162ce38d3909ee1dec2b5ff8420a9068e1a376963b8"
        ),
        "source_bundle_semantic_sha256": (
            "199884c2c57a1765539bd7bad6211a584447637fde6d116804161312d4a0aa27"
        ),
        "source_registration_event_sequence": 219,
        "source_review_event_sequence": 221,
        "shadow_reference_path": "10y-base",
        "source_initial_capital_usd": 10000.0,
        "source_actual_weight_trade_band": 0.025,
        "maximum_completed_quarter_age_days": 186,
        "risk_change_sessions": 63,
        "risk_off_scale": 0.5,
        "invested_target_budget": 0.98,
        "source_frozen_file_count": 520,
        "original_ledger_sha256": base.LEGACY_SHA256,
        "prior_ssd_journal_sequence": 432,
        "prior_ssd_journal_head_sha256": (
            "271652e16e9df7486d8f58783afeb87b97343b0f10af4d14a731e54e8e058904"
        ),
        "new_strategy_evaluations": 0,
        "new_shadow_account_runs": 0,
        "new_recommendation_artifacts": 0,
        "new_economic_factor_definitions": 0,
        "new_source_requests": 0,
    }
    if not isinstance(policy, dict) or any(
        k not in policy or type(policy[k]) is not type(v) or policy[k] != v
        for k, v in fixed.items()
    ):
        raise QuantError(
            "Keep the fixed macro representative, original source contract and no-replay scope."
        )
    if (
        any(
            not isinstance(policy.get(k), dict) or set(policy[k]) != {"path", "sha256"}
            for k in REFERENCE_KEYS
        )
        or not isinstance(policy.get("methodology"), dict)
        or any(
            policy["methodology"].get(k) is not False
            for k in (
                "source_or_strategy_pipeline_replayed",
                "aggregate_proxy_claimed_company_margin_or_independent_factor",
                "expanded_pool_or_next_meta_rule_frozen",
                "final_portfolio_gates_changed",
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError(
            "Source evidence and technical admission cannot imply financial qualification."
        )


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def legacy_record():
    path = ROOT / "runtime/research-program.sqlite3"
    if file_digest(path) != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        candidate = db.execute(
            "SELECT spec_sha,registered_at,body FROM candidates WHERE id=?", (SOURCE,)
        ).fetchone()
        review = db.execute(
            "SELECT at,evidence_sha,body FROM reviews WHERE candidate_id=?", (SOURCE,)
        ).fetchone()
        invalid = db.execute(
            "SELECT body FROM technical_failures WHERE candidate_id=?", (SOURCE,)
        ).fetchone()
        events = {
            seq: {"seq": seq, "at": at, "kind": kind, "body": json.loads(body), "sha": sha}
            for seq, at, kind, body, sha in db.execute(
                "SELECT seq,at,kind,body,sha FROM events WHERE seq IN (219,221)"
            )
        }
    if candidate is None or review is None or invalid is not None or set(events) != {219, 221}:
        raise QuantError("A missing or technically invalid legacy policy cannot enter the pool.")
    return {
        "spec_sha": candidate[0],
        "registered_at": candidate[1],
        "spec": json.loads(candidate[2]),
        "reviewed_at": review[0],
        "evidence_sha": review[1],
        "review": json.loads(review[2]),
        "events": events,
    }


def check_identity(policy, registration, entry, bundle, ledger, source_results, publication):
    spec, review = entry["spec"], ledger["review"]
    reg = entry["registration"]
    registered, fenced, started = (
        pd.Timestamp(reg["registered_at"]),
        pd.Timestamp(publication["verified_on_remote_at"]),
        pd.Timestamp(source_results["started_at"]),
    )
    if (
        spec != ledger["spec"]
        or digest_json(spec) != ledger["spec_sha"]
        or ledger["spec_sha"] != policy["source_specification_semantic_sha256"]
        or digest_json(bundle) != ledger["evidence_sha"]
        or ledger["evidence_sha"] != policy["source_bundle_semantic_sha256"]
        or bundle["candidate_spec_sha256"] != ledger["spec_sha"]
        or spec["id"] != SOURCE
        or spec["configuration"] != {"id": SOURCE, "profit_measure": "output_share"}
        or reg["candidate_id"] != SOURCE
        or reg["spec_sha256"] != ledger["spec_sha"]
        or reg["registered_at"] != ledger["registered_at"]
        or ledger["registered_at"] != policy["source_registered_at"]
        or ledger["events"][219]["kind"] != "candidate"
        or ledger["events"][219]["body"] != reg
        or ledger["events"][219]["at"] != ledger["registered_at"]
        or ledger["events"][221]["kind"] != "review"
        or ledger["events"][221]["body"] != review
        or ledger["events"][221]["at"] != ledger["reviewed_at"]
        or review["candidate_id"] != SOURCE
        or review["live_strategy_update"] is not False
        or review["evidence_sha256"] != ledger["evidence_sha"]
        or review["status"] not in ("rejected_historical", "historically_qualified")
        or any(
            value.get("order_authority") is not False
            for value in (spec, bundle, review, registration, source_results, publication)
        )
        or spec["leveraged_products_allowed"] is not False
        or bundle["leveraged_products_allowed"] is not False
        or not spec["asset_leverage"]
        or any(v != 1.0 for v in spec["asset_leverage"].values())
        or bundle["market"] != spec["market"]
        or review["accounting_engine"] != bundle["accounting_engine"]
        or bundle["accounting_engine"]
        != {
            "id": "cash_funded_noop_v2",
            "source_sha256": file_digest(ROOT / "src/us_quant/cash_funded_accounting_v2.py"),
        }
        or registration["registered_before_strategy_accounts"] is not True
        or publication["publication_fence_predates_new_strategy_accounts"] is not True
        or publication["published_commit"] != source_results["published_registration_commit"]
        or publication["observed_remote_head"] != publication["published_commit"]
        or any(t.tzinfo is None or pd.isna(t) for t in (registered, fenced, started))
        or not registered <= fenced <= started
        or pd.Timestamp(bundle["completed_at"]) > pd.Timestamp(ledger["reviewed_at"])
    ):
        raise QuantError(
            "The actual legacy identity, funding permission or publication sequence changed."
        )


def check_macro_tables(common, comparisons, original, desired, market_index):
    pairs = decision_pairs(market_index)
    queries = pd.DatetimeIndex(sorted(set(pairs) | set(pairs.values())))
    dates = pd.DatetimeIndex(list(pairs))
    flags = {
        "snapshot_available_two_nyse_sessions_before_query": True,
        "quarter_label_claimed_first_release": False,
        "same_quarter_selected_in_both_original_query_snapshots": True,
        "macro_proxy_claimed_company_margin": False,
    }
    if (
        not common.index.equals(queries)
        or not comparisons.index.equals(dates)
        or not original.index.equals(market_index)
        or not desired.index.equals(market_index)
        or not original.columns.equals(desired.columns)
        or "BIL" not in original
        or not original.dropna(how="all").index.equals(dates)
        or not desired.isna().equals(original.isna())
        or any(
            not pd.api.types.is_bool_dtype(common[k]) or not (common[k] == v).all()
            for k, v in flags.items()
        )
        or not np.isfinite(common[["CPATAX", "GDP", "profit_share"]]).all().all()
        or (common["GDP"] <= 0).any()
        or not np.array_equal(common["profit_share"], common["CPATAX"] / common["GDP"])
        or not np.isfinite(comparisons).all().all()
    ):
        raise QuantError(
            "Cached macro queries, same-quarter ratios or nominal target lattice changed."
        )
    for query, row in common.iterrows():
        cutoff = previous_session(previous_session(query))
        quarter = pd.Timestamp(row["observation_quarter"])
        completed = quarter + pd.offsets.QuarterEnd(startingMonth=3)
        age = (query - completed).days
        if (
            row["vintage_cutoff"] != str(cutoff.date())
            or quarter.day != 1
            or quarter.month not in (1, 4, 7, 10)
            or row["completed_period_end"] != str(completed.date())
            or completed > cutoff
            or not 0 <= age <= 186
            or row["completed_period_age_calendar_days"] != age
            or quarter > pd.Timestamp(row["independently_latest_profit_quarter"])
            or quarter > pd.Timestamp(row["independently_latest_output_quarter"])
        ):
            raise QuantError(
                "A cached quarterly observation violates its original causal cutoff or age."
            )
    maximum_target_error = 0.0
    triggered = 0
    non_bil = original.columns.drop("BIL")
    for day, earlier in pairs.items():
        row = comparisons.loc[day]
        for field in ("CPATAX", "profit_share"):
            if (
                row[f"{field}_level"] != common.loc[day, field]
                or row[f"{field}_reference_level"] != common.loc[earlier, field]
                or row[f"{field}_change"] != row[f"{field}_level"] - row[f"{field}_reference_level"]
            ):
                raise QuantError(
                    "An own-query profit comparison cannot use a different vintage or quarter."
                )
        expected = original.loc[day].copy()
        if row["dollar_change"] > 0 and row["profit_share_change"] < 0:
            expected.loc[non_bil] *= 0.5
            expected["BIL"] += float(original.loc[day, non_bil].sum()) * 0.5
            triggered += 1
        error = float(abs(desired.loc[day] - expected).max())
        if (
            not np.isfinite(desired.loc[day]).all()
            or (desired.loc[day] < 0).any()
            or abs(float(desired.loc[day].sum()) - 0.98) > 1e-12
            or error > 1e-12
        ):
            raise QuantError(
                "A cached nominal recommendation differs from the frozen source policy."
            )
        maximum_target_error = max(maximum_target_error, error)
    return {
        "common_query_rows_checked": len(common),
        "monthly_nominal_rows_checked": len(dates),
        "profit_share_confirmed_dollar_risk_off_months": triggered,
        "maximum_cached_rule_target_difference": maximum_target_error,
        "raw_snapshot_selector_or_original_target_generator_rerun": False,
        "new_target_artifact_generated": False,
    }


def check_native_book(account, weights, independent, issued, nominal, market, dates):
    if (
        not account.index.equals(dates)
        or not weights.index.equals(dates)
        or not independent.index.equals(dates)
        or not weights.columns.equals(market.close.columns)
        or not issued.index.equals(market.close.index)
        or not issued.columns.equals(market.close.columns)
        or not np.isfinite(account).all().all()
        or not np.isfinite(weights).all().all()
        or (account["equity"] <= 0).any()
        or (account["cash"] < 0).any()
        or (weights < 0).any().any()
        or (account["gross_exposure"] < 0).any()
        or (account["gross_exposure"] > 1).any()
        or (account[["cost", "orders", "turnover"]] < 0).any().any()
        or (account["orders"] != account["orders"].astype(int)).any()
        or not account["risk_free"].equals(market.risk_free.loc[dates])
        or not np.allclose(weights.sum(axis=1), account["gross_exposure"], rtol=0, atol=1e-12)
        or float(
            abs(weights.sum(axis=1) * account["equity"] + account["cash"] - account["equity"]).max()
        )
        > 1e-6
    ):
        raise QuantError(
            "A cached native book is incomplete, borrowed, misaligned or not its original RF."
        )
    capital = float(account.iloc[0]["equity"]) / (1 + float(account.iloc[0]["return"]))
    difference = float(abs(account["equity"] - independent["equity"]).max())
    if (
        not np.isfinite(independent).all().all()
        or abs(capital - 10000) > 1e-8
        or difference > 1e-6
        or not np.allclose(account["return"], independent["return"], rtol=0, atol=1e-10)
    ):
        raise QuantError("The cached strategy and independent BT disagree on actual funded wealth.")
    active = issued.dropna(how="all")
    anchor = previous_session(dates[0])
    if (
        active.empty
        or active.index[0] != anchor
        or (issued.isna().any(axis=1) & ~issued.isna().all(axis=1)).any()
        or not np.isfinite(active).all().all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Keep actual issued target funding and the real prewindow initial anchor.")
    for day, target in active.iterrows():
        known = nominal.loc[:day].dropna(how="all")
        if (
            known.empty
            or day != anchor
            and not is_month_end(day)
            or float(abs(target - known.iloc[-1]).max()) > 1e-12
        ):
            raise QuantError(
                "A native request is not its actually known monthly nominal recommendation."
            )
    return {
        "sessions_checked": len(dates),
        "inferred_initial_capital_usd": capital,
        "cached_BT_maximum_NAV_difference_usd": difference,
        "minimum_cash_usd": float(account["cash"].min()),
        "maximum_gross_exposure": float(account["gross_exposure"].max()),
        "cached_issued_targets_checked": len(active),
        "original_cash_funding_verification_proof_reused": True,
        "old_account_BT_or_primary_metrics_recomputed": False,
    }


def declared_inputs(policy):
    validate(policy)
    values = {
        k: read_json(base.checked_reference(policy[k]))
        for k in (
            "source_registration",
            "source_results",
            "source_pre_account_proof",
            "source_reproducibility",
            "source_publication",
            "source_bundle",
        )
    }
    entries = [e for e in values["source_registration"]["candidates"] if e["spec"]["id"] == SOURCE]
    if len(entries) != 1:
        raise QuantError("Require one actual original profit-share registration.")
    entry = entries[0]
    refs = [policy[k] for k in REFERENCE_KEYS]
    refs.extend({"path": k, "sha256": v} for k, v in entry["spec"]["frozen_files"].items())
    refs.extend((entry["candidate_spec"], entry["readiness"]))
    refs.extend(
        r
        for p in values["source_bundle"]["paths"]
        for r in p.values()
        if isinstance(r, dict) and set(r) == {"path", "sha256"}
    )
    unique = {}
    for ref in refs:
        if ref["path"] in unique and unique[ref["path"]]["sha256"] != ref["sha256"]:
            raise QuantError("Conflicting source input hashes must not be overwritten.")
        unique[ref["path"]] = {"path": ref["path"], "sha256": ref["sha256"]}
    return entry, values, [unique[k] for k in sorted(unique)]


def load_profit_expert(policy):
    entry, values, refs = declared_inputs(policy)
    for ref in refs:
        base.checked_reference(ref)
    registration, bundle = values["source_registration"], values["source_bundle"]
    ledger = legacy_record()
    check_identity(
        policy,
        registration,
        entry,
        bundle,
        ledger,
        values["source_results"],
        values["source_publication"],
    )
    candidate_file = read_json(base.checked_reference(entry["candidate_spec"]))
    source_readiness = read_json(base.checked_reference(entry["readiness"]))
    source_policy_file = read_json(base.checked_reference(registration["policy_reference"]))
    if (
        candidate_file != entry["spec"]
        or digest_json(candidate_file) != entry["candidate_spec"]["body_sha256"]
        or digest_json(source_readiness) != entry["registration"]["readiness_sha256"]
        or digest_json(source_readiness) != entry["readiness"]["body_sha256"]
        or source_policy_file != registration["policy"]
        or digest_json(source_policy_file) != registration["policy_reference"]["body_sha256"]
        or registration["pre_account_proof"]["sha256"]
        != policy["source_pre_account_proof"]["sha256"]
    ):
        raise QuantError(
            "The original registered spec, readiness or pre-account proof was relabelled."
        )
    try:
        original_blob = subprocess.run(
            [
                "git",
                "show",
                f"{values['source_publication']['published_commit']}:research/workbench/{policy['source_registration']['path']}",
            ],
            cwd=base.REPO,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise QuantError("The original published registration Git blob is unavailable.") from error
    if sha256(original_blob).hexdigest() != policy["source_registration"]["sha256"]:
        raise QuantError("The original published registration differs from its actual Git blob.")
    spec = entry["spec"]
    if len(spec["frozen_files"]) != 520:
        raise QuantError("Do not remove original source dependencies from admission.")
    pre, repro = values["source_pre_account_proof"], values["source_reproducibility"]
    if (
        pre["candidate_returns_or_metrics_computed"] is not False
        or pre["actual_path_dependent_trade_targets_computed"] is not False
        or pre["independent_raw_quarterly_snapshots_checked"] != 450
        or pre["independent_common_quarter_joins_checked"] != 225
        or pre["original_latest_quarter_mismatches_handled_without_mixing"] != 87
        or pre["prepared_source_target_sha256"][SOURCE]
        != policy["source_nominal_targets"]["sha256"]
        or pre["prepared_common_quarter_queries_sha256"] != policy["saved_common_queries"]["sha256"]
        or pre["prepared_risk_comparisons_sha256"] != policy["saved_comparisons"]["sha256"]
        or pre["original_source_desired_sha256"] != policy["saved_original_targets"]["sha256"]
        or repro[
            "all_common_quarter_raw_sources_comparisons_targets_feedback_funding_and_gates_checked"
        ]
        is not True
        or repro["actual_publication_fence_predates_new_feedback_accounts"] is not True
        or repro["initial_capital_drawdown_peak_included"] is not True
        or repro["results_sha256"] != policy["source_results"]["sha256"]
    ):
        raise QuantError("The original independent cached source/account proof changed.")
    source_policy = registration["policy"]
    readiness = read_json(base.checked_reference(source_policy["profit_readiness"]))
    manifest = read_json(base.checked_reference(source_policy["profit_manifest"]))
    if (
        readiness["full_source_readiness"] is not True
        or readiness["actual_vintage_snapshots"] != 450
        or readiness["fresh_query_selections"] != 450
        or readiness["blocked_query_selections"] != 0
        or readiness["queries_with_different_latest_profit_output_quarters"] != 87
        or readiness["complete_historical_methodology_or_intraday_first_release_chain_verified"]
        is not False
        or manifest["all_requested_batches_complete"] is not True
        or manifest["full_source_readiness"] is not True
    ):
        raise QuantError("Preserve the actual source scope and historical release limitations.")
    source_market = review_market({"market": spec["market"]}, ROOT)
    admitted = read_json(base.checked_reference(policy["admitted_market_registration"]))
    shared = review_market(
        {"market": admitted["entries"][0]["specification"]["frozen"]["market"]}, ROOT
    )
    if not source_market.close.index.equals(
        shared.close.index
    ) or not source_market.risk_free.equals(shared.risk_free):
        raise QuantError("Original and current admitted market calendars or RF changed.")
    for name in ("open", "close", "raw_close", "volume"):
        left, right = getattr(source_market, name), getattr(shared, name)
        if not set(left.columns).issubset(right.columns) or not left.equals(
            right.loc[:, left.columns]
        ):
            raise QuantError("A cached legacy price/volume cell differs from the admitted panel.")
    if set(spec["asset_leverage"]) != set(source_market.close.columns):
        raise QuantError("Every source instrument needs its actual unleveraged permission.")
    nominal = frame(policy["source_nominal_targets"])
    common = frame(policy["saved_common_queries"])
    macro = check_macro_tables(
        common,
        frame(policy["saved_comparisons"]),
        frame(policy["saved_original_targets"]),
        nominal,
        source_market.close.index,
    )
    if (
        macro["common_query_rows_checked"] != 225
        or macro["monthly_nominal_rows_checked"] != 131
        or int(
            (
                common["independently_latest_profit_quarter"]
                != common["independently_latest_output_quarter"]
            ).sum()
        )
        != 87
    ):
        raise QuantError("All original macro queries and recommendation opportunities must remain.")
    paths, shadow_ref = {}, None
    for path in bundle["paths"]:
        key = f"{path['years']}y-{path['scenario']}"
        if (
            key not in base.PATH_KEYS
            or key in paths
            or path["capital_usd"] != 10000
            or path["commission_per_order"] != 1
            or (path["cost_bps"], path["delay_sessions"])
            != ((5, 1) if path["scenario"] == "base" else (20, 2))
        ):
            raise QuantError(
                "Keep all four actual source path identities and execution assumptions."
            )
        dates = sessions("2016-10-06" if path["years"] == 10 else "2021-10-06", "2026-10-05")
        paths[key] = check_native_book(
            frame(path["strategy"]),
            frame(path["weights"]),
            frame(path["strategy_bt"]),
            frame(path["targets"]),
            nominal,
            source_market,
            dates,
        )
        if key == "10y-base":
            shadow_ref = path["strategy"]
    if set(paths) != base.PATH_KEYS or shadow_ref is None:
        raise QuantError("A partial historical policy cannot become a complete strategy expert.")
    admitted_at = utc_now()
    introduced = pd.Timestamp(admitted_at)
    if (
        introduced.tzinfo is None
        or pd.isna(introduced)
        or introduced < pd.Timestamp(ledger["registered_at"])
    ):
        raise QuantError("Actual expert admission cannot precede original policy registration.")
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(SOURCE, GROUP, admitted_at),
        nominal,
        frame(shadow_ref),
        tuple(shared.close.columns),
    )
    initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
    if initial.target is None or initial.target_known_at > initial.as_of or initial.score_mature:
        raise QuantError("Known historical recommendation and score maturity must stay distinct.")
    return (
        expert,
        {
            "id": SOURCE,
            "mechanism_group": GROUP,
            "adapter": "monthly",
            "introduced_at": admitted_at,
            "actual_expert_admitted_at": admitted_at,
            "source_policy_registered_at": ledger["registered_at"],
            "source_registration_event_sequence": 219,
            "source_review_event_sequence": 221,
            "source_specification_semantic_sha256": ledger["spec_sha"],
            "source_bundle_semantic_sha256": ledger["evidence_sha"],
            "target_reference": policy["source_nominal_targets"],
            "shadow_reference": shadow_ref,
            "first_known_historical_recommendation": str(nominal.dropna(how="all").index[0].date()),
            "original_standalone_financial_verdict_preserved": ledger["review"]["status"],
            "source_frozen_files_checked": len(spec["frozen_files"]),
            "saved_macro_rule_checks": macro,
            "cached_native_paths": paths,
            "source_market_12_assets_and_RF_match_admitted_14": True,
            "existing_raw_snapshot_selector_and_funding_proofs_reused": True,
            "source_intraday_first_release_or_full_historical_method_chain_verified": False,
            "source_publication_is_independent_first_publication_timestamp_proof": False,
            "counterfactual_historical_availability": True,
            "shadow_state_is_not_main_capital_or_net_wealth_guarantee": True,
        },
        refs,
    )


def admit():
    runtime_guard()
    for path in (INTENT, OUTPUT, PUBLIC):
        if path.exists():
            raise QuantError("Preserve the existing complete or partial admission, not repeat it.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        policy = read_json(POLICY)
        _, _, refs = declared_inputs(policy)
        xml = FOLDER / "offline-tests-v1.xml"
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        names = {
            case.get("name", "").split("[", 1)[0]
            for suite in suites
            for case in suite.findall("testcase")
        }
        required = {
            "test_actual_legacy_identity_does_not_require_financial_pass",
            "test_saved_common_quarter_rule_uses_two_prior_sessions_and_exact_queries",
            "test_cached_book_matches_existing_BT_without_account_reexecution",
            "test_existing_monthly_adapter_preserves_known_targets_and_net_prefixes",
        }
        if (
            not required.issubset(names)
            or sum(int(s.get("tests", 0)) for s in suites) < len(required)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Require completed independent legacy-admission contract tests.")
        own = [
            base.reference(p)
            for p in (POLICY, Path(__file__), TEST, xml, FOLDER / "offline-tests-v1.log")
        ]
        before = base.journal_events(base.JOURNAL)
        if len(before) != 432 or before[-1]["sha256"] != policy["prior_ssd_journal_head_sha256"]:
            raise QuantError("Admit against the actual unchanged current research history.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 38584,
                "actual_delivery": "system_idle",
                "implementation_and_test_inputs": own,
                "source_inputs": refs,
                "source_or_strategy_entrypoints_replayed": False,
                "new_strategy_evaluations": 0,
                "new_shadow_account_runs": 0,
                "order_authority": False,
            },
        )
        current.progress(
            "admitting_legacy_profit_expert",
            active_compute_job={"pid": os.getpid(), "kind": "read_only_profit_admission"},
        )
        try:
            _, report, checked = load_profit_expert(policy)
            admitted_at = report["actual_expert_admitted_at"]
            for ref in [*refs, *own]:
                base.checked_reference(ref)
            if (
                base.journal_events(base.JOURNAL) != before
                or file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256
            ):
                raise QuantError("A read-only expert admission changed protected research history.")
            runtime_guard()
            body = {
                "admitted_at": admitted_at,
                "admission_policy": base.reference(POLICY),
                "intent": base.reference(INTENT),
                "expert": report,
                "source_inputs_checked": len(checked),
                "old_ledger_sha256_unchanged": base.LEGACY_SHA256,
                "journal_events_unchanged": len(before),
                "journal_head_sha256_unchanged": before[-1]["sha256"],
                "new_strategy_evaluations": 0,
                "new_shadow_account_runs": 0,
                "new_recommendation_artifacts": 0,
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
                    "source_policy_registered_at": report["source_policy_registered_at"],
                    "source_registration": policy["source_registration"],
                    "source_results": policy["source_results"],
                    "private_admission": base.reference(OUTPUT),
                    "source_frozen_files_checked": report["source_frozen_files_checked"],
                    "common_query_rows_checked": report["saved_macro_rule_checks"][
                        "common_query_rows_checked"
                    ],
                    "monthly_nominal_rows_checked": report["saved_macro_rule_checks"][
                        "monthly_nominal_rows_checked"
                    ],
                    "cached_native_funded_paths_checked": len(report["cached_native_paths"]),
                    "source_market_12_assets_and_RF_match_admitted_14": True,
                    "existing_raw_snapshot_and_funding_proofs_reused_without_reexecution": True,
                    "original_standalone_financial_rejection_preserved": True,
                    "company_profit_margin_or_independent_factor_claimed": False,
                    "intraday_first_release_or_full_historical_method_chain_verified": False,
                    "counterfactual_historical_availability": True,
                    "expanded_pool_or_next_meta_rule_frozen": False,
                    "new_strategy_evaluations": 0,
                    "new_shadow_account_runs": 0,
                    "new_recommendation_artifacts": 0,
                    "new_source_requests": 0,
                    "original_records_reclassified": False,
                    "raw_prices_documents_targets_or_daily_accounts_exported": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "legacy_profit_expert_admitted_waiting_publication",
                active_compute_job=None,
                legacy_profit_expert_admission=base.reference(OUTPUT),
                legacy_profit_expert_public_admission=base.reference(PUBLIC),
                pending_steps=[
                    "publish_sanitized_profit_expert_admission_before_future_pool_freeze"
                ],
            )
            return base.reference(OUTPUT)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            current.progress(
                "legacy_profit_expert_admission_blocked",
                active_compute_job=None,
                current_failure=str(error),
                partial_outputs_preserved=True,
            )
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Read-only legacy macro policy admission; no trading."
    )
    parser.add_argument("command", choices=("admit", "status"))
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Archived profit expert admission cannot contact live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "admit":
            result = admit()
        else:
            if not OUTPUT.exists():
                raise QuantError("No completed profit expert admission exists.")
            result = base.reference(OUTPUT)
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Profit expert admission blocked: {error}\n")


if __name__ == "__main__":
    main()
