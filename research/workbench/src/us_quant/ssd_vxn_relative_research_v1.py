from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_hmm_growth_research_v1 as donor
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as source
from us_quant import vxn_inputs_v1 as admitted
from us_quant import vxn_relative_uncertainty_v1 as rules
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_text_atomic,
)

ROOT = base.ROOT
FOLDER = base.REPORTS / "vxn-relative-uncertainty-20261029-v1"
TARGETS = ROOT / "data/ssd-vxn-relative-uncertainty-20261029-v1"
PACKET = ROOT / "evidence/vxn_relative_uncertainty_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/vxn_relative_uncertainty_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/vxn-relative-uncertainty-v1.json",
    "src/us_quant/vxn_relative_uncertainty_v1.py",
    "src/us_quant/ssd_vxn_relative_research_v1.py",
    "tests/test_vxn_relative_uncertainty_v1.py",
    "tests/test_ssd_vxn_relative_research_v1.py",
)
SOURCE_EXTENSIONS = (
    "config/vxn-inputs-v1.json",
    "src/us_quant/vxn_inputs_v1.py",
    "tests/test_vxn_inputs_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    *SOURCE_EXTENSIONS,
    "evidence/vxn_input_readiness_20261029_v1.json",
    "evidence/hmm_growth_bil_20261029_registration_v1.json",
    "src/us_quant/relative_equity_odds_v1.py",
    "config/relative-equity-odds-v1.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def information_inputs(policy, index):
    readiness = read_json(base.checked_reference(policy["vxn_source_readiness"]))
    manifest = read_json(base.checked_reference(policy["vxn_source_manifest"]))
    source_policy = read_json(base.checked_reference(policy["vxn_source_policy"]))
    admitted.validate_policy(source_policy)
    if (
        readiness["private_manifest"] != policy["vxn_source_manifest"]
        or readiness["policy"] != policy["vxn_source_policy"]
        or manifest["policy"] != policy["vxn_source_policy"]
        or manifest["daily_closes"] != policy["vxn_daily_closes"]
        or any(row["source_admitted"] is not True for row in (readiness, manifest))
        or any(
            row["admission_scope"] != "archival_daily_closes_and_original_query_coverage_only"
            for row in (readiness, manifest)
        )
        or any(
            not row["coverage"]["full_daily_coverage"]
            or not row["coverage"]["all_original_monthly_queries_covered"]
            for row in (readiness, manifest)
        )
        or any(
            row["ratios_strategy_states_targets_or_returns_computed"] is not False
            for row in (readiness, manifest)
        )
        or manifest["old_source_files_or_forward_models_changed"] is not False
    ):
        raise QuantError(
            "Only the complete admitted historical VXN/VIX source may enter this study."
        )
    refs = [
        policy["vxn_source_readiness"],
        policy["vxn_source_manifest"],
        policy["vxn_source_policy"],
        policy["vxn_daily_closes"],
        manifest["monthly_closes"],
        manifest["validation_intent"],
        *manifest["input_refs"],
    ]
    for ref in refs:
        base.checked_reference(ref)
    information = source.frame(policy["vxn_daily_closes"])
    if (
        not information.index.equals(index)
        or list(information.columns) != ["VXN", "VIX"]
        or not np.isfinite(information).all().all()
        or (information <= 0).any().any()
    ):
        raise QuantError(
            "Keep every original dated native option close, without filling or new queries."
        )
    return information, refs


def inputs():
    events = base.journal_events(base.JOURNAL)
    if len(events) < 309 or events[308]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError(
            "Preserve all204completed configurations and the original technical failure."
        )
    data, _, _, market_source, original, benchmarks = source.inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["source_registration"]))["entries"][0]
    parent = entry["specification"]["frozen"]
    check_frozen(parent)
    donor.checked_packet(parent)
    measurement = read_json(base.checked_reference(policy["prior_measurement_policy"]))
    base.checked_reference(policy["prior_structural_odds_policy"])
    if (
        policy["source_registration"] != base.reference(donor.PACKET)
        or entry["candidate_id"] != policy["source_candidate"]
        or entry["target_proof"]["targets"] != policy["source_targets"]
        or measurement["realized_return_sessions"] != 21
        or measurement["annualization_sessions"] != 252
        or measurement["segments_per_session"] != 2
        or any(
            parent[key] != market_source[key]
            for key in ("market", "protocol", "benchmark_artifacts", "legacy")
        )
    ):
        raise QuantError("The fixed source carrier, measurement and original market cannot change.")
    desired = source.frame(policy["source_targets"])
    information, refs = information_inputs(policy, data.close.index)
    proof_ref = entry["target_proof"]["proof"]
    proof = read_json(base.checked_reference(proof_ref))
    dates = rules.eligible_dates(data, desired)
    if (
        len(data.close) != 2805
        or len(dates) != 122
        or str(dates[0].date()) != "2016-08-31"
        or len(proof["retained_model_fit_receipts"]) != 122
    ):
        raise QuantError("Keep the full original context and every saved HMM source month.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"]
        + [base.reference(ROOT / name) for name in OWN_EXTENSIONS[1:]]
        + [
            base.reference(ROOT / name)
            for name in (
                "src/us_quant/relative_equity_odds_v1.py",
                "src/us_quant/vxn_inputs_v1.py",
                "tests/test_vxn_inputs_v1.py",
            )
        ],
        "policies": parent["policies"]
        + [
            base.reference(rules.POLICY),
            policy["vxn_source_policy"],
            policy["prior_measurement_policy"],
            policy["prior_structural_odds_policy"],
        ],
        "source_candidate": policy["source_candidate"],
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "vxn_source_readiness": policy["vxn_source_readiness"],
        "vxn_source_manifest": policy["vxn_source_manifest"],
        "vxn_daily_closes": policy["vxn_daily_closes"],
        "prior_completed_configurations": 204,
        "prior_journal_sequence": 309,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "underlying_matched_relative_option_uncertainty_stock_composition",
        "nonstock_targets_and_total_equity_preserved": True,
        "hmm_refitted_or_states_recomputed": False,
        "parent_target_values_or_macro_option_values_used": True,
        "parent_information_role": "saved_HMM_targets_only_not_a_new_macro_or_old_option_signal",
        "original_policy_role": "frozen_HMM_total_stock_and_cash_budget_only",
        "prior_evidence": parent["prior_evidence"]
        + refs
        + [
            policy["source_registration"],
            policy["source_targets"],
            proof_ref,
            proof["model_audits"],
            *proof["retained_model_fit_receipts"],
            base.reference(donor.PUBLIC_RESULTS),
            base.reference(FOLDER / "research-intent.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "vxn-relative-offline-tests-v1.xml",
                "vxn-relative-offline-tests-v1.log",
            )
        ],
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, desired, information, frozen, original, benchmarks


def verify_saved_targets(data, desired, information, targets, audits):
    dates = rules.eligible_dates(data, desired)
    nonstock = desired.columns.difference(rules.STOCKS)
    if (
        len(dates) != 122
        or len(audits) != 122
        or not targets.index.equals(desired.index)
        or not targets.columns.equals(desired.columns)
        or not targets.dropna().index.equals(dates)
        or not np.array_equal(targets.loc[dates, nonstock], desired.loc[dates, nonstock])
        or [row["decision_session"] for row in audits] != dates.strftime("%Y-%m-%d").tolist()
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
    ):
        raise QuantError("Preserve every source date, undefined prehistory and nonstock target.")
    maximum_target = maximum_share = maximum_variation = maximum_ratio_relative = 0.0
    for day, audit in zip(dates, audits, strict=True):
        equity = rules.source_equity(desired.loc[day])
        oracle, share, variations = rules.independent_row(
            data, desired.loc[day], information.loc[day], day
        )
        error = float(np.max(abs(targets.loc[day] - oracle)))
        share_error = abs(audit["SOXX_share_of_stock_budget"] - share)
        if not np.isfinite(error + share_error) or max(error, share_error) > 1e-12:
            raise QuantError("Independent relative-uncertainty allocation differs.")
        maximum_target, maximum_share = max(maximum_target, error), max(maximum_share, share_error)
        if (
            audit["VXN_quote_session"] != str(day.date())
            or audit["VIX_quote_session"] != str(day.date())
            or audit["VXN_native_quote"] != information.loc[day, "VXN"]
            or audit["VIX_native_quote"] != information.loc[day, "VIX"]
            or audit["information_clock_after_16_15_ET"] is not True
            or any(
                audit[key] is not False
                for key in (
                    "hmm_refitted_or_states_recomputed",
                    "exact_portfolio_or_SOXX_implied_variance_claimed",
                    "true_variance_risk_premium_claimed",
                )
            )
            or abs(audit["source_equity_budget_preserved"] - equity) > 1e-12
            or abs(float(targets.loc[day, list(rules.STOCKS)].sum()) - equity) > 1e-12
            or audit["BIL_target_unchanged"] != desired.loc[day, "BIL"]
            or audit["GLD_target_unchanged"] != desired.loc[day, "GLD"]
        ):
            raise QuantError("Source capital, dated quotes or causal interpretation changed.")
        position = data.close.index.get_loc(day)
        for symbol, market, quote in (("SPY", "SPX", "VIX"), ("QQQ", "NDX", "VXN")):
            record = audit["observed_variation"][symbol]
            difference = abs(
                record["two_segment_squared_log_variation_annualized"] - variations[symbol]
            )
            implied = (float(information.loc[day, quote]) / 100) ** 2
            ratio = implied / variations[symbol]
            ratio_relative = abs(audit[f"{market}_implied_to_observed_ratio"] - ratio) / max(
                1.0, abs(ratio)
            )
            if (
                record["symbol"] != symbol
                or record["first_prior_close"] != str(data.close.index[position - 21].date())
                or record["first_open_session"] != str(data.close.index[position - 20].date())
                or record["last_observation"] != str(day.date())
                or record["return_sessions"] != 21
                or record["segments"] != 42
                or not np.isfinite(difference + ratio_relative)
                or difference > 1e-12
                or ratio_relative > 1e-10
                or abs(audit[f"{market}_implied_variance_proxy"] - implied) > 1e-12
            ):
                raise QuantError(
                    "Independent market-proxy variation, odds or unit conversion differs."
                )
            maximum_variation = max(maximum_variation, difference)
            maximum_ratio_relative = max(maximum_ratio_relative, ratio_relative)
    return maximum_target, maximum_share, maximum_variation, maximum_ratio_relative


def prove_targets(data, desired, information, protocol):
    targets, audits = rules.build_targets(data, desired, information, read_json(rules.POLICY))
    target = targets[IDS[0]]
    errors = verify_saved_targets(data, desired, information, target, audits[IDS[0]])
    dates = desired.dropna().index
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if (
            anchor not in known.index
            or known.isna().any().any()
            or any(not (known[name] > 1e-10).all() for name in rules.CORE)
        ):
            raise QuantError("All original anchors and four genuine factor holdings must remain.")
        coverage.append({"years": window["years"], "positive_factor_symbols": list(rules.CORE)})
    prefixes = []
    for position in (0, 12, 36, 60, len(dates) - 1):
        runtime_guard()
        end = dates[position]
        clipped = MarketData(
            data.open.loc[:end],
            data.close.loc[:end],
            data.raw_close.loc[:end],
            data.volume.loc[:end],
            data.risk_free.loc[:end],
        )
        early, _ = rules.build_targets(
            clipped, desired.loc[:end], information.loc[:end], read_json(rules.POLICY)
        )
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Future quotes or prices altered an earlier source-date target.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            IDS[0]: {
                "checked_at": utc_now(),
                "monthly_decisions": 122,
                "original_calendar_opportunities": 131,
                "additional_source_months_dropped": 0,
                "undefined_prehistory_months": 9,
                "first_eligible_source_month": "2016-08-31",
                "maximum_independent_target_difference": errors[0],
                "maximum_independent_share_difference": errors[1],
                "maximum_independent_variation_difference": errors[2],
                "maximum_independent_ratio_relative_difference": errors[3],
                "exact_nonstock_and_total_stock_budget_preserved": True,
                "formal_factor_coverage": coverage,
                "prefix_truncations": prefixes,
                "source_model_refitting": False,
                "original_source_signal_recomputed": False,
                "candidate_accounts_or_metrics_computed": False,
            }
        },
    )


def checked_packet(frozen):
    common.output_guard(PACKET)
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 1
        or packet["raw_inputs_and_target_matrices_remain_private"] is not True
        or any(
            packet.get(key) is not False
            for key in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "order_authority",
                "independent_forward_validation",
            )
        )
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(row["target_proof_at"]) > created for row in entries)
        or any(
            created > pd.Timestamp(row["at"])
            for row in base.matching_events("path_started", IDS[0])
        )
    ):
        raise QuantError(
            "Keep the actual unchanged pre-account VXN relative-uncertainty registration."
        )
    return packet


def failure(stage, error):
    identifier = IDS[0]
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if (
        not expired
        and base.matching_events("registered", identifier)
        and not base.matching_events("review", identifier)
        and not base.matching_events("technical_failure", identifier)
    ):
        base.append_event(
            base.JOURNAL,
            "technical_failure",
            identifier,
            {
                "stage": stage,
                "reason": str(error),
                "partial_outputs_retained": True,
                "candidate_accounts_started": bool(
                    base.matching_events("path_started", identifier)
                ),
            },
        )
    common.progress(
        "vxn_relative_expired_preserved" if expired else "vxn_relative_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, information, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 27
            or prior["new_completed_strategy_evaluations"] != 26
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve27registrations,26completions andtheoriginalfailure.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("The actual VXN specification cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve a failed preparation without replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_vxn_relative",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proofs = prove_targets(
                    data, desired, information, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                directory = TARGETS / identifier
                path = directory / "targets.csv"
                common.output_guard(path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Preserve conflicting partial VXN targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(
                    directory / "information-audits.json", {"decisions": audits[identifier]}
                )
                proofs[identifier]["information_audits"] = base.reference(
                    directory / "information-audits.json"
                )
                common.retain_receipt(directory / "proof.json", proofs[identifier], ("checked_at",))
                runtime_guard()
                base.append_event(
                    base.JOURNAL,
                    "target_proof",
                    identifier,
                    {
                        "registered_event_sha256": base.matching_events("registered", identifier)[
                            0
                        ]["sha256"],
                        "targets": base.reference(path),
                        "proof": base.reference(directory / "proof.json"),
                    },
                )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("vxn_relative_preparation", error)
                raise
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [common.packet_entry(candidate, frozen)],
                "new_registered_candidates": 1,
                "candidate_accounts_or_metrics_computed": False,
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        if not base.matching_events("path_started", identifier):
            common.progress(
                "vxn_relative_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                vxn_relative_registration=base.reference(PACKET),
                pending_steps=["publish_actual_VXN_registration", "four_original_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen, original, commit):
    inherited = base.published_inputs(original, commit)
    parent = verify_publication(ROOT, frozen["source_registration"], set(donor.EXTENSIONS), commit)
    information = verify_publication(
        ROOT, frozen["vxn_source_readiness"], set(SOURCE_EXTENSIONS), commit
    )
    own = verify_publication(ROOT, base.reference(PACKET), set(EXTENSIONS), commit)
    events = [
        row
        for row in base.matching_events("publication")
        if row["body"].get("study_id") == frozen["study_id"]
    ]
    if events:
        if len(events) != 1 or events[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with this study's actual original publication commit.")
        return events[0]["body"]["publication"]
    receipt = {
        **own,
        "original_dependency_publication": inherited,
        "HMM_carrier_dependency_publication": parent,
        "VXN_source_dependency_publication": information,
    }
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    identifier = IDS[0]
    reviews, failed = (
        base.matching_events("review", identifier),
        base.matching_events("technical_failure", identifier),
    )
    paths = {
        row["body"]["path_key"]: base.verify_path(
            row["body"]["receipt"], frozen, identifier, row["body"]["path_key"]
        )
        for row in base.matching_events("path_completed", identifier)
    }
    if reviews:
        if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
            raise QuantError("The completed VXN review and its four paths disagree.")
        return {identifier: reviews[0]["body"]}
    return {
        identifier: {
            "candidate_id": identifier,
            "status": "technical_failure" if failed else "pending",
            "paths": paths,
            "failure_event_sha256": failed[0]["sha256"] if failed else None,
            "failure_stage": failed[0]["body"]["stage"] if failed else None,
        }
    }


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, information, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        common.output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Completed evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        identifier = IDS[0]
        common.progress(
            "evaluating_vxn_relative",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            vxn_relative_publication=published,
        )
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(event["targets"])
                proof = read_json(base.checked_reference(event["proof"]))
                audits = read_json(base.checked_reference(proof["information_audits"]))["decisions"]
                verify_saved_targets(data, desired, information, targets, audits)
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_vxn_relative", active_candidate=identifier, active_path=key
                        )
                        paths[key] = source.evaluate_path(
                            data, targets, frozen, identifier, window, scenario, benchmarks
                        )
                check_frozen(frozen)
                base.legacy_snapshot(read_json(base.POLICY))
                base.append_event(
                    base.JOURNAL,
                    "review",
                    identifier,
                    {
                        "candidate_id": identifier,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed"
                        if all(row["all_point_gates_pass"] for row in paths.values())
                        else "rejected",
                        "paths": paths,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            common.retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            common.progress(
                "vxn_relative_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                vxn_relative_results=base.reference(RESULTS),
                pending_steps=["publish_all_four_VXN_outcomes"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("vxn_relative_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "vxn_relative_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only the actual terminal VXN allocation outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("An unfinished study is not a completed result.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for identifier, row in actual.items():
        paths = []
        for key, item in row["paths"].items():
            year, scenario = key.split("-")
            year = int(year[:-1])
            paths.append(
                {
                    **{
                        name: item[name]
                        for name in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": year,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        name: lookup[year, scenario, name] for name in ("SPY", "QQQ")
                    },
                }
            )
        reviews.append(
            {
                "candidate_id": identifier,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [row for row in base.journal_events(base.JOURNAL) if row["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": base.reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": base.reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return base.reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="One frozen VXN relative-uncertainty study; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("This VXN allocation study cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published preregistration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = common.counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"VXN relative research blocked: {error}\n")


if __name__ == "__main__":
    main()
