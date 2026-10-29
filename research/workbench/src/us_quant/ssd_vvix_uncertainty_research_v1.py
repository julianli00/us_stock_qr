from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_split_session_research_v1 as donor
from us_quant import ssd_variance_disagreement_research_v1 as source
from us_quant import vvix_uncertainty_confirmation_v1 as rules
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
FOLDER = base.REPORTS / "vvix-uncertainty-confirmation-20261029-v1"
TARGETS = ROOT / "data/ssd-vvix-uncertainty-confirmation-20261029-v1"
PACKET = ROOT / "evidence/vvix_uncertainty_confirmation_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/vvix_uncertainty_confirmation_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/vvix-uncertainty-confirmation-v1.json",
    "src/us_quant/vvix_uncertainty_confirmation_v1.py",
    "src/us_quant/ssd_vvix_uncertainty_research_v1.py",
    "tests/test_vvix_uncertainty_confirmation_v1.py",
    "tests/test_ssd_vvix_uncertainty_research_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    "evidence/split_session_variation_20261028_registration_v1.json",
    "src/us_quant/anfci_residual_confirmation_v1.py",
    "config/vvix-inputs-v1.json",
    "src/us_quant/vvix_inputs_v1.py",
    "tests/test_vvix_inputs_v1.py",
    "evidence/vvix_input_readiness_20261028_v1.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def information_inputs(policy):
    public = read_json(base.checked_reference(policy["vvix_readiness"]))
    manifest = read_json(base.checked_reference(policy["vvix_manifest"]))
    coverage = manifest["coverage"]
    if (
        public["private_manifest"] != policy["vvix_manifest"]
        or public["source_admitted"] is not True
        or manifest["source_admitted"] is not True
        or manifest["admission_scope"] != "fixed original archival daily closes only"
        or public["coverage"] != coverage
        or coverage["full_daily_coverage"] is not True
        or coverage["all_original_monthly_queries_covered"] is not True
        or manifest["historical_intraday_first_publication_or_complete_method_chain_verified"]
        is not False
        or manifest["strategy_states_targets_ratios_or_returns_computed"] is not False
        or manifest["existing_comparator_requests"] != 0
    ):
        raise QuantError("Use only the declared archival VVIX admission, not a stronger PIT claim.")
    for name in ("VVIX", "VIX"):
        row = coverage["series"][name]
        if (
            row["context_sessions"] != 2805
            or row["observed_daily_values"] != 2805
            or row["original_monthly_queries"] != 131
            or row["observed_monthly_values"] != 131
            or row["missing_daily_dates"]
            or row["missing_monthly_dates"]
        ):
            raise QuantError("The source's full original daily/monthly coverage changed.")
    source_policy = read_json(base.checked_reference(manifest["policy"]))
    if source_policy["new_raw_csv"] != policy["vvix_raw_csv"]:
        raise QuantError("The independent native quote source changed.")
    refs = [
        policy["vvix_readiness"],
        policy["vvix_manifest"],
        policy["vvix_raw_csv"],
        manifest["policy"],
        manifest["intent"],
        manifest["aligned_closes"],
        manifest["original_monthly_closes"],
        *read_json(base.checked_reference(manifest["intent"]))["inputs"],
    ]
    for ref in refs:
        base.checked_reference(ref)
    quotes = source.frame(manifest["aligned_closes"])
    if list(quotes.columns) != ["VVIX", "VIX"]:
        raise QuantError("Use the actual admitted index identities and unchanged native units.")
    return quotes["VVIX"], refs, manifest["aligned_closes"]


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 216 or events[215]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all prior evaluations and the original technical failure.")
    data, _, _, parent, original, benchmarks = donor.inputs()
    donor.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["source_registration"]))["entries"][0]
    if (
        policy["source_registration"] != base.reference(donor.PACKET)
        or entry["candidate_id"] != policy["source_candidate"]
        or entry["target_proof"]["targets"] != policy["source_targets"]
    ):
        raise QuantError("Read the exact saved fifteenth target matrix, not its generator.")
    desired = source.frame(policy["source_targets"])
    quotes, source_refs, quote_ref = information_inputs(policy)
    if (
        len(data.close) != 2805
        or len(desired.dropna()) != 131
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or not quotes.index.equals(data.close.index)
        or min(data.close.index.get_loc(day) for day in desired.dropna().index) < rules.LOOKBACK
    ):
        raise QuantError("Every original monthly opportunity needs both actual VVIX observations.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"]
        + [base.reference(ROOT / "src/us_quant/anfci_residual_confirmation_v1.py")]
        + [base.reference(ROOT / name) for name in OWN_EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "source_candidate": policy["source_candidate"],
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "vvix_readiness": policy["vvix_readiness"],
        "vvix_manifest": policy["vvix_manifest"],
        "vvix_closes": quote_ref,
        "prior_completed_configurations": 196,
        "prior_journal_sequence": 216,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "VVIX_quarter_increase_confirms_original_dollar_half_guard",
        "decision_information_clock": policy["decision_information_clock"],
        "information_period_sessions": rules.LOOKBACK,
        "archival_source_not_authenticated_intraday_PIT": True,
        "old_source_generator_or_ANFCI_admission_replayed": False,
        "prior_evidence": parent["prior_evidence"]
        + source_refs
        + [
            policy["source_registration"],
            policy["source_targets"],
            entry["target_proof"]["proof"],
            base.reference(donor.PUBLIC_RESULTS),
            base.reference(FOLDER / "research-intent.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "vvix-confirmation-offline-tests-v1.xml",
                "vvix-confirmation-offline-tests-v1.log",
            )
        ],
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, desired, quotes, frozen, original, benchmarks


def native_quotes():
    path = base.checked_reference(read_json(rules.POLICY)["vvix_raw_csv"])
    values = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["DATE", "VVIX"]:
            raise QuantError("The independent oracle needs the actual native VVIX schema.")
        for row in reader:
            day = pd.Timestamp(datetime.strptime(row["DATE"], "%m/%d/%Y"))
            value = Decimal(row["VVIX"])
            if day in values or not value.is_finite() or value <= 0:
                raise QuantError("Independent VVIX values are invalid or duplicated.")
            values[day] = value
    return values


def verify_saved_targets(data, desired, quotes, targets, audits):
    opportunities = desired.dropna().index
    if (
        len(opportunities) != 131
        or len(audits) != 131
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(opportunities)
        or [row["decision_session"] for row in audits]
        != opportunities.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Every original saved source opportunity must remain represented.")
    raw, maximum = native_quotes(), 0.0
    for day, audit in zip(opportunities, audits, strict=True):
        prior = data.close.loc[:day].index[-64]
        if day not in raw or prior not in raw:
            raise QuantError("A required native quote cannot be filled or bypassed.")
        oracle = rules.independent_row(desired.loc[day], raw[day], raw[prior])
        increased = raw[day] > raw[prior]
        guarded = abs(float(desired.loc[day, "BIL"]) - 0.49) <= 1e-12
        if (
            audit["VVIX_reference_session"] != str(prior.date())
            or audit["VVIX_current_native_level"] != float(raw[day])
            or audit["VVIX_prior_native_level"] != float(raw[prior])
            or Decimal(str(quotes.loc[day])) != raw[day]
            or Decimal(str(quotes.loc[prior])) != raw[prior]
            or audit["VVIX_level_increased"] is not increased
            or audit["original_dollar_guard_active"] is not guarded
            or audit["uncertainty_guard_confirmed"] is not (guarded and increased)
            or audit["candidate_target_created"] is not True
            or audit["information_clock"] != "after_16_15_ET_index_close"
            or audit["VVIX_index_return_or_variance_premium_claimed"] is not False
        ):
            raise QuantError(
                "Saved uncertainty metadata disagrees with the independent native oracle."
            )
        difference = float(np.max(abs(targets.loc[day] - oracle)))
        if not np.isfinite(difference) or difference > 1e-12:
            raise QuantError("Independent VVIX budget transformation differs from saved targets.")
        maximum = max(maximum, difference)
    return maximum


def prove_targets(data: MarketData, desired: pd.DataFrame, quotes: pd.Series, protocol: dict):
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, desired, quotes, policy)
    target = targets[IDS[0]]
    maximum = verify_saved_targets(data, desired, quotes, target, audits)
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if (
            anchor not in known.index
            or known.isna().any().any()
            or any(not (known[name] > 1e-10).all() for name in rules.CORE)
        ):
            raise QuantError("Keep both original funded anchors and four genuine factor families.")
        coverage.append({"years": window["years"], "positive_factor_symbols": list(rules.CORE)})
    prefixes = []
    for position in (252, 503, 1006, 1509, len(data.close) - 2):
        end = data.close.index[position]
        sliced = MarketData(
            data.open.loc[:end],
            data.close.loc[:end],
            data.raw_close.loc[:end],
            data.volume.loc[:end],
            data.risk_free.loc[:end],
        )
        early, _ = rules.build_targets(sliced, desired.loc[:end], quotes.loc[:end], policy)
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Future source observations altered an earlier VVIX target.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "monthly_decisions": 131,
            "VVIX_comparison_sessions": 63,
            "independent_native_quote_queries_checked": 262,
            "maximum_independent_target_difference": maximum,
            "source_monthly_dates_and_relative_composition_preserved": True,
            "decision_after_index_close_before_original_delayed_opens": True,
            "intraday_first_publication_or_method_history_authenticated": False,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "VVIX_index_return_or_variance_premium_claimed": False,
        },
    )


def checked_packet(frozen: dict) -> dict:
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
            for identifier in IDS
            for row in base.matching_events("path_started", identifier)
        )
    ):
        raise QuantError("Keep the actual unchanged pre-account VVIX-confirmation registration.")
    return packet


def failure(stage: str, error: Exception) -> None:
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
        "vvix_uncertainty_expired_preserved" if expired else "vvix_uncertainty_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, options, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 19
            or prior["new_completed_strategy_evaluations"] != 18
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError(
                "Preserve 19 registrations, 18 completions and the old failure."
            )
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered VVIX-confirmation rule cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve a failed preparation without same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_split_session",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(
                    data, desired, options, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                directory = TARGETS / identifier
                path = directory / "targets.csv"
                common.output_guard(path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Preserve conflicting partial VVIX-confirmation targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "uncertainty-audits.json", {"decisions": audits})
                proof["uncertainty_audits"] = base.reference(directory / "uncertainty-audits.json")
                common.retain_receipt(directory / "proof.json", proof, ("checked_at",))
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
                failure("vvix_uncertainty_preparation", error)
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
                "vvix_uncertainty_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                vvix_uncertainty_registration=base.reference(PACKET),
                pending_steps=["publish_vvix_uncertainty_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    inherited = base.published_inputs(original, commit)
    prior = verify_publication(
        ROOT, frozen["daily_proxy_registration"], set(source.EXTENSIONS), commit
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
        "daily_proxy_dependency_publication": prior,
    }
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen: dict) -> dict:
    result = {}
    for identifier in IDS:
        reviews = base.matching_events("review", identifier)
        failed = base.matching_events("technical_failure", identifier)
        paths = {
            row["body"]["path_key"]: base.verify_path(
                row["body"]["receipt"], frozen, identifier, row["body"]["path_key"]
            )
            for row in base.matching_events("path_completed", identifier)
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("Completed VVIX-confirmation review and paths disagree.")
            result[identifier] = reviews[0]["body"]
        else:
            result[identifier] = {
                "candidate_id": identifier,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit: str) -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, options, frozen, original, benchmarks = inputs()
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
        common.progress(
            "evaluating_vvix_uncertainty",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            vvix_uncertainty_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                target_event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(target_event["targets"])
                proof = read_json(base.checked_reference(target_event["proof"]))
                audits = read_json(base.checked_reference(proof["uncertainty_audits"]))["decisions"]
                verify_saved_targets(data, desired, options, targets, audits)
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_vvix_uncertainty",
                            active_candidate=identifier,
                            active_path=key,
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
                "vvix_uncertainty_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                vvix_uncertainty_results=base.reference(RESULTS),
                pending_steps=["publish_all_vvix_uncertainty_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("vvix_uncertainty_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "vvix_uncertainty_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal VVIX-confirmation outcome.")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One VVIX-confirmation information study; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The VVIX-confirmation study cannot request live sources.")

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
        parser.exit(2, f"VVIX-confirmation research blocked: {error}\n")


if __name__ == "__main__":
    main()
