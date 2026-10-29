from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import core_activity_momentum_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_split_session_research_v1 as donor
from us_quant import ssd_variance_disagreement_research_v1 as source
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
FOLDER = base.REPORTS / "core-activity-momentum-20261029-v1"
TARGETS = ROOT / "data/ssd-core-activity-momentum-20261029-v1"
PACKET = ROOT / "evidence/core_activity_momentum_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/core_activity_momentum_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/core-activity-momentum-v1.json",
    "src/us_quant/core_activity_momentum_v1.py",
    "src/us_quant/ssd_core_activity_research_v1.py",
    "tests/test_core_activity_momentum_v1.py",
    "tests/test_ssd_core_activity_research_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    "evidence/split_session_variation_20261028_registration_v1.json",
    "src/us_quant/anfci_residual_confirmation_v1.py",
    "src/us_quant/core_activity_inputs_v1.py",
    "tests/test_core_activity_inputs_v1.py",
    "evidence/core_activity_source_readiness_20261029_v1.json",
    "config/factor-core-momentum.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def information_inputs(policy, market):
    public = read_json(base.checked_reference(policy["activity_source_readiness"]))
    manifest = read_json(base.checked_reference(policy["activity_source_manifest"]))
    if (
        public["private_manifest"] != policy["activity_source_manifest"]
        or public["source_admitted"] is not True
        or manifest["source_admitted"] is not True
        or manifest["admission_scope"] != "original_four_style_fund_reported_activity_fields_only"
        or manifest["original_market"] != market
        or public["funds"] != manifest["funds"]
        or [row["symbol"] for row in manifest["funds"]] != list(rules.CORE)
        or any(
            row["sessions"] != 2805
            or row["reported_split_events"] != 0
            or row["reported_share_volume_native_CSV_registered_exact"] is not True
            for row in manifest["funds"]
        )
        or manifest["new_scores_ranks_targets_or_accounts_computed"] is not False
        or manifest["old_prices_Rf_volumes_replaced"] is not False
    ):
        raise QuantError("Use the admitted native reported activity fields, not substituted data.")
    refs = [
        policy["activity_source_readiness"],
        policy["activity_source_manifest"],
        manifest["source_intent"],
        manifest["validation_intent"],
        *read_json(base.checked_reference(manifest["validation_intent"]))["inputs"],
    ]
    for ref in refs:
        base.checked_reference(ref)
    return manifest, refs


def inputs():
    events = base.journal_events(base.JOURNAL)
    if len(events) < 240 or events[239]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Keep the prior198completed configurations and original failure.")
    data, _, _, parent, original, benchmarks = donor.inputs()
    donor.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["source_registration"]))["entries"][0]
    prior_rank = read_json(base.checked_reference(policy["prior_rank_policy"]))
    if (
        policy["source_registration"] != base.reference(donor.PACKET)
        or entry["candidate_id"] != policy["source_candidate"]
        or entry["target_proof"]["targets"] != policy["source_targets"]
        or prior_rank["ranked_factor_shares"] != list(rules.SHARES)
    ):
        raise QuantError("The parent budget or inherited fixed rank shares changed.")
    desired = source.frame(policy["source_targets"])
    activity, refs = information_inputs(policy, parent["market"])
    dates = rules.eligible_dates(data, desired)
    if (
        len(data.close) != 2805
        or len(desired.dropna()) != 131
        or len(dates) != 128
        or str(dates[0].date()) != "2016-02-29"
    ):
        raise QuantError("Keep the exact original calendar and declared pre-account warmup.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"]
        + [base.reference(ROOT / "src/us_quant/anfci_residual_confirmation_v1.py")]
        + [base.reference(ROOT / name) for name in OWN_EXTENSIONS[1:]],
        "policies": parent["policies"]
        + [base.reference(rules.POLICY), policy["prior_rank_policy"]],
        "source_candidate": policy["source_candidate"],
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "activity_source_readiness": policy["activity_source_readiness"],
        "activity_source_manifest": policy["activity_source_manifest"],
        "prior_completed_configurations": 198,
        "prior_journal_sequence": 240,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "fixed_paired_uniform_time_and_reported_activity_core_momentum",
        "noncore_targets_and_combined_core_budget_preserved": True,
        "prior_evidence": parent["prior_evidence"]
        + refs
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
            for name in ("core-activity-offline-tests-v1.xml", "core-activity-offline-tests-v1.log")
        ],
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, desired, activity, frozen, original, benchmarks


def verify_saved_targets(data, desired, targets, audits, candidate):
    dates = rules.eligible_dates(data, desired)
    noncore = desired.columns.difference(rules.CORE)
    if (
        len(dates) != 128
        or len(audits) != 128
        or not targets.index.equals(desired.index)
        or not targets.columns.equals(desired.columns)
        or not targets.dropna().index.equals(dates)
        or not np.array_equal(targets.loc[dates, noncore], desired.loc[dates, noncore])
        or [row["decision_session"] for row in audits] != dates.strftime("%Y-%m-%d").tolist()
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
    ):
        raise QuantError("Keep the eligible dates, undefined prehistory and every noncore target.")
    target_error, score_error = 0.0, 0.0
    for day, audit in zip(dates, audits, strict=True):
        oracle, scores, order = rules.independent_row(
            data, desired.loc[day], day, candidate["score"]
        )
        difference = float(np.max(abs(targets.loc[day] - oracle)))
        if not np.isfinite(difference) or difference > 1e-12:
            raise QuantError("Independent core target reconstruction differs.")
        target_error = max(target_error, difference)
        position = data.close.index.get_loc(day)
        if (
            audit["first_prior_close"] != str(data.close.index[position - 126].date())
            or audit["last_score_and_activity_session"]
            != str(data.close.index[position - 21].date())
            or audit["return_sessions"] != 105
            or audit["method"] != candidate["score"]
            or audit["ranking"] != order
            or audit["ranking_scores"] != {name: round(value, 12) for name, value in scores.items()}
            or audit["all_noncore_targets_preserved"] is not True
            or audit["activity_is_actual_flow_or_constituent_turnover"] is not False
        ):
            raise QuantError("The saved ranking, lag or activity interpretation changed.")
        for name in rules.CORE:
            error = abs(audit["scores"][name] - scores[name])
            if not np.isfinite(error) or error > 1e-12:
                raise QuantError("Independent unrounded scores differ before ranking.")
            score_error = max(score_error, error)
        budget = float(desired.loc[day, list(rules.CORE)].sum())
        if (
            abs(audit["source_core_budget"] - budget) > 1e-12
            or abs(float(targets.loc[day, list(rules.CORE)].sum()) - budget) > 1e-12
        ):
            raise QuantError("Ranking cannot alter the combined source factor budget.")
    return target_error, score_error


def prove_targets(data, desired, information, protocol):
    targets, audits = rules.build_targets(data, desired, read_json(rules.POLICY))
    proofs = {}
    for candidate in rules.CANDIDATES:
        identifier = candidate["id"]
        target_error, score_error = verify_saved_targets(
            data, desired, targets[identifier], audits[identifier], candidate
        )
        coverage = []
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            known = targets[identifier].ffill().loc[anchor : window["end"]]
            if (
                anchor not in known.index
                or known.isna().any().any()
                or any(not (known[name] > 1e-10).all() for name in rules.CORE)
            ):
                raise QuantError("All formal anchors and four genuine style holdings must remain.")
            coverage.append({"years": window["years"], "positive_factor_symbols": list(rules.CORE)})
        proofs[identifier] = {
            "checked_at": utc_now(),
            "monthly_decisions": 128,
            "original_source_opportunities": 131,
            "undefined_prehistory_months": 3,
            "first_eligible_source_month": "2016-02-29",
            "maximum_independent_target_difference": target_error,
            "maximum_independent_unrounded_score_difference": score_error,
            "exact_noncore_and_combined_core_budget_preserved": True,
            "formal_factor_coverage": coverage,
            "prefix_truncations": [],
            "original_source_signal_recomputed": False,
            "candidate_accounts_or_metrics_computed": False,
        }
    for position in (252, 503, 1006, 1509, len(data.close) - 2):
        end = data.close.index[position]
        clipped = MarketData(
            data.open.loc[:end],
            data.close.loc[:end],
            data.raw_close.loc[:end],
            data.volume.loc[:end],
            data.risk_free.loc[:end],
        )
        early, _ = rules.build_targets(clipped, desired.loc[:end], read_json(rules.POLICY))
        for identifier in IDS:
            if not np.allclose(
                early[identifier], targets[identifier].loc[:end], rtol=0, atol=1e-12, equal_nan=True
            ):
                raise QuantError("Later activity altered an earlier target.")
            proofs[identifier]["prefix_truncations"].append(str(end.date()))
    for proof in proofs.values():
        proof["checked_at"] = utc_now()
    return targets, audits, proofs


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
        or packet["new_registered_candidates"] != len(IDS)
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
        raise QuantError("Keep the actual unchanged pre-account Core-activity registration.")
    return packet


def failure(stage, error, identifiers=None):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    for identifier in IDS if identifiers is None else identifiers:
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
        "core_activity_expired_preserved" if expired else "core_activity_blocked",
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
            prior["new_registered_candidates"] != 21
            or prior["new_completed_strategy_evaluations"] != 20
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve21registrations,20completions andtheoriginalfailure.")
        for candidate in rules.CANDIDATES:
            spec = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", candidate["id"])
            if existing and (len(existing) != 1 or existing[0]["body"] != spec):
                raise QuantError("A paired candidate's actual specification cannot change.")
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], spec)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Preserve a failed paired preparation without replay.")
        prepared = [bool(base.matching_events("target_proof", identifier)) for identifier in IDS]
        if any(prepared) and not all(prepared):
            raise QuantError("Preserve a partial proof set; do not recompute completed targets.")
        if not all(prepared):
            common.progress(
                "preparing_registered_core_activity",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proofs = prove_targets(
                    data, desired, information, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                for identifier in IDS:
                    directory = TARGETS / identifier
                    path = directory / "targets.csv"
                    common.output_guard(path)
                    content = targets[identifier].to_csv(float_format="%.17g")
                    if path.exists():
                        if file_digest(path) != digest_bytes(content.encode()):
                            raise QuantError("Preserve conflicting partial paired targets.")
                    else:
                        write_text_atomic(path, content)
                    common.retain_receipt(
                        directory / "score-audits.json", {"decisions": audits[identifier]}
                    )
                    proofs[identifier]["score_audits"] = base.reference(
                        directory / "score-audits.json"
                    )
                    common.retain_receipt(
                        directory / "proof.json", proofs[identifier], ("checked_at",)
                    )
                for identifier in IDS:
                    runtime_guard()
                    directory = TARGETS / identifier
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        identifier,
                        {
                            "registered_event_sha256": base.matching_events(
                                "registered", identifier
                            )[0]["sha256"],
                            "targets": base.reference(directory / "targets.csv"),
                            "proof": base.reference(directory / "proof.json"),
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("core_activity_preparation", error)
                raise
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [
                    common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES
                ],
                "new_registered_candidates": len(IDS),
                "candidate_accounts_or_metrics_computed": False,
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        if not any(base.matching_events("path_started", identifier) for identifier in IDS):
            common.progress(
                "core_activity_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                core_activity_registration=base.reference(PACKET),
                pending_steps=["publish_both_core_activity_registrations", "eight_funded_paths"],
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
                raise QuantError("Completed Core-activity review and paths disagree.")
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
            "evaluating_core_activity",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            core_activity_publication=published,
        )
        identifier = None
        try:
            for candidate in rules.CANDIDATES:
                identifier = candidate["id"]
                if not base.matching_events("review", identifier) and not base.matching_events(
                    "technical_failure", identifier
                ):
                    target_event = base.matching_events("target_proof", identifier)[0]["body"]
                    targets = source.frame(target_event["targets"])
                    proof = read_json(base.checked_reference(target_event["proof"]))
                    audits = read_json(base.checked_reference(proof["score_audits"]))["decisions"]
                    verify_saved_targets(data, desired, targets, audits, candidate)
                    paths = {}
                    protocol = read_json(base.PROTOCOL)
                    for window in protocol["windows"]:
                        for scenario in protocol["accounts"]["scenarios"]:
                            runtime_guard()
                            check_frozen(frozen)
                            key = f"{window['years']}y-{scenario['id']}"
                            common.progress(
                                "evaluating_core_activity",
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
                "core_activity_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                core_activity_results=base.reference(RESULTS),
                pending_steps=["publish_all_core_activity_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("core_activity_evaluation", error, (identifier,) if identifier else IDS)
            raise
        except KeyboardInterrupt:
            common.progress(
                "core_activity_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal Core-activity outcome.")
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
    parser = argparse.ArgumentParser(description="One Core-activity information study; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The Core-activity study cannot request live sources.")

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
        parser.exit(2, f"Core-activity research blocked: {error}\n")


if __name__ == "__main__":
    main()
