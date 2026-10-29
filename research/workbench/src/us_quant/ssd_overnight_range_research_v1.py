from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import overnight_range_product_v1 as rules
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
FOLDER = base.REPORTS / "overnight-range-product-20261029-v1"
TARGETS = ROOT / "data/ssd-overnight-range-product-20261029-v1"
PACKET = ROOT / "evidence/overnight_range_product_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/overnight_range_product_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/overnight-range-product-v1.json",
    "src/us_quant/overnight_range_product_v1.py",
    "src/us_quant/ssd_overnight_range_research_v1.py",
    "tests/test_overnight_range_product_v1.py",
    "tests/test_ssd_overnight_range_research_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    "evidence/split_session_variation_20261028_registration_v1.json",
    "src/us_quant/anfci_residual_confirmation_v1.py",
    "src/us_quant/spy_range_inputs_v1.py",
    "tests/test_spy_range_inputs_v1.py",
    "evidence/spy_ohlc_range_readiness_20261029_v1.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def information_inputs(policy, market):
    public = read_json(base.checked_reference(policy["range_source_readiness"]))
    manifest = read_json(base.checked_reference(policy["range_source_manifest"]))
    proof = manifest["source_validation"]
    if (
        public["private_manifest"] != policy["range_source_manifest"]
        or public["source_admitted"] is not True
        or manifest["source_admitted"] is not True
        or manifest["admission_scope"] != "fixed_original_2805_SPY_daily_raw_OHLC_bars_only"
        or manifest["original_market"] != market
        or public["source_validation"] != proof
        or proof["context_sessions"] != 2805
        or proof["independently_checked_native_fields"] != 16830
        or proof["raw_OHLC_containment_verified"] is not True
        or proof["common_adjustment_factor_verified"] is not True
        or proof["registered_open_close_Rf_or_volume_replaced"] is not False
        or proof["source_gaps_filled_or_original_dates_dropped"] is not False
        or manifest["new_variance_or_strategy_targets_computed"] is not False
        or manifest["primary_paper_full_text_verified"] is not False
    ):
        raise QuantError("Use the exact admitted archived fields, not new prices or paper claims.")
    refs = [
        policy["range_source_readiness"],
        policy["range_source_manifest"],
        manifest["range_bars"],
        manifest["intent"],
        *read_json(base.checked_reference(manifest["intent"]))["inputs"],
    ]
    for ref in refs:
        base.checked_reference(ref)
    bars = source.frame(manifest["range_bars"])
    if list(bars.columns) != list(rules.FIELDS):
        raise QuantError("The source's four raw intraday fields changed.")
    return bars, refs, manifest["range_bars"]


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 228 or events[227]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all197completed configurations and old failure records.")
    data, _, options, parent, original, benchmarks = donor.inputs()
    donor.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["source_registration"]))["entries"][0]
    if (
        policy["source_registration"] != base.reference(donor.PACKET)
        or entry["candidate_id"] != policy["source_candidate"]
        or entry["target_proof"]["targets"] != policy["source_targets"]
    ):
        raise QuantError("The new measurement must use the exact saved fifteenth budget.")
    desired = source.frame(policy["source_targets"])
    bars, range_refs, range_ref = information_inputs(policy, parent["market"])
    if (
        len(data.close) != 2805
        or len(desired.dropna()) != 131
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or not bars.index.equals(data.close.index)
        or not options.index.equals(desired.dropna().index)
    ):
        raise QuantError("Every original monthly opportunity and price field must remain.")
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
        "range_source_readiness": policy["range_source_readiness"],
        "range_source_manifest": policy["range_source_manifest"],
        "range_bars": range_ref,
        "prior_completed_configurations": 197,
        "prior_journal_sequence": 228,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "original_overnight_square_plus_raw_intraday_range_products",
        "source_BIL_and_non_BIL_budget_preserved": True,
        "primary_full_paper_verified": False,
        "prior_evidence": parent["prior_evidence"]
        + range_refs
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
                "overnight-range-offline-tests-v1.xml",
                "overnight-range-offline-tests-v1.log",
            )
        ],
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, desired, {"bars": bars, "options": options}, frozen, original, benchmarks


def verify_saved_targets(data, desired, information, targets, audits):
    opportunities = desired.dropna().index
    if (
        len(opportunities) != 131
        or len(audits) != 131
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(opportunities)
        or not np.array_equal(targets["BIL"], desired["BIL"], equal_nan=True)
        or [row["decision_session"] for row in audits]
        != opportunities.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep the original target calendar and exact defensive BIL budget.")
    target_error, proxy_error = 0.0, 0.0
    for day, audit in zip(opportunities, audits, strict=True):
        oracle, components = rules.independent_row(data, desired.loc[day], information, day)
        difference = float(np.max(abs(targets.loc[day] - oracle)))
        if not np.isfinite(difference) or difference > 1e-12:
            raise QuantError("Independent target reconstruction differs.")
        target_error = max(target_error, difference)
        position = data.close.index.get_loc(day)
        if (
            audit["first_prior_close"] != str(data.close.index[position - 21].date())
            or audit["first_intraday_session"] != str(data.close.index[position - 20].date())
            or audit["last_price_observation"] != str(day.date())
            or audit["source_BIL_target_preserved"] != float(desired.loc[day, "BIL"])
            or audit["raw_intraday_common_adjustment_invariance"] is not True
            or audit["unbiased_variance_or_true_premium_claimed"] is not False
            or audit["minimum_daily_range_product"] < 0
        ):
            raise QuantError("The saved input-window or proxy interpretation changed.")
        for name, expected in components.items():
            error = abs(audit[name] - expected)
            if not np.isfinite(error) or error > 1e-12 * max(1.0, abs(expected)):
                raise QuantError("Independent measurement differs before allocation clipping.")
            proxy_error = max(proxy_error, error)
        old_budget = float(desired.loc[day, [*rules.STOCKS, "GLD"]].sum())
        new_budget = float(targets.loc[day, [*rules.STOCKS, "GLD"]].sum())
        if (
            abs(old_budget - new_budget) > 1e-12
            or abs(audit["source_non_BIL_budget_preserved"] - old_budget) > 1e-12
        ):
            raise QuantError("New extrema information cannot expand the defended budget.")
    return target_error, proxy_error


def prove_targets(data: MarketData, desired: pd.DataFrame, information: dict, protocol: dict):
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, desired, information, policy)
    target = targets[IDS[0]]
    target_error, proxy_error = verify_saved_targets(data, desired, information, target, audits)
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if (
            anchor not in known.index
            or known.isna().any().any()
            or any(not (known[name] > 1e-10).all() for name in rules.CORE)
        ):
            raise QuantError("Both full funded anchors and all four style families must remain.")
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
        early, _ = rules.build_targets(
            sliced,
            desired.loc[:end],
            {name: frame.loc[:end] for name, frame in information.items()},
            policy,
        )
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Future extrema altered an earlier allocation.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "monthly_decisions": 131,
            "sessions_per_proxy": 21,
            "maximum_independent_target_difference": target_error,
            "maximum_independent_unclipped_proxy_difference": proxy_error,
            "source_BIL_exact_and_non_BIL_budget_preserved": True,
            "source_price_fields_and_relative_stock_composition_preserved": True,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "primary_full_paper_or_unbiased_variance_claimed": False,
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
        raise QuantError("Keep the actual unchanged pre-account Overnight-range registration.")
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
        "overnight_range_expired_preserved" if expired else "overnight_range_blocked",
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
            prior["new_registered_candidates"] != 20
            or prior["new_completed_strategy_evaluations"] != 19
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve20registrations,19completions andtheoldfailure.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered Overnight-range rule cannot change.")
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
                        raise QuantError("Preserve conflicting partial Overnight-range targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "range-audits.json", {"decisions": audits})
                proof["range_audits"] = base.reference(directory / "range-audits.json")
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
                failure("overnight_range_preparation", error)
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
                "overnight_range_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                overnight_range_registration=base.reference(PACKET),
                pending_steps=["publish_overnight_range_registration", "four_funded_paths"],
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
                raise QuantError("Completed Overnight-range review and paths disagree.")
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
            overnight_range_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                target_event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(target_event["targets"])
                proof = read_json(base.checked_reference(target_event["proof"]))
                audits = read_json(base.checked_reference(proof["range_audits"]))["decisions"]
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
                "overnight_range_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                overnight_range_results=base.reference(RESULTS),
                pending_steps=["publish_all_overnight_range_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("overnight_range_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "overnight_range_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal Overnight-range outcome.")
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
        description="One Overnight-range information study; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The Overnight-range study cannot request live sources.")

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
        parser.exit(2, f"Overnight-range research blocked: {error}\n")


if __name__ == "__main__":
    main()
