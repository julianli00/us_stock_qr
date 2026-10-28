from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import nearest_band_execution_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as source
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

ROOT = base.ROOT
FOLDER = base.REPORTS / "band-projection-20261028-v1"
PACKET = ROOT / "evidence/band_projection_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/band_projection_20261028_results_v1.json"
EXTENSIONS = (
    "config/nearest-band-execution-v1.json",
    "src/us_quant/nearest_band_execution_v1.py",
    "src/us_quant/ssd_band_projection_research_v1.py",
    "tests/test_nearest_band_execution_v1.py",
    "tests/test_ssd_band_projection_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 142 or events[141]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve the actual twelve-study journal prefix.")
    data, _, _, parent, original, benchmarks = source.inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    entry = packet["entries"][0]
    if (
        entry["candidate_id"] != policy["source_candidate"]
        or entry["target_proof"]["targets"] != policy["source_targets"]
    ):
        raise QuantError(
            "Use the exact original eleventh desired target matrix, not recreated signals."
        )
    desired = source.frame(policy["source_targets"])
    if (
        not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or len(desired.dropna()) != 131
    ):
        raise QuantError(
            "All original monthly desired targets and asset/calendar identities are required."
        )
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "source_candidate": policy["source_candidate"],
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "source_targets_role": "desired_only_new_feedback_targets_after_publication",
        "original_policy_role": "unchanged_fee_funding_clock_with_new_projection_destination",
        "execution_policy": base.reference(rules.POLICY),
        "prior_completed_configurations": 190,
        "prior_journal_sequence": 142,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "prior_evidence": parent["prior_evidence"]
        + [
            policy["source_registration"],
            policy["source_targets"],
            entry["target_proof"]["proof"],
            base.reference(source.PUBLIC_RESULTS),
            base.reference(FOLDER / "fixed-fee-contract-witness.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "band-projection-offline-tests-v1.xml",
                "band-projection-offline-tests-v1.log",
            )
        ],
        "minimum_fee_or_future_cost_claim": False,
    }
    check_frozen(frozen)
    return data, desired, frozen, original, benchmarks


def source_proof(data, desired, frozen: dict, protocol: dict) -> dict:
    defined = desired.dropna()
    if (
        len(defined) != 131
        or not all(is_month_end(day) for day in defined.index)
        or not np.isfinite(defined).all().all()
        or (defined < 0).any().any()
        or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("The exact desired-source matrix is incomplete or changed.")
    families = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = desired.ffill().loc[anchor : window["end"]]
        if anchor not in known.index or known.isna().any().any():
            raise QuantError("Keep the full original seeded formal account windows.")
        initial = known.loc[anchor]
        names = [name for name in source.rules.CORE if initial[name] > 1e-10]
        if names != list(source.rules.CORE):
            raise QuantError("Fresh initial source deployment must use all four genuine factors.")
        families.append({"years": window["years"], "initial_positive_factor_symbols": names})
    return {
        "checked_at": utc_now(),
        "monthly_desired_decisions": 131,
        "source_target_bytes_unchanged": frozen["source_targets"],
        "formal_factor_coverage": families,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "actual_path_dependent_issued_targets_computed": False,
        "synthetic_projection_and_fee_contract_evidence": frozen["validation_evidence"],
        "minimum_fee_or_future_cost_claim": False,
    }


def verify_feedback(data, desired, result, issued, decisions, window, scenario) -> dict:
    dates = data.close.loc[window["start"] : window["end"]].index
    anchor = previous_session(dates[0])
    expected_dates = [anchor, *[day for day in dates if is_month_end(day)]]
    if [row["signal_session"] for row in decisions] != [str(day.date()) for day in expected_dates]:
        raise QuantError("Projection cannot change the original monthly decision calendar.")
    known = desired.ffill()
    fill_dates, signal_dates = [], []
    maximum = 0.0
    for day, row in zip(expected_dates, decisions, strict=True):
        initial = day == anchor
        actual = np.zeros(len(desired.columns)) if initial else result.weights.loc[day].to_numpy()
        center = known.loc[day].to_numpy()
        gap = float(np.max(np.abs(actual - center)))
        trigger = initial or gap + 1e-12 >= 0.025
        if (
            row["initial_capital"] is not initial
            or abs(row["maximum_observed_source_gap"] - gap) > 1e-12
            or not np.allclose(row["actual_closing_weights"], actual, rtol=0, atol=1e-12)
            or not np.allclose(row["source_desired_weights"], center, rtol=0, atol=1e-12)
            or (row["reason"] != "hold") != trigger
        ):
            raise QuantError("Actual feedback inputs differ from the original source-band trigger.")
        execution = day
        for _ in range(scenario["delay_sessions"]):
            execution = next_session(execution)
        if row["execution_session"] != str(execution.date()):
            raise QuantError("A projected request was retimed or filled early.")
        if trigger:
            expected = center if initial else rules.independent_projection(actual, center)
            recorded = np.asarray(row["issued_projected_target"], dtype=float)
            maximum = max(maximum, float(np.max(np.abs(recorded - expected))))
            if maximum > 1e-12 or not np.allclose(issued.loc[day], expected, rtol=0, atol=1e-12):
                raise QuantError(
                    "Issued projected target differs from the independent breakpoint oracle."
                )
            if not initial:
                rules.certificate(
                    actual, center, recorded, row["projection_certificate"]["dual_lambda"]
                )
            signal_dates.append(day)
            fill_dates.append(execution)
        elif (
            row["issued_projected_target"] is not None or row["projection_certificate"] is not None
        ):
            raise QuantError("A hold decision cannot contain an unrecorded projection trade.")
    if (
        len(fill_dates) != len(set(fill_dates))
        or not issued.dropna().index.equals(pd.DatetimeIndex(signal_dates))
        or any(day not in fill_dates for day in result.frame.index[result.frame["orders"] > 0])
    ):
        raise QuantError("A projected fill lacks its original dated request or overwrote one.")
    return {
        "checked_actual_monthly_decisions_and_initial": len(decisions),
        "issued_requests": len(signal_dates),
        "maximum_independent_projection_difference": maximum,
        "all_actual_gaps_targets_and_delays_checked": True,
        "minimum_fee_or_future_cost_claim": False,
    }


def evaluate_path(data, desired, frozen, identifier, window, scenario, benchmarks) -> dict:
    key = f"{window['years']}y-{scenario['id']}"
    events = base.journal_events(base.JOURNAL)
    completed = [
        row
        for row in events
        if row["kind"] == "path_completed"
        and row["candidate_id"] == identifier
        and row["body"]["path_key"] == key
    ]
    if completed:
        if len(completed) != 1:
            raise QuantError("Multiple completed paths cannot be cherry-picked.")
        return base.verify_path(completed[0]["body"]["receipt"], frozen, identifier, key)
    starts = [
        row
        for row in events
        if row["kind"] == "path_started"
        and row["candidate_id"] == identifier
        and row["body"]["path_key"] == key
    ]
    if starts:
        prior = starts[-1]
        receipt_path = ROOT / prior["body"]["directory"] / "complete.json"
        if receipt_path.exists():
            recovered = base.verify_path(base.reference(receipt_path), frozen, identifier, key)
            base.append_event(
                base.JOURNAL,
                "path_completed",
                identifier,
                {
                    "path_key": key,
                    "receipt": base.reference(receipt_path),
                    "recovered_actual_complete_receipt": True,
                },
            )
            return recovered
        if not any(
            row["kind"] == "path_interrupted"
            and row["body"].get("started_event_sha256") == prior["sha256"]
            for row in events
        ):
            base.append_event(
                base.JOURNAL,
                "path_interrupted",
                identifier,
                {
                    "path_key": key,
                    "started_event_sha256": prior["sha256"],
                    "partial_output_preserved": True,
                    "reason": "Exclusive lock reacquired without a complete path receipt.",
                },
            )
    runtime_guard()
    directory = base.ACCOUNTS / identifier / f"{key}-attempt-{len(starts) + 1:03d}"
    base.append_event(
        base.JOURNAL,
        "path_started",
        identifier,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    base.storage_guard()
    new_output_directory(directory)
    result, issued, decisions = rules.run(
        data,
        desired,
        read_json(rules.POLICY),
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback = verify_feedback(data, desired, result, issued, decisions, window, scenario)
    independent = base.independent_equity(
        data,
        issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(result.frame["return"], result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]
        for row in benchmarks["benchmarks"]
    }
    criteria = base.gates(
        own,
        lookup[window["years"], scenario["id"], "SPY"],
        lookup[window["years"], scenario["id"], "QQQ"],
        lookup[window["years"], "base", "SPY"],
        scenario["id"],
    )
    disclosure = base.execution_statistics(result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": result.frame,
        "weights.csv": result.weights,
        "issued.csv": issued,
        "independent-bt.csv": independent,
    }
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": identifier,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": disclosure,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"]
        - lookup[window["years"], scenario["id"], "SPY"]["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"]
        - lookup[window["years"], scenario["id"], "QQQ"]["cagr"],
        "stress_vs_base_cost_qqq_diagnostic": own["cagr"]
        - lookup[window["years"], "base", "QQQ"]["cagr"]
        if scenario["id"] == "stress"
        else None,
        "independent_bt_max_equity_difference_usd": difference,
        "independent_projected_feedback": feedback,
        "artifacts": [base.reference(directory / name) for name in (*frames, "decisions.json")],
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        identifier,
        {
            "path_key": key,
            "receipt": base.reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def checked_packet(frozen: dict) -> dict:
    common.output_guard(PACKET)
    value = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(value["created_at"])
    if (
        value["schema_version"] != 2
        or value["program_id"] != "ssd_continuous_research_v1"
        or value["study_id"] != frozen["study_id"]
        or value["entries"] != entries
        or value["new_registered_candidates"] != 1
        or value["raw_inputs_and_target_matrices_remain_private"] is not True
        or any(
            value.get(key) is not False
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
        raise QuantError("Keep the actual pre-account projection registration.")
    return value


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
        "band_projection_expired_preserved" if expired else "band_projection_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 12
            or prior["new_completed_strategy_evaluations"] != 12
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve all twelve completed predecessor studies.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered projection version cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve the terminal failed version, without replay.")
        if not base.matching_events("target_proof", identifier):
            try:
                runtime_guard()
                proof = source_proof(data, desired, frozen, read_json(base.PROTOCOL))
                check_frozen(frozen)
                path = FOLDER / "source-target-proof.json"
                common.retain_receipt(path, proof, ("checked_at",))
                base.append_event(
                    base.JOURNAL,
                    "target_proof",
                    identifier,
                    {
                        "registered_event_sha256": base.matching_events("registered", identifier)[
                            0
                        ]["sha256"],
                        "targets": frozen["source_targets"],
                        "proof": base.reference(path),
                    },
                )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("projection_source_preparation", error)
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
                "band_projection_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                band_projection_registration=base.reference(PACKET),
                pending_steps=["publish_projection_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    prior = verify_publication(ROOT, frozen["source_registration"], set(source.EXTENSIONS), commit)
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
        "original_dependency_publication": original_receipt,
        "source_dependency_publication": prior,
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
                raise QuantError(
                    "Completed projection review and actual account evidence disagree."
                )
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
        data, desired, frozen, original, benchmarks = inputs()
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
            "evaluating_band_projection",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            band_projection_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_band_projection",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = evaluate_path(
                            data, desired, frozen, identifier, window, scenario, benchmarks
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
                "band_projection_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                band_projection_results=base.reference(RESULTS),
                pending_steps=["publish_all_band_projection_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("band_projection_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "band_projection_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal projection outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Unfinished evidence cannot be presented as complete.")
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
                            "independent_projected_feedback",
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
            "minimum_fee_or_future_cost_claim": False,
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
        description="One nearest-band execution experiment; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The frozen-source execution experiment cannot request live data.")

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
        parser.exit(2, f"Band-projection research blocked: {error}\n")


if __name__ == "__main__":
    main()
