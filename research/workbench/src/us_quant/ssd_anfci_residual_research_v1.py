from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import anfci_residual_confirmation_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_split_session_research_v1 as donor
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

ROOT = base.ROOT
SOURCE = donor.source
FOLDER = base.REPORTS / "anfci-residual-confirmation-20261028-v1"
TARGETS = ROOT / "data/ssd-anfci-residual-confirmation-20261028-v1"
PACKET = ROOT / "evidence/anfci_residual_confirmation_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/anfci_residual_confirmation_20261028_results_v1.json"
EXTENSIONS = (
    "config/anfci-residual-confirmation-v1.json",
    "src/us_quant/anfci_residual_confirmation_v1.py",
    "src/us_quant/ssd_anfci_residual_research_v1.py",
    "tests/test_anfci_residual_confirmation_v1.py",
    "tests/test_ssd_anfci_residual_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = SOURCE.check_frozen


def information_inputs(policy: dict) -> tuple[list[dict], list[dict]]:
    public = read_json(base.checked_reference(policy["anfci_readiness"]))
    manifest = read_json(base.checked_reference(policy["anfci_manifest"]))
    validation = read_json(base.checked_reference(policy["anfci_validation"]))
    if (
        public["source_admitted"] is not False
        or manifest["source_admitted"] is not False
        or validation["source_admitted"] is not False
        or public["actual_vintage_count"] != 225
        or public["fresh_query_count"] != 223
        or public["stale_query_count"] != 2
        or public["freshness_limit_calendar_days"] != 14
        or manifest["validation"] != policy["anfci_validation"]
        or manifest["batches"] != validation["batches"]
    ):
        raise QuantError("The original full-source nonadmission must remain unchanged.")
    records = validation["query_readings"]
    all_dates = pd.DatetimeIndex(pd.to_datetime([row["query_session"] for row in records]))
    rules.align_readings(records, all_dates)
    if len(records) != 225 or sum(not row["fresh"] for row in records) != 2:
        raise QuantError("Retain all original source queries, including both stale ones.")
    refs = [policy[name] for name in ("anfci_readiness", "anfci_manifest", "anfci_validation")] + [
        manifest["policy"],
        manifest["freeze"],
        public["original_contract"],
        public["original_failure"],
    ]
    for freeze_ref in (manifest["freeze"], public["original_contract"]):
        refs.extend(read_json(base.checked_reference(freeze_ref))["inputs"])
    refs.extend(
        ref for batch in manifest["batches"] for ref in (batch["request"], batch["response"])
    )
    for ref in refs:
        base.checked_reference(ref)
    return records, refs


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 190 or events[189]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all sixteen completed predecessors and source history.")
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
        raise QuantError("Use the exact frozen fifteenth donor matrix without regeneration.")
    desired = SOURCE.frame(policy["source_targets"])
    if (
        len(data.close) != 2805
        or len(desired.dropna()) != 131
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
    ):
        raise QuantError("Preserve the complete original donor and market calendars.")
    records, source_refs = information_inputs(policy)
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "source_candidate": policy["source_candidate"],
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "prior_completed_configurations": 194,
        "prior_journal_sequence": 190,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "fresh_own_vintage_residual_tightness_confirms_original_dollar_guard",
        "source_full_225_query_admission_passed": False,
        "information_policy": "fresh_current_level_or_explicit_no_new_request",
        "prior_evidence": parent["prior_evidence"]
        + source_refs
        + [
            policy["source_registration"],
            policy["source_targets"],
            entry["target_proof"]["proof"],
            base.reference(donor.PUBLIC_RESULTS),
            base.reference(ROOT / "config/financial-conditions-guard.json"),
            base.reference(ROOT / "src/us_quant/financial_conditions_guard.py"),
            base.reference(ROOT / "evidence/simplex_wealth_average_20261028_results_v1.json"),
            base.reference(FOLDER / "research-intent.json"),
            base.reference(FOLDER / "primary-definition-native-event.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "anfci-residual-offline-tests-v1.xml",
                "anfci-residual-offline-tests-v1.log",
            )
        ],
        "new_source_requests": 0,
        "source_age_limit_relaxed": False,
        "formal_return_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, desired, records, frozen, original, benchmarks


def prove_targets(data: MarketData, desired: pd.DataFrame, records: list[dict], protocol: dict):
    targets, audits = rules.build_targets(data, desired, records, read_json(rules.POLICY))
    target = targets[IDS[0]]
    opportunities = desired.dropna().index
    expected = rules.align_readings(records, opportunities)
    gaps = opportunities[~expected["fresh"]]
    if (
        len(audits) != 131
        or len(target.dropna()) != 130
        or gaps.strftime("%Y-%m-%d").tolist() != ["2022-12-30"]
        or not target.dropna().index.equals(opportunities[expected["fresh"]])
    ):
        raise QuantError("Preserve all131 opportunities and the one actual source abstention.")
    lookup = {row["query_session"]: row for row in records}
    maximum = 0.0
    for day in opportunities:
        oracle = rules.independent_row(desired.loc[day], lookup[str(day.date())], day)
        if not np.allclose(target.loc[day], oracle, rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError(
                "Independent source eligibility or residual-budget transformation differs."
            )
        if not oracle.isna().all():
            maximum = max(maximum, float(np.max(abs(target.loc[day] - oracle))))
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        prior = opportunities[opportunities <= anchor]
        if not len(prior) or target.loc[prior[-1]].isna().any():
            raise QuantError("Initialization cannot use a stale or neutral synthetic seed.")
        known = target.ffill().loc[anchor : window["end"]]
        if (
            anchor not in known.index
            or known.isna().any().any()
            or any(not (known[name] > 1e-10).all() for name in rules.CORE)
        ):
            raise QuantError("Keep both full formal windows and four genuine factor families.")
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
        past = [row for row in records if row["query_session"] <= str(end.date())]
        early, _ = rules.build_targets(sliced, desired.loc[:end], past, read_json(rules.POLICY))
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Later source readings changed an earlier decision.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "original_monthly_opportunities": 131,
            "fresh_target_decisions": 130,
            "source_abstention_count": 1,
            "source_abstention_dates": ["2022-12-30"],
            "maximum_independent_target_difference": maximum,
            "source_full_225_query_admission_passed": False,
            "source_age_limit_relaxed": False,
            "stale_numeric_readings_used": False,
            "formal_returns_removed": False,
            "original_source_signal_recomputed": False,
            "candidate_accounts_or_metrics_computed": False,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
        },
    )


def verify_feedback(
    data, targets, audits, result, issued, decisions, window, scenario
) -> tuple[dict, list]:
    dates = data.close.loc[window["start"] : window["end"]].index
    anchor = previous_session(dates[0])
    opportunities = [anchor, *[day for day in dates if is_month_end(day)]]
    gap_dates = {
        pd.Timestamp(row["decision_session"])
        for row in audits
        if not row["candidate_target_created"]
    }
    eligible = [day for day in opportunities if day == anchor or day not in gap_dates]
    paused = [day for day in opportunities if day != anchor and day in gap_dates]
    if (
        not result.frame.index.equals(dates)
        or [row["signal_session"] for row in decisions] != [str(day.date()) for day in eligible]
        or any(not targets.loc[day].isna().all() for day in paused)
        or any(targets.loc[day].isna().any() for day in eligible if day != anchor)
        or any(not issued.loc[day].isna().all() for day in paused)
    ):
        raise QuantError("Every source abstention must preserve returns and issue no new request.")
    known, fills, expected_issued = targets.ffill(), [], []
    for day, row in zip(eligible, decisions, strict=True):
        initial = day == anchor
        current = pd.Series(0.0, index=targets.columns) if initial else result.weights.loc[day]
        target = known.loc[day]
        gap = float(np.max(np.abs(current - target)))
        trade = initial or gap + 1e-12 >= 0.025
        if (
            row["initial_capital"] is not initial
            or (row["reason"] != "hold") != trade
            or abs(row["maximum_observed_target_gap"] - gap) > 1e-12
        ):
            raise QuantError("Independent actual-holding gap verification failed.")
        execution = day
        for _ in range(scenario["delay_sessions"]):
            execution = next_session(execution)
        if row["execution_session"] != str(execution.date()):
            raise QuantError("A request was retimed or filled before its original delay.")
        if trade:
            if not np.allclose(issued.loc[day], target, rtol=0, atol=1e-12):
                raise QuantError("Issued requests differ from the actual fresh known target.")
            expected_issued.append(day)
            fills.append(execution)
    if (
        len(fills) != len(set(fills))
        or not issued.dropna().index.equals(pd.DatetimeIndex(expected_issued))
        or any(day not in fills for day in result.frame.index[result.frame["orders"] > 0])
    ):
        raise QuantError("A trade has no dated request or a pending request was overwritten.")
    metadata = {row["decision_session"]: row for row in audits}
    abstentions = [
        {
            "signal_session": str(day.date()),
            "reason": metadata[str(day.date())]["source_abstention_reason"],
            "age_calendar_days": metadata[str(day.date())]["age_calendar_days"],
            "no_new_request": True,
            "actual_return_session_present": True,
            "actual_return": float(result.frame.loc[day, "return"]),
            "actual_orders_on_day": int(result.frame.loc[day, "orders"]),
            "actual_cost_on_day": float(result.frame.loc[day, "cost"]),
            "pending_fills_not_canceled": True,
        }
        for day in paused
    ]
    return {
        "checked_original_monthly_opportunities_and_initial": len(opportunities),
        "checked_actual_monthly_decisions_and_initial": len(decisions),
        "source_abstention_count": len(paused),
        "source_abstention_signal_dates": [str(day.date()) for day in paused],
        "all_abstentions_had_no_new_request_and_kept_return": True,
        "issued_requests": len(fills),
        "all_actual_gaps_targets_and_delays_checked": True,
    }, abstentions


def evaluate_path(data, targets, audits, frozen, identifier, window, scenario, benchmarks):
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
            raise QuantError("Completed paths cannot be cherry-picked or replayed.")
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
    new_output_directory(directory)
    execution_policy = read_json(base.checked_reference(frozen["execution_policy"]))
    result, issued, decisions = SOURCE.monthly.run(
        data,
        targets,
        SOURCE.monthly.CANDIDATES[0],
        execution_policy,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback, abstentions = verify_feedback(
        data, targets, audits, result, issued, decisions, window, scenario
    )
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
    execution = base.execution_statistics(result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": result.frame,
        "weights.csv": result.weights,
        "issued.csv": issued,
        "independent-bt.csv": independent,
    }
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
    write_new_receipt(directory / "source-abstentions.json", {"source_abstentions": abstentions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": identifier,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": execution,
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
        "independent_monthly_feedback": feedback,
        "artifacts": [
            base.reference(directory / name)
            for name in (*frames, "decisions.json", "source-abstentions.json")
        ],
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
        raise QuantError("Keep the actual unchanged pre-account information-policy registration.")
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
        "anfci_residual_expired_preserved" if expired else "anfci_residual_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, records, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 16
            or prior["new_completed_strategy_evaluations"] != 16
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve all sixteen completed predecessors.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("Registered residual confirmation cannot change.")
        if not existing:
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve a terminal failed version without same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_anfci_residual",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(
                    data, desired, records, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                directory = TARGETS / identifier
                path = directory / "targets.csv"
                common.output_guard(path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Preserve conflicting partial residual targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "information-audits.json", {"decisions": audits})
                proof["information_audits"] = base.reference(directory / "information-audits.json")
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
                failure("residual_preparation", error)
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
                "anfci_residual_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                anfci_residual_registration=base.reference(PACKET),
                pending_steps=["publish_anfci_residual_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen, original, commit):
    inherited = base.published_inputs(original, commit)
    prior = verify_publication(ROOT, frozen["source_registration"], set(donor.EXTENSIONS), commit)
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
        "donor_dependency_publication": prior,
    }
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for identifier in IDS:
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
                raise QuantError("Completed information-policy review and actual paths disagree.")
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


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, records, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError(
                    "Completed information-policy evidence differs; never replay accounts."
                )
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        common.progress(
            "evaluating_anfci_residual",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            anfci_residual_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                regenerated, audits = rules.build_targets(
                    data, desired, records, read_json(rules.POLICY)
                )
                ref = base.matching_events("target_proof", identifier)[0]["body"]["targets"]
                targets = SOURCE.frame(ref)
                if not np.allclose(
                    targets, regenerated[identifier], rtol=0, atol=1e-12, equal_nan=True
                ):
                    raise QuantError("Actual registered residual targets changed.")
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_anfci_residual",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = evaluate_path(
                            data, targets, audits, frozen, identifier, window, scenario, benchmarks
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
                        "full_source_freshness_admission_repaired": False,
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
                "anfci_residual_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                anfci_residual_results=base.reference(RESULTS),
                pending_steps=["publish_all_anfci_residual_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("residual_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "anfci_residual_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result():
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal information-policy outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("An unfinished information-policy study is not complete.")
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
            "full_source_freshness_admission_repaired": False,
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
        description="One fresh-only ANFCI information policy;no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Frozen-input residual confirmation cannot request live sources.")

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
        parser.exit(2, f"ANFCI residual research blocked: {error}\n")


if __name__ == "__main__":
    main()
