from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import daily_variance_disagreement_v1 as rules
from us_quant import observed_weight_execution as monthly
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_program import review_market
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
FOLDER = base.REPORTS / "daily-variance-disagreement-20261028-v1"
TARGETS = ROOT / "data/ssd-daily-variance-disagreement-20261028-v1"
PACKET = ROOT / "evidence/daily_variance_disagreement_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/daily_variance_disagreement_20261028_results_v1.json"
EXTENSIONS = (
    "config/daily-variance-disagreement-v1.json",
    "src/us_quant/daily_variance_disagreement_v1.py",
    "src/us_quant/ssd_variance_disagreement_research_v1.py",
    "tests/test_daily_variance_disagreement_v1.py",
    "tests/test_ssd_variance_disagreement_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
SOURCE_PUBLIC = ROOT / "evidence/implied_execution_20261012_results.json"


def frame(reference: dict) -> pd.DataFrame:
    return pd.read_csv(
        base.checked_reference(reference),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def check_frozen(frozen: dict) -> None:
    base.check_frozen_files(frozen)
    for reference in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(reference)


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 118 or events[117]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Keep the actual ten-study journal prefix and all failed outcomes.")
    data, _, parent, benchmarks = base.study_inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    source_packet = read_json(base.checked_reference(policy["source_registration"]))
    entries = [
        row
        for row in source_packet["candidates"]
        if row["spec"]["id"] == policy["source_candidate"]
    ]
    if len(entries) != 1:
        raise QuantError("The actual original source registration is required.")
    source = entries[0]["spec"]
    original = review_market(source, ROOT)
    if (
        tuple(data.close.columns) != rules.UNIVERSE
        or not data.risk_free.equals(original.risk_free)
        or any(
            not getattr(data, key).equals(getattr(original, key))
            for key in ("open", "close", "raw_close", "volume")
        )
        or source["factor_ids"] != list(rules.FACTOR_IDS)
        or source["leveraged_products_allowed"] is not False
        or any(value != 1 for value in source["asset_leverage"].values())
    ):
        raise QuantError(
            "The new study must use the same original unleveraged source market and RF."
        )
    desired, options = frame(policy["source_targets"]), frame(policy["monthly_options"])
    if (
        len(desired.dropna()) != 131
        or len(options) != 131
        or not desired.dropna().index.equals(options.index)
        or not desired.index.equals(data.close.index)
    ):
        raise QuantError("Keep every original monthly source decision and option observation.")
    source_refs = [{"path": path, "sha256": sha} for path, sha in source["frozen_files"].items()]
    factors = parent["legacy"]["used_actual_registered_factors"]
    if [row["id"] for row in factors] != list(rules.FACTOR_IDS):
        raise QuantError(
            "Use the four actual registered source families, not inferred factor labels."
        )
    literature = base.REPORTS / "variance-premium-input-review-20261028-v1"
    prior_files = [
        rules.POLICY,
        SOURCE_PUBLIC,
        literature / "source-admissibility.json",
        literature / "crossref-metadata-native-tool-result.json",
        literature / "fed-working-paper-abstract-native-tool-result.json",
    ]
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 188,
        "prior_journal_sequence": 118,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "source_registration": policy["source_registration"],
        "source_targets": policy["source_targets"],
        "monthly_options": policy["monthly_options"],
        "source_candidate": policy["source_candidate"],
        "used_actual_registered_factors": factors,
        "used_economic_factor_families": 4,
        "original_policy_role": "monthly_pure_observed_band_execution_not_daily_feedback",
        "execution_policy": base.reference(monthly.POLICY),
        "new_economic_factor_definitions": 0,
        "literature_replication_or_true_variance_premium_claim": False,
        "prior_evidence": source_refs
        + [
            policy["source_registration"],
            policy["source_targets"],
            policy["monthly_options"],
            base.reference(monthly.POLICY),
            *[base.reference(path) for path in prior_files],
        ],
        "validation_evidence": [
            base.reference(base.REPORTS / name)
            for name in (
                "variance-disagreement-offline-tests-v1.xml",
                "variance-disagreement-offline-tests-v1.log",
            )
        ],
    }
    check_frozen(frozen)
    monthly.validate(read_json(monthly.POLICY))
    monthly.pure_band(read_json(monthly.POLICY))
    return data, desired, options, frozen, parent, benchmarks


def prove_targets(
    data: MarketData, source: pd.DataFrame, options: pd.DataFrame, protocol: dict
) -> tuple:
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, source, options, policy)
    target = targets[IDS[0]]
    if len(audits) != 131 or not target.dropna().index.equals(source.dropna().index):
        raise QuantError("Do not change the original monthly decision lattice.")
    maximum = 0.0
    for day in options.index:
        oracle = rules.independent_row(data.close, source.loc[day], options.loc[day], day)
        maximum = max(maximum, float(np.max(np.abs(target.loc[day] - oracle))))
        if maximum > 1e-12:
            raise QuantError(
                "Independent scalar proxy allocation differs from the registered rule."
            )
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if anchor not in known.index or known.isna().any().any():
            raise QuantError("Every formal return requires its original causal source seed.")
        used = [name for name in rules.CORE if (known[name] > 1e-10).any()]
        if used != list(rules.CORE):
            raise QuantError("All four genuine registered families must participate.")
        coverage.append({"years": window["years"], "factor_symbols": used})
    prefixes = []
    for position in (252, 503, 1006, 1509, len(data.close) - 2):
        end = data.close.index[position]
        prefix_data = MarketData(
            data.open.loc[:end],
            data.close.loc[:end],
            data.raw_close.loc[:end],
            data.volume.loc[:end],
            data.risk_free.loc[:end],
        )
        early, _ = rules.build_targets(prefix_data, source.loc[:end], options.loc[:end], policy)
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("A later source observation changed an earlier proxy target.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "monthly_decisions": 131,
            "maximum_independent_target_difference": maximum,
            "source_BIL_and_non_BIL_scale_preserved": True,
            "source_monthly_dates_and_relative_equity_composition_preserved": True,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "true_variance_risk_premium_or_paper_replication_claimed": False,
        },
    )


def verify_feedback(
    data: MarketData,
    desired: pd.DataFrame,
    result,
    issued: pd.DataFrame,
    decisions: list[dict],
    window: dict,
    scenario: dict,
) -> dict:
    dates = data.close.loc[window["start"] : window["end"]].index
    anchor = previous_session(dates[0])
    expected_dates = [anchor, *[day for day in dates if is_month_end(day)]]
    if [row["signal_session"] for row in decisions] != [str(day.date()) for day in expected_dates]:
        raise QuantError("The model inspected holdings on a nonmonthly or missing decision date.")
    known = desired.ffill()
    fills = []
    expected_issued = []
    for day, row in zip(expected_dates, decisions, strict=True):
        initial = day == anchor
        current = pd.Series(0.0, index=desired.columns) if initial else result.weights.loc[day]
        target = known.loc[day]
        gap = float(np.max(np.abs(current - target)))
        should_trade = initial or gap + 1e-12 >= 0.025
        if (
            row["initial_capital"] is not initial
            or (row["reason"] != "hold") != should_trade
            or abs(row["maximum_observed_target_gap"] - gap) > 1e-12
        ):
            raise QuantError("Independent actual-holding gap verification failed.")
        execution = day
        for _ in range(scenario["delay_sessions"]):
            execution = next_session(execution)
        if row["execution_session"] != str(execution.date()):
            raise QuantError("A monthly request was retimed or filled early.")
        if should_trade:
            if not np.allclose(issued.loc[day], target, rtol=0, atol=1e-12):
                raise QuantError("Issued orders do not follow the actual known monthly target.")
            expected_issued.append(day)
            fills.append(execution)
    if (
        len(fills) != len(set(fills))
        or not issued.dropna().index.equals(pd.DatetimeIndex(expected_issued))
        or any(day not in fills for day in result.frame.index[result.frame["orders"] > 0])
    ):
        raise QuantError(
            "Pending requests were overwritten or trades lack an original dated signal."
        )
    return {
        "checked_actual_monthly_decisions_and_initial": len(decisions),
        "issued_requests": len(fills),
        "all_actual_gaps_targets_and_delays_checked": True,
    }


def evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks) -> dict:
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
    execution_policy = read_json(base.checked_reference(frozen["execution_policy"]))
    result, issued, decisions = monthly.run(
        data,
        targets,
        monthly.CANDIDATES[0],
        execution_policy,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback = verify_feedback(data, targets, result, issued, decisions, window, scenario)
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
    base.storage_guard()
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
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
        raise QuantError("Keep the actual pre-account daily-proxy registration.")
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
        "daily_proxy_expired_preserved" if expired else "daily_proxy_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, source, options, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        counts = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            counts["new_registered_candidates"] != 10
            or counts["new_completed_strategy_evaluations"] != 10
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve all ten actual completed predecessors.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("Do not change a registered daily-proxy version.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve the terminal failed version without replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_daily_proxy",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(
                    data, source, options, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                directory = TARGETS / identifier
                path = directory / "targets.csv"
                common.output_guard(path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Conflicting partial proxy targets must be preserved.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "proxy-audits.json", {"decisions": audits})
                proof["proxy_audits"] = base.reference(directory / "proxy-audits.json")
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
                failure("daily_proxy_target_preparation", error)
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
                "daily_proxy_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                daily_proxy_registration=base.reference(PACKET),
                pending_steps=["publish_daily_proxy_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    source = verify_publication(
        ROOT,
        frozen["source_registration"],
        {
            "config/implied-risk-execution.json",
            "src/us_quant/implied_risk_execution.py",
            "config/observed-weight-execution.json",
            "src/us_quant/observed_weight_execution.py",
        },
        commit,
    )
    own = verify_publication(ROOT, base.reference(PACKET), set(EXTENSIONS), commit)
    events = [
        row
        for row in base.matching_events("publication")
        if row["body"].get("study_id") == frozen["study_id"]
    ]
    if events:
        if len(events) != 1 or events[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with the actual original daily-proxy publication commit.")
        return events[0]["body"]["publication"]
    receipt = {
        **own,
        "original_dependency_publication": original_receipt,
        "source_execution_dependency_publication": source,
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
                raise QuantError("The completed proxy review and actual accounts disagree.")
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
        data, source, options, frozen, original, benchmarks = inputs()
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
            "evaluating_daily_proxy",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            daily_proxy_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                regenerated, _ = rules.build_targets(data, source, options, read_json(rules.POLICY))
                targets = frame(
                    base.matching_events("target_proof", identifier)[0]["body"]["targets"]
                )
                if (
                    not targets.index.equals(data.close.index)
                    or not targets.columns.equals(data.close.columns)
                    or not np.allclose(
                        targets, regenerated[identifier], rtol=0, atol=1e-12, equal_nan=True
                    )
                ):
                    raise QuantError(
                        "Registered proxy targets differ from the frozen transformation."
                    )
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_daily_proxy", active_candidate=identifier, active_path=key
                        )
                        paths[key] = evaluate_path(
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
                "daily_proxy_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                daily_proxy_results=base.reference(RESULTS),
                pending_steps=["publish_all_daily_proxy_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("daily_proxy_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "daily_proxy_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal daily-proxy outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Unfinished evidence cannot become a completed outcome.")
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
            "true_variance_risk_premium_or_paper_replication_claimed": False,
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
        description="One coarse daily-risk proxy; no trading or VRP claim."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The frozen-input proxy study cannot request live data.")

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
        parser.exit(2, f"Daily proxy research blocked: {error}\n")


if __name__ == "__main__":
    main()
