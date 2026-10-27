from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import commitment_duration_v1 as rules
from us_quant import ssd_research_v1 as base
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
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
TARGETS = ROOT / "data/ssd-commitment-duration-20261027-v1"
RESULTS = base.REPORTS / "commitment-duration-20261027-v1/results.json"
PACKET = ROOT / "evidence/commitment_duration_20261027_registration_v1.json"
PUBLIC_RESULTS = ROOT / "evidence/commitment_duration_20261027_results_v1.json"
EXTENSIONS = (
    "config/commitment-duration-v1.json",
    "src/us_quant/commitment_duration_v1.py",
    "src/us_quant/ssd_commitment_research_v1.py",
    "tests/test_commitment_duration_v1.py",
    "tests/test_ssd_commitment_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)


def counts(events: list[dict]) -> dict:
    identities = {}
    for kind in ("registered", "review", "technical_failure"):
        rows = [event for event in events if event["kind"] == kind]
        identities[kind] = {row["candidate_id"] for row in rows}
        if len(identities[kind]) != len(rows):
            raise QuantError("Duplicate candidate lifecycle events cannot change trial counts.")
    registered, reviewed, failed = (identities[key] for key in identities)
    paths = [
        (row["candidate_id"], row["body"]["path_key"])
        for row in events
        if row["kind"] == "path_completed"
    ]
    if (
        reviewed & failed
        or not (reviewed | failed).issubset(registered)
        or len(set(paths)) != len(paths)
        or any(
            identifier not in registered or key not in base.PATH_KEYS for identifier, key in paths
        )
        or any(
            {key for candidate, key in paths if candidate == identifier} != base.PATH_KEYS
            for identifier in reviewed
        )
        or any(
            row["body"].get("status") not in {"rejected", "historical_point_gates_passed"}
            for row in events
            if row["kind"] == "review"
        )
    ):
        raise QuantError("Completed, partial and terminal-failure trials must remain distinct.")
    return {
        "new_registered_candidates": len(registered),
        "new_completed_strategy_evaluations": len(reviewed),
        "new_reviews": len(reviewed),
        "new_technical_failures": len(failed),
        "new_pending_candidates": len(registered - reviewed - failed),
        "new_account_paths": len(paths),
        "new_historical_point_gate_passes": sum(
            row["body"]["status"] == "historical_point_gates_passed"
            for row in events
            if row["kind"] == "review"
        ),
    }


def progress(phase: str, **fields) -> None:
    base.progress(phase, **counts(base.journal_events(base.JOURNAL)), **fields)


def check_frozen(frozen: dict) -> None:
    base.check_frozen_files(frozen)
    for reference in [frozen["source"][key] for key in ("packet", "targets", "proof")]:
        base.checked_reference(reference)
    for reference in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(reference)


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    counts(events)
    if len(events) < 24 or events[23]["sha256"] != rules.PRIOR_JOURNAL_HEAD:
        raise QuantError("The disclosed original new-lane journal prefix changed.")
    data, _, parent, benchmarks = base.study_inputs()
    parent_packet = base.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = next(
        row for row in parent_packet["entries"] if row["candidate_id"] == rules.SOURCE_CANDIDATE
    )
    source_reference = entry["target_proof"]["targets"]
    source = pd.read_csv(
        base.checked_reference(source_reference),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.validate_source(source)
    if not source.index.equals(data.close.index) or not source.columns.equals(data.close.columns):
        raise QuantError("The original proved intent lattice differs from admitted market data.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "source": {
            "candidate_id": rules.SOURCE_CANDIDATE,
            "packet": base.reference(base.PACKET),
            "targets": source_reference,
            "proof": entry["target_proof"]["proof"],
            "registered_event_sha256": entry["registered_event_sha256"],
            "target_proof_event_sha256": entry["target_proof_event_sha256"],
            "source_signal_recomputed": False,
        },
        "prior_completed_configurations": 180,
        "validation_evidence": [
            base.reference(base.REPORTS / name)
            for name in ("commitment-offline-tests-v1.xml", "commitment-offline-tests-v1.log")
        ],
        "prior_evidence": [
            base.reference(ROOT / name)
            for name in (
                "evidence/equity_first_20261027_results_v1.json",
                "reports/ssd-continuous-research-v1/equity-first-20261027-v1/saved-account-diagnostic-v1.json",
                "reports/ssd-continuous-research-v1/equity-first-20261027-v1/saved-holdings-pnl-bridge-v1.json",
            )
        ],
        "account_namespace": base.ACCOUNTS.relative_to(ROOT).as_posix(),
        "account_namespace_reuses_immutable_execution_not_original_candidate_ids": True,
    }
    check_frozen(frozen)
    return data, source, frozen, parent, benchmarks


def prove_targets(source: pd.DataFrame, policy: dict, protocol: dict) -> tuple[dict, dict]:
    targets = rules.build_targets(source, policy)
    oracle = rules.independent_targets(source, policy)
    differences = {}
    for identifier, target in targets.items():
        if not np.allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("The independent commitment oracle disagrees.")
        defined = target.dropna(how="all")
        if defined.empty:
            raise QuantError("A registered source must establish complete commitment warmup.")
        differences[identifier] = float(
            abs(defined - oracle[identifier].loc[defined.index]).to_numpy().max()
        )
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            formal = target.loc[anchor : window["end"]]
            if anchor not in formal.index or formal.isna().any().any():
                raise QuantError("Commitment warmup must precede every full formal account window.")
    prefixes = []
    for position in sorted({20, 252, 272, 503, 1006, 1509, 2012, len(source) - 2}):
        if not 0 <= position < len(source):
            continue
        earlier = rules.build_targets(source.iloc[: position + 1], policy)
        for identifier in targets:
            if not earlier[identifier].equals(targets[identifier].iloc[: position + 1]):
                raise QuantError("Future observations changed a historical commitment.")
        prefixes.append(str(source.index[position].date()))
    return targets, {
        "checked_at": utc_now(),
        "independent_sequential_deque_oracle_all_rows": True,
        "maximum_weight_difference": differences,
        "prefix_truncations": prefixes,
        "all_formal_initial_anchors_and_sessions_defined": True,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
    }


def output_guard(path: Path) -> None:
    base.storage_guard()
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise QuantError("New evidence must remain inside its actual SSD workbench directory.")


def retain_receipt(path: Path, body: dict, volatile: tuple[str, ...] = ()) -> dict:
    output_guard(path)
    if path.exists():
        saved = read_json(path)
        if {key: value for key, value in saved.items() if key not in volatile} != {
            key: value for key, value in body.items() if key not in volatile
        }:
            raise QuantError("Conflicting prior evidence must be preserved, not overwritten.")
        return saved
    write_new_receipt(path, body)
    return body


def record_failure(stage: str, identifiers: tuple[str, ...], exc: Exception) -> None:
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for identifier in identifiers:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                base.append_event(
                    base.JOURNAL,
                    "technical_failure",
                    identifier,
                    {
                        "stage": stage,
                        "reason": str(exc),
                        "partial_outputs_retained": True,
                        "candidate_accounts_started": bool(
                            base.matching_events("path_started", identifier)
                        ),
                    },
                )
    progress(
        "commitment_authorization_expired_preserved" if expired else "commitment_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(exc),
    )


def packet_entry(candidate: dict, frozen: dict) -> dict:
    registered = base.matching_events("registered", candidate["id"])
    prepared = base.matching_events("target_proof", candidate["id"])
    if len(registered) != 1 or len(prepared) != 1:
        raise QuantError("Require actual registration and target-proof events for each candidate.")
    registration, proof = registered[0], prepared[0]
    if (
        registration["body"] != {"candidate": candidate, "frozen": frozen}
        or proof["body"]["registered_event_sha256"] != registration["sha256"]
    ):
        raise QuantError("The commitment identity, frozen rules or source binding changed.")
    base.checked_reference(proof["body"]["targets"])
    evidence = read_json(base.checked_reference(proof["body"]["proof"]))
    clocks = [
        pd.Timestamp(value) for value in (registration["at"], evidence["checked_at"], proof["at"])
    ]
    if (
        any(clock.tzinfo is None or pd.isna(clock) for clock in clocks)
        or not clocks[0] <= clocks[1] <= clocks[2] <= pd.Timestamp(utc_now())
        or evidence["candidate_accounts_or_metrics_computed"] is not False
        or evidence["original_source_signal_recomputed"] is not False
    ):
        raise QuantError("Commitment proof must follow actual registration and precede outcomes.")
    return {
        "candidate_id": candidate["id"],
        "registered_at": registration["at"],
        "registered_event_sha256": registration["sha256"],
        "target_proof_at": proof["at"],
        "target_proof_event_sha256": proof["sha256"],
        "specification": registration["body"],
        "target_proof": proof["body"],
    }


def checked_packet(frozen: dict) -> dict:
    output_guard(PACKET)
    packet = read_json(PACKET)
    entries = [packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
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
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != len(IDS)
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
        raise QuantError("Require the complete unchanged pre-account commitment packet.")
    return packet


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, source, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = counts([event for event in events if event["candidate_id"] not in IDS])
        if (
            not events
            or events[0]["body"]["legacy"] != frozen["legacy"]
            or prior["new_registered_candidates"] != 2
            or prior["new_completed_strategy_evaluations"] != 2
        ):
            raise QuantError("The disclosed pre-study trial history changed; do not reset it.")
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", candidate["id"])
            if existing and (len(existing) != 1 or existing[0]["body"] != specification):
                raise QuantError(
                    "The registered candidate is immutable; use a new reviewed version."
                )
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Preserve terminal preparation failures; do not rerun this version.")
        pending = tuple(
            identifier for identifier in IDS if not base.matching_events("target_proof", identifier)
        )
        if pending:
            progress(
                "preparing_registered_commitment_targets",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, proof = prove_targets(
                    source, read_json(rules.POLICY), read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                for identifier in pending:
                    directory = TARGETS / identifier
                    target_path = directory / "targets.csv"
                    output_guard(target_path)
                    content = targets[identifier].to_csv(float_format="%.17g")
                    if target_path.exists():
                        if file_digest(target_path) != digest_bytes(content.encode()):
                            raise QuantError("Preserve conflicting partial commitment targets.")
                    else:
                        write_text_atomic(target_path, content)
                    retain_receipt(directory / "proof.json", proof, ("checked_at",))
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        identifier,
                        {
                            "registered_event_sha256": base.matching_events(
                                "registered", identifier
                            )[0]["sha256"],
                            "targets": base.reference(target_path),
                            "proof": base.reference(directory / "proof.json"),
                            "defined_target_sessions": len(targets[identifier].dropna(how="all")),
                            "candidate_accounts_or_metrics_computed": False,
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
                record_failure(
                    "registered_commitment_target_preparation",
                    tuple(
                        identifier
                        for identifier in pending
                        if not base.matching_events("target_proof", identifier)
                    ),
                    exc,
                )
                raise
        packet = {
            "schema_version": 2,
            "program_id": "ssd_continuous_research_v1",
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "entries": [packet_entry(candidate, frozen) for candidate in rules.CANDIDATES],
            "new_registered_candidates": len(IDS),
            "candidate_accounts_or_metrics_computed": False,
            "raw_inputs_and_target_matrices_remain_private": True,
            "investment_objective_verified": False,
            "order_authority": False,
            "independent_forward_validation": False,
        }
        retain_receipt(PACKET, packet, ("created_at",))
        checked_packet(frozen)
        if not any(base.matching_events("path_started", identifier) for identifier in IDS):
            progress(
                "commitment_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                commitment_registration=base.reference(PACKET),
                current_research="Registered commitment-duration pair; no candidate account yet",
                pending_steps=["publish_exact_commitment_packet", "four_path_commitment_accounts"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, parent: dict, commit: str) -> dict:
    parent_receipt = base.published_inputs(parent, commit)
    new_receipt = verify_publication(ROOT, base.reference(PACKET), set(EXTENSIONS), commit)
    existing = [
        event
        for event in base.matching_events("publication")
        if event["body"].get("study_id") == frozen["study_id"]
    ]
    if existing:
        if len(existing) != 1 or existing[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with this study's actual original publication commit.")
        return existing[0]["body"]["publication"]
    receipt = {**new_receipt, "parent_dependency_publication": parent_receipt}
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def candidate_states(frozen: dict) -> dict:
    result = {}
    for identifier in IDS:
        reviews = base.matching_events("review", identifier)
        failed = base.matching_events("technical_failure", identifier)
        paths = {
            event["body"]["path_key"]: base.verify_path(
                event["body"]["receipt"], frozen, identifier, event["body"]["path_key"]
            )
            for event in base.matching_events("path_completed", identifier)
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("Saved review and actual completed path receipts disagree.")
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
        data, source, frozen, parent, benchmarks = inputs()
        checked_packet(frozen)
        output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or (
                saved["candidates"] != candidate_states(frozen)
            ):
                raise QuantError("Completed cohort evidence differs; never replay it.")
            return base.reference(RESULTS)
        published = publication(frozen, parent, commit)
        protocol, execution_policy = read_json(base.PROTOCOL), read_json(base.POLICY)
        regenerated = rules.build_targets(source, read_json(rules.POLICY))
        progress(
            "evaluating_registered_commitment_pair",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            commitment_publication=published,
        )
        active = None
        try:
            for identifier in IDS:
                if base.matching_events("review", identifier) or base.matching_events(
                    "technical_failure", identifier
                ):
                    continue
                active = identifier
                proof = base.matching_events("target_proof", identifier)[0]["body"]
                targets = pd.read_csv(
                    base.checked_reference(proof["targets"]),
                    index_col=0,
                    parse_dates=True,
                    float_precision="round_trip",
                )
                if (
                    not targets.index.equals(data.close.index)
                    or not targets.columns.equals(data.close.columns)
                    or not np.allclose(
                        targets, regenerated[identifier], rtol=0, atol=1e-12, equal_nan=True
                    )
                ):
                    raise QuantError(
                        "Submitted commitments do not reproduce the published source rule."
                    )
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        progress(
                            "evaluating_registered_commitment_pair",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = base.evaluate_path(
                            data,
                            targets,
                            frozen,
                            identifier,
                            window,
                            scenario,
                            execution_policy,
                            benchmarks,
                        )
                        progress("evaluating_registered_commitment_pair", last_completed_path=key)
                check_frozen(frozen)
                base.legacy_snapshot(execution_policy)
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
            states = candidate_states(frozen)
            events = base.journal_events(base.JOURNAL)
            retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states,
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": counts(events),
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            progress(
                "commitment_pair_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                commitment_results=base.reference(RESULTS),
                pending_steps=["publish_all_commitment_results", "next_distinct_economic_question"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            record_failure(
                "registered_commitment_account_evaluation",
                (active,) if active is not None else (),
                exc,
            )
            raise
        except KeyboardInterrupt:
            progress(
                "commitment_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish a complete cohort result only after its actual terminal receipt.")
    saved = read_json(RESULTS)
    states = candidate_states(frozen)
    if saved["candidates"] != states or any(row["status"] == "pending" for row in states.values()):
        raise QuantError("Unfinished candidates cannot become a completed cohort.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for identifier, state in states.items():
        paths = []
        for key, path in state["paths"].items():
            years, scenario = key.split("-")
            years = int(years[:-1])
            paths.append(
                {
                    **{
                        name: path[name]
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
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        symbol: lookup[years, scenario, symbol] for symbol in ("SPY", "QQQ")
                    },
                }
            )
        reviews.append(
            {
                "candidate_id": identifier,
                "status": state["status"],
                "paths": paths,
                "reviewed_at": state.get("reviewed_at"),
                "failure_event_sha256": state.get("failure_event_sha256"),
                "failure_stage": state.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": state.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    events = base.journal_events(base.JOURNAL)
    cohort_counts = counts([row for row in events if row["candidate_id"] in IDS])
    body = {
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
        **cohort_counts,
        "new_completed_configurations": cohort_counts["new_completed_strategy_evaluations"],
        "global_counts_at_completion": saved["global_counts_at_completion"],
        "original_records_reclassified": False,
        "raw_prices_documents_targets_or_daily_accounts_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
        "automatic_live_deployment": False,
    }
    retain_receipt(PUBLIC_RESULTS, body, ("created_at",))
    return base.reference(PUBLIC_RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned commitment study; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Admitted-history research must not request any live data.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published commitment registration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as exc:
        parser.exit(2, f"Commitment research blocked: {exc}\n")


if __name__ == "__main__":
    main()
