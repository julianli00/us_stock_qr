from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from us_quant import channel_regimes_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.calendar import previous_session
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
TARGETS = ROOT / "data/ssd-channel-regimes-20261027-v1"
RESULTS = base.REPORTS / "channel-regimes-20261027-v1/results.json"
PACKET = ROOT / "evidence/channel_regimes_20261027_registration_v1.json"
PUBLIC_RESULTS = ROOT / "evidence/channel_regimes_20261027_results_v1.json"
EXTENSIONS = (
    "config/channel-regimes-v1.json",
    "src/us_quant/channel_regimes_v1.py",
    "src/us_quant/ssd_channel_research_v1.py",
    "tests/test_channel_regimes_v1.py",
    "tests/test_ssd_channel_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    common.counts(events)
    if len(events) < 47 or events[46]["sha256"] != rules.PRIOR_JOURNAL_HEAD:
        raise QuantError("The disclosed pre-channel trial history changed.")
    data, _, parent, original, benchmarks = common.inputs()
    common.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    diagnostic = common.RESULTS.parent
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 182,
        "prior_journal_sequence": 47,
        "prior_journal_head_sha256": rules.PRIOR_JOURNAL_HEAD,
        "inherited_source_target_reference_is_lineage_not_channel_input": True,
        "channel_inputs_are_original_admitted_prices": True,
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in ("channel-offline-tests-v1.xml", "channel-offline-tests-v1.log")
        ],
        "prior_evidence": parent["prior_evidence"]
        + [
            base.reference(path)
            for path in (
                common.PACKET,
                common.PUBLIC_RESULTS,
                diagnostic / "saved-risk-diagnostic-v1.py",
                diagnostic / "saved-risk-diagnostic-v1-intent.json",
                diagnostic / "saved-risk-diagnostic-v1.json",
            )
        ],
    }
    common.check_frozen(frozen)
    return data, frozen, original, benchmarks


def prove_targets(data, policy: dict, protocol: dict) -> tuple[dict, dict]:
    targets = rules.build_targets(data, policy)
    oracle = rules.independent_targets(data, policy)
    gaps = {}
    for identifier, target in targets.items():
        if not np.allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("The separate sequential price-channel oracle disagrees.")
        defined = target.dropna(how="all")
        if (
            defined.empty
            or target.isna().any(axis=1).iloc[252:].any()
            or not np.isfinite(defined).all().all()
            or (defined < 0).any().any()
            or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Channel targets must be fully observed, nonnegative and funded.")
        gaps[identifier] = float(
            abs(defined - oracle[identifier].loc[defined.index]).to_numpy().max()
        )
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            formal = target.loc[anchor : window["end"]]
            if anchor not in formal.index or formal.isna().any().any():
                raise QuantError("Continuous channel warmup must precede both fresh accounts.")
    prefixes = []
    for position in (63, 199, 252, 503, 1006, 1509, 2012, len(data.close) - 2):
        if not 0 <= position < len(data.close):
            continue
        earlier = replace(
            data,
            **{
                name: getattr(data, name).iloc[: position + 1]
                for name in ("open", "close", "raw_close", "volume", "risk_free")
            },
        )
        for identifier, target in rules.build_targets(earlier, policy).items():
            if not target.equals(targets[identifier].iloc[: position + 1]):
                raise QuantError("Future prices changed a historical channel state.")
        prefixes.append(str(data.close.index[position].date()))
    return targets, {
        "checked_at": utc_now(),
        "independent_sequential_price_oracle_all_rows": True,
        "maximum_weight_difference": gaps,
        "prefix_truncations": prefixes,
        "all_formal_initial_anchors_and_sessions_defined": True,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "new_channel_states_computed_from_admitted_prices": True,
    }


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
        raise QuantError("Require the unchanged actual pre-account channel registration packet.")
    return packet


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
    common.progress(
        "channel_authorization_expired_preserved" if expired else "channel_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(exc),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([event for event in events if event["candidate_id"] not in IDS])
        if (
            events[0]["body"]["legacy"] != frozen["legacy"]
            or prior["new_registered_candidates"] != 4
            or prior["new_completed_strategy_evaluations"] != 4
        ):
            raise QuantError("Preserve the actual four completed pre-channel candidates.")
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", candidate["id"])
            if existing and (len(existing) != 1 or existing[0]["body"] != specification):
                raise QuantError("A registered channel version cannot change.")
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Preserve terminal channel-preparation failures; do not replay.")
        pending = tuple(
            identifier for identifier in IDS if not base.matching_events("target_proof", identifier)
        )
        if pending:
            common.progress(
                "preparing_registered_channel_targets",
                active_compute_job={
                    "pid": os.getpid(),
                    "study_id": frozen["study_id"],
                },
            )
            try:
                runtime_guard()
                targets, proof = prove_targets(
                    data, read_json(rules.POLICY), read_json(base.PROTOCOL)
                )
                common.check_frozen(frozen)
                for identifier in pending:
                    directory = TARGETS / identifier
                    target_path = directory / "targets.csv"
                    common.output_guard(target_path)
                    content = targets[identifier].to_csv(float_format="%.17g")
                    if target_path.exists():
                        if file_digest(target_path) != digest_bytes(content.encode()):
                            raise QuantError(
                                "Conflicting partial channel targets must be preserved."
                            )
                    else:
                        write_text_atomic(target_path, content)
                    common.retain_receipt(directory / "proof.json", proof, ("checked_at",))
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
                    "registered_channel_target_preparation",
                    tuple(
                        identifier
                        for identifier in pending
                        if not base.matching_events("target_proof", identifier)
                    ),
                    exc,
                )
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
                "channel_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                channel_registration=base.reference(PACKET),
                current_research="Fixed price-channel pair registered; no channel account yet",
                pending_steps=["publish_exact_channel_packet", "four_path_channel_accounts"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    prior_receipt = verify_publication(
        ROOT, base.reference(common.PACKET), set(common.EXTENSIONS), commit
    )
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
    receipt = {
        **new_receipt,
        "parent_dependency_publication": original_receipt,
        "prior_cohort_dependency_publication": prior_receipt,
    }
    base.append_event(
        base.JOURNAL,
        "publication",
        None,
        {
            "study_id": frozen["study_id"],
            "publication": receipt,
        },
    )
    return receipt


def candidate_states(frozen: dict) -> dict:
    states = {}
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
                raise QuantError("Actual channel review and path identities disagree.")
            states[identifier] = reviews[0]["body"]
        else:
            states[identifier] = {
                "candidate_id": identifier,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return states


def evaluate(commit: str) -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        common.output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or (
                saved["candidates"] != candidate_states(frozen)
            ):
                raise QuantError("Completed channel evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        protocol, execution_policy = read_json(base.PROTOCOL), read_json(base.POLICY)
        regenerated = rules.build_targets(data, read_json(rules.POLICY))
        common.progress(
            "evaluating_registered_channel_pair",
            active_compute_job={
                "pid": os.getpid(),
                "study_id": frozen["study_id"],
            },
            channel_publication=published,
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
                    raise QuantError("Registered channel targets differ from the published rule.")
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        common.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_registered_channel_pair",
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
                common.check_frozen(frozen)
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
            events = base.journal_events(base.JOURNAL)
            common.retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": candidate_states(frozen),
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
                "channel_pair_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                channel_results=base.reference(RESULTS),
                pending_steps=["publish_all_channel_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            record_failure(
                "registered_channel_account_evaluation", (active,) if active else (), exc
            )
            raise
        except KeyboardInterrupt:
            common.progress(
                "channel_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("No complete channel publication before the actual terminal receipt.")
    saved, states = read_json(RESULTS), candidate_states(frozen)
    if saved["candidates"] != states or any(row["status"] == "pending" for row in states.values()):
        raise QuantError("Unfinished channel candidates cannot become completed outcomes.")
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
    counts = common.counts([row for row in events if row["candidate_id"] in IDS])
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
    parser = argparse.ArgumentParser(description="Fixed channel research; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Admitted-history channel research cannot request live data.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published channel registration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = common.counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as exc:
        parser.exit(2, f"Channel research blocked: {exc}\n")


if __name__ == "__main__":
    main()
