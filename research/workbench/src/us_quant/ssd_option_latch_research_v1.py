from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import option_exit_latch_v1 as rules
from us_quant import ssd_channel_research_v1 as prior
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
TARGETS = ROOT / "data/ssd-option-exit-latch-20261027-v1"
RESULTS = base.REPORTS / "option-exit-latch-20261027-v1/results.json"
PACKET = ROOT / "evidence/option_exit_latch_20261027_registration_v1.json"
PUBLIC_RESULTS = ROOT / "evidence/option_exit_latch_20261027_results_v1.json"
EXTENSIONS = (
    "config/option-exit-latch-v1.json",
    "src/us_quant/option_exit_latch_v1.py",
    "src/us_quant/ssd_option_latch_research_v1.py",
    "tests/test_option_exit_latch_v1.py",
    "tests/test_ssd_option_latch_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    common.counts(events)
    if len(events) < 70 or events[69]["sha256"] != rules.PRIOR_JOURNAL_HEAD:
        raise QuantError("The disclosed six-candidate history changed.")
    data, parent, original, benchmarks = prior.inputs()
    prior.checked_packet(parent)
    selected = [
        row
        for row in read_json(common.PACKET)["entries"]
        if row["candidate_id"] == rules.SOURCE_CANDIDATE
    ]
    if len(selected) != 1:
        raise QuantError("Require the single actual registered month-end source.")
    source_refs = {
        "packet": base.reference(common.PACKET),
        "targets": selected[0]["target_proof"]["targets"],
        "proof": selected[0]["target_proof"]["proof"],
    }
    source = pd.read_csv(
        base.checked_reference(source_refs["targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    terms = base.load_terms(base.SOURCES, data.close.index)
    if not source.columns.equals(data.close.columns):
        raise QuantError("The saved source and admitted portfolio column order differ.")
    rules.validate_inputs(source, terms)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    diagnostic = prior.RESULTS.parent
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 184,
        "prior_journal_sequence": 70,
        "prior_journal_head_sha256": rules.PRIOR_JOURNAL_HEAD,
        "commitment_source": {"candidate_id": rules.SOURCE_CANDIDATE, **source_refs},
        "saved_commitment_source_recomputed": False,
        "inherited_channel_flags_describe_prior_study_only": True,
        "current_model_inputs": ["saved_month_end_commitment_targets", "VIX", "VIX3M"],
        "channel_targets_consumed_by_this_model": False,
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in ("option-latch-offline-tests-v1.xml", "option-latch-offline-tests-v1.log")
        ],
        "prior_evidence": parent["prior_evidence"]
        + list(source_refs.values())
        + [
            base.reference(path)
            for path in (
                prior.PACKET,
                prior.PUBLIC_RESULTS,
                diagnostic / "saved-risk-diagnostic-v1.py",
                diagnostic / "saved-risk-diagnostic-v1-intent.json",
                diagnostic / "saved-risk-diagnostic-v1.json",
            )
        ],
    }
    common.check_frozen(frozen)
    return data, source, terms, frozen, original, benchmarks


def prove_targets(source: pd.DataFrame, terms: pd.DataFrame, protocol: dict) -> tuple[dict, dict]:
    policy = read_json(rules.POLICY)
    targets = rules.build_targets(source, terms, policy)
    oracle = rules.independent_targets(source, terms, policy)
    gaps = {}
    for identifier, target in targets.items():
        if not np.allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Independent sequential option-latch targets disagree.")
        defined = target.dropna(how="all")
        if (
            defined.empty
            or not target.isna().all(axis=1).equals(source.isna().all(axis=1))
            or target.loc[defined.index].isna().any().any()
            or not np.isfinite(defined).all().all()
            or (defined < 0).any().any()
            or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Preserve the actual source warmup and funded defined targets.")
        gaps[identifier] = float(
            abs(defined - oracle[identifier].loc[defined.index]).to_numpy().max()
        )
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            formal = target.loc[anchor : window["end"]]
            if anchor not in formal.index or formal.isna().any().any():
                raise QuantError("Both formal fresh accounts require complete prior intent.")
    prefixes = []
    for position in (63, 199, 252, 272, 503, 1006, 1509, 2012, len(source) - 2):
        if not 0 <= position < len(source):
            continue
        part = rules.build_targets(source.iloc[: position + 1], terms.iloc[: position + 1], policy)
        for identifier, target in part.items():
            if not target.equals(targets[identifier].iloc[: position + 1]):
                raise QuantError("Later observations changed a past option-latch state.")
        prefixes.append(str(source.index[position].date()))
    return targets, {
        "checked_at": utc_now(),
        "independent_sequential_oracle_all_rows": True,
        "maximum_weight_difference": gaps,
        "prefix_truncations": prefixes,
        "all_formal_initial_anchors_and_sessions_defined": True,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "saved_monthly_commitment_recomputed": False,
        "complete_named_option_sessions": len(terms),
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
        raise QuantError("Keep the actual pre-account option-latch registration packet.")
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
        "option_latch_authorization_expired_preserved" if expired else "option_latch_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(exc),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, source, terms, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        previous = common.counts([event for event in events if event["candidate_id"] not in IDS])
        if (
            events[0]["body"]["legacy"] != frozen["legacy"]
            or previous["new_registered_candidates"] != 6
            or previous["new_completed_strategy_evaluations"] != 6
        ):
            raise QuantError("Preserve all six actual completed predecessor hypotheses.")
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", candidate["id"])
            if existing and (len(existing) != 1 or existing[0]["body"] != specification):
                raise QuantError("A registered option-latch version cannot change.")
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Preserve terminal preparation failures instead of replaying them.")
        pending = tuple(
            identifier for identifier in IDS if not base.matching_events("target_proof", identifier)
        )
        if pending:
            common.progress(
                "preparing_registered_option_latch_targets",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, proof = prove_targets(source, terms, read_json(base.PROTOCOL))
                common.check_frozen(frozen)
                for identifier in pending:
                    directory = TARGETS / identifier
                    target_path = directory / "targets.csv"
                    common.output_guard(target_path)
                    content = targets[identifier].to_csv(float_format="%.17g")
                    if target_path.exists():
                        if file_digest(target_path) != digest_bytes(content.encode()):
                            raise QuantError("Preserve conflicting partial targets.")
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
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
                record_failure(
                    "option_latch_target_preparation",
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
                "option_latch_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                option_latch_registration=base.reference(PACKET),
                current_research="One option exit-latch hypothesis registered; no account yet",
                pending_steps=[
                    "publish_exact_option_latch_packet",
                    "four_path_option_latch_accounts",
                ],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    dependencies = [
        verify_publication(ROOT, base.reference(module.PACKET), set(module.EXTENSIONS), commit)
        for module in (common, prior)
    ]
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
        "prior_cohort_dependency_publications": dependencies,
    }
    base.append_event(
        base.JOURNAL,
        "publication",
        None,
        {"study_id": frozen["study_id"], "publication": receipt},
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
                raise QuantError("Actual option-latch review and path identities disagree.")
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
        data, source, terms, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        common.output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or (
                saved["candidates"] != candidate_states(frozen)
            ):
                raise QuantError("Completed evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        regenerated = rules.build_targets(source, terms, read_json(rules.POLICY))
        protocol, execution_policy = read_json(base.PROTOCOL), read_json(base.POLICY)
        common.progress(
            "evaluating_registered_option_latch",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            option_latch_publication=published,
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
                    raise QuantError("Registered option-latch targets differ from the frozen law.")
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        common.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_registered_option_latch",
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
                "option_latch_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                option_latch_results=base.reference(RESULTS),
                pending_steps=["publish_all_option_latch_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            record_failure("option_latch_account_evaluation", (active,) if active else (), exc)
            raise
        except KeyboardInterrupt:
            common.progress(
                "option_latch_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("No complete publication before the actual terminal result.")
    saved, states = read_json(RESULTS), candidate_states(frozen)
    if saved["candidates"] != states or any(row["status"] == "pending" for row in states.values()):
        raise QuantError("Unfinished option-latch evidence cannot become a completed outcome.")
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
    parser = argparse.ArgumentParser(description="One fixed option-exit latch; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Admitted-history research cannot request live data.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published registration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = common.counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as exc:
        parser.exit(2, f"Option-latch research blocked: {exc}\n")


if __name__ == "__main__":
    main()
