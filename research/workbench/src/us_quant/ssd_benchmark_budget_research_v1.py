from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import benchmark_tail_budget_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_joint_tail_research_v1 as previous
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
FOLDER = base.REPORTS / "benchmark-tail-budget-20261028-v1"
TARGETS = ROOT / "data/ssd-benchmark-tail-budget-20261028-v1"
PACKET = ROOT / "evidence/benchmark_tail_budget_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/benchmark_tail_budget_20261028_results_v1.json"
EXTENSIONS = (
    "config/benchmark-tail-budget-v1.json",
    "src/us_quant/benchmark_tail_budget_v1.py",
    "src/us_quant/ssd_benchmark_budget_research_v1.py",
    "tests/test_benchmark_tail_budget_v1.py",
    "tests/test_ssd_benchmark_budget_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = previous.check_frozen


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    common.counts(events)
    if len(events) < 106 or events[105]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve the disclosed nine-study journal prefix.")
    data, parent, original, benchmarks = previous.inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 187,
        "prior_journal_sequence": 106,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction_inputs": "past_prices_and_benchmark_tail_budget_not_old_targets",
        "original_policy_role": "unchanged_funded_execution_only",
        "prior_evidence": parent["prior_evidence"]
        + [
            base.reference(previous.PACKET),
            base.reference(previous.PUBLIC_RESULTS),
            base.reference(previous.FOLDER / "saved-return-shortfall-v1.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "benchmark-budget-offline-tests-v1.xml",
                "benchmark-budget-offline-tests-v1.log",
            )
        ],
    }
    check_frozen(frozen)
    return data, frozen, original, benchmarks


def prove_targets(data, protocol: dict) -> tuple:
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, policy)
    if not audits:
        raise QuantError("The full historical calendar needs actual month-end decisions.")
    maximum_gap = [0.0, 0.0, 0.0]
    for record in audits:
        day = pd.Timestamp(record["decision_session"])
        observed = rules.observed_problem(data.close, day)
        independent = rules.independent_problem(data.close, day)
        for i, (a, b) in enumerate(zip(observed, independent, strict=True)):
            maximum_gap[i] = max(maximum_gap[i], float(np.max(np.abs(a - b))))
        if max(maximum_gap) > 1e-14:
            raise QuantError("Independent sleeve/score/benchmark scenarios disagree.")
        weights = np.array([record["sleeve_weights"][name] for name in rules.SLEEVES])
        rules.verify_plan(weights, record, *independent)
        if not np.allclose(
            targets[IDS[0]].loc[day], rules.prior.scatter(weights), rtol=0, atol=1e-12
        ):
            raise QuantError("Actual ETF targets differ from the certified portfolio.")
    target = targets[IDS[0]]
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        formal = target.loc[anchor : window["end"]]
        if anchor not in formal.index or formal.isna().any().any():
            raise QuantError("Complete initial anchors and formal windows are mandatory.")
        used = [name for name in rules.CORE if (formal[name] > 1e-10).any()]
        if used != list(rules.CORE):
            raise QuantError("Unallocated labels are not genuine factor participation.")
        coverage.append({"years": window["years"], "positive_target_factor_symbols": used})
    prefixes = []
    for position in sorted({252, 503, 1006, 1509, len(data.close) - 2}):
        if not 0 <= position < len(data.close):
            continue
        day = data.close.index[position]
        prefix, _ = rules.build_targets(previous.prefix_data(data, day), policy)
        if not np.allclose(prefix[IDS[0]], target.loc[:day], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Future observations changed an earlier risk-budget allocation.")
        prefixes.append(str(day.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "decision_count": len(audits),
            "independent_sleeve_return_maximum_difference": maximum_gap[0],
            "independent_score_maximum_difference": maximum_gap[1],
            "independent_benchmark_return_maximum_difference": maximum_gap[2],
            "all_primary_secondary_primal_dual_certificates_checked": True,
            "exact_fractional_tail_budget_and_opportunity_checked": True,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "decision_reason_counts": {
                reason: sum(row["reason"] == reason for row in audits)
                for reason in sorted({row["reason"] for row in audits})
            },
            "all_formal_initial_anchors_and_sessions_defined": True,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "new_policy_targets_constructed_from_admitted_prices": True,
            "full_sample_diagnostic_weights_used": False,
        },
    )


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
        or value["new_registered_candidates"] != len(IDS)
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
        raise QuantError("Preserve the actual pre-account benchmark-budget registration.")
    return value


def failure(stage: str, identifier: str | None, error: Exception) -> None:
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if (
        identifier is not None
        and not expired
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
        "benchmark_budget_expired_preserved" if expired else "benchmark_budget_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            events[0]["body"]["legacy"] != frozen["legacy"]
            or prior["new_registered_candidates"] != 9
            or prior["new_completed_strategy_evaluations"] != 9
        ):
            raise QuantError("Keep all nine completed predecessor hypotheses unchanged.")
        verify_publication(
            ROOT,
            base.reference(previous.registry.PUBLIC),
            previous.CATALOG_FILES,
            frozen["catalog"]["published_commit"],
        )
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered benchmark-budget version cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve terminal preparation failures; no same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_benchmark_budget",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(data, read_json(base.PROTOCOL))
                check_frozen(frozen)
                directory = TARGETS / identifier
                target_path = directory / "targets.csv"
                common.output_guard(target_path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if target_path.exists():
                    if file_digest(target_path) != digest_bytes(content.encode()):
                        raise QuantError("Conflicting partial targets must not be overwritten.")
                else:
                    write_text_atomic(target_path, content)
                common.retain_receipt(directory / "optimizer-audits.json", {"decisions": audits})
                proof["optimizer_audits"] = base.reference(directory / "optimizer-audits.json")
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
                        "targets": base.reference(target_path),
                        "proof": base.reference(directory / "proof.json"),
                    },
                )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("benchmark_budget_target_preparation", identifier, error)
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
                "benchmark_budget_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                benchmark_budget_registration=base.reference(PACKET),
                pending_steps=["publish_benchmark_budget_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    dependencies = [
        verify_publication(
            ROOT, base.reference(previous.registry.PUBLIC), previous.CATALOG_FILES, commit
        ),
        verify_publication(ROOT, base.reference(previous.PACKET), set(previous.EXTENSIONS), commit),
    ]
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
        "preserved_dependency_publications": dependencies,
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
                raise QuantError("Completed benchmark-budget review and account paths disagree.")
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
        data, frozen, original, benchmarks = inputs()
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
            "evaluating_benchmark_budget",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            benchmark_budget_publication=published,
        )
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                regenerated, _ = rules.build_targets(data, read_json(rules.POLICY))
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
                    raise QuantError("Registered targets differ from the fixed growth-first law.")
                paths = {}
                protocol, execution = read_json(base.PROTOCOL), read_json(base.POLICY)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_benchmark_budget",
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
                            execution,
                            benchmarks,
                        )
                check_frozen(frozen)
                base.legacy_snapshot(execution)
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
                "benchmark_budget_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                benchmark_budget_results=base.reference(RESULTS),
                pending_steps=["publish_all_benchmark_budget_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("benchmark_budget_evaluation_or_target_reproduction", identifier, error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "benchmark_budget_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only actual terminal benchmark-budget outcomes.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Unfinished evidence is not a completed research outcome.")
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
                        )
                    },
                    "years": year,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        symbol: lookup[year, scenario, symbol] for symbol in ("SPY", "QQQ")
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
        description="One fixed benchmark-tail-budget study; no trading."
    )
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
                raise QuantError("Provide the actually published registration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = common.counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Benchmark-budget research blocked: {error}\n")


if __name__ == "__main__":
    main()
