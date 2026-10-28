from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import joint_tail_construction_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_factor_catalog_v1 as registry
from us_quant import ssd_research_v1 as base
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.ssd_family_core_research_v1 import used_factors
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_text_atomic,
)

ROOT = base.ROOT
FOLDER = base.REPORTS / "joint-tail-construction-20261028-v1"
TARGETS = ROOT / "data/ssd-joint-tail-construction-20261028-v1"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/joint_tail_construction_20261028_registration_v1.json"
PUBLIC_RESULTS = ROOT / "evidence/joint_tail_construction_20261028_results_v1.json"
EXTENSIONS = (
    "config/joint-tail-construction-v1.json",
    "src/us_quant/joint_tail_construction_v1.py",
    "src/us_quant/ssd_joint_tail_research_v1.py",
    "tests/test_joint_tail_construction_v1.py",
    "tests/test_ssd_joint_tail_research_v1.py",
)
CATALOG_FILES = {
    "config/ssd-factor-catalog-v1.json",
    "src/us_quant/ssd_factor_catalog_v1.py",
    "tests/test_ssd_factor_catalog_v1.py",
}
IDS = tuple(row["id"] for row in rules.CANDIDATES)


def check_frozen(frozen: dict) -> None:
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(ref)


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    common.counts(events)
    if len(events) < 94 or events[93]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("The disclosed eight-candidate history changed.")
    old, _, parent, benchmarks = base.study_inputs()
    data, catalog = registry.registered_catalog()
    admission = read_json(base.checked_reference(catalog["admission"]))
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    if (
        tuple(data.close.columns) != rules.UNIVERSE
        or not old.risk_free.equals(data.risk_free)
        or any(
            not getattr(old, key).equals(getattr(data, key).loc[:, old.close.columns])
            for key in registry.FIELDS
        )
    ):
        raise QuantError("Preserve the original fourteen-asset admission and exact prior data/RF.")
    factors = used_factors(catalog)
    if [row["id"] for row in factors] != list(rules.FACTOR_IDS):
        raise QuantError("Use exactly the six source-supported registered economic families.")
    code_names = [
        *EXTENSIONS[1:],
        "src/us_quant/ssd_factor_catalog_v1.py",
        "src/us_quant/ssd_commitment_research_v1.py",
        "src/us_quant/commitment_duration_v1.py",
        "src/us_quant/ssd_family_core_research_v1.py",
        "src/us_quant/family_core_expansion_v1.py",
    ]
    source_folder = base.REPORTS / "family-core-expansion-20261028-v1"
    prior_files = (
        registry.CATALOG,
        registry.PUBLIC,
        registry.ADMISSION,
        registry.FOLDER / "publication.json",
        source_folder / "saved-concentration-risk-v1.json",
        source_folder / "construction-prior-art-v1.json",
        FOLDER / "prior-art-supplement-v1.json",
        ROOT / "evidence/family_core_expansion_20261028_results_v1.json",
    )
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in code_names],
        "policies": parent["policies"]
        + [base.reference(rules.POLICY), base.reference(registry.POLICY)],
        "market": admission["market"],
        "catalog": {
            "private": base.reference(registry.CATALOG),
            "public": base.reference(registry.PUBLIC),
            "admission": catalog["admission"],
            "published_commit": policy["catalog_publication_commit"],
        },
        "prior_completed_configurations": 186,
        "prior_journal_sequence": 94,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "used_actual_registered_factors": factors,
        "used_economic_factor_families": 6,
        "legacy_used_factor_list_scope": "unchanged_original_catalog_snapshot_only",
        "original_policy_role": "unchanged_funded_execution_only",
        "construction_inputs": "past_prices_not_old_targets_or_full_sample_risk",
        "new_economic_factor_definitions": 0,
        "prior_evidence": [base.reference(path) for path in prior_files],
        "validation_evidence": [
            base.reference(base.REPORTS / name)
            for name in ("joint-tail-offline-tests-v1.xml", "joint-tail-offline-tests-v1.log")
        ],
        "account_namespace": base.ACCOUNTS.relative_to(ROOT).as_posix(),
        "account_namespace_reuses_execution_not_original_candidate_ids": True,
    }
    check_frozen(frozen)
    return data, frozen, parent, benchmarks


def prefix_data(data: MarketData, end: pd.Timestamp) -> MarketData:
    return MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )


def prove_targets(data: MarketData, protocol: dict) -> tuple:
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, policy)
    if not audits:
        raise QuantError("The full admitted history produced no completed month-end decisions.")
    matrix_gap = score_gap = hurdle_gap = 0.0
    for record in audits:
        day = pd.Timestamp(record["decision_session"])
        independent = rules.independent_problem(data.close, day)
        original = rules.observed_problem(data.close, day)
        matrix_gap = max(matrix_gap, float(np.max(np.abs(independent[0] - original[0]))))
        score_gap = max(score_gap, float(np.max(np.abs(independent[1] - original[1]))))
        hurdle_gap = max(hurdle_gap, abs(independent[2] - original[2]))
        if max(matrix_gap, score_gap, hurdle_gap) > 1e-14:
            raise QuantError("Independent price-scenario/opportunity reconstruction disagrees.")
        weights = np.array([record["sleeve_weights"][name] for name in rules.SLEEVES])
        rules.verify_plan(weights, record, *independent)
        if not np.allclose(targets[IDS[0]].loc[day], rules.scatter(weights), rtol=0, atol=1e-12):
            raise QuantError("Actual ETF targets differ from the certified joint-tail sleeve plan.")
    target = targets[IDS[0]]
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        formal = target.loc[anchor : window["end"]]
        if anchor not in formal.index or formal.isna().any().any():
            raise QuantError("Both full formal windows require an actual known prior-close target.")
        used = [name for name in rules.CORE if (formal[name] > 1e-10).any()]
        if used != list(rules.CORE):
            raise QuantError("Do not qualify an all-standby rule by unused factor labels.")
        coverage.append({"years": window["years"], "positive_target_factor_symbols": used})
    prefixes = []
    cut_positions = sorted({252, 503, 1006, 1509, len(data.close) - 2})
    for position in cut_positions:
        if not 0 <= position < len(data.close):
            continue
        day = data.close.index[position]
        prior, _ = rules.build_targets(prefix_data(data, day), policy)
        if not np.allclose(prior[IDS[0]], target.loc[:day], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Later observations changed an earlier optimizer target or month end.")
        prefixes.append(str(day.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "independent_scalar_market_problem_maximum_return_difference": matrix_gap,
            "independent_scalar_market_problem_maximum_score_difference": score_gap,
            "independent_scalar_market_problem_maximum_hurdle_difference": hurdle_gap,
            "all_primary_and_secondary_primal_dual_certificates_checked": True,
            "exact_fractional_tail_loss_independently_checked": True,
            "decision_count": len(audits),
            "standby_reason_counts": {
                reason: sum(row["reason"] == reason for row in audits)
                for reason in sorted({row["reason"] for row in audits})
            },
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "all_formal_initial_anchors_and_sessions_defined": True,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "new_policy_targets_constructed_from_admitted_prices": True,
            "full_sample_diagnostic_weights_used": False,
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
        raise QuantError("Keep the actual pre-account joint-tail registration packet.")
    return packet


def failure(stage: str, identifier: str | None, exc: Exception) -> None:
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
                "reason": str(exc),
                "partial_outputs_retained": True,
                "candidate_accounts_started": bool(
                    base.matching_events("path_started", identifier)
                ),
            },
        )
    common.progress(
        "joint_tail_authorization_expired_preserved" if expired else "joint_tail_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(exc),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            events[0]["body"]["legacy"] != frozen["legacy"]
            or prior["new_registered_candidates"] != 8
            or prior["new_completed_strategy_evaluations"] != 8
        ):
            raise QuantError("Preserve all eight completed predecessor hypotheses.")
        verify_publication(
            ROOT,
            base.reference(registry.PUBLIC),
            CATALOG_FILES,
            frozen["catalog"]["published_commit"],
        )
        candidate = rules.CANDIDATES[0]
        identifier = candidate["id"]
        specification = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != specification):
            raise QuantError("A registered joint-tail version cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, specification)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve a terminal preparation failure; do not rerun that version.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_joint_tail_targets",
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
                        raise QuantError("Preserve conflicting partial joint-tail targets.")
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
            except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
                failure("joint_tail_target_preparation", identifier, exc)
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
                "joint_tail_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                joint_tail_registration=base.reference(PACKET),
                current_research="Joint-tail/opportunity registered; no account yet",
                pending_steps=["publish_joint_tail_registration", "four_joint_tail_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    catalog_receipt = verify_publication(
        ROOT, base.reference(registry.PUBLIC), CATALOG_FILES, commit
    )
    own = verify_publication(ROOT, base.reference(PACKET), set(EXTENSIONS), commit)
    existing = [
        row
        for row in base.matching_events("publication")
        if row["body"].get("study_id") == frozen["study_id"]
    ]
    if existing:
        if len(existing) != 1 or existing[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume this study's actual original publication commit.")
        return existing[0]["body"]["publication"]
    receipt = {
        **own,
        "original_dependency_publication": original_receipt,
        "catalog_dependency_publication": catalog_receipt,
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
                raise QuantError("Actual joint-tail review and path evidence disagree.")
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
            "evaluating_registered_joint_tail",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            joint_tail_publication=published,
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
                    raise QuantError("Registered joint-tail targets differ from the fixed policy.")
                paths = {}
                protocol, execution = read_json(base.PROTOCOL), read_json(base.POLICY)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        common.progress(
                            "evaluating_registered_joint_tail",
                            active_candidate=identifier,
                            active_path=f"{window['years']}y-{scenario['id']}",
                        )
                        paths[f"{window['years']}y-{scenario['id']}"] = base.evaluate_path(
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
                "joint_tail_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                joint_tail_results=base.reference(RESULTS),
                pending_steps=["publish_all_joint_tail_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            failure("joint_tail_evaluation_or_target_reproduction", identifier, exc)
            raise
        except KeyboardInterrupt:
            common.progress(
                "joint_tail_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal joint-tail outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Unfinished joint-tail evidence is not a completed outcome.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for identifier, row in actual.items():
        paths = []
        for key, item in row["paths"].items():
            years, scenario = key.split("-")
            years = int(years[:-1])
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
    parser = argparse.ArgumentParser(description="One fixed joint-tail construction; no trading.")
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
        parser.exit(2, f"Joint-tail research blocked: {exc}\n")


if __name__ == "__main__":
    main()
