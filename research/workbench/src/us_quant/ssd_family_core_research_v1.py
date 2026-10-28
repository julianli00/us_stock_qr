from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import family_core_expansion_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_factor_catalog_v1 as registry
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
TARGETS = ROOT / "data/ssd-family-core-expansion-20261028-v1"
RESULTS = base.REPORTS / "family-core-expansion-20261028-v1/results.json"
PACKET = ROOT / "evidence/family_core_expansion_20261028_registration_v1.json"
PUBLIC_RESULTS = ROOT / "evidence/family_core_expansion_20261028_results_v1.json"
EXTENSIONS = (
    "config/family-core-expansion-v1.json",
    "src/us_quant/family_core_expansion_v1.py",
    "src/us_quant/ssd_family_core_research_v1.py",
    "tests/test_family_core_expansion_v1.py",
    "tests/test_ssd_family_core_research_v1.py",
)
CATALOG_FILES = {
    "config/ssd-factor-catalog-v1.json",
    "src/us_quant/ssd_factor_catalog_v1.py",
    "tests/test_ssd_factor_catalog_v1.py",
}
IDS = tuple(row["id"] for row in rules.CANDIDATES)


def used_factors(catalog: dict) -> list[dict]:
    records = {row["id"]: row for row in catalog["legacy"]["factors"]}
    for row in catalog["new_factors"]:
        factor = row["factor"]
        if factor["id"] in records:
            raise QuantError("An SSD definition cannot replace an original factor.")
        records[factor["id"]] = {
            "id": factor["id"],
            "family": factor["family"],
            "registered_at": row["registered_at"],
            "semantic_sha256": row["semantic_sha256"],
        }
    if not set(rules.FACTOR_IDS).issubset(records):
        raise QuantError("All six actually registered ETF families are required.")
    selected = [records[identifier] for identifier in rules.FACTOR_IDS]
    if (
        len({row["family"] for row in selected}) != 6
        or len({row["semantic_sha256"] for row in selected}) != 6
    ) or any(
        pd.Timestamp(row["registered_at"]).tzinfo is None
        or pd.Timestamp(row["registered_at"]) > pd.Timestamp(utc_now())
        for row in selected
    ):
        raise QuantError("Do not inflate the family count or invent registration clocks.")
    return selected


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    common.counts(events)
    if len(events) < 82 or events[81]["sha256"] != rules.PRIOR_JOURNAL_HEAD:
        raise QuantError("The disclosed seven-candidate history changed.")
    old_data, _, parent, original, benchmarks = common.inputs()
    parent_packet = common.checked_packet(parent)
    selected = [
        row for row in parent_packet["entries"] if row["candidate_id"] == rules.SOURCE_CANDIDATE
    ]
    if len(selected) != 1:
        raise QuantError("Use the single actual saved monthly source, not a reconstruction.")
    source_refs = {
        "packet": base.reference(common.PACKET),
        "targets": selected[0]["target_proof"]["targets"],
        "proof": selected[0]["target_proof"]["proof"],
    }
    if (
        source_refs["targets"]["sha256"] != rules.SOURCE_TARGET_SHA
        or source_refs["proof"]["sha256"] != rules.SOURCE_PROOF_SHA
    ):
        raise QuantError("The fixed saved monthly source or proof changed.")
    source = pd.read_csv(
        base.checked_reference(source_refs["targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.source_mask(source)
    data, catalog = registry.registered_catalog()
    admission = read_json(base.checked_reference(catalog["admission"]))
    publication_path = registry.FOLDER / "publication.json"
    catalog_publication = read_json(publication_path)
    if catalog_publication["published_commit"] != rules.CATALOG_COMMIT:
        raise QuantError("Bind the actual completed source/catalog publication.")
    implementation = read_json(base.checked_reference(catalog["intent"]))["implementation"]
    catalog_public = base.reference(registry.PUBLIC)
    if catalog_publication["catalog_files_verified"] != [*implementation, catalog_public]:
        raise QuantError("The source catalog's published inputs changed.")
    for reference in catalog_publication["catalog_files_verified"]:
        base.checked_reference(reference)
    if (
        not source.index.equals(data.close.index)
        or tuple(data.close.columns) != rules.UNIVERSE
        or not old_data.risk_free.equals(data.risk_free)
        or any(
            not getattr(old_data, name).equals(
                getattr(data, name).loc[:, list(old_data.close.columns)]
            )
            for name in registry.FIELDS
        )
    ):
        raise QuantError("Keep the full original market, RF and saved source calendar.")
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    factors = used_factors(catalog)
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"]
        + [base.reference(ROOT / name) for name in EXTENSIONS[1:]]
        + [reference for reference in implementation if reference["path"].endswith(".py")],
        "policies": parent["policies"]
        + [base.reference(rules.POLICY), base.reference(registry.POLICY)],
        "market": admission["market"],
        "source": {"candidate_id": rules.SOURCE_CANDIDATE, **source_refs},
        "prior_completed_configurations": 185,
        "prior_journal_sequence": 82,
        "prior_journal_head_sha256": rules.PRIOR_JOURNAL_HEAD,
        "catalog": {
            "private": base.reference(registry.CATALOG),
            "public": catalog_public,
            "admission": catalog["admission"],
            "publication": base.reference(publication_path),
            "published_commit": rules.CATALOG_COMMIT,
        },
        "used_actual_registered_factors": factors,
        "used_economic_factor_families": len({row["family"] for row in factors}),
        "legacy_used_factor_list_scope": "unchanged_original_catalog_snapshot_only",
        "original_policy_role": "unchanged_funded_execution_not_the_new_target_composition",
        "current_model_inputs": ["saved_monthly_targets", "admitted_fourteen_asset_prices"],
        "original_signal_or_monthly_commitment_recomputed": False,
        "new_economic_factor_definitions_in_this_study": 0,
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in ("family-core-offline-tests-v1.xml", "family-core-offline-tests-v1.log")
        ],
        "prior_evidence": parent["prior_evidence"]
        + list(source_refs.values())
        + [
            base.reference(registry.CATALOG),
            catalog_public,
            catalog["admission"],
            base.reference(publication_path),
            base.reference(ROOT / "evidence/option_exit_latch_20261027_results_v1.json"),
            base.reference(
                base.REPORTS / "option-exit-latch-20261027-v1/saved-risk-diagnostic-v1.json"
            ),
        ],
    }
    common.check_frozen(frozen)
    return data, source, frozen, original, benchmarks


def prove_targets(source: pd.DataFrame, protocol: dict) -> tuple[dict, dict]:
    policy = read_json(rules.POLICY)
    targets = rules.build_targets(source, policy)
    oracle = rules.independent_targets(source, policy)
    differences = {}
    for identifier, target in targets.items():
        if not np.allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("The independent source-SOXX budget oracle disagrees.")
        defined = target.dropna(how="all")
        if defined.empty:
            raise QuantError("The actual saved source has no complete warmup.")
        differences[identifier] = float(
            abs(defined - oracle[identifier].loc[defined.index]).to_numpy().max()
        )
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            formal = target.loc[anchor : window["end"]]
            if anchor not in formal.index or formal.isna().any().any():
                raise QuantError("Every full formal account needs its actual prior-session intent.")
    prefixes = []
    for position in (20, 252, 272, 503, 1006, 1509, 2012, len(source) - 2):
        if not 0 <= position < len(source):
            continue
        part = rules.build_targets(source.iloc[: position + 1], policy)
        if any(
            not target.equals(targets[identifier].iloc[: position + 1])
            for identifier, target in part.items()
        ):
            raise QuantError("Future source observations changed an earlier composition.")
        prefixes.append(str(source.index[position].date()))
    return targets, {
        "checked_at": utc_now(),
        "independent_sequential_oracle_all_rows": True,
        "maximum_weight_difference": differences,
        "prefix_truncations": prefixes,
        "all_formal_initial_anchors_and_sessions_defined": True,
        "source_warmup_and_non_core_targets_unchanged": True,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "saved_monthly_commitment_recomputed": False,
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
        raise QuantError("Keep the actual pre-account family-core registration packet.")
    return packet


def record_failure(stage: str, identifiers: tuple[str, ...], exc: Exception) -> None:
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for identifier in identifiers:
            if (
                base.matching_events("registered", identifier)
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
        "family_core_authorization_expired_preserved" if expired else "family_core_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(exc),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, source, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        previous = common.counts([event for event in events if event["candidate_id"] not in IDS])
        if (
            events[0]["body"]["legacy"] != frozen["legacy"]
            or previous["new_registered_candidates"] != 7
            or previous["new_completed_strategy_evaluations"] != 7
        ):
            raise QuantError("Preserve all seven actual completed predecessor hypotheses.")
        verify_publication(
            ROOT, base.reference(registry.PUBLIC), CATALOG_FILES, rules.CATALOG_COMMIT
        )
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", candidate["id"])
            if existing and (len(existing) != 1 or existing[0]["body"] != specification):
                raise QuantError("A registered family-core version cannot change.")
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Keep terminal preparation failures instead of replaying them.")
        pending = tuple(
            identifier for identifier in IDS if not base.matching_events("target_proof", identifier)
        )
        if pending:
            common.progress(
                "preparing_registered_family_core_targets",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, proof = prove_targets(source, read_json(base.PROTOCOL))
                common.check_frozen(frozen)
                for identifier in pending:
                    runtime_guard()
                    directory = TARGETS / identifier
                    target_path = directory / "targets.csv"
                    common.output_guard(target_path)
                    content = targets[identifier].to_csv(float_format="%.17g")
                    if target_path.exists():
                        if file_digest(target_path) != digest_bytes(content.encode()):
                            raise QuantError("Preserve conflicting partial family-core targets.")
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
                    "family_core_target_preparation",
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
                "family_core_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                family_core_registration=base.reference(PACKET),
                current_research="One composition-only family expansion registered; no account yet",
                pending_steps=[
                    "publish_exact_family_core_packet",
                    "four_path_family_core_accounts",
                ],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    dependencies = [
        verify_publication(ROOT, base.reference(common.PACKET), set(common.EXTENSIONS), commit),
        verify_publication(ROOT, base.reference(registry.PUBLIC), CATALOG_FILES, commit),
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
        "source_and_catalog_dependency_publications": dependencies,
    }
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
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
                raise QuantError("Actual family-core review and path identities disagree.")
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
        data, source, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        common.output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != candidate_states(frozen):
                raise QuantError("Completed evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        regenerated = rules.build_targets(source, read_json(rules.POLICY))
        protocol, execution_policy = read_json(base.PROTOCOL), read_json(base.POLICY)
        common.progress(
            "evaluating_registered_family_core",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            family_core_publication=published,
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
                    raise QuantError("Registered family-core targets differ from the frozen law.")
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        common.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_registered_family_core",
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
                "family_core_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                family_core_results=base.reference(RESULTS),
                pending_steps=["publish_all_family_core_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            record_failure("family_core_account_evaluation", (active,) if active else (), exc)
            raise
        except KeyboardInterrupt:
            common.progress(
                "family_core_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only the actual terminal family-core result.")
    saved, states = read_json(RESULTS), candidate_states(frozen)
    if saved["candidates"] != states or any(row["status"] == "pending" for row in states.values()):
        raise QuantError("Unfinished family-core evidence is not a completed outcome.")
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
    parser = argparse.ArgumentParser(description="One fixed family-core expansion; no trading.")
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
        parser.exit(2, f"Family-core research blocked: {exc}\n")


if __name__ == "__main__":
    main()
