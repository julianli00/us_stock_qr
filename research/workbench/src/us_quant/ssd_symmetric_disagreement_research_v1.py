from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as previous
from us_quant import symmetric_variance_disagreement_v1 as rules
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
FOLDER = base.REPORTS / "symmetric-variance-disagreement-20261028-v1"
TARGETS = ROOT / "data/ssd-symmetric-variance-disagreement-20261028-v1"
PACKET = ROOT / "evidence/symmetric_variance_disagreement_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/symmetric_variance_disagreement_20261028_results_v1.json"
EXTENSIONS = (
    "config/symmetric-variance-disagreement-v1.json",
    "src/us_quant/symmetric_variance_disagreement_v1.py",
    "src/us_quant/ssd_symmetric_disagreement_research_v1.py",
    "tests/test_symmetric_variance_disagreement_v1.py",
    "tests/test_ssd_symmetric_disagreement_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = previous.check_frozen


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 130 or events[129]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve the actual eleven-study history and all rejected paths.")
    data, source, options, parent, original, benchmarks = previous.inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 189,
        "prior_journal_sequence": 130,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "prior_evidence": parent["prior_evidence"]
        + [
            base.reference(previous.PACKET),
            base.reference(previous.PUBLIC_RESULTS),
            base.reference(previous.FOLDER / "saved-full-window-diagnostic-v1.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in (
                "symmetric-disagreement-offline-tests-v1.xml",
                "symmetric-disagreement-offline-tests-v1.log",
            )
        ],
        "construction": "symmetric_SPY_GLD_daily_disagreement_not_paper_replication",
    }
    check_frozen(frozen)
    return data, source, options, frozen, original, benchmarks


def prove_targets(data, source, options, protocol) -> tuple:
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, source, options, policy)
    target = targets[IDS[0]]
    if len(audits) != 131 or not target.dropna().index.equals(source.dropna().index):
        raise QuantError("Keep all original 131 source decision dates.")
    maximum = 0.0
    for day in options.index:
        oracle = rules.independent_row(data.close, source.loc[day], options.loc[day], day)
        maximum = max(maximum, float(np.max(np.abs(target.loc[day] - oracle))))
        if maximum > 1e-12:
            raise QuantError("Independent symmetric ratio allocation disagrees.")
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if anchor not in known.index or known.isna().any().any():
            raise QuantError("The full formal window lacks its actual initial source seed.")
        if any(not (known[name] > 1e-10).any() for name in rules.CORE):
            raise QuantError("Unused factor labels are not actual participation.")
        coverage.append({"years": window["years"], "factor_symbols": list(rules.CORE)})
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
        early, _ = rules.build_targets(sliced, source.loc[:end], options.loc[:end], policy)
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError("Future source observations changed a prior symmetric allocation.")
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
            "true_variance_risk_premia_or_paper_replication_claimed": False,
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
        raise QuantError("Preserve the actual pre-account symmetric-proxy registration.")
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
        "symmetric_proxy_expired_preserved" if expired else "symmetric_proxy_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, source, options, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 11
            or prior["new_completed_strategy_evaluations"] != 11
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Keep all eleven actual completed predecessors.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered symmetric-proxy version cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve terminal failed preparation, not same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_symmetric_proxy",
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
                        raise QuantError("Conflicting partial symmetric targets must be preserved.")
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
                failure("symmetric_proxy_target_preparation", error)
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
                "symmetric_proxy_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                symmetric_proxy_registration=base.reference(PACKET),
                pending_steps=["publish_symmetric_proxy_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    original_receipt = base.published_inputs(original, commit)
    dependencies = [
        verify_publication(
            ROOT,
            frozen["source_registration"],
            {
                "config/implied-risk-execution.json",
                "src/us_quant/implied_risk_execution.py",
                "config/observed-weight-execution.json",
                "src/us_quant/observed_weight_execution.py",
            },
            commit,
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
            raise QuantError("Resume this study's actual original publication commit.")
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
                raise QuantError("Completed symmetric-proxy review and path evidence disagree.")
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
            "evaluating_symmetric_proxy",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            symmetric_proxy_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                regenerated, _ = rules.build_targets(data, source, options, read_json(rules.POLICY))
                ref = base.matching_events("target_proof", identifier)[0]["body"]["targets"]
                targets = previous.frame(ref)
                if (
                    not targets.index.equals(data.close.index)
                    or not targets.columns.equals(data.close.columns)
                    or not np.allclose(
                        targets, regenerated[identifier], rtol=0, atol=1e-12, equal_nan=True
                    )
                ):
                    raise QuantError("Registered symmetric targets differ from the fixed law.")
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_symmetric_proxy",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = previous.evaluate_path(
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
                "symmetric_proxy_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                symmetric_proxy_results=base.reference(RESULTS),
                pending_steps=["publish_all_symmetric_proxy_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("symmetric_proxy_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "symmetric_proxy_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal symmetric-proxy outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Unfinished evidence cannot be reported as completed.")
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
            "true_variance_risk_premia_or_paper_replication_claimed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return base.reference(PUBLIC_RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="One symmetric daily-risk proxy; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The frozen-input study cannot request live data.")

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
        parser.exit(2, f"Symmetric proxy research blocked: {error}\n")


if __name__ == "__main__":
    main()
