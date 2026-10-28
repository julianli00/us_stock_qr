from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from us_quant import ordered_path_growth_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as source
from us_quant.calendar import is_month_end, previous_session
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
FOLDER = base.REPORTS / "ordered-path-growth-20261028-v1"
TARGETS = ROOT / "data/ssd-ordered-path-growth-20261028-v1"
PACKET = ROOT / "evidence/ordered_path_growth_20261028_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/ordered_path_growth_20261028_results_v1.json"
EXTENSIONS = (
    "config/ordered-path-growth-v1.json",
    "src/us_quant/ordered_path_growth_v1.py",
    "src/us_quant/ssd_ordered_path_research_v1.py",
    "tests/test_ordered_path_growth_v1.py",
    "tests/test_ssd_ordered_path_research_v1.py",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def inputs() -> tuple:
    events = base.journal_events(base.JOURNAL)
    if len(events) < 202 or events[201]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all seventeen completed predecessors.")
    data, _, _, parent, original, benchmarks = source.inputs()
    source.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    eligible = [
        day for i, day in enumerate(data.close.index) if i >= rules.LOOKBACK and is_month_end(day)
    ]
    if len(data.close) != 2805 or len(eligible) != 122 or str(eligible[0].date()) != "2016-08-31":
        raise QuantError("Keep the full original market calendar and declared warmup.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "prior_completed_configurations": 195,
        "prior_journal_sequence": 202,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "monthly_executor_registration": base.reference(source.PACKET),
        "construction": "ordered_peak_trough_LP_with_net_monthly_execution",
        "allocation_inputs": "admitted_adjusted_close_prices_only",
        "training_lookback_return_sessions": 252,
        "training_price_observations": 253,
        "optimization_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
            "solver": "highs-ds",
        },
        "inherited_macro_dependencies_used_for_allocation": False,
        "old_dollar_ANFCI_or_option_allocation_rules_used": False,
        "prior_evidence": parent["prior_evidence"]
        + [
            base.reference(source.PACKET),
            base.reference(source.PUBLIC_RESULTS),
            base.reference(ROOT / "config/joint-tail-construction-v1.json"),
            base.reference(ROOT / "config/benchmark-tail-budget-v1.json"),
            base.reference(ROOT / "src/us_quant/joint_tail_construction_v1.py"),
            base.reference(ROOT / "src/us_quant/benchmark_tail_budget_v1.py"),
            base.reference(ROOT / "evidence/simplex_wealth_average_20261028_results_v1.json"),
            base.reference(ROOT / "evidence/split_session_variation_20261028_results_v1.json"),
            base.reference(ROOT / "evidence/anfci_residual_confirmation_20261028_results_v1.json"),
            base.reference(
                base.REPORTS
                / "anfci-residual-confirmation-20261028-v1/saved-guard-assessment-v2.json"
            ),
            base.reference(FOLDER / "research-intent.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in ("ordered-path-offline-tests-v1.xml", "ordered-path-offline-tests-v1.log")
        ],
        "training_statistics_are_formal_strategy_results": False,
        "training_risk_guarantees_future_net_drawdown": False,
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, frozen, original, benchmarks


def prove_targets(data: MarketData, protocol: dict):
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, policy)
    target = targets[IDS[0]]
    expected = [
        day for i, day in enumerate(data.close.index) if i >= rules.LOOKBACK and is_month_end(day)
    ]
    if (
        len(audits) != 122
        or not target.dropna().index.equals(pd.DatetimeIndex(expected))
        or not target.iloc[: rules.LOOKBACK].isna().all().all()
    ):
        raise QuantError("The fixed trailing window and genuine monthly decisions changed.")
    maximum, risk_maximum, certificates = 0.0, 0.0, []
    for row in audits:
        day = pd.Timestamp(row["decision_session"])
        paths = rules.independent_paths(data.close, day)
        weights = np.array([row["sleeve_weights_of_total_capital"][name] for name in rules.SLEEVES])
        rules.verify_plan(weights, row, paths)
        difference = float(np.max(abs(target.loc[day] - rules.scatter(weights))))
        maximum = max(maximum, difference)
        risk_maximum = max(risk_maximum, rules.drawdown(paths, weights)[0])
        certificates.extend([row["primary_certificate"], row["secondary_certificate"]])
    if maximum > 1e-12:
        raise QuantError("Actual ETF targets differ from the independently certified sleeve plan.")
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if anchor not in known.index or known.isna().any().any():
            raise QuantError("Both original formal anchors and all return dates are required.")
        if any(not (known[name] > 1e-10).all() for name in rules.CORE):
            raise QuantError(
                "All four actual factor ETFs must have positive capital participation."
            )
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
        early, _ = rules.build_targets(sliced, policy)
        if not np.allclose(early[IDS[0]], target.loc[:end], rtol=0, atol=1e-12, equal_nan=True):
            raise QuantError(
                "Future price observations changed an earlier path-constrained target."
            )
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "monthly_decisions": len(audits),
            "first_eligible_decision": audits[0]["decision_session"],
            "training_return_observations": 252,
            "training_price_observations": 253,
            "ordered_peak_trough_constraints_per_decision": 31878,
            "maximum_independent_target_difference": maximum,
            "maximum_observed_training_gross_drawdown": risk_maximum,
            "maximum_primal_dual_gap": max(row["primal_dual_gap"] for row in certificates),
            "maximum_certificate_residual": max(
                value
                for row in certificates
                for key, value in row.items()
                if key not in {"objective", "dual_objective"}
            ),
            "training_statistics_are_formal_strategy_results": False,
            "future_or_net_drawdown_guaranteed": False,
            "old_targets_or_accounts_replayed": False,
            "original_source_signal_recomputed": False,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "candidate_accounts_or_metrics_computed": False,
        },
    )


def checked_packet(frozen):
    common.output_guard(PACKET)
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
        raise QuantError("Keep the actual pre-account ordered-path registration.")
    return packet


def failure(stage, error):
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
        "ordered_path_expired_preserved" if expired else "ordered_path_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 17
            or prior["new_completed_strategy_evaluations"] != 17
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve all seventeen actual completed predecessors.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered ordered-path version cannot change.")
        if not existing:
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve the terminal failed version without same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_ordered_path",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(data, read_json(base.PROTOCOL))
                check_frozen(frozen)
                directory, content = (
                    TARGETS / identifier,
                    targets[identifier].to_csv(float_format="%.17g"),
                )
                path = directory / "targets.csv"
                common.output_guard(path)
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Preserve conflicting partial ordered-path targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "ordered-path-audits.json", {"decisions": audits})
                proof["ordered_path_audits"] = base.reference(
                    directory / "ordered-path-audits.json"
                )
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
                failure("ordered_path_preparation", error)
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
                "ordered_path_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                ordered_path_registration=base.reference(PACKET),
                pending_steps=["publish_ordered_path_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen, original, commit):
    inherited = base.published_inputs(original, commit)
    prior = verify_publication(
        ROOT, frozen["monthly_executor_registration"], set(source.EXTENSIONS), commit
    )
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
        "monthly_executor_dependency_publication": prior,
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
                raise QuantError("The completed ordered-path review and actual paths disagree.")
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
        data, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Completed evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        common.progress(
            "evaluating_ordered_path",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            ordered_path_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                target_event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(target_event["targets"])
                proof = read_json(base.checked_reference(target_event["proof"]))
                audits = read_json(base.checked_reference(proof["ordered_path_audits"]))[
                    "decisions"
                ]
                expected = [
                    day
                    for i, day in enumerate(data.close.index)
                    if i >= rules.LOOKBACK and is_month_end(day)
                ]
                if len(audits) != 122 or not targets.dropna().index.equals(
                    pd.DatetimeIndex(expected)
                ):
                    raise QuantError("The actual registered decision cohort changed.")
                for row in audits:
                    day = pd.Timestamp(row["decision_session"])
                    weights = np.array(
                        [row["sleeve_weights_of_total_capital"][name] for name in rules.SLEEVES]
                    )
                    rules.verify_plan(weights, row, rules.independent_paths(data.close, day))
                    if not np.allclose(
                        targets.loc[day], rules.scatter(weights), rtol=0, atol=1e-12
                    ):
                        raise QuantError(
                            "Registered targets differ from retained certified decisions."
                        )
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_ordered_path", active_candidate=identifier, active_path=key
                        )
                        paths[key] = source.evaluate_path(
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
                        "training_risk_is_funded_net_or_future_guarantee": False,
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
                "ordered_path_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                ordered_path_results=base.reference(RESULTS),
                pending_steps=["publish_all_ordered_path_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("ordered_path_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "ordered_path_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result():
    base.storage_guard()
    _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal ordered-path outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("An unfinished study is not complete.")
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
            "training_risk_is_funded_net_or_future_guarantee": False,
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
        description="One ordered-path construction with original net gates."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Frozen-price ordered-path research cannot request live sources.")

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
        parser.exit(2, f"Ordered-path research blocked: {error}\n")


if __name__ == "__main__":
    main()
