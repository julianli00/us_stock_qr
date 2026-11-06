from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import execution_gap_shield_v1 as rules
from us_quant import ssd_execution_gap_envelope_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import review_market
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, lifecycle = prior.base, prior.ROOT, prior.lifecycle
reference, unique_refs, counts = prior.reference, prior.unique_refs, prior.counts
ID, KIND = rules.ID, "causal_main_capital_decision_program_v1"
CANDIDATE = {"id": ID, "weighting": "cached_Prod_with_daily_delay_aware_main_shield"}
FOLDER = base.REPORTS / "execution-gap-shield-20261106-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/execution_gap_shield_20261106_registration_v1.json"
PUBLIC = ROOT / "evidence/execution_gap_shield_20261106_results_v1.json"
OWN = (
    "config/execution-gap-shield-v1.json",
    "src/us_quant/execution_gap_shield_v1.py",
    "src/us_quant/ssd_execution_gap_shield_v1.py",
    "tests/test_execution_gap_shield_v1.py",
    "EXECUTION_GAP_SHIELD_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def events(kind):
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == kind and row["candidate_id"] == ID
    ]


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "A new shield cannot override authoritative counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "execution_gap_shield_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    source_event = read_json(base.checked_reference(catalogue["source_registered_event"]))
    rules.require(
        source_event["kind"] == "registered"
        and source_event["candidate_id"] == "expert_pool_six_confidence_rated_prod_v1"
        and catalogue["new_shield_targets_accounts_or_real_scenarios_computed"] is False
        and catalogue["old_learner_source_target_or_account_replayed"] is False
        and closed["actual_authoritative_counts"]["current_journal_event_count"] == 1490,
        "Keep the exact admitted source stream and the original closed J1490 history.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, source_event["body"]["frozen"], refs


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(ref)


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    suites = ElementTree.parse(base.checked_reference(proof["test_outputs"][0])).getroot()
    suites = suites.findall(".//testsuite")
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")),
        "Freeze only a successful exact-source synthetic validation.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_real_dynamic_packet_entry_defers_historical_outputs",
            "test_pending_request_never_replaced",
            "test_actual_account_prefix_is_unchanged",
            "test_inside_binary64_not_rounded_above_upper",
        }.issubset(names),
        "Exercise the real packet Interface, chronology and funding seams.",
    )
    return [*proof["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve a complete or partial first shield freeze.",
        )
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1490
            and history[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(e["candidate_id"] == ID for e in history),
            "Freeze one unused identity without altering J1490.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, parent, refs = source_inputs()
        validation = validation_refs()
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": (
                "daily_actual_main_delay_reference_and_explicit_band_override"
            ),
            "construction": "cached_six_Prod_nominal_with_actual_main_delay_aware_shield",
            "decision_program_kind": KIND,
            "prior_journal_sequence": 1490,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 303,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs([*parent["validation_evidence"], *validation]),
            "finite_reference_is_future_or_full_month_guarantee": False,
            "historical_new_shield_outputs_before_publication": False,
            "old_learner_source_target_or_account_replayed": False,
            "old_headroom_V1_repaired_retried_or_executed": False,
            "source_expert_pool_or_admission_changed": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_shield_targets_accounts_or_real_scenarios_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_dynamic_program_before_real_outputs")
    return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(ref)
    rules.require(
        sys.version == intent["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == intent["package_versions"],
        "Preserve the accepted frozen interpreter and exact package set.",
    )
    catalogue, _, refs = source_inputs()
    rules.require(refs == intent["source_references"], "Keep every exact source reference.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": catalogue["source_market"]}, ROOT)
    nominal = pd.read_csv(
        base.checked_reference(catalogue["source_nominal_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        nominal.index.equals(data.close.index) and nominal.columns.equals(data.close.columns),
        "No cropped, filled or relabeled source target lattice.",
    )
    rules.ArchivedOpinionPolicy(nominal)
    benchmark_ref = [
        ref
        for ref in catalogue["source_benchmark_artifacts"]
        if ref["path"].endswith("/results.json")
    ]
    rules.require(len(benchmark_ref) == 1, "Use the original unique completed benchmark set.")
    benchmarks = read_json(base.checked_reference(benchmark_ref[0]))
    rules.require(
        len(benchmarks["benchmarks"]) == 8,
        "Keep both full-capital tradable benchmarks on all four original paths.",
    )
    return data, nominal, frozen, benchmarks


def packet_body(frozen, registered, readiness):
    return {
        "schema_version": 3,
        "program_id": "ssd_continuous_research_v1",
        "study_id": rules.STUDY,
        "target_delivery_kind": KIND,
        "complete_fixed_cohort_ids": [ID],
        "cohort_registrations": [registered["sha256"]],
        "entries": [
            {
                "candidate_id": ID,
                "specification": {
                    "financial_criteria_id": "ssd_dual_benchmark_net_cagr_v1",
                    "decision_program_kind": KIND,
                    "frozen": frozen,
                },
                "registered_event_sha256": registered["sha256"],
                "registered_at": registered["at"],
                "program_readiness": readiness,
                "historical_target_derivation_proof": None,
            }
        ],
        "historical_new_meta_scores_or_targets_computed": False,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "historical_target_proof_deferred_until_published_execution": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def checked_packet(frozen):
    registered = events("registered")
    rules.require(
        len(registered) == 1
        and registered[0]["body"] == {"candidate": CANDIDATE, "frozen": frozen},
        "Require the unique actual shield program registration.",
    )
    body = read_json(PACKET)
    expected = packet_body(frozen, registered[0], reference(READY))
    rules.require(
        {k: v for k, v in body.items() if k != "created_at"} == expected,
        "The real public packet cannot invent an already completed target proof.",
    )
    ready = read_json(READY)
    rules.require(
        ready["historical_target_derivation_verified"] is False
        and ready["historical_new_meta_scores_targets_accounts_computed"] is False,
        "Synthetic readiness is not historical target or account evidence.",
    )
    return body


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        _, _, frozen, _ = inputs()
        rules.require(
            not events("registered") and not events("technical_failure") and not PACKET.exists(),
            "Recover a retained registration; do not register twice.",
        )
        registered = base.append_event(
            base.JOURNAL,
            "registered",
            ID,
            {"candidate": CANDIDATE, "frozen": frozen},
        )
        write_new_receipt(
            READY,
            {
                "checked_at": utc_now(),
                "readiness_kind": "synthetic_Interface_and_archived_input_identity_only",
                "implementation": reference(INTENT),
                "validation": validation_refs(),
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            },
        )
        write_new_receipt(
            PACKET,
            {
                **packet_body(frozen, registered, reference(READY)),
                "created_at": utc_now(),
            },
        )
        checked_packet(frozen)
        progress("registered_before_outputs", next_step="publish_dynamic_program_before_funding")
    return reference(PACKET)


def publication(frozen, commit):
    checked_packet(frozen)
    checker = FunctionType(
        base.published_inputs.__code__,
        {**base.published_inputs.__globals__, "PACKET": PACKET},
        argdefs=base.published_inputs.__defaults__,
    )
    return checker(frozen, commit)


def write_run(directory, output):
    frames = {
        "account.csv": output.result.frame,
        "weights.csv": output.result.weights,
        "desired.csv": output.desired,
        "issued.csv": output.issued,
    }
    for name, frame in frames.items():
        write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "decisions.json",
        {
            "decisions": output.decisions,
            "pending_after_window": output.pending_after_window,
        },
    )
    return [reference(directory / name) for name in (*frames, "decisions.json")]


def evaluate_path(data, nominal, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    complete = [e for e in events("path_completed") if e["body"]["path_key"] == key]
    rules.require(
        not complete and not any(e["body"]["path_key"] == key for e in events("path_started")),
        "A complete or partial actual account must never be silently replayed.",
    )
    directory = FOLDER / "accounts" / key
    base.append_event(
        base.JOURNAL,
        "path_started",
        ID,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    new_output_directory(directory)
    matrix_folder = directory / "reference-inputs"
    new_output_directory(matrix_folder)
    trace = directory / "chronological-trace.jsonl"
    counter = 0
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        def retain_inputs(state, target, sample):
            nonlocal counter
            path = matrix_folder / f"{counter:06d}.npz"
            np.savez_compressed(
                path,
                starts=sample.starts.asi8,
                ends=sample.ends.asi8,
                waiting=sample.waiting,
                opening=sample.opening,
                fill_mark=sample.fill_mark,
            )
            retain(
                {
                    "kind": "reference_input",
                    "as_of": str(state.as_of.date()),
                    "scenario_operands": reference(path),
                    "actual_delay": sample.delay,
                    "main_asset_values": state.asset_values.tolist(),
                    "cash": state.cash,
                    "nav": state.nav,
                    "peak": state.peak,
                    "nominal": target.tolist(),
                }
            )
            counter += 1

        output = rules.run(
            data,
            nominal,
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            before_decision=lifecycle.runtime_guard,
            retain=retain,
            retain_inputs=retain_inputs,
        )
    feedback = rules.verify_run(
        data,
        output,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    independent = base.independent_equity(
        data,
        output.issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        output.result.frame,
        independent,
        data.risk_free.loc[output.result.frame.index],
        10000,
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], symbol] for symbol in ("SPY", "QQQ"))
    gates = base.gates(own, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    artifacts = write_run(directory, output)
    write_text_atomic(directory / "independent-bt.csv", independent.to_csv(float_format="%.17g"))
    artifacts += [reference(trace), reference(directory / "independent-bt.csv")]
    artifacts += [reference(path) for path in sorted(matrix_folder.glob("*.npz"))]
    assessments = [
        r["assessment"] for r in output.decisions if "reference_status" in r["assessment"]
    ]
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "decision_program_kind": KIND,
        "metrics": own,
        "execution": base.execution_statistics(
            output.result,
            data,
            10000,
            scenario["cost_bps_per_side"],
            1,
        ),
        "gates": gates,
        "all_point_gates_pass": all(gates.values()),
        "excess_net_cagr_vs_spy": own["cagr"] - spy["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"] - qqq["cagr"],
        "independent_bt_max_equity_difference_usd": difference,
        "actual_capital_queue_and_accounting_verification": feedback,
        "decision_reason_counts": dict(Counter(r["reason"] for r in output.decisions)),
        "finite_reference_status_counts": dict(Counter(a["reference_status"] for a in assessments)),
        "retained_reference_inputs": counter,
        "maximum_independent_scalar_money_error_USD": max(
            a["maximum_scalar_money_error_USD"] for a in assessments
        ),
        "artifacts": artifacts,
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        ID,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def prefix_audits(data, nominal):
    folder = FOLDER / "capital-prefix-audits-v1"
    new_output_directory(folder)
    source = FOLDER / "accounts/10y-base"
    saved = {
        name: pd.read_csv(
            source / name,
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    decisions = read_json(source / "decisions.json")["decisions"]
    result = []
    for position in read_json(rules.POLICY)["prefix_positions"]:
        lifecycle.runtime_guard()
        day = saved["account.csv"].index[position]
        clipped = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        output = rules.run(
            clipped,
            nominal.loc[:day],
            "2016-10-06",
            str(day.date()),
            cost_bps=5,
            delay=1,
            before_decision=lifecycle.runtime_guard,
        )
        for name, frame in (
            ("account.csv", output.result.frame),
            ("weights.csv", output.result.weights),
            ("desired.csv", output.desired),
            ("issued.csv", output.issued),
        ):
            expected, actual = saved[name].loc[:day], frame.loc[:day]
            rules.require(
                expected.index.equals(actual.index)
                and expected.columns.equals(actual.columns)
                and np.array_equal(
                    expected.to_numpy(dtype=float),
                    actual.to_numpy(dtype=float),
                    equal_nan=True,
                ),
                "The full new-policy account/target prefix must remain exactly unchanged.",
            )
        expected_decisions = [r for r in decisions if pd.Timestamp(r["signal_session"]) <= day]
        rules.require(
            output.decisions == expected_decisions,
            "A truncated new-policy run changed an earlier capital or reference decision.",
        )
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        result.append(
            {
                "last_session": str(day.date()),
                "formal_return_position": position,
                "all_frame_target_position_and_decision_prefixes_exact": True,
                "artifacts": write_run(directory, output),
            }
        )
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefixes": result,
            "new_policy_auxiliary_prefix_runs": 5,
            "independent_economic_trials": 0,
            "source_learner_or_old_account_replayed": False,
        },
    )
    return reference(folder / "completion.json")


def record_failure(stage, error):
    interrupted = lifecycle.phase(datetime.now(timezone.utc)) != "research"
    if events("registered") and not events("review") and not events("technical_failure"):
        if not interrupted:
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                ID,
                {
                    "stage": stage,
                    "reason": str(error),
                    "partial_outputs_retained": True,
                },
            )
    path = FOLDER / f"{stage}-failure-v1.json"
    if not path.exists():
        write_new_receipt(
            path,
            {
                "failed_at": utc_now(),
                "stage": stage,
                "reason": str(error),
                "authorization_interrupted": interrupted,
                "partial_outputs_retained": True,
            },
        )
    progress(
        "interrupted" if interrupted else "technical_failure_preserved",
        failure=str(error),
        next_step="preserve_terminal_or_interrupted_identity_without_retuning",
    )


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not RESULTS.exists()
            and not events("technical_failure")
            and not events("review")
            and not events("path_started"),
            "Do not replay completed,failed or partial shield accounts.",
        )
        data, nominal, frozen, benchmarks = inputs()
        published = publication(frozen, commit)
        start = read_json(FOLDER / "actual-funding-start-v1.json")
        checkpoint = start["actual_continuity_checkpoint_readback"]
        rules.require(
            checkpoint["saved_at"] < start["started_at"]
            and start["actual_preregistration_commit"] == commit,
            "Actual prefunding checkpoint/publication must precede historical output.",
        )
        base.append_event(
            base.JOURNAL,
            "publication",
            None,
            {
                "study_id": rules.STUDY,
                "verification": published,
                "candidate_accounts_previously_started": False,
                "historical_new_meta_outputs_previously_computed": False,
            },
        )
        try:
            protocol = read_json(base.PROTOCOL)
            paths = {}
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, nominal, frozen, window, scenario, benchmarks)
            audits = prefix_audits(data, nominal)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "actual_published_delay_shield_targets_accounts_and_prefixes_v1",
                    "proof_stage": "after_four_actual_published_program_paths",
                    "new_main_capital_prefix_audit": audits,
                    "all_original_formal_dates_and_same_policy_parameters_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {k: reference(FOLDER / "accounts" / k / "complete.json") for k in paths},
                "all_four_paths_same_candidate_and_frozen_parameters": True,
                "independent_forward_validation": False,
                "order_authority": False,
            }
            base.append_event(base.JOURNAL, "review", ID, review)
            write_new_receipt(
                RESULTS,
                {
                    "completed_at": utc_now(),
                    "frozen": frozen,
                    "publication": published,
                    "review": review,
                    "paths": paths,
                    "new_main_capital_prefix_audit": audits,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_all_actual_results_without_source_replay")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(set(saved["paths"]) == base.PATH_KEYS, "Export the complete actual study.")
        for key, ref in saved["review"]["paths"].items():
            base.verify_path(ref, saved["frozen"], ID, key)
        fields = (
            "path_key",
            "metrics",
            "execution",
            "gates",
            "all_point_gates_pass",
            "independent_bt_max_equity_difference_usd",
            "actual_capital_queue_and_accounting_verification",
            "decision_reason_counts",
            "finite_reference_status_counts",
            "retained_reference_inputs",
            "maximum_independent_scalar_money_error_USD",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "actual_delay_aware_main_shield_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "paths": {k: {name: v[name] for name in fields} for k, v in saved["paths"].items()},
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "new_policy_auxiliary_prefix_runs": 5,
                "all_real_prefix_audits_exact": True,
                "finite_reference_is_future_or_full_month_guarantee": False,
                "old_learner_source_target_or_account_replayed": False,
                "raw_prices_targets_holdings_daily_books_or_private_scenarios_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Separate actual-main delayed shield research.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published dynamic-program preregistration SHA.")
    result = (
        evaluate(args.commit)
        if args.action == "evaluate"
        else {
            "freeze": freeze,
            "register": register,
            "export": export,
        }[args.action]()
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
