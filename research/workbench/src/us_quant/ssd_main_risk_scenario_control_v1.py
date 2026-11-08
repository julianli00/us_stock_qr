from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import main_risk_scenario_control_v1 as rules
from us_quant import scenario_main_execution_v1 as execution
from us_quant import ssd_retained_main_risk_v1 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import (
    digest_json,
    new_output_directory,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = rules.ROOT
FOLDER = ROOT / "reports/ssd-continuous-research-v1/main-risk-scenario-control-20261108-v1"
CATALOGUE = FOLDER / "source-inputs-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/main_risk_scenario_control_20261108_registration_v1.json"
PUBLIC = ROOT / "evidence/main_risk_scenario_control_20261108_results_v1.json"
ID, KIND = rules.ID, "causal_main_capital_decision_program_v1"
CANDIDATE = {"id": ID, "weighting": "current_inventory_finite_scenario_viability"}
base, lifecycle = prior.saved.base, prior.lifecycle
OWN = (
    "config/main-risk-scenario-control-v1.json",
    "src/us_quant/main_risk_scenario_control_v1.py",
    "src/us_quant/scenario_main_execution_v1.py",
    "src/us_quant/ssd_main_risk_scenario_control_v1.py",
    "tests/test_main_risk_scenario_control_v1.py",
    "MAIN_RISK_SCENARIO_CONTROL_V1.md",
    "tests/conftest.py",
)
content_path = prior.content_path
read_json = prior.read_json
reference = prior.reference
checked_reference = prior.checked_reference
file_digest = prior.file_digest
safe_file = prior.saved.safe_file
counts, unique_refs = prior.saved.counts, prior.saved.unique_refs
PROTECTED = prior.saved.PROTECTED


def bound(function):
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def events(kind):
    return bound(prior.saved.framework.events)(kind)


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Never replace authoritative financial counts with scenario bookkeeping.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "main_risk_scenario_control_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **fields,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


class ArchiveOpinions:
    def __init__(self, path: Path):
        self.feed = prior.ArchiveFeed(content_path(path))
        origins = self.feed.origins
        expected = (
            pd.Timestamp("2016-10-05"),
            *(
                d
                for d in (
                    sessions("2016-10-06", origins[-1])
                    if origins[-1] > pd.Timestamp("2016-10-05")
                    else ()
                )
                if is_month_end(d)
            ),
        )
        rules.require(origins == expected, "No native monthly source origin may be skipped.")
        for index, point in enumerate(self.feed.points):
            point.checked()
            rules.require(
                point.complete_intervals == max(0, index - 1),
                "Keep original maturity, including the unassimilated partial first month.",
            )

    def observe(self, day: pd.Timestamp) -> rules.OpinionSet:
        point = self.feed.latest(day)
        result = rules.OpinionSet(point.origin, np.asarray(point.targets), point.known_at)
        result.check(day)
        return result


def source_inputs():
    catalogue = read_json(CATALOGUE)
    rules.validate(read_json(rules.POLICY))
    rules.require(
        catalogue["study_id"] == rules.STUDY
        and catalogue["actual_authoritative_counts"]["current_journal_event_count"] == 1554
        and catalogue["forecast_or_allocations_or_new_accounts_computed"] is False
        and catalogue["permitted_checks"]
        == {"archival_references": 16, "permitted_live": 15, "permitted_user_live": 5},
        "Preserve the admitted J1554 metadata capture and the prohibited-content exclusion.",
    )
    refs = unique_refs(
        [
            reference(CATALOGUE),
            catalogue["last_complete_risk_cycle"],
            catalogue["last_complete_financial_cycle"],
            catalogue["source_opinions"],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
        ]
    )
    for ref in refs:
        checked_reference(ref)
    return catalogue, catalogue["parent_frozen_specification"], refs


def validation_refs():
    versions = sorted(FOLDER.glob("validation-source-v*.json"))
    rules.require(bool(versions), "Successful exact-source synthetic evidence is required.")
    proof = read_json(versions[-1])
    suites = (
        ElementTree.parse(checked_reference(proof["test_outputs"][0]))
        .getroot()
        .findall(".//testsuite")
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / name) for name in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        and {
            "test_negative_gap_envelope_contains_original_cash_engine",
            "test_archive_and_memory_opinions_cross_same_interface",
            "test_actual_packet_defers_historical_capital",
            "test_future_source_and_prices_do_not_change_daily_prefix",
            "test_prohibited_content_rejected_before_open",
            "test_driver_terminal_failure_preserves_outputs",
        }.issubset(names),
        "Validate the complete clock, accounting, packet, protection and failure Interfaces.",
    )
    refs = []
    for version in versions:
        record = read_json(version)
        refs += [
            reference(version),
            *record["test_outputs"],
            record["validation_runner"],
            *record["versioned_prototypes"],
        ]
    return unique_refs(refs)


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        checked_reference(ref)


def freeze():
    prior._guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Do not repeat a complete or partial first financial freeze.",
        )
        catalogue, parent, refs = source_inputs()
        history = base.journal_events(base.JOURNAL)
        rules.require(
            counts() == catalogue["actual_authoritative_counts"]
            and len(history) == 1554
            and history[-1]["sha256"]
            == catalogue["actual_authoritative_counts"]["current_journal_head_sha256"]
            and not any(row["candidate_id"] == ID for row in history),
            "Freeze one unused financial identity without changing the predecessor.",
        )
        deps, validation = bound(prior.saved.dependencies)(), validation_refs()
        inherited = (
            "protocol",
            "catalog",
            "legacy",
            "volatility_sources",
            "source_policy_and_expert_admission_clocks",
            "possible_economic_factor_families",
            "eligible_actual_registered_factors",
            "minimum_economic_factor_families",
        )
        frozen = {
            **{k: parent[k] for k in inherited},
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[2]),
            "execution_adapter_role": "new_daily_actual_inventory_one_pending_cash_book",
            "expert_pool_definition_role": "unchanged_canonical_six_complete_admitted_sources",
            "construction": "finite_matured_price_scenario_current_inventory_viability",
            "decision_program_kind": KIND,
            "decision_information_clock": "latest_monthly_source_then_daily_actual_main_close",
            "declared_financial_cohort_size": 1,
            "expert_ids": list(rules.EXPERTS),
            "market": catalogue["source_market"],
            "benchmark_artifacts": catalogue["source_benchmark_artifacts"],
            "code": unique_refs(
                [*parent["code"], *[r for r in deps if r["path"].startswith(("src/", "tests/"))]]
                + [reference(ROOT / OWN[5])]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": refs,
            "validation_evidence": validation,
            "prior_journal_sequence": 1554,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 308,
            "protected_archival_reference_is_new_live_validation": False,
            "archival_reference_count": 16,
            "permitted_live_reference_count": 15,
            "permitted_user_live_reference_count": 5,
            "source_probabilities_or_regret_recomputed": False,
            "old_source_learner_or_account_replayed": False,
            "historical_scenario_feasibility_is_future_drawdown_guarantee": False,
            "historical_new_meta_outputs_before_publication": False,
            "independent_forward_validation": False,
            "order_authority": False,
        }
        check_frozen(frozen)
        prior._permitted_refs(catalogue)
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "python_version": sys.version,
                "package_versions": prior._packages(),
                "actual_frozen_method_complete": True,
                "real_forecasts_source_choices_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_checkpoint_then_one_schema3_registration")
        return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        checked_reference(ref)
    catalogue, _, refs = source_inputs()
    rules.require(
        refs == intent["source_references"]
        and sys.version == intent["python_version"]
        and prior._packages() == intent["package_versions"]
        and intent["actual_frozen_method_complete"] is True,
        "Keep the complete frozen method, inputs, interpreter and exact package inventory.",
    )
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return catalogue, frozen


def checkpoint(path):
    receipt = read_json(path)
    cp = receipt["actual_continuity_checkpoint_readback"]
    created = pd.Timestamp(read_json(INTENT)["created_at"])
    saved, checked = pd.Timestamp(cp["saved_at"]), pd.Timestamp(receipt["checked_at"])
    rules.require(
        receipt["entire_state_readback_equal"] is True
        and receipt["frozen_implementation"] == reference(INTENT)
        and cp["session_id"] == lifecycle.SESSION
        and type(cp["revision"]) is int
        and type(cp["covered_seq"]) is int
        and cp["revision"] > 932
        and cp["covered_seq"] > 211082
        and all(t.tzinfo is not None and pd.notna(t) for t in (created, saved, checked))
        and created < saved <= checked <= pd.Timestamp(utc_now()),
        "An actual whole-state checkpoint readback must follow this frozen implementation.",
    )
    return receipt


def packet_body(frozen, registered, readiness):
    return bound(prior.saved.framework.packet_body)(frozen, registered, readiness)


def checked_packet(frozen):
    return bound(prior.saved.framework.checked_packet)(frozen)


def register():
    prior._guard()
    with lifecycle.research_job():
        catalogue, frozen = inputs()
        checkpoint(FOLDER / "actual-preregistration-checkpoint-v1.json")
        rules.require(
            not events("registered")
            and not events("technical_failure")
            and not PACKET.exists()
            and not READY.exists()
            and counts() == catalogue["actual_authoritative_counts"],
            "Do not replay a partial or complete financial registration.",
        )
        registered = base.append_event(
            base.JOURNAL, "registered", ID, {"candidate": CANDIDATE, "frozen": frozen}
        )
        write_new_receipt(
            READY,
            {
                "checked_at": utc_now(),
                "readiness_kind": "synthetic_Interface_and_archived_input_identity_only",
                "implementation": reference(INTENT),
                "validation": validation_refs(),
                "actual_frozen_method_complete": True,
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            },
        )
        write_new_receipt(
            PACKET,
            {**packet_body(frozen, registered, reference(READY)), "created_at": utc_now()},
        )
        checked_packet(frozen)
        progress(
            "registered_before_outputs", next_step="publish_program_then_prefunding_checkpoint"
        )
        return reference(PACKET)


def publication(frozen, commit):
    checked_packet(frozen)
    return bound(base.published_inputs)(frozen, commit)


def funding_fence(start, commit):
    remote = read_json(checked_reference(start["actual_remote_publication"]))
    saved = pd.Timestamp(start["actual_continuity_checkpoint_readback"]["saved_at"])
    published = pd.Timestamp(remote["verified_at"])
    registered = pd.Timestamp(read_json(PACKET)["created_at"])
    rules.require(
        start["actual_preregistration_commit"] == remote["actual_preregistration_commit"] == commit
        and remote["historical_outputs_previously_started"] is False
        and all(t.tzinfo is not None and pd.notna(t) for t in (registered, published, saved))
        and registered < published < saved <= pd.Timestamp(start["checked_at"]),
        "A distinct actual checkpoint must follow remote program verification before funding.",
    )


def _stream(path, rows):
    with content_path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")


def _records(path, cutoff=None):
    with content_path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            day = pd.Timestamp(row.get("day", row.get("signal_session")))
            if cutoff is not None and day > cutoff:
                break
            yield row


def write_run(directory, output):
    for name, frame in (
        ("account.csv", output.result.frame),
        ("weights.csv", output.result.weights),
        ("desired.csv", output.issued),
        ("issued.csv", output.issued),
    ):
        write_text_atomic(content_path(directory / name), frame.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "pending-after-window.json",
        {"pending": output.pending_after_window, "main_decisions": output.decision_count},
    )
    return [
        reference(directory / name)
        for name in (
            "account.csv",
            "weights.csv",
            "desired.csv",
            "issued.csv",
            "pending-after-window.json",
        )
    ]


def same_frame(left, right):
    return (
        left.index.equals(right.index)
        and left.columns.equals(right.columns)
        and np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
        and np.array_equal(np.signbit(left.to_numpy()), np.signbit(right.to_numpy()))
    )


def verify_run(data, output, start, end, cost, delay, decisions, retain_comparison):
    replay = execution.cash_replay(data, output, start, end, cost, delay)
    exact_frame = same_frame(output.result.frame, replay.frame)
    exact_positions = same_frame(output.result.weights, replay.weights)
    report = {
        "all_financial_frame_coordinates_exact": exact_frame,
        "all_actual_position_coordinates_exact": exact_positions,
        "main_initial_capital_and_peak_usd": 10000.0,
    }
    retain_comparison(output.result, replay, report)
    rules.require(exact_frame and exact_positions, "Retained exact cash comparison failed.")
    expected = pd.DataFrame(np.nan, index=output.issued.index, columns=output.issued.columns)
    frame, peak, count = output.result.frame, 10000.0, 0
    maximum_error = 0.0
    reasons = Counter()
    for row in decisions:
        day = pd.Timestamp(row["signal_session"])
        nav = 10000.0 if count == 0 else float(frame.loc[day, "equity"])
        peak = max(peak, nav)
        rules.require(
            row["main_NAV"] == nav
            and row["main_peak"] == peak
            and row["initial_capital"] == (count == 0),
            "Every scenario decision must consume actual fresh main NAV and its running peak.",
        )
        if row["target"] is not None:
            expected.loc[day] = row["target"]
        detail = row["assessment"]
        origin = pd.Timestamp(detail["source_origin"])
        rules.require(
            origin <= day
            and all(pd.Timestamp(d) <= origin for d in detail["source_native_known_at"])
            and not detail["original_source_clock_refreshed_daily"]
            and all(pd.Timestamp(d) <= day for d in detail.get("price_path_ends", [])),
            "The daily main review cannot refresh a source or consume unmatured scenarios.",
        )
        for action in detail.get("actions", []):
            maximum_error = max(
                maximum_error, action["assessment"]["independent_relative_wealth_error"]
            )
        reasons[detail["status"]] += 1
        count += 1
    rules.require(
        count == output.decision_count == len(frame) + 1
        and np.array_equal(expected.to_numpy(), output.issued.to_numpy(), equal_nan=True),
        "Retain all decisions and the exact originally issued target coordinates.",
    )
    return {
        **report,
        "daily_main_decisions": count,
        "decision_status_counts": dict(reasons),
        "maximum_independent_scenario_wealth_error": maximum_error,
        "source_clock_not_refreshed_daily": True,
        "new_source_learner_or_shadow_runs": 0,
        "scenario_feasibility_is_future_drawdown_guarantee": False,
    }


def _run(data, opinions, window, scenario, trace):
    controller = rules.Controller(
        rules.PriceScenarioFeed(data.open, data.close),
        opinions,
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    with content_path(trace).open("x", encoding="utf-8") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
            stream.flush()

        return execution.run(
            data,
            controller,
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            retain=retain,
            before=lifecycle.runtime_guard,
            keep_decisions=False,
        )


def evaluate_path(data, catalogue, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    rules.require(
        not any(e["body"]["path_key"] == key for e in events("path_started")),
        "Never replay a complete or partial main-capital path.",
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
    trace = directory / "chronological-trace.jsonl"
    output = _run(
        data,
        ArchiveOpinions(checked_reference(catalogue["source_opinions"])),
        window,
        scenario,
        trace,
    )
    artifacts = write_run(directory, output)
    comparison = directory / "cash-comparison"
    new_output_directory(comparison)
    feedback = verify_run(
        data,
        output,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
        (r for r in _records(trace) if r["kind"] == "main_decision_resolved"),
        lambda actual, replay, report: prior.saved.retain_cash_comparison(
            comparison, actual, replay, report
        ),
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
    independent_path = directory / "independent-bt.csv"
    write_text_atomic(independent_path, independent.to_csv(float_format="%.17g"))
    metrics = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    other, difference = base.independent_metrics(
        output.result.frame, independent, data.risk_free.loc[output.result.frame.index], 10000
    )
    base.check_metrics(other, metrics)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], s] for s in ("SPY", "QQQ"))
    gates = base.gates(metrics, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "decision_program_kind": KIND,
        "metrics": metrics,
        "gates": gates,
        "all_point_gates_pass": all(gates.values()),
        "execution": base.execution_statistics(
            output.result, data, 10000, scenario["cost_bps_per_side"], 1
        ),
        "independent_bt_max_equity_difference_usd": difference,
        "actual_capital_and_source_selection_verification": feedback,
        "artifacts": artifacts
        + [reference(trace), reference(independent_path)]
        + [reference(p) for p in sorted(comparison.iterdir())],
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


def _clip(data, day):
    return MarketData(
        data.open.loc[:day],
        data.close.loc[:day],
        data.raw_close.loc[:day],
        data.volume.loc[:day],
        data.risk_free.loc[:day],
    )


def prefixes(data, catalogue):
    folder = FOLDER / "capital-and-source-prefix-audits-v1"
    new_output_directory(folder)
    source = FOLDER / "accounts/10y-base"
    frames = {
        name: pd.read_csv(
            checked_reference(reference(source / name)),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    results = []
    original_opinions = ArchiveOpinions(checked_reference(catalogue["source_opinions"]))
    for position in read_json(rules.POLICY)["prefix_positions"]:
        lifecycle.runtime_guard()
        day = frames["account.csv"].index[position]
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        tape = directory / "source-prefix.jsonl"
        with (
            content_path(catalogue_path := checked_reference(catalogue["source_opinions"])).open(
                encoding="utf-8"
            ) as stream,
            content_path(tape).open("x", encoding="utf-8") as target_stream,
        ):
            for line in stream:
                if pd.Timestamp(json.loads(line)["decision_session"]) <= day:
                    target_stream.write(line)
        clipped_opinions = ArchiveOpinions(tape)
        a, b = original_opinions.observe(day), clipped_opinions.observe(day)
        source_exact = (
            a.origin == b.origin
            and a.known_at == b.known_at
            and np.array_equal(a.targets, b.targets)
        )
        trace = directory / "chronological-trace.jsonl"
        output = _run(
            _clip(data, day),
            clipped_opinions,
            {"start": "2016-10-06", "end": str(day.date())},
            {"cost_bps_per_side": 5.0, "delay_sessions": 1},
            trace,
        )
        artifacts = write_run(directory, output)
        expected_trace = directory / "expected-trace.jsonl"
        _stream(expected_trace, _records(source / "chronological-trace.jsonl", day))
        exact = source_exact and file_digest(trace) == file_digest(expected_trace)
        for name, actual in (
            ("account.csv", output.result.frame),
            ("weights.csv", output.result.weights),
            ("desired.csv", output.issued),
            ("issued.csv", output.issued),
        ):
            expected = frames[name].loc[:day]
            write_text_atomic(
                directory / ("expected-" + name), expected.to_csv(float_format="%.17g")
            )
            exact = exact and same_frame(actual.loc[:day], expected)
            artifacts.append(reference(directory / ("expected-" + name)))
        write_new_receipt(
            directory / "comparison.json",
            {
                "last_session": str(day.date()),
                "source_prefix_exact": source_exact,
                "all_capital_scenario_and_source_trace_prefixes_exact": exact,
                "both_operands_retained_before_assertion": True,
                "original_source_tape": reference(catalogue_path),
                "artifacts": artifacts
                + [reference(trace), reference(expected_trace), reference(tape)],
            },
        )
        rules.require(exact, "Future prices or source records changed an earlier main prefix.")
        results.append(reference(directory / "comparison.json"))
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefixes": results,
            "new_economic_trials": 0,
            "new_source_prefix_checks": 5,
            "new_main_capital_prefix_runs": 5,
            "old_source_learner_or_account_replayed": False,
        },
    )
    return reference(folder / "completion.json")


def record_failure(stage, error):
    return bound(prior.saved.framework.record_failure)(stage, error)


def evaluate(commit):
    prior._guard()
    with lifecycle.research_job():
        rules.require(
            not RESULTS.exists()
            and not events("technical_failure")
            and not events("review")
            and not events("path_started")
            and not (FOLDER / "evaluation-started-v1.json").exists(),
            "Preserve every completed, failed or partially started financial identity.",
        )
        catalogue, frozen = inputs()
        published = publication(frozen, commit)
        start = checkpoint(FOLDER / "actual-funding-start-v1.json")
        funding_fence(start, commit)
        write_new_receipt(FOLDER / "evaluation-started-v1.json", {"started_at": utc_now()})
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
        progress("running", active_compute_job={"pid": os.getpid(), "study_id": rules.STUDY})
        try:
            data = prior.saved.review_market({"market": catalogue["source_market"]}, ROOT)
            refs = [
                r
                for r in catalogue["source_benchmark_artifacts"]
                if r["path"].endswith("/results.json")
            ]
            rules.require(len(refs) == 1, "Use the unique admitted dual-benchmark result set.")
            benchmarks = read_json(checked_reference(refs[0]))
            rules.require(
                len(benchmarks["benchmarks"]) == 8, "Preserve all original benchmark paths."
            )
            protocol, paths = read_json(base.PROTOCOL), {}
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(
                        data, catalogue, frozen, window, scenario, benchmarks
                    )
            proof = prefixes(data, catalogue)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "actual_inventory_scenarios_daily_capital_and_monthly_source_v1",
                    "proof_stage": "after_four_actual_published_program_paths",
                    "prefix_audits": proof,
                    "all_original_formal_dates_and_fixed_parameters_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(r["all_point_gates_pass"] for r in paths.values())
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
                    "prefix_audits": proof,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_actual_results_and_close_cycle")
        return reference(RESULTS)


def export():
    prior._guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(set(saved["paths"]) == base.PATH_KEYS, "Export only the complete four paths.")
        for key, ref in saved["review"]["paths"].items():
            bound(base.verify_path)(ref, saved["frozen"], ID, key)
        fields = (
            "path_key",
            "metrics",
            "execution",
            "gates",
            "all_point_gates_pass",
            "independent_bt_max_equity_difference_usd",
            "actual_capital_and_source_selection_verification",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "actual_inventory_scenario_control_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "paths": {k: {f: v[f] for f in fields} for k, v in saved["paths"].items()},
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "new_source_prefix_checks": 5,
                "new_main_capital_prefix_runs": 5,
                "all_real_prefix_audits_exact": True,
                "scenario_feasibility_is_future_drawdown_guarantee": False,
                "old_source_learner_or_account_replayed": False,
                "raw_prices_scenarios_targets_holdings_or_daily_books_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        return reference(PUBLIC)


def capture():
    prior._guard()
    with prior.lifecycle.research_job():
        policy = read_json(rules.POLICY)
        rules.validate(policy)
        rules.require(not FOLDER.exists(), "Preserve a retained or partial new study capture.")
        new_output_directory(FOLDER)
        write_new_receipt(FOLDER / "capture-started-v1.json", {"started_at": utc_now()})
        risk = read_json(prior.FOLDER / "cycle-closure-v1.json")
        financial = read_json(checked_reference(risk["last_complete_financial_cycle"]))
        source = read_json(checked_reference(financial["private_result"]))
        actual = prior.saved.counts()
        rules.require(
            actual == risk["actual_authoritative_counts"]
            and actual["current_journal_event_count"] == 1554,
            "Use the closed J1554 predecessor, not an old quota or stale journal.",
        )
        baseline = read_json(prior.CATALOGUE)
        market = source["frozen"]["market"]
        tape = baseline["source_records"]
        for ref in [*market.values(), tape]:
            checked_reference(ref)
        protection = prior._permitted_refs(baseline)
        write_new_receipt(
            CATALOGUE,
            {
                "captured_at": utc_now(),
                "study_id": rules.STUDY,
                "actual_authoritative_counts": actual,
                "last_complete_risk_cycle": reference(prior.FOLDER / "cycle-closure-v1.json"),
                "last_complete_financial_cycle": risk["last_complete_financial_cycle"],
                "source_market": market,
                "source_opinions": tape,
                "source_benchmark_artifacts": source["frozen"]["benchmark_artifacts"],
                "parent_frozen_specification": source["frozen"],
                "protected_reference_metadata": baseline["protected_reference_metadata"],
                "permitted_checks": protection,
                "forecast_or_allocations_or_new_accounts_computed": False,
                "external_paper_status": (
                    "Springer DOI10.1007/s10479-018-2947-3 returned406; "
                    "no access workaround; not a verified algorithm implementation source"
                ),
                "research_owned_explicit_scenario_law": True,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        write_json(
            ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
            {
                "updated_at_utc": utc_now(),
                "phase": "main_risk_scenario_control_captured_unfrozen",
                "study_id": rules.STUDY,
                "active_compute_job": None,
                **actual,
                "new_financial_trials": 0,
                "last_complete_cycle": reference(prior.FOLDER / "cycle-closure-v1.json"),
                "last_complete_financial_cycle": risk["last_complete_financial_cycle"],
                "next_step": "complete_registration_evaluation_wrappers_and_synthetic_freeze",
                "forecast_or_allocations_or_new_accounts_computed": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        return reference(CATALOGUE)


def main():
    parser = argparse.ArgumentParser(description="Versioned current-main-inventory scenario study.")
    parser.add_argument("action", choices=("capture", "freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate":
        rules.require(bool(args.commit), "Use the actual remotely verified preregistration commit.")
        result = evaluate(args.commit)
    else:
        result = globals()[args.action]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
