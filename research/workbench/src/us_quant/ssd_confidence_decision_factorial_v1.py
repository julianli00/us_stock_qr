from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import confidence_decision_factorial_v1 as rules
from us_quant import ssd_main_risk_confidence_prod_v1 as previous
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

base, ROOT, lifecycle = previous.base, previous.ROOT, previous.lifecycle
reference, unique_refs, counts = previous.reference, previous.unique_refs, previous.counts
FOLDER = base.REPORTS / "confidence-decision-factorial-20261107-v1"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/confidence_decision_factorial_20261107_v1.json"
OWN = (
    "config/confidence-decision-factorial-v1.json",
    "src/us_quant/confidence_decision_factorial_v1.py",
    "src/us_quant/ssd_confidence_decision_factorial_v1.py",
    "tests/test_confidence_decision_factorial_v1.py",
    "CONFIDENCE_DECISION_FACTORIAL_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "A saved-decision contrast cannot alter financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "confidence_decision_factorial_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    rules.require(
        counts() == closed["actual_authoritative_counts"]
        and counts()["current_journal_event_count"] == 1518
        and catalogue["new_factorial_projections_or_contribution_labels_computed"] is False
        and catalogue["original_learner_weights_update_generator_or_account_replayed"] is False
        and set(catalogue["four_saved_pairs"]) == set(rules.PATHS),
        "Keep the original complete J1518 closure and every captured original/recent pair.",
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
            *[
                v
                for case in catalogue["four_saved_pairs"].values()
                for v in case.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
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
    return catalogue, refs


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
        "Freeze actual successful three-axis projection,clock and record-Adapter checks.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_all_three_axes_add_exactly_over_all_orders",
            "test_original_seed_and_actual_main_Adapters_cross_same_Interface",
            "test_retains_input_before_endpoint_failure",
            "test_future_records_do_not_change_prefix",
            "test_original_learning_and_account_entrypoints_never_called",
        }.issubset(names),
        "Exercise actual archived formats and exact contrast seams.",
    )
    return [
        *proof["test_outputs"],
        reference(FOLDER / "validation-source-v1.json"),
        proof["validation_runner"],
        *proof["versioned_prototypes"],
    ]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial factorial diagnostic freeze.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, refs = source_inputs()
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": dependencies(),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_refs(),
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_authoritative_counts": counts(),
                "new_factorial_projections_or_contribution_labels_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_one_saved_decision_inspection")
    return reference(INTENT)


def _hex(values):
    array = np.asarray(values)
    rules.require(
        array.ndim == 1 and array.dtype.kind in "fiu", "Preserve numeric saved coordinates."
    )
    return tuple(float(v).hex() for v in array)


def _frame(ref):
    frame = pd.read_csv(
        base.checked_reference(ref),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        not frame.empty
        and isinstance(frame.index, pd.DatetimeIndex)
        and frame.index.tz is None
        and frame.index.is_unique
        and frame.index.is_monotonic_increasing
        and tuple(frame.columns) == rules.ASSETS
        and frame.index.equals(sessions(frame.index[0], frame.index[-1])),
        "A saved recommendation cannot lose its original asset coordinates or calendar.",
    )
    return frame


def _snapshot(day, audit, opinions, target, known):
    state = audit["learner_state"]
    rules.require(
        state["complete_intervals"] == audit["complete_learning_intervals"],
        "Preserve the actual source learner maturity without a new state update.",
    )
    return rules.SavedSnapshot(
        day,
        tuple(state["unnormalized_mass_decimal"]),
        tuple(state["learning_rates_decimal"]),
        state["complete_intervals"],
        _hex(audit["current_confidences"]),
        tuple(tuple(row) for row in opinions),
        tuple(known),
        _hex(audit["expert_weights"]),
        tuple(target),
    )


class OriginalSavedAdapter:
    def __init__(self, catalogue):
        source = read_json(base.checked_reference(catalogue["original_global_audits"]))
        self.targets = _frame(catalogue["original_master_targets"])
        self.snapshots = {}
        observed_audits, records = [], 0
        with base.checked_reference(catalogue["original_global_records"]).open() as stream:
            while line := stream.readline():
                before = json.loads(line)
                after_line = stream.readline()
                rules.require(bool(after_line), "Retain both original source-record halves.")
                after = json.loads(after_line)
                day = before["decision_session"]
                audit = after["audit"]
                rules.require(
                    before["phase"] == "before_decision"
                    and after["phase"] == "after_decision"
                    and day == after["decision_session"] == audit["decision_session"]
                    and day not in self.snapshots
                    and before["known_target_dates"] == audit["known_target_dates"],
                    "The original saved source game cannot acquire an invented capital assessment.",
                )
                self.snapshots[day] = _snapshot(
                    day,
                    audit,
                    before["known_targets_hex"],
                    after["unified_target_hex"],
                    before["known_target_dates"],
                )
                observed_audits.append(audit)
                records += 2
        self.dates = pd.DatetimeIndex(list(self.snapshots))
        active = self.targets.dropna(how="all")
        rules.require(
            records == 242
            and len(self.snapshots) == 121
            and len(source["periods"]) == 119
            and observed_audits == source["decisions"]
            and self.dates.is_unique
            and self.dates.is_monotonic_increasing
            and self.dates.equals(active.index)
            and not (self.targets.isna().any(axis=1) & ~self.targets.isna().all(axis=1)).any(),
            "Retain all121original snapshots/242records and their actual stored target dates.",
        )

    def snapshot_at(self, capital_signal):
        day = rules.trading_day(capital_signal)
        position = self.dates.searchsorted(day, side="right") - 1
        rules.require(position >= 0, "A future master opinion cannot initialize past capital.")
        return self.snapshots[str(self.dates[position].date())]

    def cached_target(self, snapshot):
        return _hex(self.targets.loc[snapshot.as_of].to_numpy())


class ActualMainSavedAdapter:
    def __init__(self, case):
        self.rows = read_json(base.checked_reference(case["recent_capital_decisions"]))["decisions"]
        self.desired, self.issued = (
            _frame(case[name]) for name in ("recent_desired", "recent_issued")
        )
        self.by_day = {row["signal_session"]: row for row in self.rows}
        rules.require(
            len(self.by_day) == len(self.rows) == case["expected_capital_decisions"]
            and self.desired.index.equals(self.issued.index),
            "Keep every actual recent review and both stored target lattices.",
        )

    def snapshot_at(self, capital_signal):
        rules.require(
            capital_signal in self.by_day, "A missing actual capital review cannot be filled."
        )
        row = self.by_day[capital_signal]
        audit = row["assessment"]
        return _snapshot(
            capital_signal,
            audit,
            audit["known_source_targets_hex"],
            _hex(row["desired_target"]),
            audit["source_target_known_at"],
        )


class PairedArchiveAdapter:
    def __init__(self, catalogue, *, paths=rules.PATHS, last_signal=None):
        rules.require(
            bool(paths) and len(paths) == len(set(paths)) and all(k in rules.PATHS for k in paths),
            "Inspect an explicit original path set,not an outcome-selected subset.",
        )
        self.catalogue, self.paths, self.last_signal = catalogue, tuple(paths), last_signal
        self.original = OriginalSavedAdapter(catalogue)

    def __iter__(self):
        for key in self.paths:
            case = self.catalogue["four_saved_pairs"][key]
            old_rows = read_json(base.checked_reference(case["original_capital_decisions"]))[
                "decisions"
            ]
            old_issued = _frame(case["original_issued"])
            recent = ActualMainSavedAdapter(case)
            dates = sessions(case["formal_start"], case["formal_end"])
            expected = [
                str(previous_session(dates[0]).date()),
                *[str(d.date()) for d in dates if is_month_end(d)],
            ]
            rules.require(
                len(dates) == case["formal_return_rows"]
                and len(expected) == case["expected_capital_decisions"]
                and [r["signal_session"] for r in old_rows]
                == [r["signal_session"] for r in recent.rows]
                == expected
                and old_issued.index.equals(self.original.targets.index)
                and recent.issued.index.equals(self.original.targets.index),
                "No old/new capital review may disappear or change the original source lattice.",
            )
            for old, new in zip(old_rows, recent.rows, strict=True):
                signal = old["signal_session"]
                if self.last_signal is not None and signal > self.last_signal:
                    break
                rules.require(
                    old["initial_capital"] is new["initial_capital"]
                    and old["execution_session"] == new["execution_session"],
                    "Do not replace an original initial/monthly capital clock or delay.",
                )
                original_snapshot = self.original.snapshot_at(signal)
                old_target = old_issued.loc[signal]
                new_target = recent.issued.loc[signal]
                yield rules.SavedPair(
                    key,
                    signal,
                    old["initial_capital"],
                    original_snapshot,
                    recent.snapshot_at(signal),
                    old["reason"],
                    new["reason"],
                    self.original.cached_target(original_snapshot),
                    _hex(recent.desired.loc[signal].to_numpy()),
                    None if old_target.isna().all() else _hex(old_target.to_numpy()),
                    None if new_target.isna().all() else _hex(new_target.to_numpy()),
                )


def _retainer(stream, counter):
    def retain(row):
        lifecycle.runtime_guard()
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        counter[0] += 1

    return retain


def prefix_inspections(catalogue, whole):
    folder = FOLDER / "saved-decision-prefixes-v1"
    new_output_directory(folder)
    results = []
    for key in rules.PATHS:
        full = [r for r in whole["retained_row_hashes"] if r["path_key"] == key]
        for position in read_json(rules.POLICY)["prefix_positions"]:
            count = (
                len(full)
                if position == "full"
                else (len(full) // 2 if position == "half" else position)
            )
            day = full[count - 1]["signal_session"]
            trace = folder / f"{key}-{count:03d}.jsonl"
            counter = [0]
            with trace.open("x") as stream:
                inspected = rules.inspect(
                    PairedArchiveAdapter(catalogue, paths=(key,), last_signal=day),
                    retain=_retainer(stream, counter),
                )
            receipt = {
                "path_key": key,
                "last_signal": day,
                "requested_position": position,
                "actual_prefix_count": inspected["pairs"],
                "expected_row_hashes": full[:count],
                "actual_row_hashes": inspected["retained_row_hashes"],
                "all_prefix_inputs_and_resolutions_exact": (
                    inspected["retained_row_hashes"] == full[:count]
                ),
                "retained_before_acceptance_assertion": True,
                "retained_input_resolution_rows": counter[0],
                "trace": reference(trace),
            }
            write_new_receipt(folder / f"{key}-{count:03d}.json", receipt)
            rules.require(
                receipt["all_prefix_inputs_and_resolutions_exact"]
                and inspected["pairs"] == count
                and counter[0] == 2 * count,
                "Future saved records changed a prior static decision contrast.",
            )
            results.append(reference(folder / f"{key}-{count:03d}.json"))
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefix_inspections": results,
            "new_financial_trials": 0,
            "old_learner_or_account_replayed": False,
        },
    )
    return reference(folder / "completion.json")


def analyze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
            "Never replay the first complete or partial real saved-decision analysis.",
        )
        intent = read_json(INTENT)
        for ref in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        rules.require(
            sys.version == intent["python_version"]
            and counts() == intent["actual_authoritative_counts"]
            and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
            == intent["package_versions"],
            "Keep the actual immutable sources,exact environment and financial predecessor.",
        )
        catalogue, refs = source_inputs()
        rules.require(refs == intent["source_references"], "No source reference may change.")
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "An actual saved checkpoint must precede new real diagnostic projections.",
        )
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": reference(INTENT),
            },
        )
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        counter = [0]
        try:
            with trace.open("x") as stream:
                diagnostic = rules.inspect(
                    PairedArchiveAdapter(catalogue),
                    retain=_retainer(stream, counter),
                )
            criteria = intent["criteria"]
            rules.require(
                diagnostic["pairs"] == criteria["saved_pairs"]
                and counter[0] == criteria["expected_main_input_resolution_rows"]
                and all(
                    diagnostic["paths"][key]["overall"]["cases"] == count
                    for key, count in criteria["expected_decisions"].items()
                ),
                "Keep every original comparison and input/resolution record.",
            )
            prefixes = prefix_inspections(catalogue, diagnostic)
            rules.require(
                counts() == intent["actual_authoritative_counts"], "No financial trial occurred."
            )
            write_new_receipt(
                OUTPUT,
                {
                    "completed_at": utc_now(),
                    "criteria_sha256": digest_json(criteria),
                    "analysis_intent": reference(INTENT),
                    "diagnostic": diagnostic,
                    "retained_input_resolution_rows": counter[0],
                    "retained_actual_records": reference(trace),
                    "prefix_inspections": prefixes,
                    "source_references": refs,
                    "actual_authoritative_counts": counts(),
                    "new_financial_trials": 0,
                    "old_learner_weights_update_generator_account_BT_or_diagnosis_replayed": False,
                    "new_executable_recommendations_or_accounts": 0,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure = FOLDER / "analysis-failure-v1.json"
            write_new_receipt(
                failure,
                {
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retained_input_resolution_rows": counter[0],
                    "partial_outputs_preserved": True,
                    "actual_authoritative_counts": counts(),
                    "new_financial_trials": 0,
                    "order_authority": False,
                },
            )
            progress("failed_partial_preserved", failure=reference(failure))
            raise
        progress(
            "complete",
            diagnostic=reference(OUTPUT),
            next_step="export_actual_descriptive_aggregate",
        )
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        rules.require(
            not (FOLDER / "analysis-failure-v1.json").exists()
            and saved["actual_authoritative_counts"] == counts()
            and saved["retained_input_resolution_rows"] == 728,
            "Publish complete actual diagnostics,not successful-looking partial results.",
        )
        for ref in saved["source_references"]:
            base.checked_reference(ref)
        base.checked_reference(saved["retained_actual_records"])
        prefixes = read_json(base.checked_reference(saved["prefix_inspections"]))
        rules.require(
            len(prefixes["prefix_inspections"]) == 20, "Keep all declared prefix inspections."
        )
        for ref in prefixes["prefix_inspections"]:
            base.checked_reference(ref)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "saved_confidence_decision_factorial_contrast",
                "study_id": rules.STUDY,
                "completed_at": saved["completed_at"],
                "criteria": read_json(rules.POLICY),
                "analysis_intent_sha256": file_digest(INTENT),
                "private_diagnostic_sha256": file_digest(OUTPUT),
                "diagnostic": {
                    k: v for k, v in saved["diagnostic"].items() if k != "retained_row_hashes"
                },
                "retained_actual_input_resolution_rows": saved["retained_input_resolution_rows"],
                "actual_prefix_inspections": 20,
                "all_retained_input_and_result_prefixes_exact": True,
                "actual_authoritative_counts": counts(),
                "new_financial_trials": 0,
                "new_executable_recommendations_or_accounts": 0,
                "old_learner_weights_update_generator_account_BT_or_diagnosis_replayed": False,
                "source_CAGR_Sharpe_drawdown_or_qualification_recomputed": False,
                "dated_raw_operands_or_eight_view_vectors_public": False,
                "causal_wealth_effect_or_independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "exported", public_result=reference(PUBLIC), next_step="publish_descriptive_evidence"
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect saved main-confidence decision differences."
    )
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    action = parser.parse_args().action
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[action](), indent=2))
