from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from importlib.metadata import distributions
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree

from us_quant import source_utility_geometry_v1 as rules
from us_quant import ssd_expert_switching_loss_v1 as completed
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

base, ROOT, lifecycle = completed.base, completed.ROOT, completed.lifecycle
reference, unique_refs, counts = completed.reference, completed.unique_refs, completed.counts
FOLDER = base.REPORTS / "source-utility-geometry-20261107-v1"
INTENT, OUTPUT = FOLDER / "analysis-intent-v1.json", FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/source_utility_geometry_20261107_v1.json"
OWN = (
    "config/source-utility-geometry-v1.json",
    "src/us_quant/source_utility_geometry_v1.py",
    "src/us_quant/ssd_source_utility_geometry_v1.py",
    "tests/test_source_utility_geometry_v1.py",
    "SOURCE_UTILITY_GEOMETRY_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def progress(phase: str, **extra) -> None:
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "Utility geometry cannot alter authoritative financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "source_utility_geometry_" + phase,
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
        and counts()["current_journal_event_count"] == 1542
        and tuple(catalogue["source_expert_ids"]) == rules.EXPERT_IDS
        and catalogue["new_utility_projection_or_magnitude_share_labels_computed"] is False
        and catalogue["source_rate_forecast_or_target_regenerated"] is False,
        "Keep the complete J1542 predecessor and unchanged six-source records.",
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
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "capture-failed-attempt-v1.py",
                    "capture-failure-v1.json",
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
    first = read_json(FOLDER / "validation-source-v1.json")
    proof = read_json(FOLDER / "validation-source-v2.json")
    suites = ElementTree.parse(base.checked_reference(proof["test_outputs"][0])).getroot()
    suites = suites.findall(".//testsuite")
    rules.require(
        suites
        and first["exit_code"] == 1
        and first["junit_totals"]["tests"] == 33
        and first["junit_totals"]["failures"] == 30
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")),
        "Freeze actual successful utility Interface and record-Adapter checks.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_exact_parallel_orthogonal_and_energy",
            "test_common_rate_cancels_magnitude_not_displacement",
            "test_archive_and_memory_cross_same_Interface",
            "test_future_records_do_not_change_prefix",
            "test_old_source_and_account_entrypoints_never_called",
            "test_retains_input_before_invalid_endpoint",
        }.issubset(names),
        "Test actual archived formats, source clocks and descriptive-only geometry.",
    )
    return [
        *proof["test_outputs"],
        reference(FOLDER / "validation-source-v2.json"),
        reference(FOLDER / "validation-source-v1.json"),
        proof["validation_runner"],
        *proof["versioned_prototypes"],
        *first["test_outputs"],
        *first["versioned_prototypes"],
    ]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial source-utility freeze.",
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
                "new_utility_projection_or_magnitude_share_labels_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_checkpoint_then_one_retained_utility_analysis")
    return reference(INTENT)


def _hex(values, length: int) -> tuple[str, ...]:
    rules.require(
        isinstance(values, list | tuple)
        and len(values) == length
        and all(type(v) in (float, int) and math.isfinite(v) for v in values),
        "Preserve finite complete binary64 source coordinates.",
    )
    return tuple(float(v).hex() for v in values)


def _pairs(ref: dict, reader: Callable[[dict], Path], *, read_rate: bool = False):
    result = {}
    records = 0
    with reader(ref).open(encoding="utf-8") as stream:
        while line := stream.readline():
            before = json.loads(line)
            after_line = stream.readline()
            rules.require(bool(after_line), "Retain both halves of every source record.")
            after = json.loads(after_line)
            origin = before["decision_session"]
            rules.require(
                before["phase"] == "before_decision"
                and after["phase"] == "after_decision"
                and after["decision_session"] == origin
                and after["audit"]["decision_session"] == origin
                and before["source_native_known_at"] == after["audit"]["source_native_known_at"]
                and origin not in result,
                "Stored source-record order, clock or identity changed.",
            )
            result[origin] = {
                "matured": before["matured_source_interval"],
                "endpoint": rules.StoredEndpoint(
                    tuple(tuple(row) for row in before["source_targets_hex"]),
                    tuple(before["source_native_known_at"]),
                    _hex(after["audit"]["expert_weights"], rules.N),
                    tuple(after["unified_target_hex"]),
                ),
                "depth": after["audit"]["complete_source_intervals"],
                "rate": after["audit"]["state"]["learning_rate_decimal"] if read_rate else None,
            }
            records += 2
    rules.require(bool(result), "A source tape cannot be silently empty.")
    return result, records


class ArchiveAdapter:
    def __init__(
        self,
        catalogue: dict,
        *,
        reader: Callable[[dict], Path] = base.checked_reference,
        last_origin: str | None = None,
    ):
        teacher = read_json(reader(catalogue["teacher_periods"]))
        self.months = tuple(
            rules.UtilityMonth(
                p["start_after"],
                p["end"],
                _hex(p["canonical_log_relatives"], rules.N),
                _hex(p["input_loss_labels"], rules.N),
            )
            for p in teacher["periods"]
        )
        old, old_count = _pairs(catalogue["wealth_records"], reader)
        new, new_count = _pairs(catalogue["adaptive_loss_records"], reader, read_rate=True)
        origins = tuple(a["decision_session"] for a in teacher["decisions"])
        rules.require(
            tuple(catalogue["source_expert_ids"]) == rules.EXPERT_IDS
            and tuple(old) == tuple(new) == origins
            and old_count == new_count == 2 * len(origins),
            "Keep every original source origin and both complete tapes; no cropping.",
        )
        self.snapshots = {}
        for origin in origins:
            depth = sum(m.end <= origin for m in self.months)
            rules.require(
                type(old[origin]["depth"]) is type(new[origin]["depth"]) is int
                and old[origin]["depth"] == new[origin]["depth"] == depth,
                "Saved source maturity disagrees with its original complete-month history.",
            )
            for values, field in (
                (old[origin], "original_log_relatives_hex"),
                (new[origin], "original_loss_labels_hex"),
            ):
                matured = values["matured"]
                if depth == 0:
                    rules.require(matured is None, "An initial partial month cannot train.")
                else:
                    month = self.months[depth - 1]
                    expected = (
                        month.log_hex if field == "original_log_relatives_hex" else month.loss_hex
                    )
                    rules.require(
                        matured is not None
                        and matured["start_after"] == month.start_after
                        and matured["end"] == month.end == origin
                        and tuple(matured[field]) == expected,
                        "Original source-log/loss primitives were changed or rematured.",
                    )
            rate = new[origin]["rate"]
            self.snapshots[origin] = rules.UtilitySnapshot(
                origin,
                self.months[:depth],
                rate,
                old[origin]["endpoint"],
                new[origin]["endpoint"],
            )
        self.origins = (
            origins if last_origin is None else tuple(d for d in origins if d <= last_origin)
        )
        rules.require(bool(self.origins), "A requested source prefix contains no origin.")

    def observe(self, origin: str) -> rules.UtilitySnapshot:
        rules.require(origin in self.origins, "The saved origin is not in this Adapter's scope.")
        return self.snapshots[origin]


def _retainer(stream, counter: list[int]):
    def retain(value):
        stream.write(
            json.dumps(rules.encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        counter[0] += 1

    return retain


def prefix_inspections(catalogue: dict, full: dict):
    folder = FOLDER / "prefix-inspections-v1"
    new_output_directory(folder)
    inspections = []
    for cutoff in read_json(rules.POLICY)["prefix_cutoffs"]:
        lifecycle.runtime_guard()
        trace, counter = folder / f"{cutoff}.jsonl", [0]
        with trace.open("x", encoding="utf-8") as stream:
            observed = rules.inspect(
                ArchiveAdapter(catalogue, last_origin=cutoff),
                retain=_retainer(stream, counter),
                before_origin=lifecycle.runtime_guard,
            )
        expected_origins = [r for r in full["origin_row_hashes"] if r["origin"] <= cutoff]
        expected_cells = [r for r in full["cell_row_hashes"] if r["origin"] <= cutoff]
        receipt = {
            "cutoff": cutoff,
            "actual_source_origins": observed["source_origins"],
            "actual_past_month_views": observed["past_month_utility_views"],
            "expected_origin_rows": expected_origins,
            "actual_origin_rows": observed["origin_row_hashes"],
            "expected_cell_rows": expected_cells,
            "actual_cell_rows": observed["cell_row_hashes"],
            "exact_prefix": observed["origin_row_hashes"] == expected_origins
            and observed["cell_row_hashes"] == expected_cells,
            "retained_input_resolution_rows": counter[0],
            "retained_before_assertion": True,
            "trace": reference(trace),
            "new_financial_trials": 0,
        }
        path = folder / f"{cutoff}.json"
        write_new_receipt(path, receipt)
        rules.require(
            receipt["exact_prefix"]
            and counter[0] == 2 * (len(expected_origins) + len(expected_cells)),
            "Future records changed an earlier static source-utility view.",
        )
        inspections.append(reference(path))
    path = folder / "completion.json"
    write_new_receipt(
        path,
        {"completed_at": utc_now(), "prefix_inspections": inspections, "new_financial_trials": 0},
    )
    return reference(path)


def analyze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
            "Never replay the first complete or partial real source-utility analysis.",
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
            "Keep frozen inputs, exact packages and the complete financial predecessor.",
        )
        catalogue, refs = source_inputs()
        rules.require(refs == intent["source_references"], "A source reference changed.")
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "An actual saved checkpoint must precede new real utility labels.",
        )
        begun = FOLDER / "analysis-started-v1.json"
        write_new_receipt(
            begun,
            {"started_at": utc_now(), "pid": os.getpid(), "analysis_intent": reference(INTENT)},
        )
        progress(
            "running",
            active_compute_job={"pid": os.getpid(), "start": reference(begun)},
        )
        trace, counter = FOLDER / "actual-inputs-resolved-v1.jsonl", [0]
        try:
            with trace.open("x", encoding="utf-8") as stream:
                diagnostic = rules.inspect(
                    ArchiveAdapter(catalogue),
                    retain=_retainer(stream, counter),
                    before_origin=lifecycle.runtime_guard,
                )
            policy = intent["criteria"]
            rules.require(
                diagnostic["source_origins"] == policy["source_origins"]
                and diagnostic["complete_source_months"] == policy["complete_source_months"]
                and diagnostic["past_month_utility_views"] == policy["past_month_utility_views"]
                and counter[0] == policy["expected_retained_input_resolution_rows"],
                "Keep all original origins and every declared source-utility comparison.",
            )
            prefixes = prefix_inspections(catalogue, diagnostic)
            rules.require(
                counts() == intent["actual_authoritative_counts"], "A financial trial occurred."
            )
            write_new_receipt(
                OUTPUT,
                {
                    "completed_at": utc_now(),
                    "criteria_sha256": digest_json(policy),
                    "analysis_intent": reference(INTENT),
                    "diagnostic": diagnostic,
                    "retained_input_resolution_rows": counter[0],
                    "retained_actual_records": reference(trace),
                    "prefix_inspections": prefixes,
                    "source_references": refs,
                    "actual_authoritative_counts": counts(),
                    "new_financial_trials": 0,
                    "old_learner_source_target_or_account_replayed": False,
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
        progress("complete", diagnostic=reference(OUTPUT), next_step="export_actual_aggregate")
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        rules.require(
            not (FOLDER / "analysis-failure-v1.json").exists()
            and saved["actual_authoritative_counts"] == counts()
            and saved["retained_input_resolution_rows"] == 14522,
            "Publish complete actual utility diagnostics, not partial results.",
        )
        for ref in saved["source_references"]:
            base.checked_reference(ref)
        base.checked_reference(saved["retained_actual_records"])
        prefixes = read_json(base.checked_reference(saved["prefix_inspections"]))
        rules.require(
            len(prefixes["prefix_inspections"]) == 5, "Keep every declared source prefix."
        )
        for ref in prefixes["prefix_inspections"]:
            base.checked_reference(ref)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "retained_source_utility_geometry",
                "study_id": rules.STUDY,
                "completed_at": saved["completed_at"],
                "criteria": read_json(rules.POLICY),
                "analysis_intent_sha256": file_digest(INTENT),
                "private_diagnostic_sha256": file_digest(OUTPUT),
                "diagnostic": {
                    k: v
                    for k, v in saved["diagnostic"].items()
                    if k not in ("origin_row_hashes", "cell_row_hashes")
                },
                "retained_actual_input_resolution_rows": saved["retained_input_resolution_rows"],
                "actual_prefix_inspections": 5,
                "all_retained_prefixes_exact": True,
                "actual_authoritative_counts": counts(),
                "new_financial_trials": 0,
                "new_executable_recommendations_or_accounts": 0,
                "old_learner_source_target_or_account_replayed": False,
                "dated_raw_logs_losses_rates_utilities_targets_public": False,
                "causal_wealth_effect_or_independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress("exported", public_result=reference(PUBLIC), next_step="publish_actual_evidence")
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect retained source utility units.")
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    action = parser.parse_args().action
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[action](), indent=2))
