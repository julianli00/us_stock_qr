from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from itertools import islice
from types import FunctionType
from xml.etree import ElementTree

from us_quant import nested_growth_allocation_audit_v1 as rules
from us_quant import ssd_growth_macro_defense_hedge_v1 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    read_json,
    utc_now,
    write_json,
)

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "nested-growth-allocation-audit-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/nested_growth_allocation_audit_20261106_v1.json"
OWN = (
    "config/nested-growth-allocation-audit-v1.json",
    "src/us_quant/nested_growth_allocation_audit_v1.py",
    "src/us_quant/ssd_nested_growth_allocation_audit_v1.py",
    "tests/test_nested_growth_allocation_audit_v1.py",
    "tests/conftest.py",
    "NESTED_GROWTH_ALLOCATION_AUDIT_V1.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    previous = read_json(base.checked_reference(inputs["outer_completed_cycle"]))
    refs = [
        *previous["exact_evidence_references"],
        *[
            reference(FOLDER / name)
            for name in (
                "source-inputs-v1.json",
                "design-decision-v1.json",
                "actual-invocation-v1.json",
                "preservation-baseline-v1.json",
                "capture-preflight-v1.py",
            )
        ],
        *[
            inputs[k]
            for k in (
                "outer_input_resolved_tape",
                "outer_implementation_freeze",
                "outer_prepared_targets",
                "outer_information_audits",
                "outer_private_result",
                "outer_public_result",
                "outer_completed_cycle",
                "child_input_resolved_tape",
                "child_admission_review",
            )
        ],
        *inputs["outer_registration_packets"],
    ]
    for key in ("registration", "results", "source_cycle", "source_implementation_freeze"):
        refs.append(inputs["child_source_record"][key])
    return unique_refs(refs)


def check_frozen():
    frozen = read_json(INTENT)
    for ref in (
        frozen["implementation_dependencies"]
        + frozen["source_references"]
        + frozen["validation_evidence"]
    ):
        base.checked_reference(ref)
    rules.require(
        frozen["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN],
        "Preserve the exact first frozen source-scoped allocation audit.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep the accepted full Python version and every captured package row.",
    )
    return frozen


def progress(phase, **fields):
    counts = source.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "An allocation-mass audit cannot alter authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "nested_growth_allocation_audit_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial nested-audit freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        rules.validate(read_json(rules.POLICY))
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            source.current_counts(base.journal_events(base.JOURNAL)) == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1320,
            "Freeze the audit against its actual unchanged J1320 baseline.",
        )
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, _ = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful source identity, exact algebra, overlap and prefix checks.",
        )
        refs = source_refs()
        for ref in refs:
            base.checked_reference(ref)
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "policy": reference(rules.POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "implementation_dependencies": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "actual_nested_expansion_or_concentration_computed": False,
                "new_financial_trials": 0,
                "source_graph_and_criteria_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def pairs(reference):
    with base.checked_reference(reference).open() as stream:
        while True:
            before = stream.readline()
            if not before:
                break
            after = stream.readline()
            rules.require(bool(after), "A partial source input/resolution pair cannot be rescued.")
            yield json.loads(before), json.loads(after)


def joined_records():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    for outer, child in zip(
        pairs(inputs["outer_input_resolved_tape"]),
        pairs(inputs["child_input_resolved_tape"]),
        strict=True,
    ):
        yield {
            "outer_before": outer[0],
            "outer_after": outer[1],
            "child_before": child[0],
            "child_after": child[1],
        }


def audit():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "audit-attempt-v1.json").exists(),
        "Preserve the first complete or partial actual nested audit instead of replaying it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(FOLDER / "audit-attempt-v1.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        calendar = sessions("2016-10-05", "2026-10-05")
        dates = [calendar[0], *[d for d in calendar if d > calendar[0] and is_month_end(d)]]
        rows, checks = [], []
        previous = [0.25] * 4
        log = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with log.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for i, record in enumerate(joined_records()):
                lifecycle.runtime_guard()
                rules.require(
                    i < len(dates)
                    and record["outer_before"]["decision_index"] == i
                    and record["outer_before"]["as_of"] == str(dates[i].date())
                    and record["outer_before"]["previous_weights"] == previous,
                    "Keep every original matched initial/month-end source origin.",
                )
                retain({"record": "nested_allocation_input", "source": record})
                current = rules.decompose(record, policy)
                checked = rules.verify(record, current, policy)
                rows.append(current)
                checks.append(checked)
                previous = record["outer_after"]["audit"]["rolling_expert_weights"]
                retain(
                    {
                        "record": "nested_allocation_resolved",
                        "audit": current,
                        "independent": checked,
                    }
                )
        rules.require(len(rows) == len(dates) == 121, "No omitted or additional source origins.")
        prefixes = []
        for position in (0, 6, 12, 60, 120):
            lifecycle.runtime_guard()
            for i, record in enumerate(islice(joined_records(), position + 1)):
                rules.require(
                    rules.decompose(record, policy) == rows[i],
                    "A later retained input changed a previous expansion or concentration.",
                )
            prefixes.append(str(dates[position].date()))
        counts = source.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"],
            "Readonly nesting cannot append a financial lifecycle event.",
        )
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "implementation_freeze": reference(INTENT),
                "summary": rules.summarize(rows, checks),
                "audit_rows": rows,
                "actual_input_resolved_records": reference(log),
                "actual_input_resolved_record_count": 2 * len(rows),
                "exact_prefixes": prefixes,
                "actual_authoritative_counts": counts,
                "new_financial_trials": 0,
                "source_algorithms_or_main_returns_recomputed": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_audit_complete",
            private_result=reference(RESULTS),
            implementation_freeze=reference(INTENT),
            new_financial_trials=0,
        )
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "summary": saved["summary"],
                "actual_input_resolved_record_count": saved["actual_input_resolved_record_count"],
                "exact_prefixes": saved["exact_prefixes"],
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "new_financial_trials": 0,
                "source_algorithms_or_main_returns_recomputed": False,
                "raw_prices_targets_returns_or_individual_allocation_rows_exported": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Readonly complete nested allocation-mass audit.")
    parser.add_argument("action", choices=("freeze", "audit", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "audit": audit, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
