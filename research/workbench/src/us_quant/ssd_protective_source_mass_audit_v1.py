from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from itertools import islice
from types import FunctionType
from xml.etree import ElementTree

from us_quant import protective_source_mass_audit_v1 as rules
from us_quant import ssd_protective_six_adaptive_hedge_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "protective-source-mass-audit-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/protective_source_mass_audit_20261106_v1.json"
OWN = (
    "config/protective-source-mass-audit-v1.json",
    "src/us_quant/protective_source_mass_audit_v1.py",
    "src/us_quant/ssd_protective_source_mass_audit_v1.py",
    "tests/test_protective_source_mass_audit_v1.py",
    "tests/conftest.py",
    "PROTECTIVE_SOURCE_MASS_AUDIT_V1.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    closed = read_json(base.checked_reference(inputs["source_completed_cycle"]))
    failed = read_json(FOLDER / "validation-source-v1.json")
    rules.require(failed["exit_code"] == 1, "Preserve the original failed synthetic batch.")
    refs = [
        *closed["exact_evidence_references"],
        reference(FOLDER / "validation-source-v1.json"),
        *failed["test_outputs"],
        *failed["versioned_prototypes"],
        *[
            reference(FOLDER / name)
            for name in (
                "source-inputs-v1.json",
                "design-decision-v1.json",
                "actual-invocation-v1.json",
                "preservation-baseline-v1.json",
                "capture-preflight-v1.py",
                "run-analysis-v1.py",
            )
        ],
        *[
            value
            for value in inputs.values()
            if isinstance(value, dict) and "path" in value and "sha256" in value
        ],
        *inputs["prior_allocation_decomposition"],
    ]
    rules.require(
        inputs["retained_source_clocks_identical"] is True
        and inputs["five_source_clocks"] == inputs["six_source_clocks"][:5]
        and inputs["five_expert_ids"] == inputs["six_expert_ids"][:5]
        and inputs["six_expert_ids"][-1] == "semiconductor_ratchet_protection"
        and inputs["new_allocation_decomposition_labels_computed"] is False
        and inputs["source_learning_targets_or_accounts_replayed"] is False,
        "Keep the exact admitted source scopes and the controlled five-plus-one order.",
    )
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
        "Preserve the first frozen paired allocation audit.",
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
        all(key not in fields or fields[key] == value for key, value in counts.items()),
        "A readonly allocation audit cannot change authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "protective_source_mass_audit_" + phase,
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
        "Preserve the first complete or partial audit freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        rules.validate(read_json(rules.POLICY))
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            source.current_counts(base.journal_events(base.JOURNAL)) == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1464,
            "Freeze the diagnostic against unchanged J1464/301completed/zero qualifiers.",
        )
        validation = read_json(FOLDER / "validation-source-v2.json")
        xml, _ = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(key, 0)) for s in suites for key in ("failures", "errors", "skipped")
            ),
            "Bind successful exact algebra,clock,maturity,source and prefix checks.",
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
                    reference(FOLDER / "validation-source-v2.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "actual_allocation_decomposition_labels_computed": False,
                "new_financial_trials": 0,
                "source_clock_or_financial_criteria_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def pairs(ref):
    with base.checked_reference(ref).open() as stream:
        while True:
            before = stream.readline()
            if not before:
                break
            after = stream.readline()
            rules.require(bool(after), "A partial source input/resolution pair cannot be rescued.")
            yield json.loads(before), json.loads(after)


def joined_records():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    for index, (five, six) in enumerate(
        zip(
            pairs(inputs["five_input_resolved_tape"]),
            pairs(inputs["six_input_resolved_tape"]),
            strict=True,
        )
    ):
        yield {
            "decision_index": index,
            "five_before": five[0],
            "five_after": five[1],
            "six_before": six[0],
            "six_after": six[1],
        }


def audit():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "audit-attempt-v1.json").exists(),
        "Preserve the first complete or partial actual audit instead of replaying it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["implementation_freeze"] == reference(INTENT)
            and start["actual_counts_before_child"] == frozen["actual_authoritative_counts"]
            and start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "A real saved preanalysis checkpoint must precede new diagnostic labels.",
        )
        write_new_receipt(
            FOLDER / "audit-attempt-v1.json",
            {
                "started_at": utc_now(),
                "preexisting_actual_start": reference(FOLDER / "actual-analysis-start-v1.json"),
            },
        )
        policy = read_json(rules.POLICY)
        rows, checks = [], []
        previous = {"five": None, "six": None}
        log = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with log.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for index, record in enumerate(joined_records()):
                lifecycle.runtime_guard()
                rules.require(index < 121, "No extra source origin can enter the audit.")
                for name in previous:
                    before, after = record[name + "_before"], record[name + "_after"]
                    if previous[name] is not None:
                        rules.require(
                            before["state"] == previous[name]["adaptive_state"]
                            and [float(v).hex() for v in previous[name]["expert_weights"]]
                            == before["previous_weights_hex"],
                            "A source state transition cannot reset or skip history.",
                        )
                    previous[name] = after["audit"]
                retain({"record": "matched_five_six_allocation_input", "source": record})
                current = rules.decompose(record, policy)
                checked = rules.verify(record, current, policy)
                rows.append(current)
                checks.append(checked)
                retain(
                    {
                        "record": "allocation_identity_resolved",
                        "audit": current,
                        "independent": checked,
                    }
                )
        rules.require(len(rows) == 121, "Keep every original initial/month-end source origin.")
        prefixes = []
        for position in (0, 2, 12, 60, 120):
            lifecycle.runtime_guard()
            for index, record in enumerate(islice(joined_records(), position + 1)):
                rules.require(
                    digest_json(rules.decompose(record, policy)) == digest_json(rows[index]),
                    "Future retained inputs changed a previous signed allocation identity.",
                )
            prefixes.append(rules.decision_dates()[position])
        counts = source.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"],
            "Readonly diagnosis cannot append a financial lifecycle event.",
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
                "matched_original_source_record_count": 4 * len(rows),
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
                "matched_original_source_record_count": saved[
                    "matched_original_source_record_count"
                ],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Readonly controlled source allocation-mass audit."
    )
    parser.add_argument("action", choices=("freeze", "audit", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "audit": audit, "export": export}[args.action]()))
