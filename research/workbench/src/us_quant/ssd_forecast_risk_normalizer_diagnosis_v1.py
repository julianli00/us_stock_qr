from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import forecast_risk_normalizer_diagnosis_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_seven_expert_meta_v1 import unique_refs
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

ROOT, reference = base.ROOT, base.reference
FOLDER = base.REPORTS / "forecast-risk-normalizer-diagnosis-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/forecast_risk_normalizer_diagnosis_20261106_v1.json"
OWN = (
    "config/forecast-risk-normalizer-diagnosis-v1.json",
    "src/us_quant/forecast_risk_normalizer_diagnosis_v1.py",
    "src/us_quant/ssd_forecast_risk_normalizer_diagnosis_v1.py",
    "tests/test_forecast_risk_normalizer_diagnosis_v1.py",
    "FORECAST_RISK_NORMALIZER_DIAGNOSIS_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def packages():
    return [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]


def source_refs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    keys = (
        "source_allocation_tape",
        "source_allocation_freeze",
        "source_economic_policy",
        "source_completed_financial_result",
        "source_completed_cycle",
    )
    for key in keys:
        base.checked_reference(policy[key])
    source_freeze = read_json(base.checked_reference(policy["source_allocation_freeze"]))
    source_policy = read_json(base.checked_reference(policy["source_economic_policy"]))
    if (
        source_policy["switch_cost_amortization_sessions"] != 21
        or source_policy["selection_score_decimal_places"] != 12
        or source_policy["independent_score_tolerance"] != rules.TOLERANCE
    ):
        raise QuantError(
            "Diagnose the actual frozen score law, not another cost or risk convention."
        )
    refs = unique_refs(
        [
            *[policy[k] for k in keys],
            *source_freeze["complete_dependency_inputs"],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return refs


def source_audits():
    path = base.checked_reference(read_json(rules.POLICY)["source_allocation_tape"])
    with path.open() as stream:
        while line := stream.readline():
            resolved = stream.readline()
            if not resolved:
                raise QuantError("A partial saved allocation tape cannot be filled or skipped.")
            before, after = json.loads(line), json.loads(resolved)
            if (
                before["record"] != "saved_forecast_allocation_input"
                or after["record"] != "saved_forecast_allocation_resolved"
                or before["source_before"]["decision_index"] != after["audit"]["decision_index"]
                or before["source_before"]["as_of"] != after["audit"]["decision_session"]
            ):
                raise QuantError(
                    "Only complete original allocation input/result records are admitted."
                )
            yield after["audit"]


def progress(phase, **fields):
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "forecast_risk_normalizer_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **current_counts(base.journal_events(base.JOURNAL)),
            "order_authority": False,
        },
    )


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first sign-diagnosis freeze.")
    with lifecycle.research_job():
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 176855
            or design["new_real_sign_classifications_or_outcomes_computed"] is not False
            or design["alternative_rankings_or_new_targets"] != 0
        ):
            raise QuantError("Declare the bounded identity question before real classifications.")
        history = base.journal_events(base.JOURNAL)
        if (
            len(history) != 1274
            or history[-1]["sha256"] != read_json(rules.POLICY)["prior_journal_head_sha256"]
        ):
            raise QuantError("Keep the actual closed J1274/285/0 financial state.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or validation["synthetic_only"] is not True
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Bind the exact sign, zero/tie, independent numeric and prefix checks."
            )
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "policy": reference(rules.POLICY),
                "design": reference(FOLDER / "design-decision-v1.json"),
                "invocation": reference(FOLDER / "actual-invocation-v1.json"),
                "preservation_baseline": reference(FOLDER / "preservation-baseline-v1.json"),
                "complete_dependency_inputs": dependencies(),
                "source_references": source_refs(),
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "package_versions": packages(),
                "python_version": sys.version,
                "prior_journal_sequence": len(history),
                "prior_journal_head_sha256": history[-1]["sha256"],
                "new_real_sign_classifications_or_outcomes_computed": False,
                "alternative_rankings_targets_or_financial_trials": 0,
            },
        )
        progress(
            "frozen",
            diagnostic_freeze=reference(INTENT),
            next_step="one_readonly_all_pair_sign_identity_pass_no_alternative_allocation",
        )
    return reference(INTENT)


def check_frozen():
    frozen = read_json(INTENT)
    for key in ("complete_dependency_inputs", "source_references", "validation_evidence"):
        for ref in frozen[key]:
            base.checked_reference(ref)
    for key in ("policy", "design", "invocation", "preservation_baseline"):
        base.checked_reference(frozen[key])
    if frozen["package_versions"] != packages() or frozen["python_version"] != sys.version:
        raise QuantError("Keep the exact accepted SSD interpreter and package inventory.")
    history = base.journal_events(base.JOURNAL)
    if (
        len(history) != frozen["prior_journal_sequence"]
        or history[-1]["sha256"] != frozen["prior_journal_head_sha256"]
    ):
        raise QuantError("A read-only diagnosis cannot mutate any financial trial history.")
    return frozen


def run():
    lifecycle.runtime_guard()
    if RESULTS.exists() or (FOLDER / "analysis-attempt-v1.json").exists():
        raise QuantError("Never replay a completed or partial real sign diagnosis.")
    with lifecycle.research_job():
        check_frozen()
        write_new_receipt(
            FOLDER / "analysis-attempt-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "freeze": reference(INTENT)},
        )
        progress("running", active_compute_job={"pid": os.getpid(), "study": str(FOLDER)})
        start = time.monotonic()
        tape = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with tape.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            summary, decisions = rules.diagnose(
                source_audits(),
                retain=retain,
                before_case=lifecycle.runtime_guard,
            )
        if len(decisions) != 121:
            raise QuantError("Retain every one of the original121decision origins.")
        prefixes = []
        from itertools import islice

        for i in (0, 6, 12, 60, 120):
            lifecycle.runtime_guard()
            _, earlier = rules.diagnose(islice(source_audits(), i + 1))
            if earlier != decisions[: i + 1]:
                raise QuantError("Future records changed a prior sign identity classification.")
            prefixes.append(decisions[i]["decision_session"])
        count = 0
        with tape.open() as stream:
            for i, original in enumerate(source_audits()):
                before, after = json.loads(next(stream)), json.loads(next(stream))
                if before != {
                    "record": "normalizer_input",
                    "index": i,
                    "source_audit": original,
                } or after != {"record": "normalizer_resolved", "index": i, "result": decisions[i]}:
                    raise QuantError(
                        "The actual retained sign inputs/results disagree with source bytes."
                    )
                count += 2
            if stream.read():
                raise QuantError("Unexpected extra sign-diagnosis records cannot be ignored.")
        check_frozen()
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "freeze": reference(INTENT),
                "summary": summary,
                "decisions": decisions,
                "actual_input_resolved_records": reference(tape),
                "retained_record_count": count,
                "exact_prefixes": prefixes,
                "elapsed_seconds": time.monotonic() - start,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "financial_journal_changed": False,
                "alternative_rankings_targets_or_financial_trials": 0,
            },
        )
        saved = read_json(RESULTS)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "summary": summary,
                "implementation_freeze": reference(INTENT),
                "retained_record_count": count,
                "exact_prefixes": prefixes,
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "financial_journal_changed": False,
                "alternative_rankings_targets_or_financial_trials": 0,
                "raw_prices_source_forecasts_pair_rows_or_main_books_exported": False,
            },
        )
        progress(
            "complete_unpublished",
            result=reference(RESULTS),
            public_result=reference(PUBLIC),
            next_step="publish_complete_aggregate_identity_no_causal_drawdown_claim",
        )
    return reference(RESULTS)


def main():
    parser = argparse.ArgumentParser(description="Read-only signed risk-normalizer identity audit.")
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps(initialize() if args.action == "freeze" else run()))


if __name__ == "__main__":
    main()
