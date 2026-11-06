from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import expert_active_forecast_diagnosis_v1 as rules
from us_quant import expert_contextual_contest_v1 as contexts
from us_quant import ssd_eight_expert_meta_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import file_digest, read_json, utc_now, write_json

ROOT, base, lifecycle = source.ROOT, source.base, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "expert-active-forecast-diagnosis-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_active_forecast_diagnosis_20261106_v1.json"
OWN = (
    "config/expert-active-forecast-diagnosis-v1.json",
    "src/us_quant/expert_active_forecast_diagnosis_v1.py",
    "src/us_quant/ssd_expert_active_forecast_diagnosis_v1.py",
    "tests/test_expert_active_forecast_diagnosis_v1.py",
    "EXPERT_ACTIVE_FORECAST_DIAGNOSIS_V1.md",
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


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    if base.checked_reference(policy["canonical_source_policy"]) != source.rules.POLICY:
        raise QuantError("Observe the original eight, not the failed compact pool or new sleeves.")
    data, experts, frozen, _ = source.inputs()
    if (
        tuple(e.identity.identifier for e in experts) != source.rules.EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in experts) != source.rules.MECHANISMS
    ):
        raise QuantError("All eight original admitted experts must remain in this diagnosis.")
    original = read_json(source.INTENT)
    known_books = {v["path"]: v for v in frozen["benchmark_artifacts"]}
    benchmark_returns = {}
    for symbol in rules.BENCHMARKS:
        ref = policy["benchmark_books"][symbol]
        if known_books.get(ref["path"]) != ref:
            raise QuantError("Do not replace original funded NET benchmarks with price proxies.")
        book = source.seven.frame(ref)
        if (
            len(book) != 2512
            or str(book.index[0].date()) != "2016-10-06"
            or str(book.index[-1].date()) != "2026-10-05"
            or not book["risk_free"].equals(data.risk_free.loc[book.index])
        ):
            raise QuantError("Keep the full original tradable benchmark dates and RF convention.")
        benchmark_returns[symbol] = book["return"]
    refs = unique_refs(
        [
            *original["source_references"],
            *original["complete_dependency_inputs"],
            *frozen["code"],
            *frozen["policies"],
            *frozen["prior_evidence"],
            *frozen["validation_evidence"],
            *policy["benchmark_books"].values(),
            policy["canonical_source_policy"],
            reference(source.INTENT),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, pd.DataFrame(benchmark_returns), refs, frozen


def decision_inputs(data, experts, benchmarks):
    dates = source.rules.decision_sessions(data.close.index, read_json(source.rules.POLICY))
    for position, day in enumerate(dates):
        observed = [e.observe(day, 126) for e in experts]
        known = observed[0].net_returns.index
        yield {
            "record": "forecast_input",
            "index": position,
            "as_of": str(day.date()),
            "information_clock": source.rules.CLOCK,
            "context": contexts.context_inputs(data.close, day),
            "observations": [
                {
                    "id": o.identity.identifier,
                    "mechanism_group": o.identity.mechanism_group,
                    "introduced_at": o.identity.introduced_at,
                    "as_of": str(o.as_of.date()),
                    "target_available": o.target is not None,
                    "target_known_at": str(o.target_known_at.date())
                    if o.target_known_at is not None
                    else None,
                    "return_dates": o.net_returns.index.strftime("%Y-%m-%d").tolist(),
                    "NET_returns": o.net_returns.tolist(),
                    "RF": o.risk_free.tolist(),
                    "score_mature": o.score_mature,
                }
                for o in observed
            ],
            "benchmark_dates": known.strftime("%Y-%m-%d").tolist(),
            "benchmarks": {b: benchmarks.loc[known, b].tolist() for b in rules.BENCHMARKS},
            "binary64_values_truncated_to_fixed_decimals": False,
        }


def progress(phase, **fields):
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "active_forecast_diagnosis_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **current_counts(base.journal_events(base.JOURNAL)),
            "order_authority": False,
        },
    )


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    xml, _ = [base.checked_reference(v) for v in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
        or proof["synthetic_only"] is not True
        or not suites
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Bind the exact synthetic chronology, oracle and no-funding checks.")
    return [*proof["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first diagnostic freeze.")
    with lifecycle.research_job():
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 175233
            or design["new_real_contexts_labels_forecasts_or_outcomes_computed"] is not False
            or design["new_allocations_or_financial_trials"] != 0
        ):
            raise QuantError("Declare the monetary-rate question before any real new output.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, _, refs, parent = source_inputs()
        history = base.journal_events(base.JOURNAL)
        policy = read_json(rules.POLICY)
        if (
            len(history) != policy["prior_journal_sequence"]
            or history[-1]["sha256"] != policy["prior_journal_head_sha256"]
        ):
            raise QuantError("Freeze against the actual closed J1246 financial baseline.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "policy": reference(rules.POLICY),
                "design": reference(FOLDER / "design-decision-v1.json"),
                "invocation": reference(FOLDER / "actual-invocation-v1.json"),
                "preservation_baseline": reference(FOLDER / "preservation-baseline-v1.json"),
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "source_policy_and_expert_admission_clocks": parent[
                    "source_policy_and_expert_admission_clocks"
                ],
                "validation_evidence": validation_refs(),
                "package_versions": packages(),
                "python_version": sys.version,
                "prior_journal_sequence": len(history),
                "prior_journal_head_sha256": history[-1]["sha256"],
                "new_real_contexts_labels_forecasts_or_outcomes_computed": False,
                "new_allocations_or_financial_trials": 0,
            },
        )
        progress(
            "frozen",
            diagnostic_freeze=reference(INTENT),
            next_step="one_readonly_prequential_diagnosis_no_allocations_or_funds",
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


def verify_records(path, data, experts, benchmarks, audits):
    captured, values = [], 0
    with path.open() as stream:
        for position, expected in enumerate(decision_inputs(data, experts, benchmarks)):
            before, after = json.loads(next(stream)), json.loads(next(stream))
            if before != expected or after != {
                "record": "forecast_resolved",
                "index": position,
                "audit": audits[position],
            }:
                raise QuantError(
                    "Actual raw source/benchmark/context or resolved evidence changed."
                )
            values += sum(2 * len(o["NET_returns"]) for o in before["observations"])
            captured.append(before)
        if stream.read():
            raise QuantError("No unexplained extra retained forecast records are allowed.")
    return captured, {
        "actual_retained_records": 2 * len(captured),
        "source_NET_RF_binary64_values_checked": values,
    }


def run():
    lifecycle.runtime_guard()
    if RESULTS.exists() or (FOLDER / "analysis-attempt-v1.json").exists():
        raise QuantError("Never replay a complete or partial real forecast diagnosis.")
    with lifecycle.research_job():
        frozen = check_frozen()
        data, experts, benchmarks, refs, _ = source_inputs()
        if refs != frozen["source_references"]:
            raise QuantError("The frozen source closure or admitted views changed.")
        write_new_receipt(
            FOLDER / "analysis-attempt-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "freeze": reference(INTENT)},
        )
        progress("running", active_compute_job={"pid": os.getpid(), "study": str(FOLDER)})
        started = time.monotonic()
        path = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with path.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            summary, audits = rules.analyze(
                decision_inputs(data, experts, benchmarks),
                retain=retain,
                before_decision=lifecycle.runtime_guard,
            )
        captured, retained = verify_records(path, data, experts, benchmarks, audits)
        independent = rules.verify(captured, audits, summary)
        prefixes = []
        for i in (0, 6, 12, 60, 120):
            lifecycle.runtime_guard()
            if i < len(captured):
                _, earlier = rules.analyze(captured[: i + 1])
                if earlier != audits[: i + 1]:
                    raise QuantError("A future input changed an earlier prediction or error grade.")
                prefixes.append(captured[i]["as_of"])
        if len(audits) != 121 or summary["complete_learning_months"] != 119:
            raise QuantError("Preserve all original decisions and full exchange months.")
        check_frozen()
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "freeze": reference(INTENT),
                "actual_inputs_resolved": reference(path),
                "summary": summary,
                "verification": {**retained, **independent, "exact_prefixes": prefixes},
                "elapsed_seconds": time.monotonic() - started,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "financial_journal_changed": False,
                "new_allocations_or_financial_trials": 0,
                "source_models_signals_targets_or_accounts_replayed": False,
            },
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "actual_native_event_seq": 175233,
                "completed_at": read_json(RESULTS)["completed_at"],
                "implementation_freeze_sha256": file_digest(INTENT),
                "summary": summary,
                "verification": {**retained, **independent, "exact_prefixes": prefixes},
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "raw_prices_daily_books_or_forecast_rows_exported": False,
                "financial_journal_changed": False,
                "new_allocations_or_financial_trials": 0,
                "all_models_and_all_eight_sources_reported": True,
                "source_native_RF_identity_claimed_without_admitted_views": False,
                "global_pool_changed": False,
                "new_source_requests": 0,
                "new_RF_views": 0,
                "source_models_signals_targets_or_accounts_replayed": False,
            },
        )
        progress(
            "complete_unpublished",
            result=reference(RESULTS),
            public_result=reference(PUBLIC),
            next_step="publish_sanitized_diagnostic_and_close_no_financial_trials",
        )
    return reference(RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only complete-expert active-rate prediction."
    )
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps(initialize() if args.action == "freeze" else run()))


if __name__ == "__main__":
    main()
