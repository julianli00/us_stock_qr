from __future__ import annotations

import argparse
import json
import os
import time

from us_quant import anfci_inputs_v2 as source
from us_quant import ssd_anfci_acquisition_v1 as transport
from us_quant import ssd_commitment_research_v1 as common
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now

base, original = source.base, source.original
FREEZE = source.DATA / "source-contract-freeze.json"
MANIFEST = source.DATA / "manifest.json"
FAILURE = source.DATA / "source-failure.json"
PRIVATE = source.REPORTS / "validation.json"


def frozen_inputs() -> dict:
    policy, correction = source.settings()
    original.checked_plan(policy)
    paths = [
        source.POLICY,
        source.ROOT / "src/us_quant/anfci_inputs_v2.py",
        source.ROOT / "src/us_quant/ssd_anfci_acquisition_v2.py",
        source.ROOT / "tests/test_anfci_inputs_v2.py",
        source.ROOT / "tests/test_ssd_anfci_acquisition_v2.py",
        source.REPORTS / "offline-tests-v2.xml",
        source.REPORTS / "offline-tests-v2.log",
    ]
    refs = [base.reference(path) for path in paths] + [
        correction[name]
        for name in (
            "original_contract",
            "original_failure",
            "diagnostic",
            "retained_attempt",
            "retained_request",
            "retained_response",
        )
    ]
    for ref in refs:
        base.checked_reference(ref)
    return {
        "source_id": correction["source_id"],
        "inputs": refs,
        "original_source_contract": correction["original_contract"],
        "original_source_failure": correction["original_failure"],
        "technical_missing_encoding_correction_only": True,
        "first_archive_predates_v2_correction": True,
        "first_archive_redownload_allowed": False,
        "remaining_requests_require_this_freeze": True,
        "source_data_timing_freshness_or_numeric_parameters_changed": False,
        "ANFCI_strategy_outcomes_computed": False,
    }


def check_freeze() -> dict:
    value = read_json(FREEZE)
    if {key: item for key, item in value.items() if key != "created_at"} != frozen_inputs():
        raise QuantError("The separately frozen v2 correction changed.")
    return value


def prepare() -> dict:
    runtime_guard()
    with research_job(source.ROOT / "runtime/ssd-continuous-research-v1.lock"):
        expected = frozen_inputs()
        if FREEZE.exists():
            check_freeze()
            return base.reference(FREEZE)
        new_output_directory(source.DATA)
        write_new_receipt(FREEZE, {"created_at": utc_now(), **expected})
        return base.reference(FREEZE)


def retained_first(correction: dict, dates: list[str], policy: dict) -> tuple[dict, dict]:
    attempt = read_json(base.checked_reference(correction["retained_attempt"]))
    receipt = read_json(base.checked_reference(correction["retained_request"]))
    if (
        attempt["dates"] != dates
        or receipt["exchanges"][-1]["response_body"] != correction["retained_response"]
    ):
        raise QuantError("Reuse exactly the first successful source response, not another cohort.")
    original.verify_request(receipt, dates, policy, attempt["created_at"])
    base.checked_reference(correction["retained_response"])
    return attempt, receipt


def collect() -> dict:
    runtime_guard()
    with research_job(source.ROOT / "runtime/ssd-continuous-research-v1.lock"):
        check_freeze()
        policy, correction = source.settings()
        plan = original.checked_plan(policy)
        if MANIFEST.exists():
            return base.reference(MANIFEST)
        if FAILURE.exists():
            raise QuantError("The v2 source attempt failed; no automatic retry.")
        before = base.journal_events(base.JOURNAL)
        if len(before) != 190 or before[-1]["sha256"] != transport.JOURNAL_HEAD:
            raise QuantError("Preserve all original research events.")
        values, batches = {}, []
        common.progress(
            "acquiring_anfci_native_csv_v2",
            active_compute_job={"pid": os.getpid(), "kind": "ANFCI_source_v2"},
        )
        try:
            for number, dates in enumerate(plan["batches"], 1):
                runtime_guard()
                check_freeze()
                if number == 1:
                    attempt, receipt = retained_first(correction, dates, policy)
                    receipt_ref = correction["retained_request"]
                else:
                    directory = source.DATA / f"batch-{number:02d}"
                    attempt_path, receipt_path = (
                        directory / "attempt.json",
                        directory / "request-complete.json",
                    )
                    if attempt_path.exists():
                        attempt = read_json(attempt_path)
                        if not receipt_path.exists():
                            raise QuantError(
                                "A prior v2 source attempt is incomplete or failed; no retry."
                            )
                        if attempt["dates"] != dates or attempt["freeze"] != base.reference(FREEZE):
                            raise QuantError("Preserve the exact source request and v2 freeze.")
                        receipt = read_json(receipt_path)
                    else:
                        new_output_directory(directory)
                        attempt = {
                            "created_at": utc_now(),
                            "batch": number,
                            "dates": dates,
                            "freeze": base.reference(FREEZE),
                            "expected_form_fields": original.form_fields(dates, policy),
                            "source_POST_attempt_budget": 1,
                        }
                        write_new_receipt(attempt_path, attempt)
                        try:
                            receipt = transport.fetch_one(directory, dates, policy)
                            write_new_receipt(receipt_path, receipt)
                        except (QuantError, OSError, ValueError) as error:
                            write_new_receipt(
                                directory / "attempt-failed.json",
                                {
                                    "at": utc_now(),
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "attempt": base.reference(attempt_path),
                                    "partial_responses_retained": True,
                                    "automatic_retry_performed": False,
                                },
                            )
                            raise
                    receipt_ref = base.reference(receipt_path)
                request_proof = original.verify_request(
                    receipt, dates, policy, attempt["created_at"]
                )
                for exchange in receipt["exchanges"]:
                    base.checked_reference(exchange["response_body"])
                response = receipt["exchanges"][-1]["response_body"]
                frame, proof = source.parse_archive(base.checked_reference(response), dates, policy)
                for day in dates:
                    if day in values:
                        raise QuantError("A required vintage was duplicated.")
                    values[day] = frame[day].dropna().iloc[-1:]
                item = {
                    "batch": number,
                    "dates": dates,
                    "request": receipt_ref,
                    "response": response,
                    "request_proof": request_proof,
                    "parse_proof": proof,
                    "reused_successful_v1_archive_without_request": number == 1,
                }
                common.retain_receipt(source.DATA / f"batch-{number:02d}-validated.json", item)
                batches.append(item)
                common.progress(
                    "acquiring_anfci_native_csv_v2",
                    completed_anfci_batches=number,
                    active_compute_job={"pid": os.getpid(), "kind": "ANFCI_source_v2"},
                )
                if number < len(plan["batches"]):
                    time.sleep(2)
            if sorted(values) != plan["vintage_cutoffs"]:
                raise QuantError("All225 original vintages remain required.")
            assessment = original.assess_queries(values, plan["query_sessions"], policy)
            if assessment["query_count"] != 225:
                raise QuantError("No original query can be omitted.")
            check_freeze()
            if (
                base.journal_events(base.JOURNAL) != before
                or file_digest(source.ROOT / "runtime/research-program.sqlite3")
                != base.LEGACY_SHA256
            ):
                raise QuantError("Source correction altered protected research history.")
            write_new_receipt(
                PRIVATE,
                {
                    "created_at": utc_now(),
                    "freeze": base.reference(FREEZE),
                    "batches": batches,
                    **assessment,
                    "source_admitted": assessment["all_queries_fresh"],
                    "first_archive_reused": True,
                    "new_v2_POSTs": 5,
                    "total_unique_source_POSTs": 6,
                    "ANFCI_strategy_outcomes_computed": False,
                    "new_strategy_evaluations": 0,
                },
            )
            write_new_receipt(
                MANIFEST,
                {
                    "created_at": utc_now(),
                    "source_id": correction["source_id"],
                    "policy": base.reference(source.POLICY),
                    "freeze": base.reference(FREEZE),
                    "validation": base.reference(PRIVATE),
                    "batches": batches,
                    "source_admitted": assessment["all_queries_fresh"],
                    "original_contract_failure_and_raw_response_preserved": True,
                    "original_ledger_and_journal_unchanged": True,
                    "raw_inputs_remain_private": True,
                    "new_strategy_evaluations": 0,
                },
            )
            write_new_receipt(
                source.PUBLIC,
                {
                    "schema_version": 1,
                    "source_id": correction["source_id"],
                    "created_at": utc_now(),
                    "source": policy["source"],
                    "series": policy["series"],
                    "units": policy["units"],
                    "frequency": policy["frequency"],
                    "policy": base.reference(source.POLICY),
                    "original_contract": correction["original_contract"],
                    "original_failure": correction["original_failure"],
                    "freeze": base.reference(FREEZE),
                    "private_manifest": base.reference(MANIFEST),
                    "actual_archive_count": len(batches),
                    "actual_vintage_count": len(values),
                    "unique_source_POSTs": 6,
                    "reused_v1_archive_count": 1,
                    "new_v2_POSTs": 5,
                    "first_archive_redownloaded": False,
                    "native_missing_tokens": list(source.MISSING),
                    "finite_numeric_values_changed": False,
                    "fresh_query_count": assessment["fresh_query_count"],
                    "stale_query_count": assessment["stale_query_count"],
                    "maximum_observation_age_calendar_days": assessment[
                        "maximum_observation_age_calendar_days"
                    ],
                    "freshness_limit_calendar_days": 14,
                    "stale_query_metadata": assessment["stale_queries"],
                    "archive_integrity_and_exact_vintage_coverage_verified": True,
                    "source_admitted": assessment["all_queries_fresh"],
                    "raw_values_published": False,
                    "original_ledger_and_journal_unchanged": True,
                    "new_strategy_evaluations": 0,
                    "new_factor_definitions": 0,
                    "strategy_rule_selected": False,
                    "order_authority": False,
                },
            )
            common.progress(
                "anfci_native_csv_validation_complete",
                active_compute_job=None,
                anfci_readiness=base.reference(source.PUBLIC),
                current_research="ANFCI vintage validation complete;no strategy selected",
                pending_steps=["report_actual_anfci_readiness", "select_supported_next_question"],
            )
            return base.reference(source.PUBLIC)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            if not FAILURE.exists():
                write_new_receipt(
                    FAILURE,
                    {
                        "failed_at": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "freeze": base.reference(FREEZE),
                        "completed_batches": len(batches),
                        "all_partial_outputs_retained": True,
                        "automatic_retry_allowed": False,
                        "source_admitted": False,
                        "new_strategy_evaluations": 0,
                    },
                )
            common.progress(
                "anfci_native_csv_blocked",
                active_compute_job=None,
                source_failure=str(error),
                source_partial_outputs_preserved=True,
            )
            raise
        except KeyboardInterrupt:
            common.progress(
                "anfci_native_csv_interrupted",
                active_compute_job=None,
                source_partial_outputs_preserved=True,
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Versioned native ANFCI CSV correction;no trading."
    )
    parser.add_argument("command", choices=("freeze", "collect"))
    args = parser.parse_args()
    try:
        print(json.dumps(prepare() if args.command == "freeze" else collect(), indent=2))
    except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"ANFCI v2 source operation blocked: {error}\n")


if __name__ == "__main__":
    main()
