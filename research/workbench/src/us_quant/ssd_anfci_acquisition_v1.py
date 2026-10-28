from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit

from us_quant import anfci_inputs_v1 as source
from us_quant import ssd_commitment_research_v1 as common
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
)

base = source.base
FREEZE = source.DATA / "source-contract-freeze.json"
MANIFEST = source.DATA / "manifest.json"
FAILURE = source.DATA / "source-failure.json"
PRIVATE = source.REPORTS / "validation.json"
JOURNAL_HEAD = "9109b83e7425b310895446f2b9fdeb344003805cd25cd15d1efe7b049196db71"


class NoAutomaticRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def frozen_inputs() -> dict:
    policy = read_json(source.POLICY)
    source.checked_plan(policy)
    boundary = source.REPORTS / "browser-preflight-boundary.json"
    if file_digest(boundary) != "4ffa9958e6a98b9dd1b4aa7cec8afa78b783a4e5db7b9fbeddb665f48ee91783":
        raise QuantError("Preserve the actual pre-data browser failure and transport decision.")
    paths = [
        source.POLICY,
        Path(__file__),
        source.ROOT / "src/us_quant/anfci_inputs_v1.py",
        source.ROOT / "tests/test_anfci_inputs_v1.py",
        source.ROOT / "tests/test_ssd_anfci_acquisition_v1.py",
        source.REPORTS / "offline-tests-v1.xml",
        source.REPORTS / "offline-tests-v1.log",
        boundary,
    ]
    refs = [base.reference(path) for path in paths] + [
        policy[name] for name in ("source_review", "query_plan", "form_contract")
    ]
    for ref in refs:
        base.checked_reference(ref)
    return {
        "source_id": policy["source_id"],
        "inputs": refs,
        "python_version": list(sys.version_info[:3]),
        "old_ledger_sha256": base.LEGACY_SHA256,
        "journal_event_count_before_acquisition": 190,
        "journal_head_before_acquisition": JOURNAL_HEAD,
        "source_observations_acquired_before_freeze": False,
        "transport": "anonymous_public_form_http",
        "no_custom_UA_proxy_TLS_or_credentials": True,
        "source_refusal_bypassed": False,
        "new_strategy_evaluations": 0,
    }


def prepare() -> dict:
    runtime_guard()
    with research_job(source.ROOT / "runtime/ssd-continuous-research-v1.lock"):
        expected = frozen_inputs()
        if FREEZE.exists():
            saved = read_json(FREEZE)
            if {key: value for key, value in saved.items() if key != "created_at"} != expected:
                raise QuantError("A frozen ANFCI source contract cannot change.")
            return base.reference(FREEZE)
        events = base.journal_events(base.JOURNAL)
        if (
            len(events) != 190
            or events[-1]["sha256"] != expected["journal_head_before_acquisition"]
        ):
            raise QuantError("The pre-acquisition research history changed.")
        if file_digest(source.ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
            raise QuantError("The original research ledger changed.")
        new_output_directory(source.DATA)
        write_new_receipt(FREEZE, {"created_at": utc_now(), **expected})
        return base.reference(FREEZE)


def check_freeze() -> dict:
    value = read_json(FREEZE)
    if {key: item for key, item in value.items() if key != "created_at"} != frozen_inputs():
        raise QuantError("Source acquisition code, tests or policy changed after freezing.")
    return value


def fetch_one(directory: Path, dates: list[str], policy: dict, opener=None) -> dict:
    opener = opener or urllib.request.build_opener(NoAutomaticRedirect())
    encoded = urlencode(source.form_fields(dates, policy))
    url, method, body = policy["url"], "POST", encoded.encode()
    exchanges = []
    for hop in range(policy["maximum_same_origin_download_redirects"] + 1):
        runtime_guard()
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/x-www-form-urlencoded"} if body else {},
        )
        started = utc_now()
        try:
            response = opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        path = directory / f"response-{hop:02d}.bin"
        with response, path.open("xb") as stream:
            status = response.status
            headers = {key.lower(): value for key, value in response.headers.items()}
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                stream.write(chunk)
                total += len(chunk)
                if total > policy["maximum_archive_bytes"]:
                    raise QuantError(
                        "Source response exceeded the frozen byte limit; partial kept."
                    )
        exchange = {
            "started_at": started,
            "completed_at": utc_now(),
            "url": request.full_url,
            "method": request.get_method(),
            "encoded_body": request.data.decode() if request.data is not None else None,
            "status": status,
            "headers": headers,
            "response_body": base.reference(path),
            "response_bytes": total,
        }
        write_new_receipt(directory / f"exchange-{hop:02d}.json", exchange)
        exchanges.append(exchange)
        if status == 200:
            return {"exchanges": exchanges, "automatic_retries": 0}
        if status not in policy["allowed_redirect_statuses"]:
            raise QuantError(f"ANFCI source HTTP{status}; stop without retry or source workaround.")
        destination = urljoin(url, headers.get("location", ""))
        parsed = urlsplit(destination)
        if parsed.scheme != "https" or parsed.netloc != "alfred.stlouisfed.org":
            raise QuantError("A source redirect leaves the original secure origin; stop.")
        url, method, body = destination, "GET", None
    raise QuantError("The source exceeded the frozen same-origin download redirect limit.")


def collect() -> dict:
    runtime_guard()
    with research_job(source.ROOT / "runtime/ssd-continuous-research-v1.lock"):
        contract = check_freeze()
        policy = read_json(source.POLICY)
        plan = source.checked_plan(policy)
        if MANIFEST.exists():
            return base.reference(MANIFEST)
        if FAILURE.exists():
            raise QuantError(
                "This source attempt failed; preserve its evidence without automatic retry."
            )
        before = base.journal_events(base.JOURNAL)
        if (
            len(before) != 190
            or before[-1]["sha256"] != contract["journal_head_before_acquisition"]
        ):
            raise QuantError("The source study must preserve all existing research events.")
        common.progress(
            "acquiring_anfci_vintages",
            active_compute_job={"pid": os.getpid(), "kind": "ANFCI_source_only"},
        )
        values, batches = {}, []
        try:
            for number, dates in enumerate(plan["batches"], 1):
                runtime_guard()
                check_freeze()
                directory = source.DATA / f"batch-{number:02d}"
                attempt_path = directory / "attempt.json"
                receipt_path = directory / "request-complete.json"
                if attempt_path.exists():
                    attempt = read_json(attempt_path)
                    if not receipt_path.exists():
                        raise QuantError(
                            "A prior source attempt is incomplete or failed; no retry."
                        )
                    if attempt["dates"] != dates or attempt["freeze"] != base.reference(FREEZE):
                        raise QuantError(
                            "Preserve the original source attempt and requested dates."
                        )
                    receipt = read_json(receipt_path)
                else:
                    new_output_directory(directory)
                    attempt = {
                        "created_at": utc_now(),
                        "batch": number,
                        "dates": dates,
                        "freeze": base.reference(FREEZE),
                        "expected_form_fields": source.form_fields(dates, policy),
                        "source_POST_attempt_budget": 1,
                    }
                    write_new_receipt(attempt_path, attempt)
                    try:
                        receipt = fetch_one(directory, dates, policy)
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
                request_proof = source.verify_request(receipt, dates, policy, attempt["created_at"])
                for exchange in receipt["exchanges"]:
                    base.checked_reference(exchange["response_body"])
                response = receipt["exchanges"][-1]["response_body"]
                frame, proof = source.parse_archive(base.checked_reference(response), dates, policy)
                for day in dates:
                    if day in values:
                        raise QuantError("A requested vintage is duplicated between batches.")
                    actual = frame[day].dropna()
                    values[day] = actual.iloc[-1:]
                item = {
                    "batch": number,
                    "dates": dates,
                    "request": base.reference(receipt_path),
                    "response": response,
                    "request_proof": request_proof,
                    "parse_proof": proof,
                }
                common.retain_receipt(directory / "validated.json", item)
                batches.append(item)
                common.progress(
                    "acquiring_anfci_vintages",
                    completed_anfci_batches=number,
                    active_compute_job={"pid": os.getpid(), "kind": "ANFCI_source_only"},
                )
                if number < len(plan["batches"]):
                    time.sleep(2)
            if sorted(values) != plan["vintage_cutoffs"]:
                raise QuantError("The complete225-vintage cohort is required without substitution.")
            assessment = source.assess_queries(values, plan["query_sessions"], policy)
            if assessment["query_count"] != 225:
                raise QuantError("Do not drop any original information query.")
            check_freeze()
            if (
                base.journal_events(base.JOURNAL) != before
                or file_digest(source.ROOT / "runtime/research-program.sqlite3")
                != base.LEGACY_SHA256
            ):
                raise QuantError("Source acquisition altered protected research history.")
            write_new_receipt(
                PRIVATE,
                {
                    "created_at": utc_now(),
                    "freeze": base.reference(FREEZE),
                    "batches": batches,
                    **assessment,
                    "source_admitted": assessment["all_queries_fresh"],
                    "new_strategy_evaluations": 0,
                    "new_factor_definitions": 0,
                    "strategy_rule_selected": False,
                    "order_authority": False,
                },
            )
            write_new_receipt(
                MANIFEST,
                {
                    "created_at": utc_now(),
                    "source_id": policy["source_id"],
                    "policy": base.reference(source.POLICY),
                    "freeze": base.reference(FREEZE),
                    "validation": base.reference(PRIVATE),
                    "batches": batches,
                    "source_admitted": assessment["all_queries_fresh"],
                    "original_ledger_and_journal_unchanged": True,
                    "new_strategy_evaluations": 0,
                    "raw_inputs_remain_private": True,
                },
            )
            write_new_receipt(
                source.PUBLIC,
                {
                    "schema_version": 1,
                    "source_id": policy["source_id"],
                    "created_at": utc_now(),
                    "source": policy["source"],
                    "series": policy["series"],
                    "units": policy["units"],
                    "frequency": policy["frequency"],
                    "policy": base.reference(source.POLICY),
                    "freeze": base.reference(FREEZE),
                    "private_manifest": base.reference(MANIFEST),
                    "actual_archive_count": len(batches),
                    "actual_vintage_count": len(values),
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
                "anfci_source_validation_complete",
                active_compute_job=None,
                anfci_readiness=base.reference(source.PUBLIC),
                current_research="Actual ANFCI archive validation complete;no strategy selected",
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
                "anfci_source_blocked",
                active_compute_job=None,
                source_failure=str(error),
                source_partial_outputs_preserved=True,
            )
            raise
        except KeyboardInterrupt:
            common.progress(
                "anfci_source_interrupted",
                active_compute_job=None,
                source_partial_outputs_preserved=True,
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="One frozen ANFCI source acquisition;no trading.")
    parser.add_argument("command", choices=("freeze", "collect"))
    args = parser.parse_args()
    try:
        result = prepare() if args.command == "freeze" else collect()
        print(json.dumps(result, indent=2, allow_nan=False))
    except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"ANFCI source operation blocked: {error}\n")


if __name__ == "__main__":
    main()
