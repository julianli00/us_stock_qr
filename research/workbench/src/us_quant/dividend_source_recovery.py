from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from us_quant.candidate_receipts import input_reference, write_new_receipt
from us_quant.config import QuantError
from us_quant.dividend_growth_sources import history_from_payload, validate_policy as validate_source_policy
from us_quant.research_program import safe_file
from us_quant.research_weekly import verify_publication
from us_quant.storage import digest_json, file_digest, new_output_directory, read_json, utc_now, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = "config/dividend-source-recovery.json"
REFERENCES = ("source_policy", "source_validator", "original_source_manifest", "original_source_evidence")


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "bounded_dividend_source_recovery_20261012",
        "scope": "historical_source_acquisition_only", "directory": "data/dividend-source-recovery-20261012",
        "registration": "evidence/dividend_source_recovery_20261012_registration.json", "symbols": ["VIG", "SDY"],
        "minimum_cooldown_hours": 24, "maximum_new_requests_per_invocation": 1,
        "maximum_attempts_per_symbol": 1, "maximum_total_new_requests": 2,
        "stop_after_any_refusal_transport_failure_or_invalid_payload": True,
        "original_url_method_headers_and_curl_transport_unchanged": True,
        "capture_response_headers_without_changing_request_headers": True,
        "original_retry_after_header_was_not_captured": True,
        "local_24hour_gate_claimed_provider_retry_authorization": False,
        "data_start": "2015-08-10", "as_of": "2026-10-05", "required_market_sessions": 2805,
        "earliest_separate_hypothesis_cycle": "2026-10-24T09:00:00+08:00",
        "source_history_is_forward_observation": False, "new_registered_factor_definitions": 0,
        "new_registered_candidates": 0, "new_strategy_evaluations": 0,
        "portfolio_states_targets_or_returns_computed": False, "existing_W42_or_forward_rules_changed": False,
        "automatic_live_deployment": False, "investment_objective_verified": False, "order_authority": False,
    }
    if (not isinstance(policy, dict)
        or any(policy.get(key) != value or type(value) is bool and policy.get(key) is not value
               for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)):
        raise QuantError("Keep the fixed two-fund source-only scope,one-request budget,cooldown and stop-on-refusal boundary.")


def moment(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise QuantError("Source recovery timestamps must be actual timezone-aware instants.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QuantError("Source recovery timestamps must be actual timezone-aware instants.")
    return parsed.astimezone(timezone.utc)


def fingerprint(root: Path) -> str:
    return digest_json({
        name: file_digest(safe_file(root, name))
        for name in ("src/us_quant/dividend_source_recovery.py", "src/us_quant/dividend_growth_sources.py",
                     "src/us_quant/candidate_receipts.py", "src/us_quant/research_weekly.py",
                     "src/us_quant/data.py", "src/us_quant/multifactor_stability.py",
                     "src/us_quant/calendar.py", "src/us_quant/storage.py")
    })


def original_inputs(root: Path, policy: dict) -> tuple[dict, dict, list[dict]]:
    validate_policy(policy)
    inputs = {name: read_json(safe_file(root, policy[name]["path"], policy[name]["sha256"])) for name in REFERENCES
              if name != "source_validator"}
    safe_file(root, policy["source_validator"]["path"], policy["source_validator"]["sha256"])
    source_policy, manifest, evidence = (inputs[name] for name in
                                        ("source_policy", "original_source_manifest", "original_source_evidence"))
    validate_source_policy(source_policy)
    if (manifest["policy_sha256"] != policy["source_policy"]["sha256"]
        or manifest["source_validator_sha256"] != policy["source_validator"]["sha256"]
        or evidence["source_manifest_sha256"] != policy["original_source_manifest"]["sha256"]
        or evidence["all_required_fund_price_histories_ready"] is not False):
        raise QuantError("Preserve the original authenticated dividend source failure rather than rewriting readiness.")
    parent = Path(policy["original_source_manifest"]["path"]).parent
    for name, digest in manifest["source_files"].items():
        safe_file(root, (parent/name).as_posix(), digest)
    requests = []
    for symbol in policy["symbols"]:
        path = safe_file(root, (parent/f"{symbol}-raw.request.json").as_posix(),
                         manifest["source_files"][f"{symbol}-raw.request.json"])
        old = read_json(path)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?" + urlencode({
            "period1": int(pd.Timestamp(policy["data_start"], tz="UTC").timestamp()),
            "period2": int((pd.Timestamp(policy["as_of"], tz="UTC")+pd.Timedelta(days=1)).timestamp()),
            "interval": "1d", "events": "div,splits",
        })
        if (old["url"] != url or old["method"] != "GET" or old["advertised_form_fields"] is not None
            or old["http_success"] is not False or old["response"]["http_code"] != 429
            or old["exit_code"] != 0 or old["access_gates_bypassed"] is not False):
            raise QuantError("Recovery may only reuse the exact ordinary429request,not another endpoint or access refusal.")
        requests.append({"symbol": symbol, "url": url, "old_finished_at": old["finished_at"],
                         "old_request": {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}})
    return source_policy, manifest, requests


def register(root: Path = ROOT) -> dict:
    policy, policy_ref = input_reference(root, root/POLICY_PATH)
    _, _, requests = original_inputs(root, policy)
    output = root/policy["directory"]
    if output.exists() or (root/policy["registration"]).exists():
        raise QuantError("Preserve an existing source-only recovery registration or partial directory.")
    deadline = max(moment(row["old_finished_at"]) for row in requests)+timedelta(hours=24)
    registration = {
        "schema_version": 1, "registered_at": utc_now(), "policy": policy, "policy_reference": policy_ref,
        "controller_sha256": fingerprint(root), "source_requests": requests,
        "first_request_not_before_utc": deadline.isoformat(), "provider_retry_permission_certified": False,
        "new_requests_at_registration": 0, "new_strategy_evaluations": 0, "order_authority": False,
    }
    new_output_directory(output)
    write_new_receipt(output/"registration.json", registration)
    write_new_receipt(root/policy["registration"], registration)
    return registration


def attempts(root: Path, policy: dict) -> list[dict]:
    directory = root/policy["directory"]/"attempts"
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise QuantError("Unexpected or escaped recovery state cannot be ignored.")
        intent = read_json(safe_file(root, (path/"intent.json").relative_to(root).as_posix()))
        record_path = path/"result.json"
        if not record_path.exists():
            raise QuantError("An incomplete source request is preserved;manual review is required before any further request.")
        record = read_json(safe_file(root, record_path.relative_to(root).as_posix()))
        if record.get("status") != "complete_source_history":
            raise QuantError("A previous source recovery failed;automatic retries and later fund requests are stopped.")
        if record["symbol"] != intent["symbol"] or moment(record["finished_at"]) < moment(intent["started_at"]):
            raise QuantError("Original recovery intent and actual completion chronology changed.")
        for name, digest in record["files"].items():
            safe_file(root, (path/name).relative_to(root).as_posix(), digest)
        result.append(record)
    if (len(result) > 2 or [row["symbol"] for row in result] != policy["symbols"][:len(result)]
        or any(moment(right["started_at"])-moment(left["finished_at"]) < timedelta(hours=24)
               for left, right in zip(result, result[1:]))):
        raise QuantError("The fixed symbol order,one-attempt budget or global24hour spacing changed.")
    return result


def eligibility(registration: dict, records: list[dict], now: datetime) -> dict:
    if now.tzinfo is None or now.utcoffset() is None:
        raise QuantError("Eligibility requires an actual aware clock,not a source period label.")
    if len(records) == 2:
        return {"action": "all_sources_acquired", "new_requests": 0, "complete_histories": 2}
    deadline = moment(registration["first_request_not_before_utc"])
    if records:
        deadline = max(deadline, moment(records[-1]["next_request_not_before_utc"]))
    symbol = registration["policy"]["symbols"][len(records)]
    return {"action": "not_due" if now.astimezone(timezone.utc) < deadline else "request_eligible",
            "symbol": symbol, "not_before_utc": deadline.isoformat(), "new_requests": 0,
            "complete_histories": len(records)}


def state(root: Path = ROOT) -> tuple[dict, dict, list[dict]]:
    policy = read_json(safe_file(root, POLICY_PATH))
    _, _, requests = original_inputs(root, policy)
    registration, reference = input_reference(root, root/policy["registration"])
    private = read_json(safe_file(root, f"{policy['directory']}/registration.json", reference["sha256"]))
    if (private != registration or registration["policy"] != policy or registration["controller_sha256"] != fingerprint(root)
        or registration["source_requests"] != requests
        or registration["policy_reference"]["sha256"] != file_digest(root/POLICY_PATH)):
        raise QuantError("Keep the exact original registered policy,source requests and controller bytes.")
    return registration, reference, attempts(root, policy)


def retry_after(headers: str, finished: datetime) -> datetime:
    values = [line.split(":", 1)[1].strip() for line in headers.splitlines()
              if line.lower().startswith("retry-after:")]
    if len(values) > 1:
        raise QuantError("Conflicting Retry-After directives require manual source review.")
    deadline = finished+timedelta(hours=24)
    if not values:
        return deadline
    value = values[0]
    if value.isascii() and value.isdecimal():
        directive = finished+timedelta(seconds=int(value))
    else:
        try:
            directive = parsedate_to_datetime(value)
        except (TypeError, ValueError) as exc:
            raise QuantError("Malformed Retry-After cannot be ignored or replaced by a guessed permission.") from exc
        if directive is None or directive.tzinfo is None:
            raise QuantError("Retry-After needs a numeric delay or an actual aware HTTP date.")
    return max(deadline, directive.astimezone(timezone.utc))


def fetch(path: Path, url: str) -> dict:
    body, headers = path/"raw.json", path/"response-headers.txt"
    command = ["curl", "--silent", "--show-error", "--location", "--connect-timeout", "15", "--max-time", "180",
               "--output", str(body), "--dump-header", str(headers), "--write-out", "%{json}", url]
    started = utc_now()
    response = subprocess.run(command, capture_output=True, text=True, check=False)
    finished = utc_now()
    write_text_atomic(path/"curl.stdout.txt", response.stdout)
    write_text_atomic(path/"curl.stderr.txt", response.stderr)
    metadata = json.loads(response.stdout) if response.stdout else None
    record = {"started_at": started, "finished_at": finished, "url": url, "method": "GET",
              "request_headers_or_user_agent_changed": False, "exit_code": response.returncode,
              "stderr": response.stderr, "response": metadata, "access_gates_bypassed": False}
    write_new_receipt(path/"request.json", record)
    if (response.returncode != 0 or metadata is None or metadata["http_code"] != 200
        or metadata["url_effective"] != url or not body.is_file() or not headers.is_file()):
        raise QuantError("Ordinary recovery source request refused or failed;the complete attempt is preserved without retry.")
    return record


def acquire(commit: str, root: Path = ROOT) -> dict:
    registration, reference, _ = state(root)
    policy = registration["policy"]
    directory = root/policy["directory"]
    with (directory/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = attempts(root, policy)
        allowed = eligibility(registration, records, datetime.now(timezone.utc))
        if allowed["action"] != "request_eligible":
            return allowed
        published = verify_publication(root, reference, {
            POLICY_PATH, "src/us_quant/dividend_source_recovery.py",
            "config/dividend-growth-sources.json", "src/us_quant/dividend_growth_sources.py",
            "evidence/dividend_growth_source_readiness_20261011.json",
        }, commit)
        symbol = allowed["symbol"]
        source = next(row for row in registration["source_requests"] if row["symbol"] == symbol)
        path = directory/"attempts"/f"{len(records)+1:02}-{symbol}"
        path.parent.mkdir(exist_ok=True, mode=0o700)
        new_output_directory(path)
        started = utc_now()
        write_new_receipt(path/"intent.json", {"symbol": symbol, "started_at": started, "url": source["url"],
                                              "publication": published, "new_requests": 1, "order_authority": False})
        try:
            request = fetch(path, source["url"])
            source_policy = read_json(root/policy["source_policy"]["path"])
            frame, proof = history_from_payload(read_json(path/"raw.json"), symbol, source_policy)
            next_request = retry_after((path/"response-headers.txt").read_text(), moment(request["finished_at"]))
            write_text_atomic(path/f"{symbol}.csv", frame.to_csv(float_format="%.17g"))
            finished = utc_now()
            next_request = max(next_request, moment(finished)+timedelta(hours=24))
            files = {name: file_digest(path/name)
                     for name in ("intent.json", "raw.json", "response-headers.txt", "request.json",
                                  "curl.stdout.txt", "curl.stderr.txt", f"{symbol}.csv")}
            result = {"status": "complete_source_history", "symbol": symbol, "started_at": started,
                      "finished_at": finished, "next_request_not_before_utc": next_request.isoformat(),
                      "source_proof": proof, "files": files, "source_only": True,
                      "new_strategy_evaluations": 0, "order_authority": False}
            write_new_receipt(path/"result.json", result)
            return {"action": "source_acquired", "symbol": symbol, "new_requests": 1,
                    "complete_histories": len(records)+1, "actual_sessions": len(frame),
                    "new_strategy_evaluations": 0, "order_authority": False}
        except (QuantError, OSError, ValueError, subprocess.SubprocessError) as exc:
            write_new_receipt(path/"result.json", {"status": "blocked_preserved", "symbol": symbol,
                              "started_at": started, "finished_at": utc_now(), "reason": str(exc),
                              "automatic_further_requests_allowed": False, "order_authority": False})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded ordinary dividend source recovery without strategy evaluation.")
    parser.add_argument("action", choices=("register", "status", "acquire"))
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        if args.action == "register":
            result = register()
        elif args.action == "status":
            registration, _, records = state()
            result = eligibility(registration, records, datetime.now(timezone.utc))
        else:
            if args.published_commit is None:
                raise QuantError("An actual published source-recovery commit is required before a source request.")
            result = acquire(args.published_commit)
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Dividend source recovery blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
