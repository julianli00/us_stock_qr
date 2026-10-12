from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from us_quant import dividend_source_recovery as recovery
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT/"config/dividend-source-recovery.json")


def registration(policy):
    return {"policy": policy, "first_request_not_before_utc": "2026-10-12T09:29:43.277441+00:00",
            "source_requests": [{"symbol": symbol, "url": f"https://example.invalid/{symbol}"}
                                for symbol in policy["symbols"]]}


def test_actual_old_failure_cooldown_is_not_shortened_or_a_future_price_observation(policy):
    registered = registration(policy)
    deadline = recovery.moment(registered["first_request_not_before_utc"])
    assert recovery.eligibility(registered, [], deadline-timedelta(microseconds=1))["action"] == "not_due"
    assert recovery.eligibility(registered, [], deadline)["action"] == "request_eligible"
    assert recovery.eligibility(registered, [], deadline)["symbol"] == "VIG"
    assert policy["source_history_is_forward_observation"] is False


def test_global_second_fund_waits_24hours_and_honors_a_later_server_directive(policy):
    registered = registration(policy)
    now = recovery.moment("2026-10-13T10:00:00+00:00")
    records = [{"next_request_not_before_utc": (now+timedelta(days=2)).isoformat()}]
    result = recovery.eligibility(registered, records, now)
    assert result["action"] == "not_due" and result["symbol"] == "SDY" and result["new_requests"] == 0
    assert recovery.eligibility(registered, records, now+timedelta(days=2))["action"] == "request_eligible"
    assert recovery.eligibility(registered, records*2, now)["action"] == "all_sources_acquired"


@pytest.mark.parametrize("value", ["not-a-date", "2026-10-12T09:00:00", None])
def test_unaware_or_invalid_timestamps_cannot_authorize_source_requests(value):
    with pytest.raises(QuantError):
        recovery.moment(value)


def test_unaware_eligibility_clock_is_rejected(policy):
    with pytest.raises(QuantError):
        recovery.eligibility(registration(policy), [], datetime(2026, 10, 13))


@pytest.mark.parametrize("headers,hours", [
    ("HTTP/2 200\n", 24),
    ("HTTP/2 200\nRetry-After: 60\n", 24),
    ("HTTP/2 200\nretry-after: 172800\n", 48),
    ("HTTP/2 200\nRetry-After: Wed, 14 Oct 2026 10:00:00 GMT\n", 48),
])
def test_captured_retry_after_can_only_extend_the_local_24hour_gate(headers, hours):
    finished = datetime(2026, 10, 12, 10, tzinfo=timezone.utc)
    assert recovery.retry_after(headers, finished) == finished+timedelta(hours=hours)


@pytest.mark.parametrize("headers", [
    "Retry-After: bad date\n", "Retry-After: -1\n",
    "Retry-After: 60\nRetry-After: 120\n",
])
def test_malformed_or_conflicting_retry_directives_stop_instead_of_assuming_permission(headers):
    with pytest.raises(QuantError):
        recovery.retry_after(headers, datetime(2026, 10, 12, tzinfo=timezone.utc))


@pytest.mark.parametrize("field,value", [
    ("minimum_cooldown_hours", 1), ("maximum_new_requests_per_invocation", 2),
    ("maximum_attempts_per_symbol", 2), ("maximum_total_new_requests", 5),
    ("stop_after_any_refusal_transport_failure_or_invalid_payload", False),
    ("original_url_method_headers_and_curl_transport_unchanged", False),
    ("as_of", "2026-10-09"), ("earliest_separate_hypothesis_cycle", "2026-10-17T09:00:00+08:00"),
    ("new_registered_factor_definitions", 1), ("new_registered_candidates", 1),
    ("portfolio_states_targets_or_returns_computed", True), ("order_authority", True),
])
def test_source_scope_cannot_be_expanded_to_bypass_limits_or_register_early_factors(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        recovery.validate_policy(policy)


def test_current_frozen_policy_validates_without_changing_strategy_parameters(policy):
    recovery.validate_policy(policy)
    assert policy["required_market_sessions"] == 2805
    assert policy["new_strategy_evaluations"] == 0


def test_not_due_acquire_invokes_neither_publication_network_nor_creates_an_attempt(tmp_path, monkeypatch, policy):
    registered = registration(policy)
    registered["first_request_not_before_utc"] = "2999-01-01T00:00:00+00:00"
    directory = tmp_path/policy["directory"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(recovery, "state", lambda root: (registered, {}, []))
    monkeypatch.setattr(recovery, "attempts", lambda root, candidate_policy: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("No publication or source request is allowed before the real cooldown.")

    monkeypatch.setattr(recovery, "verify_publication", forbidden)
    monkeypatch.setattr(recovery, "fetch", forbidden)
    result = recovery.acquire("0"*40, tmp_path)
    assert result["action"] == "not_due" and result["new_requests"] == 0
    assert not (directory/"attempts").exists()


def test_unpublished_due_call_cannot_create_intent_or_send_a_request(tmp_path, monkeypatch, policy):
    registered = registration(policy)
    directory = tmp_path/policy["directory"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(recovery, "state", lambda root: (registered, {}, []))
    monkeypatch.setattr(recovery, "attempts", lambda root, candidate_policy: [])
    monkeypatch.setattr(recovery, "eligibility", lambda *args: {"action": "request_eligible", "symbol": "VIG"})

    def blocked(*args):
        raise QuantError("Not actually published.")

    monkeypatch.setattr(recovery, "verify_publication", blocked)
    monkeypatch.setattr(recovery, "fetch", lambda *args: pytest.fail("No source request before publication."))
    with pytest.raises(QuantError, match="published"):
        recovery.acquire("0"*40, tmp_path)
    assert not (directory/"attempts").exists()


def test_http429_preserves_default_get_transport_and_raw_attempt_without_automatic_retry(tmp_path, monkeypatch):
    commands = []

    def refuse(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--output")+1]).write_text("Edge: Too Many Requests")
        Path(command[command.index("--dump-header")+1]).write_text("HTTP/2 429\nRetry-After: 86400\n")
        return CompletedProcess(command, 0, json.dumps({"http_code": 429, "url_effective": command[-1]}), "")

    monkeypatch.setattr(recovery.subprocess, "run", refuse)
    with pytest.raises(QuantError, match="refused"):
        recovery.fetch(tmp_path, "https://query2.finance.yahoo.com/v8/finance/chart/VIG")
    assert len(commands) == 1
    assert not any(flag in commands[0] for flag in ("--retry", "--header", "--user-agent", "--insecure"))
    assert read_json(tmp_path/"request.json")["response"]["http_code"] == 429
    assert (tmp_path/"raw.json").read_text() == "Edge: Too Many Requests"


def test_invalid_curl_metadata_still_preserves_actual_stdout_stderr_and_response(tmp_path, monkeypatch):
    def malformed(command, **kwargs):
        Path(command[command.index("--output")+1]).write_text("original response")
        Path(command[command.index("--dump-header")+1]).write_text("HTTP/2 200\n")
        return CompletedProcess(command, 0, "invalid json", "original transport diagnostics")

    monkeypatch.setattr(recovery.subprocess, "run", malformed)
    with pytest.raises(ValueError):
        recovery.fetch(tmp_path, "https://query2.finance.yahoo.com/v8/finance/chart/VIG")
    assert (tmp_path/"curl.stdout.txt").read_text() == "invalid json"
    assert (tmp_path/"curl.stderr.txt").read_text() == "original transport diagnostics"
    assert (tmp_path/"raw.json").read_text() == "original response"


def test_incomplete_or_failed_attempt_stops_every_later_request(tmp_path, policy):
    directory = tmp_path/policy["directory"]/"attempts"/"01-VIG"
    directory.mkdir(parents=True)
    write_json(directory/"intent.json", {"symbol": "VIG", "started_at": "2026-10-13T01:00:00+00:00"})
    with pytest.raises(QuantError, match="incomplete"):
        recovery.attempts(tmp_path, policy)
    write_json(directory/"result.json", {"status": "blocked_preserved", "symbol": "VIG"})
    with pytest.raises(QuantError, match="failed"):
        recovery.attempts(tmp_path, policy)


def test_success_receipt_hashes_and_global_spacing_are_required(tmp_path, policy):
    first = tmp_path/policy["directory"]/"attempts"/"01-VIG"
    first.mkdir(parents=True)
    (first/"raw.json").write_text("frozen synthetic fixture")
    intent = {"symbol": "VIG", "started_at": "2026-10-13T01:00:00+00:00"}
    write_json(first/"intent.json", intent)
    record = {"status": "complete_source_history", "symbol": "VIG", "started_at": intent["started_at"],
              "finished_at": "2026-10-13T01:00:10+00:00",
              "next_request_not_before_utc": "2026-10-14T01:00:10+00:00",
              "files": {"raw.json": file_digest(first/"raw.json")}}
    write_json(first/"result.json", record)
    assert recovery.attempts(tmp_path, policy) == [record]
    (first/"raw.json").write_text("changed fixture")
    with pytest.raises(QuantError):
        recovery.attempts(tmp_path, policy)


def test_source_acquisition_failure_is_persisted_and_cannot_be_retried(tmp_path, monkeypatch, policy):
    registered = registration(policy)
    directory = tmp_path/policy["directory"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(recovery, "state", lambda root: (registered, {}, []))
    monkeypatch.setattr(recovery, "attempts", lambda root, candidate_policy: [])
    monkeypatch.setattr(recovery, "eligibility", lambda *args: {"action": "request_eligible", "symbol": "VIG"})
    monkeypatch.setattr(recovery, "verify_publication", lambda *args: {"published_commit": "0"*40})

    def refuse(*args):
        raise QuantError("HTTP429")

    monkeypatch.setattr(recovery, "fetch", refuse)
    with pytest.raises(QuantError, match="HTTP429"):
        recovery.acquire("0"*40, tmp_path)
    failed = read_json(directory/"attempts"/"01-VIG"/"result.json")
    assert failed["status"] == "blocked_preserved"
    assert failed["automatic_further_requests_allowed"] is False
    assert not (directory/"attempts"/"02-SDY").exists()
