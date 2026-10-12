from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from test_research_program import candidate_spec, make_bundle, readiness

from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram
from us_quant.review_receipts import record_or_recover
from us_quant.storage import digest_json, file_digest, read_json, write_json
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    winning = getattr(request, "param", False)
    policy = read_json(ROOT / "config/research-program.json")
    write_json(tmp_path / "config/research-program.json", policy)
    monkeypatch.setattr(
        "us_quant.research_program.registered_targets",
        lambda spec, data, start, end, cost, delay, root, cache: (
            buy_and_hold_signals(data.close, "TEST", start) * 0.98
        ),
    )
    monkeypatch.setattr(
        "us_quant.research_program.ResearchProgram.evaluate_registered",
        lambda *args: pytest.fail("Review receipts cannot create a different evaluation"),
    )
    program = ResearchProgram(
        tmp_path / "runtime/research-program.sqlite3",
        policy,
        root=tmp_path,
        create=True,
    )
    try:
        registration = program.register_candidate(
            candidate_spec(tmp_path, winning=winning),
            readiness(policy, tmp_path, ready=True),
        )
    finally:
        program.close()
    bundle = make_bundle(tmp_path, registration, winning=winning)
    path = tmp_path / "reports/original/bundle.json"
    write_json(path, bundle)
    return tmp_path, "test_multifactor", path, tmp_path / "reports/receipts/review.json", winning


def state(root):
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        read_json(root / "config/research-program.json"),
        root=root,
    )
    try:
        return program.status()
    finally:
        program.close()


def run(prepared, **kwargs):
    root, candidate, bundle, receipt, _ = prepared
    return record_or_recover(candidate, bundle, receipt, root=root, **kwargs)


@pytest.mark.parametrize("prepared", [True, False], indirect=True)
def test_complete_source_bound_review_records_actual_historical_status_once(prepared):
    before = state(prepared[0])
    result = run(prepared)
    after = state(prepared[0])
    assert result["action"] == "review_recorded"
    assert result["new_reviews_recorded"] == result["new_strategy_evaluations"] == 1
    assert result["historical_gates_passed"] is prepared[-1]
    assert result["historical_status"] == (
        "historical_qualified_awaiting_forward" if prepared[-1] else "rejected_historical"
    )
    assert after["total_evaluated_configurations"] == before["total_evaluated_configurations"] + 1
    assert len(after["candidate_reviews"]) == 1
    assert not result["independent_forward_validation"]
    assert not result["investment_objective_verified"]
    assert not result["order_authority"]
    assert read_json(prepared[3])["review"]["status"] == result["historical_status"]


def test_post_review_pre_receipt_failure_recovers_with_readonly_replay_without_recounting(
    prepared, monkeypatch
):
    original = ResearchProgram.review

    def fail(path, receipt):
        raise OSError("synthetic post-review disk error")

    with pytest.raises(OSError, match="post-review"):
        run(prepared, receipt_writer=fail)
    committed = state(prepared[0])
    assert len(committed["candidate_reviews"]) == 1
    assert not prepared[3].exists()
    audits = []

    def audit_only(self, *args, **kwargs):
        assert kwargs.get("audit_existing") is True
        result = original(self, *args, **kwargs)
        audits.append(result)
        return result

    monkeypatch.setattr("us_quant.research_program.ResearchProgram.review", audit_only)
    recovered = run(prepared, allow_new_review=False)
    assert recovered["action"] == "recovered_review"
    assert recovered["new_strategy_evaluations"] == recovered["new_reviews_recorded"] == 0
    assert recovered["existing_review_read_only_reaudited"]
    assert len(audits) == 1 and audits[0]["read_only_reaudit"]
    receipt = read_json(prepared[3])
    assert receipt["review"] == committed["candidate_reviews"][0]
    assert receipt["review"]["at"] != audits[0]["at"]
    assert receipt["registration_event"]["at"] == receipt["registration"]["registered_at"]
    assert state(prepared[0]) == committed


def test_interruption_after_review_commit_before_return_can_recover_the_same_event(
    prepared, monkeypatch
):
    original = ResearchProgram.review

    def commit_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("synthetic interruption after commit")

    monkeypatch.setattr("us_quant.research_program.ResearchProgram.review", commit_then_fail)
    with pytest.raises(OSError, match="after commit"):
        run(prepared)
    committed = state(prepared[0])
    assert not prepared[3].exists()
    monkeypatch.setattr("us_quant.research_program.ResearchProgram.review", original)
    recovered = run(prepared, allow_new_review=False)
    assert recovered["action"] == "recovered_review"
    assert state(prepared[0]) == committed


def test_interrupted_postwrite_and_repeated_receipts_are_byte_identical(prepared):
    def write_then_fail(path, receipt):
        write_new_receipt(path, receipt)
        raise OSError("synthetic failure after receipt write")

    with pytest.raises(OSError, match="after receipt"):
        run(prepared, receipt_writer=write_then_fail)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    repeated = run(prepared)
    assert repeated["action"] == "already_recorded"
    assert repeated["new_strategy_evaluations"] == 0
    assert file_digest(prepared[3]) == digest
    assert state(prepared[0]) == before


def test_recover_only_never_submits_a_new_review(prepared):
    before = state(prepared[0])
    with pytest.raises(QuantError, match="unreviewed"):
        run(prepared, allow_new_review=False)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_horizon",
        "duplicate_horizon",
        "malformed_year",
        "missing_independent",
        "missing_weights",
        "missing_market",
        "missing_hash",
        "null_hash",
        "changed_curve",
        "changed_rule",
        "changed_cutoff",
        "changed_capital",
        "changed_cost",
    ],
)
def test_missing_changed_or_unsupported_account_paths_do_not_become_completed_reviews(
    prepared, problem
):
    bundle = read_json(prepared[2])
    if problem == "missing_horizon":
        bundle["paths"].pop()
    elif problem == "duplicate_horizon":
        bundle["paths"][1] = bundle["paths"][0]
    elif problem == "malformed_year":
        bundle["paths"][0]["years"] = [10]
    elif problem == "missing_independent":
        bundle["paths"][0].pop("strategy_bt")
    elif problem == "missing_weights":
        bundle["paths"][0].pop("weights")
    elif problem == "missing_market":
        bundle["market"].pop("risk_free")
    elif problem == "missing_hash":
        bundle["paths"][0]["strategy"].pop("sha256")
    elif problem == "null_hash":
        bundle["paths"][0]["strategy"]["sha256"] = None
    elif problem == "changed_curve":
        (prepared[0] / bundle["paths"][0]["strategy"]["path"]).write_text("changed curve")
    elif problem == "changed_rule":
        write_json(prepared[0] / "frozen-rule.json", {"different_rule": True})
    elif problem == "changed_cutoff":
        bundle["as_of"] = "2026-10-02"
    elif problem == "changed_capital":
        bundle["paths"][0]["capital_usd"] = 100000
    else:
        bundle["paths"][0]["cost_bps"] = 0
    write_json(prepared[2], bundle)
    before = state(prepared[0])
    with pytest.raises(QuantError):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


def test_a_complete_bundle_cannot_belong_to_an_unregistered_candidate(prepared):
    root, _, bundle, receipt, _ = prepared
    before = state(root)
    with pytest.raises(QuantError, match="Register the actual"):
        record_or_recover("unknown_candidate", bundle, receipt, root=root)
    assert state(root) == before
    assert not receipt.exists()


def test_a_second_bundle_does_not_replace_a_completed_review_or_its_receipt(prepared):
    run(prepared)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    bundle = read_json(prepared[2])
    bundle["unrequested_metadata"] = "new evidence attempt"
    write_json(prepared[2], bundle)
    with pytest.raises(QuantError, match="different evidence"):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


@pytest.mark.parametrize("problem", ["targets", "strategy", "risk_free"])
def test_rehashed_evidence_still_requires_frozen_targets_accounts_and_risk_free(prepared, problem):
    bundle = read_json(prepared[2])
    item = bundle["paths"][0]
    names = ("strategy", "spy") if problem == "risk_free" else (problem,)
    for name in names:
        path = prepared[0] / item[name]["path"]
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if problem == "targets":
            frame["TEST"] *= 0.5
        elif problem == "strategy":
            frame["equity"] *= 1.1
        else:
            frame["risk_free"] = -0.01
        frame.to_csv(path)
        item[name]["sha256"] = file_digest(path)
    write_json(prepared[2], bundle)
    before = state(prepared[0])
    with pytest.raises(QuantError, match="registered strategy|replay|risk.free|inconsistent"):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


def test_changed_file_after_review_blocks_recovery_without_replacing_original_status(prepared):
    def fail(path, receipt):
        raise OSError("synthetic missing receipt")

    with pytest.raises(OSError, match="missing receipt"):
        run(prepared, receipt_writer=fail)
    bundle = read_json(prepared[2])
    (prepared[0] / bundle["paths"][0]["targets"]["path"]).write_text("changed targets")
    before = state(prepared[0])
    with pytest.raises(QuantError, match="hash changed"):
        run(prepared, allow_new_review=False)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


def test_internally_consistent_altered_review_still_requires_original_account_metrics(prepared):
    def fail(path, receipt):
        raise OSError("synthetic missing receipt")

    with pytest.raises(OSError, match="missing receipt"):
        run(prepared, receipt_writer=fail)
    root = prepared[0]
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        read_json(root / "config/research-program.json"),
        root=root,
    )
    try:
        event = program.db.execute("SELECT * FROM events WHERE kind='review'").fetchone()
        body = json.loads(event["body"])
        body["paths"][0]["metrics"]["cagr"] += 0.1
        encoded = json.dumps(body, sort_keys=True, allow_nan=False)
        event_digest = digest_json(
            {
                "at": event["at"],
                "kind": event["kind"],
                "body": body,
                "previous_sha": event["previous_sha"],
            }
        )
        with program.db:
            program.db.execute(
                "UPDATE events SET body=?,sha=? WHERE seq=?", (encoded, event_digest, event["seq"])
            )
            program.db.execute(
                "UPDATE reviews SET body=? WHERE candidate_id=?", (encoded, prepared[1])
            )
        program.verify()
    finally:
        program.close()
    before = state(root)
    with pytest.raises(QuantError):
        run(prepared, allow_new_review=False)
    assert state(root) == before
    assert not prepared[3].exists()


def test_new_reviews_cannot_use_an_unversioned_accounting_fallback(prepared):
    bundle = read_json(prepared[2])
    bundle.pop("accounting_engine")
    write_json(prepared[2], bundle)
    before = state(prepared[0])
    with pytest.raises(QuantError, match="accounting engine"):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


@pytest.mark.parametrize("value", [None, [], "not-a-time", "2026-10-11", "2100-01-01T00:00:00Z"])
def test_review_completion_requires_an_actual_timezone_aware_instant(prepared, value):
    bundle = read_json(prepared[2])
    bundle["completed_at"] = value
    write_json(prepared[2], bundle)
    before = state(prepared[0])
    with pytest.raises(QuantError, match="timezone-aware|fake clock"):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


@pytest.mark.parametrize("problem", ["status", "event", "before_review", "authority", "partial"])
def test_review_receipts_cannot_relabel_a_rejection_or_its_completion_time(prepared, problem):
    run(prepared)
    receipt = read_json(prepared[3])
    if problem == "status":
        receipt["review"]["status"] = "historical_qualified_awaiting_forward"
    elif problem == "event":
        receipt["review_event"]["sha256"] = "0" * 64
    elif problem == "before_review":
        receipt["recorded_at"] = receipt["registration"]["registered_at"]
        assert pd.Timestamp(receipt["recorded_at"]) < pd.Timestamp(receipt["review"]["at"])
    elif problem == "authority":
        receipt["order_authority"] = True
    if problem == "partial":
        prepared[3].write_text("{partial")
    else:
        write_json(prepared[3], receipt)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    with pytest.raises(QuantError):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


def test_existing_unbacked_receipt_does_not_submit_the_review(prepared):
    write_json(prepared[3], {"unbacked": True})
    before, digest = state(prepared[0]), file_digest(prepared[3])
    with pytest.raises(QuantError, match="no actual"):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


def test_noop_writer_cannot_return_success_but_committed_review_is_recoverable(prepared):
    with pytest.raises(QuantError, match="missing"):
        run(prepared, receipt_writer=lambda *args: None)
    committed = state(prepared[0])
    assert len(committed["candidate_reviews"]) == 1
    recovered = run(prepared, allow_new_review=False)
    assert recovered["action"] == "recovered_review"
    assert state(prepared[0]) == committed


def test_concurrent_calls_record_one_review_and_keep_one_identical_receipt(prepared):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(prepared), range(2)))
    assert sorted(row["action"] for row in results) == ["already_recorded", "review_recorded"]
    assert sum(row["new_reviews_recorded"] for row in results) == 1
    assert results[0]["receipt_sha256"] == results[1]["receipt_sha256"]
    assert len(state(prepared[0])["candidate_reviews"]) == 1


@pytest.mark.parametrize(
    "path", ["../receipt.json", "config/receipt.json", "reports/original/bundle.json"]
)
def test_review_output_cannot_escape_or_overwrite_input_evidence(prepared, path):
    root, candidate, bundle, _, _ = prepared
    before = state(root)
    with pytest.raises(QuantError):
        record_or_recover(candidate, bundle, Path(path), root=root)
    assert state(root) == before
