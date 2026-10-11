from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from us_quant.candidate_receipts import register_or_recover, write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, timestamp
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def prepared(tmp_path, market_factory, monkeypatch):
    policy = read_json(ROOT / "config/research-program.json")
    write_json(tmp_path / "config/research-program.json", policy)
    program = ResearchProgram(
        tmp_path / "runtime/research-program.sqlite3",
        policy,
        root=tmp_path,
        create=True,
    )
    program.close()
    market = market_factory("2026-01-02", "2026-10-05", ("SPY", "TEST"))
    references = {}
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        path = tmp_path / "data" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        getattr(market, name).to_csv(path)
        references[name] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": file_digest(path),
        }
    rule, proof = tmp_path / "data/rule.json", tmp_path / "data/proof.json"
    write_json(rule, {"rule": "synthetic unit test,not research performance"})
    write_json(proof, {"synthetic": True})
    spec = {
        "id": "receipt_test_candidate",
        "factor_ids": ["price_momentum", "quality_exposure", "value_exposure"],
        "frozen_files": {"data/rule.json": file_digest(rule)},
        "asset_leverage": {"SPY": 1.0, "TEST": 1.0},
        "evaluation_as_of": "2026-10-05",
        "market": references,
        "history_status": "exposed_history_not_independent_holdout",
        "leveraged_products_allowed": False,
        "order_authority": False,
    }
    readiness = {
        "schema_version": 1,
        "checked_at": timestamp().isoformat(),
        "capabilities": {
            key: {
                "verified": True,
                "reason": "synthetic unit fixture",
                "evidence_path": "data/proof.json",
                "evidence_sha256": file_digest(proof),
            }
            for key in policy["required_stock_data"]
        },
    }
    spec_path, ready_path = tmp_path / "data/spec.json", tmp_path / "data/readiness.json"
    write_json(spec_path, spec)
    write_json(ready_path, readiness)
    monkeypatch.setattr(
        "us_quant.candidate_receipts.registered_generator", lambda *args: ("synthetic", {})
    )
    monkeypatch.setattr(
        "us_quant.research_program.ResearchProgram.evaluate_registered",
        lambda *args: pytest.fail("Registration receipts must never evaluate a strategy"),
    )
    return tmp_path, spec_path, ready_path, tmp_path / "reports/receipts/candidate.json"


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
    root, spec, ready, receipt = prepared
    return register_or_recover(spec, ready, receipt, root=root, **kwargs)


def test_new_registration_receipt_is_bound_to_the_actual_event_not_an_evaluation(prepared):
    before = state(prepared[0])
    result = run(prepared)
    receipt = read_json(prepared[3])
    after = state(prepared[0])
    assert result["action"] == "registered"
    assert result["new_candidate_registration"] is True
    assert receipt["registration"]["spec_sha256"] == digest_json(read_json(prepared[1]))
    assert receipt["registration"]["readiness_sha256"] == digest_json(read_json(prepared[2]))
    assert receipt["registration_event"]["sha256"] == after["event_chain_sha256"]
    assert len(after["registered_candidates"]) == len(before["registered_candidates"]) + 1
    assert after["total_evaluated_configurations"] == before["total_evaluated_configurations"]
    assert not receipt["portfolio_returns_computed"]
    assert not receipt["investment_objective_verified"]
    assert not receipt["order_authority"]


def test_post_commit_pre_receipt_failure_recovers_exact_registration_without_second_admission(
    prepared, monkeypatch
):
    def fail_before_write(path, receipt):
        raise OSError("synthetic disk failure after candidate commit")

    with pytest.raises(OSError, match="after candidate commit"):
        run(prepared, receipt_writer=fail_before_write)
    committed = state(prepared[0])
    assert len(committed["registered_candidates"]) == 1
    assert not prepared[3].exists()
    monkeypatch.setattr(
        "us_quant.research_program.ResearchProgram.register_candidate",
        lambda *args, **kwargs: pytest.fail("Do not re-register a committed candidate"),
    )
    result = run(prepared, allow_new_registration=False)
    assert result["action"] == "recovered_registration"
    assert not result["new_candidate_registration"]
    assert state(prepared[0]) == committed
    assert read_json(prepared[3])["registration"]["registered_at"] == result["actual_registered_at"]


def test_failure_after_complete_receipt_write_is_idempotent_without_rewriting(prepared):
    def fail_after_write(path, receipt):
        write_new_receipt(path, receipt)
        raise OSError("synthetic interruption after durable write")

    with pytest.raises(OSError, match="after durable write"):
        run(prepared, receipt_writer=fail_after_write)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    result = run(prepared)
    assert result["action"] == "already_recorded"
    assert not result["new_candidate_registration"]
    assert file_digest(prepared[3]) == digest
    assert state(prepared[0]) == before


def test_recovery_keeps_original_readiness_even_when_a_new_check_would_be_stale(
    prepared, monkeypatch
):
    def fail(path, receipt):
        raise OSError("post-commit failure")

    with pytest.raises(OSError):
        run(prepared, receipt_writer=fail)
    before = state(prepared[0])
    future = timestamp() + timedelta(days=9)
    monkeypatch.setattr("us_quant.candidate_receipts.timestamp", lambda: future)
    monkeypatch.setattr(
        "us_quant.research_program.readiness_blockers",
        lambda *args, **kwargs: pytest.fail("Recovery is not a new data-admission decision"),
    )
    recovered = run(prepared, allow_new_registration=False)
    assert recovered["action"] == "recovered_registration"
    assert state(prepared[0]) == before


def test_a_new_registration_cannot_reuse_stale_readiness(prepared):
    ready = read_json(prepared[2])
    ready["checked_at"] = (timestamp() - timedelta(days=9)).isoformat()
    write_json(prepared[2], ready)
    before = state(prepared[0])
    with pytest.raises(QuantError, match="freshly"):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


@pytest.mark.parametrize("changed", ["specification", "readiness", "source"])
def test_recovery_never_changes_registered_inputs_or_rewrites_the_old_receipt(prepared, changed):
    run(prepared)
    before, receipt_digest = state(prepared[0]), file_digest(prepared[3])
    if changed == "source":
        write_json(prepared[0] / "data/rule.json", {"different_rule": True})
    else:
        path = prepared[1] if changed == "specification" else prepared[2]
        body = read_json(path)
        body["different"] = True
        write_json(path, body)
    with pytest.raises(QuantError):
        run(prepared, allow_new_registration=False)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == receipt_digest


@pytest.mark.parametrize(
    "problem", ["event", "authority", "timestamp", "future", "naive", "extra", "partial"]
)
def test_tampered_or_partial_receipts_are_retained_and_explicitly_rejected(prepared, problem):
    run(prepared)
    receipt = read_json(prepared[3])
    if problem == "event":
        receipt["registration_event"]["sha256"] = "0" * 64
    elif problem == "authority":
        receipt["order_authority"] = True
    elif problem == "timestamp":
        receipt["recorded_at"] = (
            pd.Timestamp(receipt["registration"]["registered_at"]) - pd.Timedelta(seconds=1)
        ).isoformat()
    elif problem == "future":
        receipt["recorded_at"] = (timestamp() + timedelta(days=1)).isoformat()
    elif problem == "naive":
        receipt["recorded_at"] = "2026-10-01"
    elif problem == "extra":
        receipt["unexpected"] = "unsupported result"
    if problem == "partial":
        prepared[3].write_text("{partial")
    else:
        write_json(prepared[3], receipt)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    with pytest.raises(QuantError):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


def test_existing_unbacked_receipt_cannot_trigger_candidate_registration(prepared):
    write_json(prepared[3], {"unbacked_receipt": True})
    before, digest = state(prepared[0]), file_digest(prepared[3])
    with pytest.raises(QuantError, match="no actual"):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


def test_recover_only_mode_never_admits_an_unknown_candidate(prepared):
    before = state(prepared[0])
    with pytest.raises(QuantError, match="unknown candidate"):
        run(prepared, allow_new_registration=False)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


def test_frozen_future_eligibility_is_checked_before_first_registration(prepared, monkeypatch):
    eligible = timestamp() + timedelta(days=1)
    monkeypatch.setattr(
        "us_quant.candidate_receipts.registered_generator",
        lambda *args: ("synthetic", {"eligible_not_before": eligible.isoformat()}),
    )
    before = state(prepared[0])
    with pytest.raises(QuantError, match="not eligible"):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()


def test_two_concurrent_invocations_do_not_duplicate_registration_or_overwrite(prepared):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(prepared), range(2)))
    assert sorted(row["action"] for row in results) == ["already_recorded", "registered"]
    assert sum(row["new_candidate_registration"] for row in results) == 1
    assert len(state(prepared[0])["registered_candidates"]) == 1
    assert results[0]["receipt_sha256"] == results[1]["receipt_sha256"]


def test_atomic_receipt_writer_never_replaces_an_existing_file_and_cleans_its_temporary(prepared):
    path = prepared[3]
    write_new_receipt(path, {"original": True})
    digest = file_digest(path)
    with pytest.raises(QuantError, match="do not overwrite"):
        write_new_receipt(path, {"different": True})
    assert file_digest(path) == digest
    assert list(path.parent.iterdir()) == [path]


def test_a_writer_return_without_an_actual_receipt_is_not_success(prepared):
    with pytest.raises(QuantError, match="missing"):
        run(prepared, receipt_writer=lambda *args: None)
    assert not prepared[3].exists()
    assert len(state(prepared[0])["registered_candidates"]) == 1
    assert run(prepared, allow_new_registration=False)["action"] == "recovered_registration"


@pytest.mark.parametrize(
    "path", ["config/receipt.json", "data/spec.json", "reports/receipt.txt", "../receipt.json"]
)
def test_receipt_path_cannot_overwrite_inputs_or_escape_the_workspace(prepared, path):
    root, spec, ready, _ = prepared
    before = state(root)
    with pytest.raises(QuantError):
        register_or_recover(spec, ready, Path(path), root=root)
    assert state(root) == before


def test_symlink_receipts_cannot_overwrite_another_file(prepared):
    root, spec, ready, receipt = prepared
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.symlink_to(spec)
    digest, before = file_digest(spec), state(root)
    with pytest.raises(QuantError, match="isolated"):
        register_or_recover(spec, ready, receipt, root=root)
    assert file_digest(spec) == digest
    assert state(root) == before


def test_raw_input_whitespace_change_cannot_silently_rewrite_an_existing_receipt(prepared):
    run(prepared)
    before, digest = state(prepared[0]), file_digest(prepared[3])
    prepared[1].write_text(json.dumps(read_json(prepared[1]), separators=(",", ":")))
    with pytest.raises(QuantError, match="existing receipt differs"):
        run(prepared)
    assert state(prepared[0]) == before
    assert file_digest(prepared[3]) == digest


@pytest.mark.parametrize("raw", ["[]", "{partial", '{"invalid": NaN}'])
def test_malformed_or_nonfinite_input_is_an_explicit_error_not_a_default(prepared, raw):
    before = state(prepared[0])
    prepared[1].write_text(raw)
    with pytest.raises(QuantError):
        run(prepared)
    assert state(prepared[0]) == before
    assert not prepared[3].exists()
