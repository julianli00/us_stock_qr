from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
from test_conditional_hedge_expert_v1 import market

from us_quant import ssd_conditional_hedge_native_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


def test_native_declaration_references_the_actual_frozen_primary_without_editing_it():
    study.check_primary()
    policy = read_json(study.POLICY)
    assert policy["candidates"] == [study.CANDIDATE]
    assert policy["primary_law"] == study.reference(study.law.POLICY)
    assert policy["pool_inclusion_automatic_after_technical_admission"] is False


def test_rebound_native_registration_uses_only_new_scoped_identity_and_no_global_patch():
    original = {
        key: study.template.register.__globals__[key]
        for key in ("ID", "FOLDER", "CANDIDATE", "PACKET", "lifecycle")
    }
    new = study.rebound(study.template.register)
    assert new.__globals__["ID"] == study.ID
    assert new.__globals__["PACKET"] == study.PACKET
    assert new.__globals__["FOLDER"] == study.FOLDER
    assert study.PACKET != study.template.PACKET
    for key, value in original.items():
        assert study.template.register.__globals__[key] is value
    assert new.__globals__["lifecycle"] is study.lifecycle


def test_native_dependency_closure_contains_primary_and_actual_shared_accounting():
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_expert_target_consensus_v1.py" in paths
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
    assert "src/us_quant/strategy_experts_v1.py" in paths


def preparation():
    close = market()
    records = []
    target, audits = study.law.build_targets(
        close, read_json(study.law.POLICY), retain=records.append
    )
    return close, target, audits, records


def test_whole_native_monthly_lattice_and_all_exact_before_after_inputs_match_oracle():
    close, target, audits, records = preparation()
    proof = study.verify_targets(close, target, audits, records)
    assert proof["genuine_month_end_decisions"] == len(audits)
    assert proof["retained_input_resolved_records"] == 2 * len(audits)
    assert proof["maximum_independent_integer_target_difference"] == 0


@pytest.mark.parametrize("kind", ["hex", "past", "clock", "selected", "resolved"])
def test_source_retention_or_independent_choice_mismatch_stops_native_proof(kind):
    close, target, audits, records = preparation()
    audits, records = copy.deepcopy(audits), copy.deepcopy(records)
    if kind == "hex":
        records[0]["used_source_prices_binary64_hex"]["SPY"][0] = float(1).hex()
    elif kind == "past":
        records[0]["SPY_year_prior_source_date"] = "1900-01-01"
    elif kind == "clock":
        records[0]["information_clock"] = "16_00_ET"
    elif kind == "selected":
        audits[0]["selected_assets"] = ["BIL"]
        records[1]["audit"] = audits[0]
    else:
        records[1]["complete_target"][0] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(close, target, audits, records)


def test_real_source_generation_cannot_precede_actual_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "events", lambda kind: [])
    with pytest.raises(QuantError, match="registration must precede"):
        study.prepare(None, None, {})
    assert not (tmp_path / "native-preparation-attempt-v1.json").exists()


def test_partial_preparation_is_preserved_instead_of_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    write_json(tmp_path / "native-preparation-attempt-v1.json", {"partial": True})
    with pytest.raises(QuantError, match="not automatically repeated"):
        study.prepare(None, None, {})


def test_native_remote_fence_failure_does_not_start_a_source_account(tmp_path, monkeypatch):
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(study, "RESULTS", tmp_path / "absent.json")

    def failed(*args):
        raise QuantError("Synthetic source preregistration failed")

    monkeypatch.setattr(study, "publication", failed)
    monkeypatch.setattr(
        study, "evaluate_path", lambda *args: pytest.fail("A source account started")
    )
    with pytest.raises(QuantError, match="preregistration failed"):
        study.evaluate("0" * 40)


def test_native_completed_source_path_recovery_never_reexecutes_old_or_new_book(monkeypatch):
    monkeypatch.setattr(
        study,
        "events",
        lambda kind: [
            {"body": {"path_key": "10y-base", "receipt": {"path": "existing", "sha256": "old"}}}
        ]
        if kind == "path_completed"
        else [],
    )
    monkeypatch.setattr(study.base, "verify_path", lambda *args: {"verified_existing": True})
    assert study.evaluate_path(None, None, {}, {"years": 10}, {"id": "base"}, None) == {
        "verified_existing": True
    }
