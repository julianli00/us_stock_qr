import copy
from contextlib import nullcontext

import pytest
from test_credit_quarter_policy_v1 import market

from us_quant import ssd_credit_quarter_policy_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


def test_reused_lifecycle_is_rebound_to_new_identity_without_old_global_mutation():
    names = ("ID", "FOLDER", "CANDIDATE", "PACKET", "lifecycle")
    before = {name: study.template.register.__globals__[name] for name in names}
    new = study.bound(study.template.register)
    for name in names:
        assert new.__globals__[name] is getattr(study, name)
        assert study.template.register.__globals__[name] is before[name]
    assert study.PACKET != study.native.PACKET
    assert study.ID != study.native.ID


def test_scientific_dependency_closure_contains_real_credit_and_funding_implementations():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
    assert "src/us_quant/expert_initial_monthly_execution_v1.py" in paths
    assert "src/us_quant/credit_relative_input_v1.py" in paths


def preparation():
    close, credit = market()
    records = []
    targets, audits = study.law.build_targets(
        close, credit, read_json(study.POLICY), retain=records.append
    )
    return close, credit, targets, audits, records


def test_actual_target_proof_interface_checks_all_retained_input_and_resolved_rows():
    close, credit, targets, audits, records = preparation()
    checked = study.verify_targets(close, credit, targets, audits, records)
    assert checked["genuine_month_end_decisions"] == len(audits)
    assert checked["retained_input_resolved_records"] == 2 * len(audits)
    assert checked["maximum_independent_credit_target_difference"] == 0


@pytest.mark.parametrize("failure", ["input", "clock", "target", "selection"])
def test_mutated_source_or_resolved_evidence_stops_proof(failure):
    close, credit, targets, audits, records = preparation()
    records, audits = copy.deepcopy(records), copy.deepcopy(audits)
    if failure == "input":
        records[0]["credit_relative_binary64_hex"] = "0x0.0p+0"
    elif failure == "clock":
        records[0]["credit_source_end"] = records[0]["decision_session"]
    elif failure == "target":
        records[1]["complete_target"][0] += 0.1
    else:
        audits[0]["selected_assets"] = ["BIL"]
        records[1]["audit"] = audits[0]
    with pytest.raises(QuantError):
        study.verify_targets(close, credit, targets, audits, records)


def test_no_real_policy_target_can_precede_actual_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "events", lambda kind: [])
    with pytest.raises(QuantError, match="registration must precede"):
        study.prepare(None, None, {})
    assert not (tmp_path / "preparation-attempt-v1.json").exists()


def test_partial_preparation_is_preserved_instead_of_automatically_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    write_json(tmp_path / "preparation-attempt-v1.json", {"partial": True})
    with pytest.raises(QuantError, match="cannot be automatically repeated"):
        study.prepare(None, None, {})


def test_failed_actual_remote_fence_never_starts_a_credit_account(tmp_path, monkeypatch):
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(study, "RESULTS", tmp_path / "absent.json")

    def failed(*args):
        raise QuantError("Synthetic credit preregistration failed")

    monkeypatch.setattr(study, "publication", failed)
    monkeypatch.setattr(
        study, "evaluate_path", lambda *args: pytest.fail("A funded account started")
    )
    with pytest.raises(QuantError, match="preregistration failed"):
        study.evaluate("0" * 40)


def test_complete_credit_account_recovery_reads_receipt_without_execution(monkeypatch):
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
