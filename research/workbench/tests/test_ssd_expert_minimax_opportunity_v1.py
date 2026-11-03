import copy
import json
from contextlib import nullcontext

import pandas as pd
import pytest
from test_expert_minimax_opportunity_v1 import ASSETS, examples, policy

from us_quant import expert_minimax_opportunity_v1 as rules
from us_quant import ssd_expert_minimax_opportunity_v1 as study
from us_quant.config import QuantError


def retained_fixture(tmp_path):
    calendar, experts = examples("2017-01-05")
    rows = []
    targets, audits = rules.build_targets(experts, calendar, ASSETS, policy(), retain=rows.append)
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, experts, targets, audits, rows


def test_retained_native_operands_scalar_compounding_all_pairs_and_rational_target(tmp_path):
    path, experts, targets, audits, _ = retained_fixture(tmp_path)
    proof = study.verify_retained(path, experts, targets, audits)
    assert proof["verified_decisions"] == 4
    assert proof["verified_actual_input_resolved_records"] == 8
    assert proof["mature_decisions"] == 2
    assert proof["unscored_decisions"] == 2
    assert proof["all_28_pairs_per_mature_decision_verified"]
    assert proof["maximum_independent_scalar_relative_difference"] < 1e-12
    assert proof["maximum_independent_rational_target_difference"] < 1e-12


@pytest.mark.parametrize("kind", ["net", "rf", "order", "state", "target", "duplicate_pair"])
def test_retained_or_resolved_tampering_is_not_a_valid_target_proof(tmp_path, kind):
    path, experts, targets, audits, rows = retained_fixture(tmp_path)
    if kind == "net":
        rows[4]["observations"][0]["new_NET_binary64_hex"][0] = (0.25).hex()
    elif kind == "rf":
        rows[4]["observations"][0]["new_RF_binary64_hex"][0] = (0.0).hex()
    elif kind == "order":
        rows[4]["observations"][0]["asset_order"] = list(reversed(ASSETS))
    elif kind == "state":
        rows[6]["previous_matured_months_binary64_hex"] = []
    elif kind == "target":
        targets.iloc[-1, 0] = 0.1
        rows[-1]["target_binary64_hex"] = [
            float(x).hex() for x in targets.loc[pd.Timestamp(audits[-1]["decision_session"])]
        ]
    else:
        rows[5]["audit"]["selection"]["pair_records"].append(
            copy.deepcopy(rows[5]["audit"]["selection"]["pair_records"][0])
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained(path, experts, targets, audits)


def test_extra_or_incomplete_resolved_records_are_not_silently_discarded(tmp_path):
    path, experts, targets, audits, rows = retained_fixture(tmp_path)
    path.write_text(path.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(QuantError, match="extra or partial"):
        study.verify_retained(path, experts, targets, audits)


def test_financial_function_binding_uses_only_new_study_identity_without_global_mutation():
    original = study.financial.evaluate
    old_id = original.__globals__["ID"]
    old_folder = original.__globals__["FOLDER"]
    own = study.bound(original)
    assert own.__code__ is original.__code__
    assert own.__globals__["ID"] == study.ID
    assert own.__globals__["FOLDER"] == study.FOLDER
    assert own.__globals__["RESULTS"] == study.RESULTS
    assert own.__globals__["PACKET"] == study.PACKET
    assert own.__globals__["inputs"] is study.inputs
    assert own.__globals__["evaluate_path"] is study.evaluate_path
    assert own.__globals__["publication"] is study.publication
    assert own.__globals__["lifecycle"] is study.lifecycle
    assert original.__globals__["ID"] == old_id != study.ID
    assert original.__globals__["FOLDER"] == old_folder != study.FOLDER


def test_actual_registration_precedes_preparation_and_synthetic_failure_is_preserved(monkeypatch):
    recorded, failures = [], []
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, (), {"study_id": "synthetic"}, None))
    monkeypatch.setattr(
        study, "events", lambda kind: [row for row in recorded if row["kind"] == kind]
    )
    monkeypatch.setattr(
        study.base,
        "append_event",
        lambda journal, kind, candidate, body: (
            recorded.append({"kind": kind, "candidate": candidate, "body": body})
        ),
    )
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study, "record_failure", lambda stage, error: failures.append((stage, str(error)))
    )

    def stop_after_registration(*args):
        assert len(recorded) == 1
        assert recorded[0]["kind"] == "registered"
        assert recorded[0]["candidate"] == study.ID
        raise QuantError("synthetic stop before any actual score or account")

    monkeypatch.setattr(study, "prepare", stop_after_registration)
    with pytest.raises(QuantError, match="synthetic stop"):
        study.register()
    assert failures == [("preparation", "synthetic stop before any actual score or account")]


def test_prepare_refuses_unregistered_input_before_rule_or_target_computation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(
        study.rules,
        "build_targets",
        lambda *args, **kwargs: pytest.fail("unregistered rule must not run"),
    )
    with pytest.raises(QuantError, match="Register once"):
        study.prepare(None, (), {})
    assert not (tmp_path / "preparation-attempt-v1.json").exists()


def test_complete_dependency_closure_binds_the_actual_rule_and_financial_adapter():
    refs = {row["path"] for row in study.dependencies()}
    assert {
        "config/expert-minimax-opportunity-v1.json",
        "src/us_quant/expert_minimax_opportunity_v1.py",
        "src/us_quant/ssd_expert_minimax_opportunity_v1.py",
        "src/us_quant/ssd_expert_target_consensus_v1.py",
        "src/us_quant/expert_initial_monthly_execution_v1.py",
        "tests/test_expert_minimax_opportunity_v1.py",
        "tests/test_ssd_expert_minimax_opportunity_v1.py",
    }.issubset(refs)
    assert "src/us_quant/macro_policy_compatibility_v1.py" not in refs
