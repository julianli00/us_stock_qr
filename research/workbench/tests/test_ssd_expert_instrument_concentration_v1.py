import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from test_expert_native_arrival_equal_v1 import ASSETS, examples

from us_quant import expert_instrument_concentration_v1 as rules
from us_quant import ssd_expert_instrument_concentration_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now


def fixture(tmp_path):
    calendar, experts = examples()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, ASSETS, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=ASSETS))
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return data, experts, path, targets, audits, rows


def test_full_current_pair_menu_and_rational_concentration_target_are_reconstructed(tmp_path):
    data, experts, path, targets, audits, _ = fixture(tmp_path)
    proof = study.verify_retained(path, data, experts, targets, audits)
    assert proof["verified_decisions"] == 4
    assert proof["actual_input_resolved_records"] == 8
    assert proof["all28_current_pair_concentration_records_verified"]
    assert proof["maximum_independent_rational_target_difference"] < 1e-12
    assert proof["maximum_independent_rational_concentration_difference"] < 1e-12


@pytest.mark.parametrize("kind", ["source", "key", "prior", "pair_weight", "target", "drop_pair"])
def test_retained_source_composition_prior_or_selection_tampering_is_rejected(tmp_path, kind):
    data, experts, path, targets, audits, rows = fixture(tmp_path)
    if kind == "source":
        rows[0]["source_opinions"][0]["target_binary64_hex"][0] = (0.25).hex()
    elif kind == "key":
        rows[1]["audit"]["selection"]["pair_records"][0]["concentration_key_12dp"] = 0.0
    elif kind == "prior":
        rows[2]["previous_expert_weights_binary64_hex"] = None
    elif kind == "pair_weight":
        rows[1]["audit"]["selection"]["pair_records"][0]["expert_weights"][0] = 0.7
    elif kind == "target":
        targets.loc["2016-10-05", "SPY"] += 0.01
    else:
        rows[1]["audit"]["selection"]["pair_records"].pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, targets, audits)


def test_extra_retained_record_or_nonmonthly_target_cannot_be_ignored(tmp_path):
    data, experts, path, targets, audits, rows = fixture(tmp_path)
    targets.loc["2016-10-12"] = targets.loc["2016-10-05"]
    with pytest.raises(QuantError, match="original capital/month-end"):
        study.verify_retained(path, data, experts, targets, audits)
    targets.loc["2016-10-12"] = float("nan")
    path.write_text(path.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(QuantError, match="extra or partial"):
        study.verify_retained(path, data, experts, targets, audits)


def test_unregistered_opinions_never_generate_current_hhi_or_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(
        study.rules,
        "build_targets",
        lambda *args, **kwargs: pytest.fail("new composition must follow actual registration"),
    )
    with pytest.raises(QuantError, match="Register once"):
        study.prepare(None, (), {})


def test_funding_scope_keeps_old_monthly_execution_and_does_not_mutate_template():
    original = study.financial.evaluate_path
    execution = original.__globals__["execution"]
    bound = study.bound(original)
    assert bound.__code__ is original.__code__
    assert bound.__globals__["ID"] == study.ID
    assert bound.__globals__["FOLDER"] == study.FOLDER
    assert bound.__globals__["events"] is study.events
    assert bound.__globals__["execution"] is execution
    assert original.__globals__["execution"] is execution
    assert original.__globals__["ID"] != study.ID


def test_unmodified_real_packet_contract_is_used_after_registration(monkeypatch, tmp_path):
    rows = []
    frozen = {"study_id": "synthetic-only"}
    target = tmp_path / "targets.csv"
    target.write_text("synthetic target bytes")
    proof = tmp_path / "proof.json"
    packet = tmp_path / "packet.json"

    def reference(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def checked(row):
        path = tmp_path / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise QuantError("retained synthetic file changed")
        return path

    def append(journal, kind, candidate, body):
        assert candidate == study.ID
        rows.append({"kind": kind, "body": body, "at": utc_now(), "sha256": "new-" + kind})

    def prepare(*args):
        assert [row["kind"] for row in rows] == ["registered"]
        proof.write_text(
            json.dumps(
                {
                    "checked_at": utc_now(),
                    "candidate_accounts_or_metrics_computed": False,
                    "original_source_signal_recomputed": False,
                }
            )
        )
        return {"targets": reference(target), "proof": reference(proof)}

    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, (), frozen, None))
    monkeypatch.setattr(study, "events", lambda kind: [r for r in rows if r["kind"] == kind])
    monkeypatch.setattr(
        study.base,
        "matching_events",
        lambda kind, candidate: [r for r in rows if r["kind"] == kind],
    )
    monkeypatch.setattr(study.base, "append_event", append)
    monkeypatch.setattr(study.base, "checked_reference", checked)
    monkeypatch.setattr(study, "prepare", prepare)
    monkeypatch.setattr(study, "PACKET", packet)
    monkeypatch.setattr(study, "reference", reference)
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study.source.common,
        "retain_receipt",
        lambda path, body, *args: (path.write_text(json.dumps(body))),
    )
    result = study.register()
    saved = json.loads(packet.read_text())
    assert result == reference(packet)
    assert [row["kind"] for row in rows] == ["registered", "target_proof"]
    assert saved["entries"][0]["candidate_id"] == study.ID
    assert saved["entries"][0]["registered_at"] <= saved["entries"][0]["target_proof_at"]


def test_complete_dependency_closure_binds_composition_not_old_native_clock_execution():
    paths = {row["path"] for row in study.dependencies()}
    assert {
        "config/expert-instrument-concentration-v1.json",
        "src/us_quant/expert_instrument_concentration_v1.py",
        "src/us_quant/ssd_expert_instrument_concentration_v1.py",
        "src/us_quant/ssd_expert_target_consensus_v1.py",
        "tests/test_ssd_expert_instrument_concentration_v1.py",
    }.issubset(paths)
    assert "src/us_quant/macro_policy_compatibility_v1.py" not in paths
