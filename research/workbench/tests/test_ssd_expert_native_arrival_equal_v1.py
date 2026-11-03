import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from test_expert_native_arrival_equal_v1 import ASSETS, examples
from test_expert_native_event_execution_v1 import daily_targets

from us_quant import expert_native_arrival_equal_v1 as rules
from us_quant import expert_native_event_execution_v1 as execution
from us_quant import ssd_expert_native_arrival_equal_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now


def fixture(tmp_path):
    calendar, experts = examples("2016-10-12")
    rows = []
    target, audits = rules.build_targets(
        experts, calendar, ASSETS, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=ASSETS))
    path = tmp_path / "source-opinions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return data, experts, path, target, audits, rows


def test_actual_query_source_operands_and_rational_mean_have_separate_proof(tmp_path):
    data, experts, path, targets, audits, _ = fixture(tmp_path)
    proof = study.verify_retained(path, data, experts, targets, audits)
    assert proof["native_decisions_verified"] == len(audits)
    assert proof["actual_query_input_resolved_records"] == 3 * len(audits)
    assert proof["source_opinion_snapshots_verified"] == 8 * len(audits)
    assert proof["source_refresh_counts_excluding_initial"] == [
        0,
        0,
        0,
        len(audits) - 1,
        0,
        0,
        0,
        0,
    ]
    assert proof["maximum_independent_rational_target_difference"] < 1e-12


@pytest.mark.parametrize(
    "kind",
    [
        "query",
        "known_date",
        "source_value",
        "arrival",
        "native_order",
        "aggregate",
        "weights",
    ],
)
def test_query_clock_source_or_resolved_output_tampering_is_rejected(tmp_path, kind):
    data, experts, path, targets, audits, rows = fixture(tmp_path)
    if kind == "query":
        rows[0]["native_query_sessions"][0] = "2016-10-05"
    elif kind == "known_date":
        rows[1]["source_opinions"][0]["source_known_at"] = "2016-10-05"
    elif kind == "source_value":
        rows[1]["source_opinions"][0]["target_binary64_hex"][0] = (0.25).hex()
    elif kind == "arrival":
        rows[1]["source_opinions"][0]["arrived_today"] = True
    elif kind == "native_order":
        rows[1]["source_opinions"][0]["source_native_order_or_execution_claimed"] = True
    elif kind == "aggregate":
        targets.loc["2016-10-05", "SPY"] += 0.01
    else:
        rows[2]["audit"]["expert_weights"][0] = 0.5
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, targets, audits)


def test_extra_or_incomplete_native_records_cannot_be_ignored(tmp_path):
    data, experts, path, targets, audits, rows = fixture(tmp_path)
    path.write_text(path.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(QuantError, match="extra or partial"):
        study.verify_retained(path, data, experts, targets, audits)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_legacy_funding_interface_delegates_both_callbacks_to_new_native_module(cost, delay):
    data, targets = daily_targets()
    policy = read_json(execution.POLICY)
    token = execution.monthly.CANDIDATES[0]
    result, issued, decisions = study.execution.run(
        data, targets, token, policy, "2016-10-06", "2016-10-20", cost, delay
    )
    direct, direct_issued, _ = execution.run(
        data, targets, policy, "2016-10-06", "2016-10-20", cost, delay
    )
    pd.testing.assert_frame_equal(result.frame, direct.frame, check_exact=True)
    assert issued.equals(direct_issued)
    proof = study.execution.original.verify_feedback(
        data,
        targets,
        result,
        issued,
        decisions,
        {"start": "2016-10-06", "end": "2016-10-20"},
        {"cost_bps_per_side": cost, "delay_sessions": delay},
    )
    assert proof["complete_frame_and_positions_reconciled_to_cash_engine"]


def test_scoped_funding_does_not_mutate_original_monthly_module_or_identity():
    original = study.financial.evaluate_path
    old_execution = original.__globals__["execution"]
    own = study.bound(original)
    assert own.__code__ is original.__code__
    assert own.__globals__["execution"] is study.execution
    assert own.__globals__["ID"] == study.ID
    assert own.__globals__["FOLDER"] == study.FOLDER
    assert own.__globals__["events"] is study.events
    assert original.__globals__["execution"] is old_execution
    assert old_execution is not study.execution


def test_unregistered_source_opinions_cannot_generate_new_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(
        study.rules,
        "build_targets",
        lambda *args, **kwargs: pytest.fail("new targets require actual registration"),
    )
    with pytest.raises(QuantError, match="Register once"):
        study.prepare(None, (), {})
    assert not (tmp_path / "preparation-attempt-v1.json").exists()


def test_real_packet_constructor_is_exercised_after_actual_registration_order(
    monkeypatch, tmp_path
):
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


def test_complete_native_financial_dependency_closure():
    refs = {row["path"] for row in study.dependencies()}
    assert {
        "config/expert-native-arrival-equal-v1.json",
        "config/expert-native-event-execution-v1.json",
        "src/us_quant/expert_native_arrival_equal_v1.py",
        "src/us_quant/expert_native_event_execution_v1.py",
        "src/us_quant/ssd_expert_native_arrival_equal_v1.py",
        "src/us_quant/ssd_expert_target_consensus_v1.py",
        "tests/test_ssd_expert_native_arrival_equal_v1.py",
    }.issubset(refs)
    assert "src/us_quant/macro_policy_compatibility_v1.py" not in refs
