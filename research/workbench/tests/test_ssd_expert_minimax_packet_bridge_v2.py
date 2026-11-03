import copy
import hashlib
import json
from contextlib import nullcontext

import pytest

from us_quant import ssd_expert_minimax_packet_bridge_v2 as bridge
from us_quant.config import QuantError
from us_quant.storage import digest_json, utc_now


def original_evidence():
    proof = {
        "checked_at": "2020-01-01T00:01:00+00:00",
        "frozen_specification_sha256": "original-specification",
        "candidate_accounts_or_metrics_computed": False,
        "source_generator_or_old_account_replayed": False,
        "verified_decisions": 121,
        "mature_decisions": 119,
        "unscored_decisions": 2,
    }
    registration = {"at": "2020-01-01T00:00:00+00:00", "sha256": "original-registration"}
    event = {
        "at": "2020-01-01T00:02:00+00:00",
        "sha256": "original-proof-event",
        "body": {"registered_event_sha256": registration["sha256"]},
    }
    return proof, registration, event


def ref(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(tmp_path, row):
    path = tmp_path / row["path"]
    if path.parent != tmp_path or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise QuantError("synthetic retained file hash changed")
    return path


def real_packet_fixture(monkeypatch, tmp_path, proof, registration_at="2020-01-02T00:00:00Z"):
    frozen = {"study_id": "synthetic-only"}
    target = tmp_path / "targets.csv"
    target.write_text("synthetic target bytes, not market data\n")
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    registration = {
        "at": registration_at,
        "sha256": "new-registered",
        "body": {"candidate": bridge.CANDIDATE, "frozen": frozen},
    }
    event = {
        "at": utc_now(),
        "sha256": "new-target-proof",
        "body": {
            "registered_event_sha256": registration["sha256"],
            "targets": ref(target),
            "proof": ref(path),
        },
    }
    rows = {"registered": [registration], "target_proof": [event]}
    monkeypatch.setattr(bridge.base, "matching_events", lambda kind, candidate: rows.get(kind, []))
    monkeypatch.setattr(bridge.base, "checked_reference", lambda row: checked(tmp_path, row))
    return frozen, target, path


def test_real_packet_constructor_reproduces_missing_v1_flag_failure(monkeypatch, tmp_path):
    old, _, _ = original_evidence()
    frozen, _, _ = real_packet_fixture(
        monkeypatch, tmp_path, old, registration_at="2020-01-01T00:00:00Z"
    )
    with pytest.raises(KeyError, match="original_source_signal_recomputed"):
        bridge.source.common.packet_entry(bridge.CANDIDATE, frozen)


def test_fresh_metadata_view_passes_the_unmodified_real_packet_contract(monkeypatch, tmp_path):
    old, registration, event = original_evidence()
    unchanged = copy.deepcopy(old)
    frozen = {"study_id": "synthetic-only"}
    view = bridge.proof_view(old, registration, event, frozen)
    actual, target, path = real_packet_fixture(monkeypatch, tmp_path, view)
    entry = bridge.source.common.packet_entry(bridge.CANDIDATE, actual)
    assert entry["candidate_id"] == bridge.ID
    assert entry["target_proof"]["targets"] == ref(target)
    assert entry["target_proof"]["proof"] == ref(path)
    assert old == unchanged
    assert "original_source_signal_recomputed" not in old
    assert view["original_source_signal_recomputed"] is False
    assert view["source_original_checked_at"] == old["checked_at"]
    assert view["source_original_registered_at"] == registration["at"]
    assert view["v2_registration_predates_cached_target_derivation"] is False
    assert view["frozen_specification_sha256"] == digest_json(frozen)
    assert view["source_original_frozen_specification_sha256"] == "original-specification"


@pytest.mark.parametrize(
    "flag",
    [
        "candidate_accounts_or_metrics_computed",
        "source_generator_or_old_account_replayed",
    ],
)
def test_view_does_not_turn_source_generation_or_funds_into_false(flag):
    old, registration, event = original_evidence()
    old[flag] = True
    with pytest.raises(QuantError):
        bridge.proof_view(old, registration, event, {})


def test_view_refuses_mismatched_original_registration():
    old, registration, event = original_evidence()
    event["body"]["registered_event_sha256"] = "different-registration"
    with pytest.raises(QuantError):
        bridge.proof_view(old, registration, event, {})


def test_real_packet_contract_rejects_stale_proof_timestamp(monkeypatch, tmp_path):
    old, _, _ = original_evidence()
    old["original_source_signal_recomputed"] = False
    frozen, _, _ = real_packet_fixture(monkeypatch, tmp_path, old)
    with pytest.raises(QuantError, match="follow actual registration"):
        bridge.source.common.packet_entry(bridge.CANDIDATE, frozen)


def test_real_packet_contract_rejects_true_source_flag(monkeypatch, tmp_path):
    old, registration, event = original_evidence()
    view = bridge.proof_view(old, registration, event, {})
    view["original_source_signal_recomputed"] = True
    frozen, _, _ = real_packet_fixture(monkeypatch, tmp_path, view)
    with pytest.raises(QuantError, match="follow actual registration"):
        bridge.source.common.packet_entry(bridge.CANDIDATE, frozen)


def test_real_packet_contract_keeps_exact_cached_target_hash(monkeypatch, tmp_path):
    old, registration, event = original_evidence()
    view = bridge.proof_view(old, registration, event, {})
    frozen, target, _ = real_packet_fixture(monkeypatch, tmp_path, view)
    target.write_text("changed cached target bytes")
    with pytest.raises(QuantError, match="hash changed"):
        bridge.source.common.packet_entry(bridge.CANDIDATE, frozen)


def test_register_uses_cached_outputs_without_any_v1_score_or_pipeline(monkeypatch, tmp_path):
    old, old_registration, old_event = original_evidence()
    target = tmp_path / "source-targets.csv"
    target.write_text("retained synthetic target bytes")
    proof = tmp_path / "source-proof.json"
    proof.write_text(json.dumps(old))
    prepared = tmp_path / "source-prepared.json"
    prepared.write_text(json.dumps({"prepared": {"targets": ref(target), "proof": ref(proof)}}))
    monkeypatch.setattr(bridge.original, "PREPARED", prepared)
    monkeypatch.setattr(bridge, "PROOF", tmp_path / "view.json")
    monkeypatch.setattr(bridge, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(bridge, "reference", ref)
    monkeypatch.setattr(bridge.base, "checked_reference", lambda row: checked(tmp_path, row))
    monkeypatch.setattr(bridge.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(bridge.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(bridge, "inputs", lambda: (None, (), {"study_id": "synthetic"}, None))
    monkeypatch.setattr(bridge, "progress", lambda *args, **kwargs: None)
    rows = []

    def append(journal, kind, candidate, body):
        assert candidate == bridge.ID
        rows.append({"kind": kind, "body": body, "at": utc_now(), "sha256": "new-" + kind})

    monkeypatch.setattr(bridge.base, "append_event", append)
    monkeypatch.setattr(bridge, "events", lambda kind: [row for row in rows if row["kind"] == kind])
    monkeypatch.setattr(
        bridge.base,
        "matching_events",
        lambda kind, candidate: [row for row in rows if row["kind"] == kind],
    )
    monkeypatch.setattr(
        bridge.original,
        "events",
        lambda kind: {
            "registered": [old_registration],
            "target_proof": [old_event],
        }.get(kind, []),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A frozen V1 pipeline or source-score-target builder must not run.")

    for name in ("initialize", "register", "prepare", "evaluate", "export"):
        monkeypatch.setattr(bridge.original, name, forbidden)
    monkeypatch.setattr(bridge.original.rules, "build_targets", forbidden)
    monkeypatch.setattr(
        bridge.source.common,
        "retain_receipt",
        lambda path, body, *args: (path.write_text(json.dumps(body))),
    )
    result = bridge.register()
    packet = json.loads(bridge.PACKET.read_text())
    assert [row["kind"] for row in rows] == ["registered", "target_proof"]
    assert result == ref(bridge.PACKET)
    assert packet["new_economic_hypotheses"] == 0
    assert packet["same_economic_hypothesis_as_terminal_v1"] is True
    assert packet["cached_original_targets_not_recomputed"] is True
    assert packet["entries"][0]["target_proof"]["targets"] == ref(target)
    assert json.loads(bridge.PROOF.read_text())["source_original_checked_at"] == old["checked_at"]
    assert "original_source_signal_recomputed" not in json.loads(proof.read_text())


def test_shared_funding_binding_is_new_identity_without_old_module_mutation():
    old = bridge.financial.evaluate
    old_id = old.__globals__["ID"]
    own = bridge.bound(old)
    assert own.__code__ is old.__code__
    assert own.__globals__["ID"] == bridge.ID != bridge.original.ID
    assert own.__globals__["inputs"] is bridge.inputs
    assert own.__globals__["evaluate_path"] is bridge.evaluate_path
    assert own.__globals__["FOLDER"] == bridge.FOLDER
    assert own.__globals__["PACKET"] == bridge.PACKET
    assert own.__globals__["RESULTS"] == bridge.RESULTS
    assert old.__globals__["ID"] == old_id
    assert bridge.CANDIDATE["economic_rule_id"] == bridge.original.ID


def test_dependency_closure_binds_both_immutable_original_and_metadata_bridge():
    refs = {row["path"] for row in bridge.dependencies()}
    assert {
        "config/expert-minimax-packet-bridge-v2.json",
        "config/expert-minimax-opportunity-v1.json",
        "src/us_quant/ssd_expert_minimax_packet_bridge_v2.py",
        "src/us_quant/ssd_expert_minimax_opportunity_v1.py",
        "src/us_quant/expert_minimax_opportunity_v1.py",
        "src/us_quant/ssd_expert_target_consensus_v1.py",
        "tests/test_ssd_expert_minimax_packet_bridge_v2.py",
    }.issubset(refs)
