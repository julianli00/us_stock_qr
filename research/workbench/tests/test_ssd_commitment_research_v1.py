from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    directory = tmp_path / "events"
    monkeypatch.setattr(base, "JOURNAL", directory)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    base.append_event(directory, "baseline", None, {"synthetic": True})
    return directory


def complete(journal, identifier):
    for key in sorted(base.PATH_KEYS):
        base.append_event(journal, "path_completed", identifier, {"path_key": key})
    base.append_event(
        journal,
        "review",
        identifier,
        {
            "status": "rejected",
            "paths": dict.fromkeys(base.PATH_KEYS, {}),
        },
    )


def test_counts_preserve_old_pair_and_include_new_pending_and_failed_versions(journal):
    for identifier in ("old_a", "old_b", *study.IDS):
        base.append_event(journal, "registered", identifier, {})
    complete(journal, "old_a")
    complete(journal, "old_b")
    base.append_event(journal, "technical_failure", study.IDS[0], {"stage": "synthetic"})
    counts = study.counts(base.journal_events(journal))
    assert counts == {
        "new_registered_candidates": 4,
        "new_completed_strategy_evaluations": 2,
        "new_reviews": 2,
        "new_technical_failures": 1,
        "new_pending_candidates": 1,
        "new_account_paths": 8,
        "new_historical_point_gate_passes": 0,
    }


@pytest.mark.parametrize("damage", ["duplicate", "double_verdict", "missing_path", "unregistered"])
def test_inconsistent_counts_fail_closed(journal, damage):
    base.append_event(journal, "registered", "candidate", {})
    complete(journal, "candidate")
    events = base.journal_events(journal)
    if damage == "duplicate":
        events.append(deepcopy(events[1]))
    elif damage == "double_verdict":
        events.append({"kind": "technical_failure", "candidate_id": "candidate", "body": {}})
    elif damage == "missing_path":
        events.pop(2)
    else:
        events.pop(1)
    with pytest.raises(QuantError):
        study.counts(events)


def test_expiration_preserves_pending_trial_not_a_fake_technical_failure(journal, monkeypatch):
    for identifier in study.IDS:
        base.append_event(journal, "registered", identifier, {})
    recorded = []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T15:59:00Z")
    monkeypatch.setattr(base, "progress", lambda phase, **fields: recorded.append((phase, fields)))
    study.record_failure("synthetic", study.IDS, QuantError("Authorization expired"))
    assert not base.matching_events("technical_failure", study.IDS[0])
    assert recorded[0][0] == "commitment_authorization_expired_preserved"
    assert recorded[0][1]["new_pending_candidates"] == 2
    assert recorded[0][1]["active_compute_job"] is None


def test_real_predeadline_failure_is_counted_once_and_retained(journal, monkeypatch):
    for identifier in study.IDS:
        base.append_event(journal, "registered", identifier, {})
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-27T18:00:00Z")
    monkeypatch.setattr(base, "progress", lambda phase, **fields: None)
    for _ in range(2):
        study.record_failure(
            "synthetic_target_proof", (study.IDS[0],), QuantError("Oracle differs")
        )
    assert study.counts(base.journal_events(journal))["new_technical_failures"] == 1
    failure = base.matching_events("technical_failure", study.IDS[0])[0]
    assert failure["body"]["reason"] == "Oracle differs"
    assert failure["body"]["partial_outputs_retained"] is True
    assert study.counts(base.journal_events(journal))["new_pending_candidates"] == 1


def test_receipt_recovery_never_overwrites_conflicting_evidence(journal):
    path = journal.parent / "result.json"
    body = {"created_at": "old", "actual": 2}
    study.retain_receipt(path, body)
    assert study.retain_receipt(path, {"created_at": "new", "actual": 2}, ("created_at",)) == body
    with pytest.raises(QuantError, match="preserved"):
        study.retain_receipt(path, {"created_at": "new", "actual": 3}, ("created_at",))
    assert read_json(path) == body


def test_evidence_symlink_is_rejected_before_any_write(journal):
    destination = journal.parent / "destination"
    destination.mkdir()
    link = journal.parent / "redirect"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(QuantError, match="actual SSD"):
        study.retain_receipt(link / "result.json", {"test": True})
    assert not (destination / "result.json").exists()


def test_publication_verifies_new_packet_not_only_the_inherited_one(journal, monkeypatch):
    calls = []
    frozen, parent = {"study_id": "synthetic_new_study"}, {"study_id": "original_study"}
    packet = {"path": "evidence/new.json", "sha256": "a" * 64}
    commit = "b" * 40
    monkeypatch.setattr(base, "reference", lambda path: packet)
    monkeypatch.setattr(
        base,
        "published_inputs",
        lambda value, sha: calls.append(("parent", value, sha)) or {"source": True},
    )

    def verify(root, reference, required, sha):
        calls.append(("new", reference, required, sha))
        return {"published_commit": sha, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    receipt = study.publication(frozen, parent, commit)
    assert calls == [("parent", parent, commit), ("new", packet, set(study.EXTENSIONS), commit)]
    assert receipt["registration_packet"] == packet
    assert receipt["parent_dependency_publication"] == {"source": True}
    study.publication(frozen, parent, commit)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, parent, "c" * 40)
    assert len(base.matching_events("publication")) == 1


def test_packet_binds_both_actual_registrations_and_proofs_before_any_account(journal, monkeypatch):
    frozen = {"study_id": "synthetic_study"}
    proofs = {}
    monkeypatch.setattr(base, "checked_reference", lambda reference: proofs[reference["path"]])
    for index, candidate in enumerate(study.rules.CANDIDATES):
        identifier = candidate["id"]
        registration = base.append_event(
            journal, "registered", identifier, {"candidate": candidate, "frozen": frozen}
        )
        proof_path = journal.parent / f"proof-{index}.json"
        write_json(
            proof_path,
            {
                "checked_at": utc_now(),
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
            },
        )
        proofs[identifier] = proof_path
        base.append_event(
            journal,
            "target_proof",
            identifier,
            {
                "registered_event_sha256": registration["sha256"],
                "targets": {"path": identifier},
                "proof": {"path": identifier},
            },
        )
    packet_path = journal.parent / "packet.json"
    monkeypatch.setattr(study, "PACKET", packet_path)
    packet = {
        "schema_version": 2,
        "program_id": "ssd_continuous_research_v1",
        "study_id": frozen["study_id"],
        "candidate_accounts_or_metrics_computed": False,
        "raw_inputs_and_target_matrices_remain_private": True,
        "investment_objective_verified": False,
        "order_authority": False,
        "independent_forward_validation": False,
        "new_registered_candidates": 2,
        "entries": [study.packet_entry(candidate, frozen) for candidate in study.rules.CANDIDATES],
        "created_at": utc_now(),
    }
    write_json(packet_path, packet)
    assert study.checked_packet(frozen) == packet
    changed = deepcopy(packet)
    changed["entries"].pop()
    write_json(packet_path, changed)
    with pytest.raises(QuantError, match="complete unchanged"):
        study.checked_packet(frozen)
    changed = deepcopy(packet)
    changed["created_at"] = (pd.Timestamp(utc_now()) + pd.Timedelta(days=1)).isoformat()
    write_json(packet_path, changed)
    with pytest.raises(QuantError, match="complete unchanged"):
        study.checked_packet(frozen)
