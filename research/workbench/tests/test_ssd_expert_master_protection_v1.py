from __future__ import annotations

import copy

import pytest

from us_quant import ssd_expert_master_protection_v1 as study
from us_quant.config import QuantError


def proof_fixture(monkeypatch):
    candidate = study.rules.CANDIDATES[0]
    nominal = {"path": "data/retained/nominal.csv", "sha256": "a" * 64}
    proof = {"path": "data/retained/proof.json", "sha256": "b" * 64}
    source = {
        "registered_event_sha256": "c" * 64,
        "target_proof": {"targets": nominal, "proof": proof},
    }
    frozen = {
        "target_proof_role": study.PROOF_ROLE,
        "state_contingent_issued_targets_precomputed": False,
        "source_nominal_target_bindings": {
            candidate["id"]: {
                "source_candidate_id": candidate["source_candidate_id"],
                "targets": nominal,
                "proof": proof,
            }
        },
        "execution_adapter": {"path": study.OWN[1], "sha256": "d" * 64},
        "validation_evidence": [{"path": "reports/tests.xml", "sha256": "e" * 64}],
    }
    checked = []
    monkeypatch.setattr(study.base, "checked_reference", lambda r: checked.append(r))
    return candidate, frozen, source, {"sha256": "f" * 64}, checked


def test_closed_loop_proof_binds_only_old_nominal_targets(monkeypatch):
    candidate, frozen, source, registration, checked = proof_fixture(monkeypatch)
    result = study.transition_proof(candidate, frozen, source, registration)
    assert result["source_nominal_targets"] == source["target_proof"]["targets"]
    assert result["source_nominal_target_proof"] == source["target_proof"]["proof"]
    assert result["controller_transition"] == frozen["execution_adapter"]
    assert result["candidate_accounts_or_metrics_computed"] is False
    assert result["state_contingent_issued_targets_precomputed"] is False
    assert result["original_source_signal_recomputed"] is False
    assert result["registered_event_sha256"] == registration["sha256"]
    assert checked == [source["target_proof"]["targets"], source["target_proof"]["proof"]]


@pytest.mark.parametrize("damage", ["future_targets", "different_source", "wrong_role"])
def test_transition_proof_rejects_relabelled_or_precomputed_state(damage, monkeypatch):
    candidate, frozen, source, registration, _ = proof_fixture(monkeypatch)
    if damage == "future_targets":
        frozen["state_contingent_issued_targets_precomputed"] = True
    elif damage == "different_source":
        frozen["source_nominal_target_bindings"][candidate["id"]]["targets"] = {
            "path": "data/unrelated.csv",
            "sha256": "0" * 64,
        }
    else:
        frozen["target_proof_role"] = "final_targets"
    with pytest.raises(QuantError, match="precomputed"):
        study.transition_proof(candidate, frozen, source, registration)


def publication_fixture(monkeypatch, found=None):
    calls = []
    commit, explicit, imported = "a" * 40, study.OWN[1], "src/us_quant/backtest.py"
    frozen = {
        "study_id": "synthetic_closed_loop",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("full_dependencies")
        return {"published_commit": sha, "unchanged_pinned_import_files_verified": 1}

    def packet(root, ref, required, sha):
        assert root == study.ROOT and sha == commit
        assert ref["path"] == study.PACKET.relative_to(study.ROOT).as_posix()
        assert required == {explicit}
        calls.append("explicit_packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": p.relative_to(study.ROOT).as_posix(),
            "sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(study, "read_json", lambda p: {"files": [f"research/workbench/{explicit}"]})
    monkeypatch.setattr(study, "matching_events", lambda kind: found or [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("fence"))
    return frozen, commit, calls


def test_publication_checks_all_dependencies_before_its_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)
    receipt = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet", "fence"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_unknown_import_failure_prevents_packet_approval_and_funded_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)

    def refuse(*args):
        raise QuantError("Unlisted actual dependency")

    monkeypatch.setattr(study, "published_inputs", refuse)
    with pytest.raises(QuantError, match="Unlisted"):
        study.publication(frozen, commit)
    assert calls == []


def test_existing_publication_cannot_move_to_a_post_outcome_commit(monkeypatch):
    found = [
        {
            "body": {
                "study_id": "synthetic_closed_loop",
                "publication": {"published_commit": "d" * 40},
            }
        }
    ]
    frozen, commit, calls = publication_fixture(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet"]


def test_duplicate_completed_paths_cannot_be_overwritten_by_dictionary_selection(monkeypatch):
    duplicate = [{"body": {"path_key": "10y-base", "receipt": {}}}] * 2
    monkeypatch.setattr(
        study,
        "matching_events",
        lambda kind, name: duplicate if kind == "path_completed" else [],
    )
    with pytest.raises(QuantError, match="Duplicate actual completion"):
        study.states({})


def test_deadline_failure_preserves_partials_without_new_research_events(monkeypatch):
    captured, events = [], []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T16:00:00+00:00")
    monkeypatch.setattr(study, "append_event", lambda *args: events.append(args))
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: captured.append(args))
    study.failure("synthetic_stage", study.IDS, QuantError("deadline"))
    assert events == []
    assert captured == [("expert_master_protection_expired",)]


def test_failure_does_not_invent_unregistered_candidate_records(monkeypatch):
    captured, events = [], []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-29T22:00:00+00:00")
    monkeypatch.setattr(study, "matching_events", lambda kind, name: [])
    monkeypatch.setattr(study, "append_event", lambda *args: events.append(args))
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: captured.append(args))
    study.failure("synthetic_stage", study.IDS, QuantError("unregistered"))
    assert events == []
    assert captured == [("expert_master_protection_blocked",)]


def test_public_result_boundary_uses_canonical_flags_without_raw_state(monkeypatch):
    frozen = {"study_id": "synthetic_closed_loop", "protocol": {"path": "config/original.json"}}
    actual = {
        name: {
            "candidate_id": name,
            "status": "technical_failure",
            "paths": {},
            "failure_event_sha256": "a" * 64,
            "failure_stage": "synthetic",
        }
        for name in study.IDS
    }
    saved = {
        "candidates": copy.deepcopy(actual),
        "finished_at": "2026-10-29T22:00:00+00:00",
        "actual_journal_event_count": 1,
        "actual_journal_head_sha256": "b" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    captured = []
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None, {"benchmarks": []}))
    monkeypatch.setattr(study, "checked_packet", lambda f: None)
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study, "states", lambda f: actual)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study.common, "counts", lambda events: {"new_completed_strategy_evaluations": 0}
    )
    monkeypatch.setattr(
        study.common, "retain_receipt", lambda path, body, volatile: captured.append(body)
    )
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "d" * 64})
    study.public_result()
    body = captured[0]
    assert body["original_records_reclassified"] is False
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["order_authority"] is body["investment_objective_verified"] is False
    assert body["reviews"][0]["failure_event_sha256"] == "a" * 64
    assert len(body["reviews"]) == 2
    assert not any(key in str(body) for key in ("known_nav_usd", "pending_executions"))
