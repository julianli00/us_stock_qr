from __future__ import annotations

import pytest

from us_quant import ssd_expert_meta_execution_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(study.base, "ROOT", tmp_path)
    monkeypatch.setattr(study.base, "REPO", tmp_path)
    monkeypatch.setattr(study.base, "JOURNAL", tmp_path / "events")
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(study.common, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "FOLDER", tmp_path / "release")
    monkeypatch.setattr(study, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(study, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: None)
    study.base.append_event(study.base.JOURNAL, "baseline", None, {"synthetic": True})
    return tmp_path


def test_publication_verifies_full_closure_but_passes_only_explicit_files(journal, monkeypatch):
    frozen = {
        "study_id": "synthetic",
        "code": [
            {"path": "src/imported.py", "sha256": "a" * 64},
            {"path": "src/extension.py", "sha256": "b" * 64},
        ],
        "policies": [{"path": "config/meta.json", "sha256": "c" * 64}],
    }
    write_json(
        journal / "config/publication_allowlist.json",
        {
            "files": ["research/workbench/src/extension.py", "research/workbench/config/meta.json"],
        },
    )
    calls = []

    def dependencies(actual, commit):
        assert actual == frozen
        calls.append("full")
        return {"unchanged_pinned_import_files_verified": 1}

    def explicit(root, packet, required, commit):
        assert required == {"src/extension.py", "config/meta.json"}
        calls.append("explicit")
        return {"published_commit": commit, "registration_packet": packet}

    monkeypatch.setattr(study.base, "published_inputs", dependencies)
    monkeypatch.setattr(study.base, "reference", lambda p: {"path": str(p), "sha256": "d" * 64})
    monkeypatch.setattr(study, "verify_publication", explicit)
    receipt = study.publication(frozen, "e" * 40)
    assert calls == ["full", "explicit"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1
    with pytest.raises(QuantError, match="original"):
        study.publication(frozen, "f" * 40)


def test_dependency_failure_does_not_append_publication(journal, monkeypatch):
    def reject(*args):
        raise QuantError("changed pinned import")

    monkeypatch.setattr(study.base, "published_inputs", reject)
    with pytest.raises(QuantError, match="pinned"):
        study.publication({"study_id": "synthetic"}, "a" * 40)
    assert not study.base.matching_events("publication")


def test_completed_execution_never_regenerates_source_scores_or_accounts(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    actual = {name: {"status": "rejected", "paths": {}} for name in study.IDS}
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda f: None)
    monkeypatch.setattr(study, "states", lambda f: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(
        study.source.rules, "build_targets", lambda *args: pytest.fail("No rescoring")
    )
    monkeypatch.setattr(
        study.source.execution, "evaluate_path", lambda *args: pytest.fail("No replay")
    )
    write_json(
        study.RESULTS,
        {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual},
    )
    assert study.evaluate("unused") == study.base.reference(study.RESULTS)


def test_registration_reuses_exact_target_bytes_without_source_generation(journal, monkeypatch):
    bindings = {}
    for candidate in study.CANDIDATES:
        target = journal / f"{candidate['source_candidate_id']}-targets.json"
        proof = journal / f"{candidate['source_candidate_id']}-proof.json"
        audit = journal / f"{candidate['source_candidate_id']}-audit.json"
        write_json(target, {"unchanged_complete_source_target": True})
        write_json(audit, {"unchanged_complete_source_audit": True})
        write_json(
            proof,
            {
                "checked_at": "2026-10-29T14:45:00Z",
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
                "information_audits": study.base.reference(audit),
            },
        )
        bindings[candidate["id"]] = {
            "source_candidate_id": candidate["source_candidate_id"],
            "targets": study.base.reference(target),
            "proof": study.base.reference(proof),
            "information_audits": study.base.reference(audit),
        }
    frozen = {"study_id": "synthetic", "source_target_bindings": bindings}
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(
        study.common,
        "counts",
        lambda events: {
            "new_registered_candidates": 32,
            "new_completed_strategy_evaluations": 29,
            "new_technical_failures": 3,
            "new_pending_candidates": 0,
        },
    )
    monkeypatch.setattr(
        study.source.rules, "build_targets", lambda *args: pytest.fail("No source generation")
    )
    monkeypatch.setattr(
        study.source, "prepare", lambda *args: pytest.fail("No completed preparation replay")
    )
    study.register()
    study.register()
    packet = read_json(study.PACKET)
    for entry in packet["entries"]:
        assert entry["target_proof"]["targets"] == bindings[entry["candidate_id"]]["targets"]
        proof = read_json(study.base.checked_reference(entry["target_proof"]["proof"]))
        assert proof["source_targets_reused_byte_for_byte"] is True
        assert proof["new_meta_scoring_or_target_generation"] is False
    assert all(len(study.base.matching_events("registered", name)) == 1 for name in study.IDS)


def test_release_cannot_change_financial_rules_or_reclassify_source_failure():
    for key, value in (
        ("financial_rule_parameter_or_gate_changes", True),
        ("reuse_exact_source_targets_and_audits", False),
        ("source_failure_stage", "signal_failure"),
    ):
        policy = read_json(study.POLICY)
        policy[key] = value
        with pytest.raises(QuantError):
            study.validate(policy)
