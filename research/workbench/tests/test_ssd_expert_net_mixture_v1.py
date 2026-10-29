from __future__ import annotations

import copy

import pytest

from us_quant import ssd_expert_net_mixture_v1 as study
from us_quant.config import QuantError


def publication_fixture(monkeypatch, found=None):
    calls = []
    commit, explicit, imported = "a" * 40, study.OWN[1], "src/us_quant/backtest.py"
    frozen = {
        "study_id": "synthetic_net_mixture",
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


def test_publication_checks_full_dependencies_before_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)
    receipt = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet", "fence"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_unlisted_dependency_blocks_packet_and_funded_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)

    def refuse(*args):
        raise QuantError("Unknown frozen dependency")

    monkeypatch.setattr(study, "published_inputs", refuse)
    with pytest.raises(QuantError, match="Unknown"):
        study.publication(frozen, commit)
    assert calls == []


def test_publication_cannot_move_to_a_post_outcome_commit(monkeypatch):
    found = [
        {
            "body": {
                "study_id": "synthetic_net_mixture",
                "publication": {"published_commit": "d" * 40},
            }
        }
    ]
    frozen, commit, calls = publication_fixture(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet"]


def test_duplicate_completed_path_cannot_be_chosen_by_dictionary_overwrite(monkeypatch):
    duplicate = [{"body": {"path_key": "10y-base", "receipt": {}}}] * 2
    monkeypatch.setattr(
        study, "matching_events", lambda kind, name: duplicate if kind == "path_completed" else []
    )
    with pytest.raises(QuantError, match="Duplicate completion"):
        study.states({})


def test_expired_failure_does_not_append_new_research_events(monkeypatch):
    events, progress = [], []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T16:00:00+00:00")
    monkeypatch.setattr(study, "append_event", lambda *a: events.append(a))
    monkeypatch.setattr(study.current, "progress", lambda *a, **k: progress.append(a))
    study.failure("synthetic_stage", QuantError("expired"))
    assert events == [] and progress == [("expert_net_mixture_expired",)]


def test_canonical_export_does_not_publish_daily_shadow_or_main_rows(monkeypatch):
    frozen = {"study_id": "synthetic_net_mixture", "protocol": {"path": "config/original.json"}}
    actual = {
        study.IDS[0]: {
            "candidate_id": study.IDS[0],
            "status": "technical_failure",
            "paths": {},
            "failure_event_sha256": "a" * 64,
            "failure_stage": "synthetic",
        }
    }
    saved = {
        "candidates": copy.deepcopy(actual),
        "finished_at": "2026-10-29T23:00:00+00:00",
        "actual_journal_event_count": 1,
        "actual_journal_head_sha256": "b" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    captured = []
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, {"benchmarks": []}))
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
    assert len(body["reviews"]) == 1 and body["reviews"][0]["failure_event_sha256"] == "a" * 64
    assert not any(
        key in str(body)
        for key in ("net_shadow_relatives", "net_excess_covariance", "expert_weights")
    )
