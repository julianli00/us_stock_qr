from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_joint_tail_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    directory = tmp_path / "events"
    monkeypatch.setattr(base, "JOURNAL", directory)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    base.append_event(directory, "baseline", None, {"legacy": {"synthetic": True}})
    return directory


def test_publication_checks_original_catalog_and_new_model(journal, monkeypatch):
    calls = []
    commit, frozen = "b" * 40, {"study_id": "synthetic_joint_tail"}
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})
    monkeypatch.setattr(
        base,
        "published_inputs",
        lambda original, sha: calls.append(("original", original, sha)) or {"source": True},
    )

    def verify(root, reference, required, sha):
        calls.append((reference["path"], required, sha))
        return {"published_commit": sha, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {"base": True}, commit)
    assert calls == [
        ("original", {"base": True}, commit),
        (str(study.registry.PUBLIC), study.CATALOG_FILES, commit),
        (str(study.PACKET), set(study.EXTENSIONS), commit),
    ]
    study.publication(frozen, {}, commit)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


@pytest.mark.parametrize("expired", [False, True])
def test_failure_record_is_idempotent_and_preserves_pending_at_cutoff(
    journal, monkeypatch, expired
):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study,
        "utc_now",
        lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-28T00:00:00Z",
    )
    monkeypatch.setattr(base, "progress", lambda *args, **kwargs: None)
    for _ in range(2):
        study.failure("synthetic", study.IDS[0], QuantError("synthetic failure"))
    counts = common.counts(base.journal_events(journal))
    assert counts["new_technical_failures"] == int(not expired)
    assert counts["new_pending_candidates"] == int(expired)


def test_missing_registration_does_not_create_fake_failure_trial(journal, monkeypatch):
    monkeypatch.setattr(base, "progress", lambda *args, **kwargs: None)
    study.failure("synthetic", study.IDS[0], QuantError("unregistered"))
    assert len(base.journal_events(journal)) == 1


def test_packet_rejects_missing_wrong_or_future_pre_account_evidence(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    identifier, candidate = study.IDS[0], study.rules.CANDIDATES[0]
    path = journal.parent / "proof.json"
    monkeypatch.setattr(base, "checked_reference", lambda ref: path)
    registration = base.append_event(
        journal, "registered", identifier, {"candidate": candidate, "frozen": frozen}
    )
    write_json(
        path,
        {
            "checked_at": utc_now(),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
        },
    )
    base.append_event(
        journal,
        "target_proof",
        identifier,
        {
            "registered_event_sha256": registration["sha256"],
            "targets": {"path": "synthetic"},
            "proof": {"path": "synthetic"},
        },
    )
    packet_path = journal.parent / "packet.json"
    monkeypatch.setattr(study, "PACKET", packet_path)
    packet = {
        "schema_version": 2,
        "program_id": "ssd_continuous_research_v1",
        "study_id": frozen["study_id"],
        "created_at": utc_now(),
        "entries": [common.packet_entry(candidate, frozen)],
        "new_registered_candidates": 1,
        "candidate_accounts_or_metrics_computed": False,
        "raw_inputs_and_target_matrices_remain_private": True,
        "investment_objective_verified": False,
        "order_authority": False,
        "independent_forward_validation": False,
    }
    write_json(packet_path, packet)
    assert study.checked_packet(frozen) == packet
    for damage in ("future", "count", "account", "entry"):
        changed = deepcopy(packet)
        if damage == "future":
            changed["created_at"] = (pd.Timestamp(utc_now()) + pd.Timedelta(days=1)).isoformat()
        elif damage == "count":
            changed["new_registered_candidates"] = 2
        elif damage == "account":
            changed["candidate_accounts_or_metrics_computed"] = True
        else:
            changed["entries"].clear()
        write_json(packet_path, changed)
        with pytest.raises(QuantError, match="joint-tail registration"):
            study.checked_packet(frozen)


def test_completed_study_does_not_resolve_or_replay_targets_accounts(journal, monkeypatch):
    frozen, states = {"study_id": "synthetic"}, {"id": {"status": "rejected", "paths": {}}}
    result = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", result)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: states)
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path)})
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No LP replay"))
    write_json(
        result, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": states}
    )
    assert study.evaluate("unused") == {"path": str(result)}
    changed = read_json(result)
    changed["frozen_specification_sha256"] = "0" * 64
    write_json(result, changed)
    with pytest.raises(QuantError, match="never replay"):
        study.evaluate("unused")


def test_actual_registration_event_precedes_target_solver(journal, monkeypatch):
    root = journal.parent
    frozen = {
        "study_id": "synthetic",
        "legacy": {"synthetic": True},
        "catalog": {"published_commit": "a" * 40},
    }
    monkeypatch.setattr(base, "ROOT", root)
    monkeypatch.setattr(study, "TARGETS", root / "targets")
    monkeypatch.setattr(study, "PACKET", root / "packet.json")
    monkeypatch.setattr(study.registry, "PUBLIC", root / "catalog.json")
    write_json(study.registry.PUBLIC, {"synthetic": True})
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(study, "verify_publication", lambda *args: {})
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {
            "new_registered_candidates": 8,
            "new_completed_strategy_evaluations": 8,
        },
    )
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    called = []

    def prove(data, protocol):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        called.append(True)
        target = pd.DataFrame(
            {"BIL": [0.98, 0.98]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
        )
        return (
            {study.IDS[0]: target},
            [],
            {
                "checked_at": utc_now(),
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
            },
        )

    monkeypatch.setattr(study, "prove_targets", prove)
    study.register()
    assert called == [True]
    assert [row["kind"] for row in base.journal_events(journal)] == [
        "baseline",
        "registered",
        "target_proof",
    ]
    study.register()
    assert called == [True]
