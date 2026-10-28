from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_family_core_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.research_program import validate_factor
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    directory = tmp_path / "events"
    monkeypatch.setattr(base, "JOURNAL", directory)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    base.append_event(directory, "baseline", None, {"synthetic": True})
    return directory


def test_publication_requires_source_catalog_and_new_packet(journal, monkeypatch):
    calls = []
    commit, frozen = "b" * 40, {"study_id": "synthetic_family_core"}
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
    receipt = study.publication(frozen, {"source": "original"}, commit)
    assert calls == [
        ("original", {"source": "original"}, commit),
        (str(common.PACKET), set(common.EXTENSIONS), commit),
        (str(study.registry.PUBLIC), study.CATALOG_FILES, commit),
        (str(study.PACKET), set(study.EXTENSIONS), commit),
    ]
    assert receipt["registration_packet"]["path"] == str(study.PACKET)
    study.publication(frozen, {}, commit)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


@pytest.mark.parametrize("expired", [False, True])
def test_terminal_failure_count_is_idempotent_and_deadline_preserves_pending(
    journal, monkeypatch, expired
):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study,
        "utc_now",
        lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-28T00:00:00Z",
    )
    updates = []
    monkeypatch.setattr(base, "progress", lambda phase, **fields: updates.append((phase, fields)))
    for _ in range(2):
        study.record_failure("synthetic", study.IDS, QuantError("synthetic stop"))
    counts = common.counts(base.journal_events(journal))
    assert counts["new_registered_candidates"] == 1
    assert counts["new_technical_failures"] == int(not expired)
    assert counts["new_pending_candidates"] == int(expired)
    assert updates[-1][1]["active_compute_job"] is None


def test_pre_registration_failure_cannot_create_a_fake_trial(journal, monkeypatch):
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-28T00:00:00Z")
    monkeypatch.setattr(base, "progress", lambda *args, **kwargs: None)
    study.record_failure("synthetic", study.IDS, QuantError("unregistered"))
    assert len(base.journal_events(journal)) == 1


def test_packet_binds_actual_registration_and_causal_target_proof(journal, monkeypatch):
    frozen = {"study_id": "synthetic_family_core"}
    proof_file = journal.parent / "proof.json"
    monkeypatch.setattr(base, "checked_reference", lambda reference: proof_file)
    candidate, identifier = study.rules.CANDIDATES[0], study.IDS[0]
    event = base.append_event(
        journal, "registered", identifier, {"candidate": candidate, "frozen": frozen}
    )
    write_json(
        proof_file,
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
            "registered_event_sha256": event["sha256"],
            "targets": {"path": identifier},
            "proof": {"path": identifier},
        },
    )
    path = journal.parent / "packet.json"
    monkeypatch.setattr(study, "PACKET", path)
    packet = {
        "schema_version": 2,
        "program_id": "ssd_continuous_research_v1",
        "study_id": frozen["study_id"],
        "candidate_accounts_or_metrics_computed": False,
        "raw_inputs_and_target_matrices_remain_private": True,
        "investment_objective_verified": False,
        "order_authority": False,
        "independent_forward_validation": False,
        "new_registered_candidates": 1,
        "entries": [common.packet_entry(candidate, frozen)],
        "created_at": utc_now(),
    }
    write_json(path, packet)
    assert study.checked_packet(frozen) == packet
    for damage in ("missing", "future", "count", "wrong_study", "account_boundary"):
        changed = deepcopy(packet)
        if damage == "missing":
            changed["entries"].clear()
        elif damage == "future":
            changed["created_at"] = (pd.Timestamp(utc_now()) + pd.Timedelta(days=1)).isoformat()
        elif damage == "count":
            changed["new_registered_candidates"] = 2
        elif damage == "wrong_study":
            changed["study_id"] = "previous_study"
        else:
            changed["candidate_accounts_or_metrics_computed"] = True
        write_json(path, changed)
        with pytest.raises(QuantError, match="family-core registration"):
            study.checked_packet(frozen)


def test_completed_accounts_are_read_without_target_or_publication_replay(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    states = {"actual": {"status": "rejected", "paths": {}}}
    path = journal.parent / "complete.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "candidate_states", lambda value: states)
    monkeypatch.setattr(base, "reference", lambda value: {"path": str(value)})
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No target replay"))
    write_json(
        path,
        {
            "frozen_specification_sha256": study.digest_json(frozen),
            "candidates": states,
        },
    )
    assert study.evaluate("unused") == {"path": str(path)}
    changed = read_json(path)
    changed["frozen_specification_sha256"] = "0" * 64
    write_json(path, changed)
    with pytest.raises(QuantError, match="never replay"):
        study.evaluate("unused")


def test_only_six_genuinely_registered_available_families_are_used():
    policy = read_json(base.ROOT / "config/research-program.json")
    proposals = read_json(base.ROOT / "config/factor-family-expansion.json")
    catalog = {
        "legacy": {
            "factors": [
                {
                    "id": row["id"],
                    "family": row["family"],
                    "semantic_sha256": validate_factor(row),
                    "registered_at": "2020-01-01T00:00:00Z",
                }
                for row in policy["seed_factors"]
            ]
        },
        "new_factors": [
            {
                "factor": row,
                "registered_at": "2020-01-01T00:00:00Z",
                "semantic_sha256": validate_factor(row),
            }
            for row in proposals["factors"]
        ],
    }
    selected = study.used_factors(catalog)
    assert [row["id"] for row in selected] == list(study.rules.FACTOR_IDS)
    assert len({row["family"] for row in selected}) == 6
    for damage in ("missing", "family", "future", "duplicate", "semantic_duplicate"):
        changed = deepcopy(catalog)
        if damage == "missing":
            changed["new_factors"].pop()
        elif damage == "family":
            changed["new_factors"][0]["factor"]["family"] = "momentum"
        elif damage == "future":
            changed["new_factors"][0]["registered_at"] = (
                pd.Timestamp(utc_now()) + pd.Timedelta(days=1)
            ).isoformat()
        elif damage == "duplicate":
            changed["new_factors"][0]["factor"]["id"] = "price_momentum"
        else:
            changed["new_factors"][0]["semantic_sha256"] = changed["legacy"]["factors"][0][
                "semantic_sha256"
            ]
        with pytest.raises(QuantError):
            study.used_factors(changed)
