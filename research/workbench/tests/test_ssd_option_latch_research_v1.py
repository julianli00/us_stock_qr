from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from us_quant import ssd_channel_research_v1 as prior
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_option_latch_research_v1 as study
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
    base.append_event(directory, "baseline", None, {"synthetic": True})
    return directory


def test_publication_checks_new_packet_and_all_preserved_dependency_packets(journal, monkeypatch):
    calls = []
    commit, frozen = "b" * 40, {"study_id": "synthetic_option_latch"}
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
        (str(prior.PACKET), set(prior.EXTENSIONS), commit),
        (str(study.PACKET), set(study.EXTENSIONS), commit),
    ]
    assert receipt["registration_packet"]["path"] == str(study.PACKET)
    study.publication(frozen, {"source": "original"}, commit)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


@pytest.mark.parametrize("expired", [False, True])
def test_failure_is_counted_once_but_deadline_preserves_pending(journal, monkeypatch, expired):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study, "utc_now", lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-27T20:00:00Z"
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


def test_single_candidate_packet_requires_actual_registration_and_causal_clocks(
    journal, monkeypatch
):
    frozen = {"study_id": "synthetic_option_latch"}
    proof_file = journal.parent / "proof.json"
    monkeypatch.setattr(base, "checked_reference", lambda reference: proof_file)
    candidate = study.rules.CANDIDATES[0]
    identifier = candidate["id"]
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
    for damage in ("missing", "future", "count", "wrong_study"):
        altered = deepcopy(packet)
        if damage == "missing":
            altered["entries"].clear()
        elif damage == "future":
            altered["created_at"] = (pd.Timestamp(utc_now()) + pd.Timedelta(days=1)).isoformat()
        elif damage == "count":
            altered["new_registered_candidates"] = 2
        else:
            altered["study_id"] = "previous_cohort"
        write_json(path, altered)
        with pytest.raises(QuantError, match="option-latch registration"):
            study.checked_packet(frozen)


def test_completed_accounts_are_recovered_without_targets_or_publication(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    states = {"actual": {"status": "rejected", "paths": {}}}
    path = journal.parent / "complete.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "candidate_states", lambda value: states)
    monkeypatch.setattr(base, "reference", lambda value: {"path": str(value)})
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No target replay"))
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": states}
    )
    assert study.evaluate("unused") == {"path": str(path)}
    changed = read_json(path)
    changed["frozen_specification_sha256"] = "0" * 64
    write_json(path, changed)
    with pytest.raises(QuantError, match="never replay"):
        study.evaluate("unused")
