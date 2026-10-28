from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_ordered_path_research_v1 as previous
from us_quant import ssd_ordered_path_research_v2 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    directory = tmp_path / "events"
    monkeypatch.setattr(base, "JOURNAL", directory)
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    base.append_event(directory, "baseline", None, {"legacy": {"synthetic": True}})
    return directory


@pytest.mark.parametrize("expired", [False, True])
def test_failure_once_and_expiry_preserves_pending(journal, monkeypatch, expired):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study, "utc_now", lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-28T00:00:00Z"
    )
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    for _ in range(2):
        study.failure("synthetic", QuantError("stop"))
    counts = common.counts(base.journal_events(journal))
    assert counts["new_technical_failures"] == int(not expired)
    assert counts["new_pending_candidates"] == int(expired)


def test_registration_precedes_targets_and_recovery_does_not_repeat(journal, monkeypatch):
    root = journal.parent
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "TARGETS", root / "targets")
    monkeypatch.setattr(study, "PACKET", root / "packet.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {
            "new_registered_candidates": 18, "new_completed_strategy_evaluations": 17,
            "new_technical_failures": 1, "new_pending_candidates": 0,
        },
    )
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    calls = []

    def proof(*args):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        calls.append(True)
        target = pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))
        return (
            {study.IDS[0]: target},
            [],
            {
                "checked_at": utc_now(),
                "candidate_accounts_or_metrics_computed": False,
                "old_targets_or_accounts_replayed": False,
                "original_source_signal_recomputed": False,
            },
        )

    monkeypatch.setattr(study, "prove_targets", proof)
    study.register()
    study.register()
    assert calls == [True]
    packet = read_json(study.PACKET)
    packet["candidate_accounts_or_metrics_computed"] = True
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_publication_requires_inherited_executor_and_current_code(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "monthly_executor_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda original, commit: {"commit": commit})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, ref, required, commit):
        calls.append((ref, required))
        return {"published_commit": commit, "registration_packet": ref}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls[0] == (frozen["monthly_executor_registration"], set(study.source.EXTENSIONS))
    assert calls[1][1] == set(study.EXTENSIONS)
    study.publication(frozen, {}, "b" * 40)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_study_never_resolves_targets_or_replays_accounts(journal, monkeypatch):
    frozen, actual = {"study_id": "synthetic"}, {"id": {"status": "rejected", "paths": {}}}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No target replay"))
    monkeypatch.setattr(
        study.source, "evaluate_path", lambda *args: pytest.fail("No account replay")
    )
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual}
    )
    assert study.evaluate("unused") == base.reference(path)


def test_accounting_proof_and_publication_functions_preserve_v1_behavior():
    for name in (
        "prove_targets", "checked_packet", "failure", "publication", "states",
        "evaluate", "public_result", "main",
    ):
        old = ast.parse(inspect.getsource(getattr(previous, name)))
        new = ast.parse(inspect.getsource(getattr(study, name)))
        assert ast.dump(new) == ast.dump(old), name
    assert set(previous.EXTENSIONS).issubset(study.EXTENSIONS)
    assert study.rules.PREVIOUS_FAILURE["path"] in study.EXTENSIONS
    assert study.IDS != previous.IDS
    for name in ("FOLDER", "TARGETS", "PACKET", "RESULTS", "PUBLIC_RESULTS"):
        assert getattr(study, name) != getattr(previous, name)


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_registered_candidates", 17),
        ("new_completed_strategy_evaluations", 18),
        ("new_technical_failures", 0),
        ("new_pending_candidates", 1),
    ],
)
def test_registration_cannot_erase_or_reclassify_previous_failure(
    journal, monkeypatch, key, value
):
    counts = {
        "new_registered_candidates": 18, "new_completed_strategy_evaluations": 17,
        "new_technical_failures": 1, "new_pending_candidates": 0,
    }
    counts[key] = value
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, frozen, None, None))
    monkeypatch.setattr(common, "counts", lambda rows: counts)
    monkeypatch.setattr(study, "prove_targets", lambda *args: pytest.fail("No real targets"))
    with pytest.raises(QuantError, match="original failed version"):
        study.register()
    assert not base.matching_events("registered", study.IDS[0])
