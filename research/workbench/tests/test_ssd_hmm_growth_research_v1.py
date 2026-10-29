from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_hmm_growth_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "events"
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "JOURNAL", path)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    base.append_event(path, "baseline", None, {"legacy": {"synthetic": True}})
    return path


@pytest.mark.parametrize("expired", [False, True])
def test_failure_once_and_expiry_preserves_pending(journal, monkeypatch, expired):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study, "utc_now", lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-29T00:00:00Z"
    )
    study.failure("synthetic", QuantError("stop"))
    study.failure("synthetic", QuantError("stop"))
    assert len(base.matching_events("technical_failure", study.IDS[0])) == int(not expired)


def setup_registration(journal, monkeypatch):
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "TARGETS", journal.parent / "targets")
    monkeypatch.setattr(study, "PACKET", journal.parent / "packet.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {
            "new_registered_candidates": 23,
            "new_completed_strategy_evaluations": 22,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def test_registration_precedes_fitting_and_does_not_repeat_completed_targets(journal, monkeypatch):
    frozen = setup_registration(journal, monkeypatch)
    calls = []

    def proof(*args):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        calls.append(True)
        return (
            {study.IDS[0]: pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))},
            [],
            {
                "checked_at": utc_now(),
                "candidate_accounts_or_metrics_computed": False,
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


def test_fit_failure_retains_partial_evidence_and_prevents_same_version_replay(
    journal, monkeypatch
):
    setup_registration(journal, monkeypatch)
    partial = journal.parent / "partial-fit.json"

    def fail(*args):
        assert base.matching_events("registered", study.IDS[0])
        write_json(partial, {"synthetic_fit": True})
        raise QuantError("unidentified regime")

    monkeypatch.setattr(study, "prove_targets", fail)
    with pytest.raises(QuantError, match="unidentified"):
        study.register()
    assert read_json(partial) == {"synthetic_fit": True}
    assert len(base.matching_events("technical_failure", study.IDS[0])) == 1
    with pytest.raises(QuantError, match="failed preparation"):
        study.register()


def test_publication_covers_the_kernel_and_inherited_floor(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "daily_proxy_registration": {"path": "prior", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, ref, required, commit):
        calls.append(required)
        return {"published_commit": commit, "registration_packet": ref}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert "src/us_quant/volatility_hmm_v1.py" in calls[-1]
    assert "tests/test_volatility_hmm_v1.py" in calls[-1]
    assert "config/ordered-path-growth-v2.json" in calls[-1]
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_evaluation_never_refits_or_replays_accounts(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    actual = {study.IDS[0]: {"status": "rejected", "paths": {}}}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No repeated publication"))
    monkeypatch.setattr(study.rules.hmm, "fit", lambda *args: pytest.fail("No refitting"))
    monkeypatch.setattr(
        study.source, "evaluate_path", lambda *args: pytest.fail("No account replay")
    )
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual}
    )
    assert study.evaluate("unused") == base.reference(path)


def test_account_stage_uses_saved_fits_and_preparation_retains_each_model():
    tree = ast.parse(inspect.getsource(study.evaluate))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "build_targets"}
        for node in calls
    )
    funded = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_path"
    ]
    assert len(funded) == 1
    assert (
        ast.unparse(funded[0])
        == "source.evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks)"
    )
    preparation = ast.parse(inspect.getsource(study.prove_targets))
    build = [
        node
        for node in ast.walk(preparation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_targets"
    ]
    assert len(build) == 1
    assert {item.arg for item in build[0].keywords} == {"retain_fit", "before_fit"}
