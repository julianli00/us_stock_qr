from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_hmm_core_session_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "events"
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "JOURNAL", path)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "progress", lambda *a, **k: None)
    base.append_event(path, "baseline", None, {"legacy": {"synthetic": True}})
    return path


@pytest.mark.parametrize("expired", [False, True])
def test_failure_is_recorded_once_for_actual_registrations_only(journal, monkeypatch, expired):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study, "utc_now", lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-29T00:00:00Z"
    )
    study.failure("synthetic", QuantError("stop"))
    study.failure("synthetic", QuantError("stop"))
    assert len(base.matching_events("technical_failure", study.IDS[0])) == int(not expired)
    assert all(
        not base.matching_events("technical_failure", identifier) for identifier in study.IDS[1:]
    )


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
            "new_registered_candidates": 24,
            "new_completed_strategy_evaluations": 23,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def test_all_three_registrations_precede_components_and_preparation_is_not_replayed(
    journal, monkeypatch
):
    frozen = setup_registration(journal, monkeypatch)
    calls = []

    def prove(*args):
        assert all(
            len(base.matching_events("registered", identifier)) == 1 for identifier in study.IDS
        )
        calls.append(True)
        target = pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))
        proof = {
            "checked_at": utc_now(),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
        }
        return (
            {identifier: target.copy() for identifier in study.IDS},
            {identifier: [] for identifier in study.IDS},
            {identifier: proof.copy() for identifier in study.IDS},
        )

    monkeypatch.setattr(study, "prove_targets", prove)
    study.register()
    study.register()
    assert calls == [True]
    assert [row["kind"] for row in base.journal_events(journal)] == [
        "baseline",
        "registered",
        "registered",
        "registered",
        "target_proof",
        "target_proof",
        "target_proof",
    ]
    packet = read_json(study.PACKET)
    assert packet["new_registered_candidates"] == len(packet["entries"]) == 3
    packet["new_registered_candidates"] = 2
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_failed_preparation_keeps_all_three_registered_attempts(journal, monkeypatch):
    setup_registration(journal, monkeypatch)

    def fail(*args):
        assert all(base.matching_events("registered", identifier) for identifier in study.IDS)
        raise QuantError("invalid adjusted open")

    monkeypatch.setattr(study, "prove_targets", fail)
    with pytest.raises(QuantError, match="invalid adjusted open"):
        study.register()
    assert all(
        len(base.matching_events("technical_failure", identifier)) == 1 for identifier in study.IDS
    )
    with pytest.raises(QuantError, match="without replay"):
        study.register()
    assert not study.PACKET.exists()


def test_publication_requires_the_hmm_carrier_and_current_candidate_closure(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "source_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, reference, required, commit):
        calls.append(required)
        return {"published_commit": commit, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls == [set(study.donor.EXTENSIONS), set(study.EXTENSIONS)]
    assert "evidence/hmm_growth_bil_20261029_registration_v1.json" in calls[-1]
    assert "src/us_quant/core_activity_momentum_v1.py" in calls[-1]
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_study_never_regenerates_targets_or_accounts(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    actual = {identifier: {"status": "rejected", "paths": {}} for identifier in study.IDS}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
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


def saved_fixture():
    index = sessions("2015-08-10", "2026-10-05")
    price = pd.DataFrame(100.0, index=index, columns=study.rules.UNIVERSE)
    data = MarketData(
        price.copy(), price.copy(), price.copy(), price * 1000, pd.Series(0.0, index=index)
    )
    desired = pd.DataFrame(np.nan, index=index, columns=price.columns)
    row = pd.Series(0.0, index=price.columns)
    row["SOXX"], row["BIL"] = 0.35, 0.28
    row.loc[list(study.rules.CORE)] = 0.35 / 4
    for i, day in enumerate(index):
        if i >= 252 and is_month_end(day):
            desired.loc[day] = row
    targets, audits = study.rules.build_targets(data, desired, read_json(study.rules.POLICY))
    return data, desired, targets, audits


def test_every_component_is_independently_checked_before_ranking():
    data, desired, targets, audits = saved_fixture()
    candidate = study.rules.CANDIDATES[2]
    identifier = candidate["id"]
    assert study.verify_saved_targets(
        data, desired, targets[identifier], audits[identifier], candidate
    ) == (0.0, 0.0, 0.0)
    audits[identifier][0]["component_scores"]["MTUM"]["overnight"] += 0.001
    with pytest.raises(QuantError, match="unrounded"):
        study.verify_saved_targets(
            data, desired, targets[identifier], audits[identifier], candidate
        )


def test_noncore_values_are_not_rebuilt_from_hmm_probabilities():
    data, desired, targets, audits = saved_fixture()
    candidate = study.rules.CANDIDATES[0]
    identifier = candidate["id"]
    day = targets[identifier].dropna().index[0]
    targets[identifier].loc[day, "SOXX"] += 1e-9
    with pytest.raises(QuantError, match="noncore"):
        study.verify_saved_targets(
            data, desired, targets[identifier], audits[identifier], candidate
        )


def test_full_proof_preserves_both_anchors_and_all_three_prefix_sequences(monkeypatch):
    data, desired, _, _ = saved_fixture()
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    targets, _, proofs = study.prove_targets(data, desired, None, read_json(base.PROTOCOL))
    assert len(targets) == 3
    for proof in proofs.values():
        assert proof["monthly_decisions"] == 122
        assert proof["additional_source_months_dropped"] == 0
        assert len(proof["prefix_truncations"]) == 5
        assert {row["years"] for row in proof["formal_factor_coverage"]} == {10, 5}
        assert proof["hmm_refitted_or_states_recomputed"] is False


def test_evaluator_visits_all_three_candidates_without_source_fitting_or_target_rebuild():
    tree = ast.parse(inspect.getsource(study.evaluate))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert any(ast.unparse(node.iter) == "rules.CANDIDATES" for node in loops)
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
    loader = ast.parse(inspect.getsource(study.inputs))
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in {"donor.inputs", "donor.verify_saved_targets", "donor.prove_targets"}
        for node in ast.walk(loader)
    )
