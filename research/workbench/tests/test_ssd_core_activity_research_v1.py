from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_core_activity_research_v1 as study
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
def test_failure_is_recorded_once_only_for_registered_candidates(journal, monkeypatch, expired):
    base.append_event(journal, "registered", study.IDS[0], {})
    monkeypatch.setattr(
        study, "utc_now", lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-29T00:00:00Z"
    )
    study.failure("synthetic", QuantError("stop"))
    study.failure("synthetic", QuantError("stop"))
    assert len(base.matching_events("technical_failure", study.IDS[0])) == int(not expired)
    assert not base.matching_events("technical_failure", study.IDS[1])


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
            "new_registered_candidates": 21,
            "new_completed_strategy_evaluations": 20,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def test_both_registrations_precede_any_score_or_target_and_no_replay(journal, monkeypatch):
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
    rows = base.journal_events(journal)
    assert [row["kind"] for row in rows] == [
        "baseline",
        "registered",
        "registered",
        "target_proof",
        "target_proof",
    ]
    packet = read_json(study.PACKET)
    assert packet["new_registered_candidates"] == len(packet["entries"]) == 2
    packet["new_registered_candidates"] = 1
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_failed_score_preparation_preserves_both_real_registrations(journal, monkeypatch):
    setup_registration(journal, monkeypatch)

    def fail(*args):
        assert all(base.matching_events("registered", identifier) for identifier in study.IDS)
        raise QuantError("invalid activity")

    monkeypatch.setattr(study, "prove_targets", fail)
    with pytest.raises(QuantError, match="invalid activity"):
        study.register()
    assert all(
        len(base.matching_events("technical_failure", identifier)) == 1 for identifier in study.IDS
    )
    with pytest.raises(QuantError, match="without replay"):
        study.register()
    assert not study.PACKET.exists()


def test_publication_requires_current_source_and_both_candidate_dependencies(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "daily_proxy_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, reference, required, commit):
        calls.append(required)
        return {"published_commit": commit, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls == [set(study.source.EXTENSIONS), set(study.EXTENSIONS)]
    assert "evidence/core_activity_source_readiness_20261029_v1.json" in calls[-1]
    assert "config/factor-core-momentum.json" in calls[-1]
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_pair_never_regenerates_targets_or_accounts(journal, monkeypatch):
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
    row["SOXX"], row["GLD"], row["BIL"] = 0.1225, 0.245, 0.49
    row.loc[list(study.rules.CORE)] = 0.030625
    for i, day in enumerate(index):
        if i >= 63 and is_month_end(day):
            desired.loc[day] = row
    targets, audits = study.rules.build_targets(data, desired, read_json(study.rules.POLICY))
    return data, desired, targets, audits


def test_scores_are_checked_before_rounding_and_every_noncore_value_is_exact():
    data, desired, targets, audits = saved_fixture()
    candidate = study.rules.CANDIDATES[1]
    identifier = candidate["id"]
    assert (
        study.verify_saved_targets(
            data, desired, targets[identifier], audits[identifier], candidate
        )[0]
        < 1e-12
    )
    audits[identifier][0]["scores"]["MTUM"] += 1e-5
    with pytest.raises(QuantError, match="unrounded"):
        study.verify_saved_targets(
            data, desired, targets[identifier], audits[identifier], candidate
        )


def test_funded_evaluator_is_nested_under_both_candidates_without_primary_rebuild():
    tree = ast.parse(inspect.getsource(study.evaluate))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert any(ast.unparse(node.iter) == "rules.CANDIDATES" for node in loops)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "build_targets" for node in calls
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
