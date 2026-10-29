from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_vxn_relative_research_v1 as study
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
def test_failure_once_and_never_extends_research_after_expiry(journal, monkeypatch, expired):
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
            "new_registered_candidates": 27,
            "new_completed_strategy_evaluations": 26,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def test_actual_registration_precedes_any_real_ratio_and_no_preparation_replay(
    journal, monkeypatch
):
    frozen = setup_registration(journal, monkeypatch)
    calls = []

    def prove(*args):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        calls.append(True)
        target = pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))
        proof = {
            "checked_at": utc_now(),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
        }
        return {study.IDS[0]: target}, {study.IDS[0]: []}, {study.IDS[0]: proof}

    monkeypatch.setattr(study, "prove_targets", prove)
    study.register()
    study.register()
    assert calls == [True]
    assert [row["kind"] for row in base.journal_events(journal)] == [
        "baseline",
        "registered",
        "target_proof",
    ]
    packet = read_json(study.PACKET)
    packet["new_registered_candidates"] = 2
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_failed_ratio_preparation_retains_the_actual_registered_failure(journal, monkeypatch):
    setup_registration(journal, monkeypatch)

    def fail(*args):
        assert base.matching_events("registered", study.IDS[0])
        raise QuantError("zero observed variation")

    monkeypatch.setattr(study, "prove_targets", fail)
    with pytest.raises(QuantError, match="zero observed variation"):
        study.register()
    assert len(base.matching_events("technical_failure", study.IDS[0])) == 1
    with pytest.raises(QuantError, match="without replay"):
        study.register()
    assert not study.PACKET.exists()


def test_publication_requires_hmm_and_actual_vxn_source_dependencies(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "source_registration": {"path": "parent", "sha256": "a" * 64},
        "vxn_source_readiness": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, reference, required, commit):
        calls.append((reference["path"], required))
        return {"published_commit": commit, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert [x[1] for x in calls] == [
        set(study.donor.EXTENSIONS),
        set(study.SOURCE_EXTENSIONS),
        set(study.EXTENSIONS),
    ]
    assert calls[1][0] == "source"
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_account_evidence_is_never_replayed(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    actual = {study.IDS[0]: {"status": "rejected", "paths": {}}}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No targets"))
    monkeypatch.setattr(study.source, "evaluate_path", lambda *args: pytest.fail("No accounts"))
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual}
    )
    assert study.evaluate("unused") == base.reference(path)


def saved_fixture():
    index = sessions("2015-08-10", "2026-10-05")
    close = pd.DataFrame(100.0, index=index, columns=study.rules.UNIVERSE)
    opening = close.copy()
    for symbol, slope in (("SPY", 0.0001), ("QQQ", 0.0002)):
        close[symbol] *= np.exp(np.arange(len(index)) * slope)
        opening[symbol] = close[symbol].shift(1).fillna(close[symbol].iloc[0]) * np.exp(slope / 2)
    data = MarketData(opening, close, close.copy(), close * 1000, pd.Series(0.0, index=index))
    desired = pd.DataFrame(np.nan, index=index, columns=close.columns)
    row = pd.Series(0.0, index=close.columns)
    row["SOXX"], row["BIL"] = 0.35, 0.28
    row.loc[list(study.rules.CORE)] = 0.35 / 4
    for i, day in enumerate(index):
        if i >= 252 and is_month_end(day):
            desired.loc[day] = row
    info = pd.DataFrame({"VXN": 20.0, "VIX": 10.0}, index=index)
    targets, audits = study.rules.build_targets(data, desired, info, read_json(study.rules.POLICY))
    return data, desired, info, targets, audits


def test_independent_market_proxies_quotes_and_nonstock_targets_are_bound():
    data, desired, info, targets, audits = saved_fixture()
    identifier = study.IDS[0]
    errors = study.verify_saved_targets(
        data, desired, info, targets[identifier], audits[identifier]
    )
    assert max(errors[:3]) < 1e-12
    assert errors[3] < 1e-10
    audits[identifier][0]["VXN_quote_session"] = "2016-08-30"
    with pytest.raises(QuantError, match="dated quotes"):
        study.verify_saved_targets(data, desired, info, targets[identifier], audits[identifier])


def test_full_proof_keeps_both_formal_anchors_and_five_prefixes(monkeypatch):
    data, desired, info, _, _ = saved_fixture()
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    _, _, proofs = study.prove_targets(data, desired, info, read_json(base.PROTOCOL))
    proof = proofs[study.IDS[0]]
    assert proof["monthly_decisions"] == 122
    assert len(proof["prefix_truncations"]) == 5
    assert {row["years"] for row in proof["formal_factor_coverage"]} == {10, 5}
    assert proof["source_model_refitting"] is False


def test_saved_evaluation_has_no_primary_or_hmm_target_generator():
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
    loader = ast.parse(inspect.getsource(study.inputs))
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in {"donor.inputs", "donor.prove_targets", "donor.verify_saved_targets"}
        for node in ast.walk(loader)
    )
