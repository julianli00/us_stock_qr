from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_overnight_range_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
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


def test_actual_registration_precedes_targets_and_recovery_does_not_repeat(journal, monkeypatch):
    root = journal.parent
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "TARGETS", root / "targets")
    monkeypatch.setattr(study, "PACKET", root / "packet.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {
            "new_registered_candidates": 20,
            "new_completed_strategy_evaluations": 19,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
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


def test_publication_requires_inherited_and_current_code(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "daily_proxy_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda original, commit: {"commit": commit})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, ref, required, commit):
        calls.append((ref, required))
        return {"published_commit": commit, "registration_packet": ref}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls[0] == (frozen["daily_proxy_registration"], set(study.source.EXTENSIONS))
    assert calls[1][1] == set(study.EXTENSIONS)
    study.publication(frozen, {}, "b" * 40)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_study_never_regenerates_targets_or_accounts(journal, monkeypatch):
    frozen, actual = {"study_id": "synthetic"}, {"id": {"status": "rejected", "paths": {}}}
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
    bars = pd.DataFrame(
        np.tile([100.0, 102, 98, 100], (len(index), 1)), index=index, columns=study.rules.FIELDS
    )
    options = pd.DataFrame({"GVZ": 15.0, "VIX": 20.0}, index=desired.dropna().index)
    information = {"bars": bars, "options": options}
    targets, audits = study.rules.build_targets(
        data, desired, information, read_json(study.rules.POLICY)
    )
    return data, desired, information, targets[study.IDS[0]], audits


def test_unclipped_measurement_is_checked_not_just_saturated_target():
    values = saved_fixture()
    assert study.verify_saved_targets(*values)[0] < 1e-12
    audits = values[-1]
    assert audits[0]["equity_share"] == 0.30
    audits[0]["combined_proxy_annualized"] *= 2
    with pytest.raises(QuantError, match="before allocation clipping"):
        study.verify_saved_targets(*values)


def test_original_defensive_budget_cannot_move_even_within_target_tolerance():
    data, desired, information, targets, audits = saved_fixture()
    day = desired.dropna().index[0]
    targets.loc[day, "BIL"] += 1e-13
    targets.loc[day, "GLD"] -= 1e-13
    with pytest.raises(QuantError, match="exact defensive BIL"):
        study.verify_saved_targets(data, desired, information, targets, audits)


def test_accounting_and_published_source_dependencies_remain_explicit():
    tree = ast.parse(inspect.getsource(study.evaluate))
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
    assert ast.unparse(funded[0]) == (
        "source.evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks)"
    )
    assert set(study.donor.EXTENSIONS).issubset(study.EXTENSIONS)
    assert "evidence/spy_ohlc_range_readiness_20261029_v1.json" in study.EXTENSIONS
    assert "src/us_quant/spy_range_inputs_v1.py" in study.EXTENSIONS
