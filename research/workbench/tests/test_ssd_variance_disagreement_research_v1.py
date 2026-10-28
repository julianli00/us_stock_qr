from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as study
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    directory = tmp_path / "events"
    monkeypatch.setattr(base, "JOURNAL", directory)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    base.append_event(directory, "baseline", None, {"legacy": {"synthetic": True}})
    return directory


@pytest.mark.parametrize("expired", [False, True])
def test_failure_is_idempotent_and_cutoff_preserves_pending(journal, monkeypatch, expired):
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


def test_missing_registration_does_not_count_failure(journal, monkeypatch):
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    study.failure("synthetic", QuantError("not registered"))
    assert len(base.journal_events(journal)) == 1


def test_actual_registration_precedes_new_target_transformation(journal, monkeypatch):
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
            "new_registered_candidates": 10,
            "new_completed_strategy_evaluations": 10,
        },
    )
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    calls = []

    def proof(*args):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        calls.append(True)
        targets = pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))
        return (
            {study.IDS[0]: targets},
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
    assert [row["kind"] for row in base.journal_events(journal)] == [
        "baseline",
        "registered",
        "target_proof",
    ]
    packet = read_json(study.PACKET)
    packet["candidate_accounts_or_metrics_computed"] = True
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_publication_verifies_original_source_execution_and_new_model(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "source_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda original, commit: {"commit": commit})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, ref, required, commit):
        calls.append((ref, required))
        return {"published_commit": commit, "registration_packet": ref}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls[0][0] == frozen["source_registration"]
    assert "src/us_quant/observed_weight_execution.py" in calls[0][1]
    assert calls[1][1] == set(study.EXTENSIONS)
    study.publication(frozen, {}, "b" * 40)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_path_never_runs_monthly_executor_again(journal, monkeypatch):
    body = {"path_key": "10y-base", "receipt": {"path": "already_complete", "sha256": "a" * 64}}
    base.append_event(journal, "registered", study.IDS[0], {})
    base.append_event(journal, "path_completed", study.IDS[0], body)
    expected = {"complete": True}
    monkeypatch.setattr(base, "verify_path", lambda *args: expected)
    monkeypatch.setattr(
        study.monthly, "run", lambda *args: pytest.fail("No completed account replay")
    )
    assert (
        study.evaluate_path(None, None, {}, study.IDS[0], {"years": 10}, {"id": "base"}, {})
        == expected
    )


@pytest.mark.parametrize("delay", [1, 2])
def test_monthly_execution_feedback_uses_actual_holdings_and_original_delays(delay):
    index = sessions("2019-01-02", "2020-04-03")
    t = np.arange(len(index))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0001 * t + 0.002 * np.sin(t / (i + 8)))
            for i, name in enumerate(study.rules.UNIVERSE)
        },
        index=index,
    )
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=index, name="risk_free"),
    )
    desired = pd.DataFrame(np.nan, index=index, columns=close.columns)
    for date in ("2019-12-31", "2020-01-31", "2020-02-28", "2020-03-31"):
        desired.loc[pd.Timestamp(date)] = 0.0
        desired.loc[pd.Timestamp(date), "BIL"] = 0.98
    policy = read_json(study.monthly.POLICY)
    window = {"start": "2020-01-08", "end": "2020-04-03"}
    scenario = {"delay_sessions": delay}
    result, issued, decisions = study.monthly.run(
        data,
        desired,
        study.monthly.CANDIDATES[0],
        policy,
        window["start"],
        window["end"],
        5.0,
        delay,
    )
    checked = study.verify_feedback(data, desired, result, issued, decisions, window, scenario)
    assert checked["all_actual_gaps_targets_and_delays_checked"] is True
    assert checked["checked_actual_monthly_decisions_and_initial"] == 4
    anchor = previous_session(window["start"])
    expected_fill = anchor
    for _ in range(delay):
        expected_fill = next_session(expected_fill)
    assert decisions[0]["execution_session"] == str(expected_fill.date())
    damaged = deepcopy(decisions)
    damaged[1]["signal_session"] = "2020-01-30"
    with pytest.raises(QuantError, match="nonmonthly"):
        study.verify_feedback(data, desired, result, issued, damaged, window, scenario)


def test_completed_study_short_circuits_source_target_regeneration(journal, monkeypatch):
    frozen, states = {"study_id": "synthetic"}, {"id": {"status": "rejected", "paths": {}}}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: states)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No target replay"))
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": states}
    )
    assert study.evaluate("unused") == base.reference(path)
