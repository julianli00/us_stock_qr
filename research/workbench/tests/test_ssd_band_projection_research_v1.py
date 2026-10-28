from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_band_projection_research_v1 as study
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.calendar import sessions
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
def test_failure_once_and_expiry_keeps_pending(journal, monkeypatch, expired):
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


def test_registration_binds_desired_source_without_feedback_accounts(journal, monkeypatch):
    root = journal.parent
    target_path = root / "desired.csv"
    target_path.write_text(",BIL\n2020-01-31,0.98\n")
    frozen = {
        "study_id": "synthetic",
        "legacy": {"synthetic": True},
        "source_targets": base.reference(target_path),
    }
    monkeypatch.setattr(study, "FOLDER", root / "new-study")
    monkeypatch.setattr(study, "PACKET", root / "packet.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {"new_registered_candidates": 12, "new_completed_strategy_evaluations": 12},
    )
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study.rules, "run", lambda *args: pytest.fail("No account before publication")
    )
    calls = []

    def proof(*args):
        assert len(base.matching_events("registered", study.IDS[0])) == 1
        calls.append(True)
        return {
            "checked_at": utc_now(),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "actual_path_dependent_issued_targets_computed": False,
        }

    monkeypatch.setattr(study, "source_proof", proof)
    study.register()
    study.register()
    assert calls == [True]
    packet = read_json(study.PACKET)
    assert packet["entries"][0]["target_proof"]["targets"] == frozen["source_targets"]
    damaged = deepcopy(packet)
    damaged["candidate_accounts_or_metrics_computed"] = True
    write_json(study.PACKET, damaged)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_publication_requires_source_and_new_execution_code(journal, monkeypatch):
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
    assert calls[0] == (frozen["source_registration"], set(study.source.EXTENSIONS))
    assert calls[1][1] == set(study.EXTENSIONS)
    study.publication(frozen, {}, "b" * 40)
    assert len(base.matching_events("publication")) == 1
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def synthetic_account(delay):
    dates = sessions("2019-01-02", "2020-04-03")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0001 * t + 0.002 * np.sin(t / (8 + i)))
            for i, name in enumerate(study.rules.UNIVERSE)
        },
        index=dates,
    )
    desired = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    for day, share in (
        ("2019-12-31", 0.2),
        ("2020-01-31", 0.4),
        ("2020-02-28", 0.2),
        ("2020-03-31", 0.4),
    ):
        desired.loc[pd.Timestamp(day)] = 0
        desired.loc[pd.Timestamp(day), "SOXX"] = share
        desired.loc[pd.Timestamp(day), "GLD"] = 0.98 - share
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    window = {"start": "2020-01-08", "end": "2020-04-03"}
    scenario = {"delay_sessions": delay}
    result, issued, decisions = study.rules.run(
        data, desired, read_json(study.rules.POLICY), window["start"], window["end"], 5.0, delay
    )
    return data, desired, result, issued, decisions, window, scenario


@pytest.mark.parametrize("delay", [1, 2])
def test_actual_feedback_is_independently_verified(delay):
    args = synthetic_account(delay)
    checked = study.verify_feedback(*args)
    assert checked["all_actual_gaps_targets_and_delays_checked"] is True
    assert checked["checked_actual_monthly_decisions_and_initial"] == 4
    assert checked["maximum_independent_projection_difference"] <= 1e-12
    data, desired, result, issued, decisions, window, scenario = args
    damaged = deepcopy(decisions)
    damaged[1]["actual_closing_weights"][0] += 0.01
    with pytest.raises(QuantError, match="feedback inputs"):
        study.verify_feedback(data, desired, result, issued, damaged, window, scenario)


def test_completed_study_does_not_run_projection_accounts_again(journal, monkeypatch):
    frozen, states = {"study_id": "synthetic"}, {"id": {"status": "rejected", "paths": {}}}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: states)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "run", lambda *args: pytest.fail("No account replay"))
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": states}
    )
    assert study.evaluate("unused") == base.reference(path)
