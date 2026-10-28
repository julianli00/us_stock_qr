from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_anfci_residual_research_v1 as study
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.backtest import BacktestResult
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json, utc_now, write_json


def synthetic_account(delay):
    dates = sessions("2022-07-01", "2023-02-10")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0002 * (i + 1) * t + 0.01 * np.sin(t / (4 + i)))
            for i, name in enumerate(study.rules.UNIVERSE)
        },
        index=dates,
    )
    data = MarketData(
        close * 0.9998,
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    desired = pd.DataFrame(np.nan, index=dates, columns=study.rules.UNIVERSE)
    records = []
    for i, day in enumerate(
        pd.to_datetime(
            ["2022-08-31", "2022-09-30", "2022-10-31", "2022-11-30", "2022-12-30", "2023-01-31"]
        )
    ):
        desired.loc[day] = 0.0
        desired.loc[day, list(study.rules.CORE)] = 0.49 / 16
        desired.loc[day, ["SOXX", "GLD", "BIL"]] = [0.49 / 4, 0.49 / 2, 0.49]
        cutoff = previous_session(day)
        observed = cutoff - pd.Timedelta(days=(cutoff.weekday() - 4) % 7)
        if day == pd.Timestamp("2022-12-30"):
            observed = pd.Timestamp("2022-12-09")
        age = (day - observed).days
        records.append(
            {
                "query_session": str(day.date()),
                "vintage_cutoff": str(cutoff.date()),
                "observation_date": str(observed.date()),
                "age_calendar_days": age,
                "fresh": age <= 14,
                "actual_latest_value": 0.2 if i % 2 == 0 else -0.2,
            }
        )
    targets, audits = study.rules.build_targets(
        data, desired, records, read_json(study.rules.POLICY)
    )
    target = targets[study.IDS[0]]
    window = {"start": "2022-11-01", "end": "2023-02-10"}
    scenario = {"delay_sessions": delay}
    execution = read_json(study.ROOT / "config/observed-weight-execution.json")
    result, issued, decisions = study.SOURCE.monthly.run(
        data,
        target,
        study.SOURCE.monthly.CANDIDATES[0],
        execution,
        window["start"],
        window["end"],
        5.0 if delay == 1 else 20.0,
        delay,
    )
    return data, target, audits, result, issued, decisions, window, scenario


@pytest.mark.parametrize("delay", [1, 2])
def test_stale_signal_is_not_rebalanced_but_all_return_days_remain(delay):
    args = synthetic_account(delay)
    proof, abstentions = study.verify_feedback(*args)
    assert proof["checked_original_monthly_opportunities_and_initial"] == 4
    assert proof["checked_actual_monthly_decisions_and_initial"] == 3
    assert proof["source_abstention_signal_dates"] == ["2022-12-30"]
    assert abstentions[0]["no_new_request"] is True
    assert abstentions[0]["actual_return_session_present"] is True
    assert "2022-12-30" in args[3].frame.index
    with pytest.raises(QuantError, match="missing decision"):
        study.SOURCE.verify_feedback(args[0], args[1], args[3], args[4], args[5], args[6], args[7])


@pytest.mark.parametrize("kind", ["request", "return_day", "eligibility"])
def test_independent_verifier_rejects_hidden_gap_behavior(kind):
    args = list(synthetic_account(1))
    if kind == "request":
        args[4] = args[4].copy()
        args[4].loc["2022-12-30"] = args[1].ffill().loc["2022-12-30"]
    elif kind == "return_day":
        args[3] = BacktestResult(args[3].frame.drop(pd.Timestamp("2022-12-30")), args[3].weights)
    else:
        args[2] = [dict(row) for row in args[2]]
        args[2][3]["candidate_target_created"] = False
    with pytest.raises(QuantError):
        study.verify_feedback(*args)


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
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {"new_registered_candidates": 16, "new_completed_strategy_evaluations": 16},
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
        "source_registration": {"path": "source", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda original, commit: {"commit": commit})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, ref, required, commit):
        calls.append((ref, required))
        return {"published_commit": commit, "registration_packet": ref}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert calls[0] == (frozen["source_registration"], set(study.donor.EXTENSIONS))
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
    monkeypatch.setattr(study, "evaluate_path", lambda *args: pytest.fail("No account replay"))
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual}
    )
    assert study.evaluate("unused") == base.reference(path)
