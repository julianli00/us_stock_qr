from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.calendar import sessions
from us_quant.prospective_data import ProspectiveArchive
from us_quant.prospective_research_accounts import (
    ACCOUNT_IDS,
    SYMBOLS,
    ResearchAccounts,
    calculate,
    chained_market,
    target_signals,
    validate_policy,
)
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic

ROOT = Path(__file__).parents[1]
INITIAL = "2026-10-09"


def moment(value):
    return datetime.fromisoformat(value)


def target_weights():
    return {
        "SPY": 0.0, "IEF": 0.0, "GLD": 0.38, "BIL": 0.0,
        "MTUM": 0.15, "VLUE": 0.15, "QUAL": 0.15, "USMV": 0.15,
    }


@pytest.fixture
def policy():
    return read_json(ROOT / "config/prospective-research-accounts.json")


def snapshot(market_factory, end):
    return market_factory("2024-10-09", end, SYMBOLS)


def test_within_snapshot_ratio_chain_is_invariant_to_adjusted_price_level_changes(market_factory):
    before = snapshot(market_factory, INITIAL)
    after = snapshot(market_factory, "2026-10-12")
    original = chained_market({pd.Timestamp(INITIAL): before, pd.Timestamp("2026-10-12"): after})
    changed = replace(after, close=after.close * 0.25, open=after.open * 0.25)
    revised = chained_market({pd.Timestamp(INITIAL): before, pd.Timestamp("2026-10-12"): changed})
    pd.testing.assert_frame_equal(original.open, revised.open)
    pd.testing.assert_frame_equal(original.close, revised.close)
    np.testing.assert_allclose(
        original.close.iloc[-1] / original.close.iloc[0],
        after.close.loc["2026-10-12"] / after.close.loc[INITIAL],
    )


def test_missing_actual_day_and_changed_previous_raw_quote_pause_not_reconstruct(market_factory):
    before = snapshot(market_factory, INITIAL)
    gap = snapshot(market_factory, "2026-10-13")
    with pytest.raises(QuantError, match="missing observed"):
        chained_market({pd.Timestamp(INITIAL): before, pd.Timestamp("2026-10-13"): gap})
    after = snapshot(market_factory, "2026-10-12")
    raw = after.raw_close.copy()
    raw.loc[INITIAL, "SPY"] += 1
    with pytest.raises(QuantError, match="raw close was revised"):
        chained_market({
            pd.Timestamp(INITIAL): before,
            pd.Timestamp("2026-10-12"): replace(after, raw_close=raw),
        })


def test_late_target_or_late_experiment_cannot_fill_an_earlier_open(market_factory):
    data = chained_market({
        pd.Timestamp(INITIAL): snapshot(market_factory, INITIAL),
        pd.Timestamp("2026-10-12"): snapshot(market_factory, "2026-10-12"),
    })
    registration = {"baseline_session": INITIAL, "registered_at": "2026-10-10T10:00:00+00:00"}
    instructions = [{
        "source_session": INITIAL, "generated_at": "2026-10-12T14:00:00+00:00",
        "weights": target_weights(),
    }]
    with pytest.raises(QuantError, match="after that open"):
        target_signals(data, instructions, registration, 1)
    instructions[0]["generated_at"] = "2026-10-10T09:00:00+00:00"
    registration["registered_at"] = "2026-10-12T14:00:00+00:00"
    with pytest.raises(QuantError, match="after that open"):
        target_signals(data, instructions, registration, 1)


def test_zero_return_anchor_initial_fees_and_delayed_independent_accounts(market_factory, policy):
    data = chained_market({
        pd.Timestamp(day): snapshot(market_factory, day)
        for day in (INITIAL, "2026-10-12", "2026-10-13")
    })
    registration = {"baseline_session": INITIAL, "registered_at": "2026-10-10T10:00:00+00:00"}
    instructions = [{
        "source_session": INITIAL, "generated_at": "2026-10-10T09:00:00+00:00",
        "weights": target_weights(),
    }]
    accounts, errors = calculate(data, instructions, registration, policy)
    assert set(accounts) == set(ACCOUNT_IDS)
    assert max(errors.values()) < 1e-8
    for name, account in accounts.items():
        assert account.frame.loc[INITIAL, "equity"] == 10000
        assert account.frame.loc[INITIAL, "return"] == 0
        assert account.frame["cash"].min() >= 0
        if name.endswith("base"):
            assert account.frame.loc["2026-10-12", "orders"] > 0
            assert account.frame.loc["2026-10-12", "cost"] > 0
        else:
            assert account.frame.loc["2026-10-12", "equity"] == 10000
            assert account.frame.loc["2026-10-12", "return"] == 0
            assert account.frame.loc["2026-10-13", "orders"] > 0
            assert account.frame.loc["2026-10-13", "cost"] > 0


@pytest.fixture
def artifacts(tmp_path, market_factory, monkeypatch):
    parent_policy = read_json(ROOT / "config/prospective-data.json")
    target_policy = read_json(ROOT / "config/prospective-target-observations.json")
    write_json(tmp_path / "config/prospective-target-observations.json", target_policy)
    parent = ProspectiveArchive(tmp_path / "data/prospective-market-v1", parent_policy)
    parent.initialize(moment("2026-10-10T08:00:00+00:00"))
    target_directory = tmp_path / "data/prospective-target-observations-v1"
    write_json(target_directory / "registration.json", {"synthetic_target_fixture_only": True})

    class Targets:
        directory = target_directory

        def __init__(self):
            self.parent = parent
            self.records = []

        def verify(self):
            return self.records

        def fingerprint(self):
            return "synthetic_target_fixture"

    targets = Targets()
    monkeypatch.setattr("us_quant.prospective_research_accounts.TargetJournal", lambda *args, **kwargs: targets)

    def add(day, acquired_at, *, record_target=True):
        data = market_factory("2024-10-09", day, tuple(parent_policy["symbols"]))

        def acquire(_policy, _day, output):
            for name in ("open", "close", "raw_close", "volume"):
                write_text_atomic(output / f"{name}.csv", getattr(data, name).to_csv(float_format="%.17g"))
            write_text_atomic(output / "risk_free.csv", data.risk_free.to_frame().to_csv(float_format="%.17g"))
            return {
                "session": day, "all_required_sources_verified": True,
                "strategy_returns_calculated": False,
            }

        parent.collect(acquire, lambda: moment(acquired_at))
        if record_target:
            folder = target_directory / "attempts" / day
            write_json(folder / "target.json", {"weights": target_weights()})
            manifest = {
                "new_target_generated": not targets.records,
                "files": {"target.json": file_digest(folder / "target.json")},
            }
            write_json(folder / "manifest.json", manifest)
            record = {
                "session": day, "observed_at": acquired_at,
                "snapshot_path": f"attempts/{day}",
                "manifest_sha256": file_digest(folder / "manifest.json"),
            }
            write_json(target_directory / "receipts" / f"{day}.json", record)
            targets.records.append(record)
        return data

    add(INITIAL, "2026-10-10T09:00:00+00:00")
    return parent, targets, add


def test_registration_before_future_open_has_zero_observations_and_never_changes_sources(
    tmp_path, policy, artifacts
):
    parent, targets, add = artifacts
    before = parent.status()
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    study.initialize(moment("2026-10-10T10:00:00+00:00"))
    status = study.advance(moment("2026-10-10T11:00:00+00:00"))
    assert status["action"] == "no_new_observed_session"
    assert status["observed_model_sessions"] == 0
    assert status["first_model_session"] == "2026-10-12"
    assert all(row["total_return"] is None and row["sharpe"] is None for row in status["accounts"].values())
    assert parent.status() == before and len(targets.records) == 1
    assert not status["actual_orders_or_broker_account"]
    with pytest.raises(QuantError, match="reset"):
        study.initialize(moment("2026-10-10T12:00:00+00:00"))


def test_advancement_only_uses_original_on_time_receipts_with_exact_costs_and_duplicate_skip(
    tmp_path, policy, artifacts, monkeypatch
):
    parent, targets, add = artifacts
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    study.initialize(moment("2026-10-10T10:00:00+00:00"))
    add("2026-10-12", "2026-10-13T01:00:00+00:00")
    first = study.advance(moment("2026-10-13T02:00:00+00:00"))
    assert first["observed_model_sessions"] == 1
    assert first["accounts"]["strategy_base"]["model_order_tickets"] == 5
    assert first["accounts"]["strategy_stress"]["equity_usd"] == 10000
    assert first["accounts"]["strategy_stress"]["cost_paid_usd"] == 0
    before = parent.status()
    add("2026-10-13", "2026-10-14T01:00:00+00:00")
    second = study.advance(moment("2026-10-14T02:00:00+00:00"))
    assert second["observed_model_sessions"] == 2
    assert second["accounts"]["strategy_stress"]["cost_paid_usd"] > 0
    assert all(row["sharpe"] is None for row in second["accounts"].values())
    new_parent = parent.status()
    monkeypatch.setattr(study, "replay", lambda *args: pytest.fail("no new data,must not replay"))
    duplicate = study.advance(moment("2026-10-14T03:00:00+00:00"))
    assert duplicate["action"] == "no_new_observed_session"
    assert duplicate["receipt_chain_head"] == second["receipt_chain_head"]
    assert parent.status() == new_parent and before["complete_snapshots"] == 2


@pytest.mark.parametrize("problem", ["missing_data_day", "missing_target_day", "future_acquisition"])
def test_missing_or_future_observations_cannot_be_reconstructed_or_recorded(
    tmp_path, policy, artifacts, problem
):
    parent, targets, add = artifacts
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    study.initialize(moment("2026-10-10T10:00:00+00:00"))
    if problem == "missing_data_day":
        add("2026-10-13", "2026-10-14T01:00:00+00:00")
        now = moment("2026-10-14T02:00:00+00:00")
    elif problem == "missing_target_day":
        add("2026-10-12", "2026-10-13T01:00:00+00:00", record_target=False)
        now = moment("2026-10-13T02:00:00+00:00")
    else:
        add("2026-10-12", "2026-10-13T02:00:00+00:00")
        now = moment("2026-10-13T01:00:00+00:00")
    with pytest.raises(QuantError, match="Missing actual|before their actual acquisition"):
        study.advance(now)
    assert study.status()["observed_model_sessions"] == 0
    assert not list((study.directory / "receipts").glob("*.json"))
    failures = list((study.directory / "attempts").glob("*/failure.json"))
    assert len(failures) == 1 and not read_json(failures[0])["account_rows_committed"]


def test_registered_start_and_recorded_cash_returns_cannot_be_silently_revised(
    tmp_path, policy, artifacts
):
    parent, targets, add = artifacts
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    study.initialize(moment("2026-10-10T10:00:00+00:00"))
    add("2026-10-12", "2026-10-13T01:00:00+00:00")
    study.advance(moment("2026-10-13T02:00:00+00:00"))
    record = study.directory / "receipts/2026-10-12.json"
    altered = read_json(record)
    altered["accounts"]["strategy_base"]["return"] += 0.5
    write_json(record, altered)
    with pytest.raises(QuantError, match="funding or return identity"):
        study.status()


def test_late_registration_or_parent_nested_directory_is_refused(tmp_path, policy, artifacts):
    parent, targets, add = artifacts
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    with pytest.raises(QuantError, match="first future open"):
        study.initialize(moment("2026-10-12T14:00:00+00:00"))
    assert not study.directory.exists()
    with pytest.raises(QuantError, match="separate"):
        ResearchAccounts(parent.directory / "account", policy, root=tmp_path)


def test_exact_63_session_reporting_boundary_never_promotes_the_rejected_source(
    tmp_path, policy, artifacts, monkeypatch
):
    study = ResearchAccounts(tmp_path / "data/research", policy, root=tmp_path)
    study.initialize(moment("2026-10-10T10:00:00+00:00"))
    dates = sessions("2026-10-12", "2027-01-20")[:63]
    returns = 0.0005 + 0.002 * np.sin(np.arange(len(dates)))
    wealth = 10000 * np.cumprod(1 + returns)
    rows = []
    for day, daily_return, equity in zip(dates, returns, wealth, strict=True):
        account = {
            "equity": float(equity), "return": float(daily_return),
            "cost": 0.0, "orders": 0.0, "risk_free": 0.00004,
        }
        rows.append({
            "session": str(day.date()),
            "accounts": {identifier: account for identifier in ACCOUNT_IDS},
        })
    monkeypatch.setattr(study, "verify", lambda: rows[:62])
    early = study.status()
    assert all(row["sharpe"] is None and row["cagr"] is None for row in early["accounts"].values())
    monkeypatch.setattr(study, "verify", lambda: rows)
    eligible = study.status()
    assert all(row["sharpe"] is not None and row["cagr"] is not None for row in eligible["accounts"].values())
    assert eligible["observed_model_sessions"] == 63
    assert not eligible["research_champion_updated"]
    assert not eligible["investment_objective_verified"]
    assert eligible["source_historical_status"] == "rejected_historical"


@pytest.mark.parametrize("field", ["costs", "delay", "sharpe", "live", "backfill"])
def test_account_policy_cannot_relax_evidence_costs_or_trading_scope(policy, field):
    if field == "costs":
        policy["scenarios"][0]["cost_bps"] = 0
    elif field == "delay":
        policy["scenarios"][0]["delay_sessions"] = 0
    elif field == "sharpe":
        policy["minimum_observed_sessions_for_reported_sharpe"] = 2
    elif field == "live":
        policy["order_authority"] = True
    else:
        policy["backfill_missing_observations"] = True
    with pytest.raises(QuantError):
        validate_policy(policy)
