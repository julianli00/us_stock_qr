from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.prospective_data import ProspectiveArchive
from us_quant.prospective_target_observations import TargetJournal, observed_target, validate_policy
from us_quant.storage import read_json, write_json, write_text_atomic

ROOT = Path(__file__).parents[1]


def moment(value):
    return datetime.fromisoformat(value)


@pytest.fixture
def policy():
    return read_json(ROOT / "config/prospective-target-observations.json")


@pytest.fixture
def parent(tmp_path, market_factory):
    policy = read_json(ROOT / "config/prospective-data.json")
    write_json(tmp_path / "config/prospective-data.json", policy)
    reference = read_json(ROOT / "evidence/macro_tilt_20261010_registration.json")
    write_json(tmp_path / "evidence/macro_tilt_20261010_registration.json", reference)
    archive = ProspectiveArchive(tmp_path / "data/prospective-market-v1", policy)
    archive.initialize(moment("2026-10-10T08:00:00+00:00"))
    return archive


def parent_fetch(market_factory):
    def collect(policy, day, output):
        data = market_factory("2024-10-09", str(day.date()), tuple(policy["symbols"]))
        for name in ("open", "close", "raw_close", "volume"):
            write_text_atomic(output / f"{name}.csv", getattr(data, name).to_csv())
        write_text_atomic(output / "risk_free.csv", data.risk_free.to_frame().to_csv())
        return {
            "session": str(day.date()),
            "all_required_sources_verified": True,
            "strategy_returns_calculated": False,
        }

    return collect


def fake_target(data, macro, spec, policy):
    month = max(day for day in data.close.index if is_month_end(day))
    return str(month.date()), {
        "SPY": 0.0, "IEF": 0.0, "GLD": 0.38, "BIL": 0.0,
        "MTUM": 0.15, "VLUE": 0.15, "QUAL": 0.15, "USMV": 0.15,
    }


@pytest.fixture
def observer(tmp_path, policy, parent, market_factory, monkeypatch):
    parent.collect(parent_fetch(market_factory), lambda: moment("2026-10-10T08:00:00+00:00"))
    source = read_json(ROOT / "config/macro-factor-tilt.json")
    monkeypatch.setattr(
        TargetJournal, "source_reference",
        lambda self: ({"configuration": source["candidates"][0]}, source),
    )
    monkeypatch.setattr(
        "us_quant.prospective_target_observations.load_macro",
        lambda path, index: (
            pd.DataFrame({"T10Y3M": 1.0, "DFII10": 0.5}, index=index), {},
        ),
    )
    journal = TargetJournal(tmp_path / "data/targets", policy, root=tmp_path)
    journal.initialize(moment("2026-10-10T09:00:00+00:00"))
    return journal


def test_first_target_is_recorded_not_filled_and_duplicate_does_not_recompute(observer):
    before = observer.parent.status()
    now = moment("2026-10-10T09:00:00+00:00")
    status = observer.collect(lambda: now, fake_target)
    assert status["new_target_observations"] == status["source_sessions_recorded"] == 1
    assert status["latest_rule_signal_session"] == "2026-09-30"
    assert not status["orders_submitted"] and not status["portfolio_initialized"]
    assert not status["strategy_returns_calculated"] and not status["investment_objective_verified"]
    record = observer.verify()[0]
    manifest = read_json(observer.directory / record["snapshot_path"] / "manifest.json")
    assert manifest["hypothetical_base_execution_session"] == "2026-10-12"
    assert manifest["hypothetical_stress_execution_session"] == "2026-10-13"
    duplicate = observer.collect(
        lambda: now, lambda *args: pytest.fail("same session must not recompute"),
    )
    assert duplicate["action"] == "already_collected"
    assert duplicate["receipt_chain_head"] == status["receipt_chain_head"]
    assert observer.parent.status() == before


def test_a_new_input_day_cannot_retune_an_already_recorded_month(observer, market_factory):
    observer.collect(lambda: moment("2026-10-10T09:00:00+00:00"), fake_target)
    first = observer.verify()[0]
    original = (observer.directory / first["snapshot_path"] / "target.json").read_bytes()
    later = moment("2026-10-13T01:00:00+00:00")
    observer.parent.collect(parent_fetch(market_factory), lambda: later)
    result = observer.collect(
        lambda: later, lambda *args: pytest.fail("no new month,keep recorded target"),
    )
    assert result["source_sessions_recorded"] == 2
    assert result["new_target_observations"] == 1
    last = observer.verify()[-1]
    assert (observer.directory / last["snapshot_path"] / "target.json").read_bytes() == original
    manifest = read_json(observer.directory / last["snapshot_path"] / "manifest.json")
    assert manifest["hypothetical_base_execution_session"] is None


def test_new_completed_month_records_a_new_target_not_retroactive_missing_days(observer, market_factory):
    observer.collect(lambda: moment("2026-10-10T09:00:00+00:00"), fake_target)
    later = moment("2026-10-31T01:00:00+00:00")
    observer.parent.collect(parent_fetch(market_factory), lambda: later)
    result = observer.collect(lambda: later, fake_target)
    assert result["new_target_observations"] == 2
    assert result["latest_rule_signal_session"] == "2026-10-30"
    assert result["missed_observation_sessions"]
    assert not (observer.directory / "receipts/2026-10-12.json").exists()


def test_target_observation_refuses_missing_parent_or_too_late_capture(observer):
    with pytest.raises(QuantError, match="matching-session"):
        observer.collect(lambda: moment("2026-10-13T01:00:00+00:00"), fake_target)
    assert observer.status()["source_sessions_recorded"] == 0
    late = observer.collect(
        lambda: moment("2026-10-12T15:00:00+00:00"),
        lambda *args: pytest.fail("late capture must not generate"),
    )
    assert late["action"] == "missed_preopen_deadline"
    assert late["source_sessions_recorded"] == 0


def test_future_input_timestamp_or_partial_target_is_rejected_before_commit(observer, market_factory):
    later_parent = moment("2026-10-14T02:00:00+00:00")
    observer.parent.collect(parent_fetch(market_factory), lambda: later_parent)
    with pytest.raises(QuantError, match="inputs existed"):
        observer.collect(lambda: moment("2026-10-14T01:00:00+00:00"), fake_target)
    assert observer.status()["source_sessions_recorded"] == 0
    with pytest.raises(QuantError, match="every instrument"):
        observer.collect(
            lambda: moment("2026-10-14T03:00:00+00:00"),
            lambda *args: ("2026-09-30", {"BIL": 0.98}),
        )
    assert observer.status()["source_sessions_recorded"] == 0


def test_changed_target_source_or_registration_is_not_accepted(observer):
    observer.collect(lambda: moment("2026-10-10T09:00:00+00:00"), fake_target)
    target = observer.directory / observer.verify()[0]["snapshot_path"] / "target.json"
    write_json(target, {"weights": {"BIL": 0.98}})
    with pytest.raises(QuantError, match="revised"):
        observer.status()


def test_earlier_month_target_cannot_be_rewritten_after_newer_results(observer, market_factory):
    observer.collect(lambda: moment("2026-10-10T09:00:00+00:00"), fake_target)
    first = observer.verify()[0]
    later = moment("2026-10-31T01:00:00+00:00")
    observer.parent.collect(parent_fetch(market_factory), lambda: later)
    observer.collect(lambda: later, fake_target)
    target = observer.directory / first["snapshot_path"] / "target.json"
    values = read_json(target)
    values["weights"]["GLD"] -= 0.03
    values["weights"]["MTUM"] += 0.03
    write_json(target, values)
    with pytest.raises(QuantError, match="hash"):
        observer.status()


def test_naive_completion_clock_fails_without_committing_a_target(observer):
    aware = moment("2026-10-10T09:00:00+00:00")
    naive = datetime(2026, 10, 10, 9, 1)
    times = iter((aware, naive, aware))
    with pytest.raises(QuantError, match="timezone-aware"):
        observer.collect(lambda: next(times), fake_target)
    assert observer.status()["source_sessions_recorded"] == 0


@pytest.mark.parametrize("key", ["compute_strategy_returns", "submit_orders", "update_research_champion"])
def test_observation_policy_cannot_enable_trading_returns_or_promotion(policy, key):
    policy[key] = True
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_original_rule_target_is_causal_and_not_a_full_fundamental_stock_strategy(market_factory):
    symbols = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")
    data = market_factory("2015-08-10", "2017-10-05", symbols)
    macro = pd.DataFrame(
        {"T10Y3M": 1.0, "DFII10": np.sin(np.arange(len(data.close)) / 80)},
        index=data.close.index,
    )
    policy = read_json(ROOT / "config/macro-factor-tilt.json")
    spec = {"configuration": policy["candidates"][0]}
    date, target = observed_target(data, macro, spec, policy)
    assert date == "2017-09-29" and sum(target.values()) == pytest.approx(0.98)
    close, opening = data.close.copy(), data.open.copy()
    future = close.index > pd.Timestamp(date)
    close.loc[future, "QUAL"] *= np.linspace(1, 1.1, future.sum())
    opening *= 1.2
    changed = MarketData(opening, close, close.copy(), data.volume, data.risk_free)
    changed_macro = macro.copy()
    changed_macro.loc[future] += 10
    assert observed_target(changed, changed_macro, spec, policy) == (date, target)
