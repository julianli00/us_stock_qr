from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import build_from_inputs, decision_pairs
from us_quant.prospective_data import ProspectiveArchive
from us_quant.prospective_dollar_guard import (
    CANDIDATE,
    SYMBOLS,
    DollarInputArchive,
    DollarResearchAccounts,
    DollarTargetJournal,
    calculate,
    chained_market,
    latest_queries,
    observed_target,
    target_signals,
    validate_profile,
)
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic

ROOT = Path(__file__).parents[1]
INITIAL = "2026-10-09"


def moment(value):
    return datetime.fromisoformat(value)


@pytest.fixture
def profile():
    return read_json(ROOT / "config/prospective-dollar-guard.json")


def vintage_fixture(data, *, rising=True):
    month, reference = latest_queries(data.close.index)
    rows = []
    for query, value in ((reference, 100.0), (month, 110.0 if rising else 90.0)):
        rows.append(
            {
                "series": "DTWEXBGS",
                "observation_date": query - pd.Timedelta(days=4),
                "release_date": query - pd.Timedelta(days=2),
                "available_session": query - pd.Timedelta(days=1),
                "unit_regime": "goods_services_january_2006_month",
                "archive_path": f"synthetic/{query.date()}.html",
                "archive_sha256": "1" * 64,
                "value": value,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rising", [False, True])
def test_latest_target_is_exactly_the_registered_rule_not_a_new_tuned_strategy(
    market_factory, rising
):
    data = market_factory("2024-10-09", INITIAL, SYMBOLS)
    vintages = vintage_fixture(data, rising=rising)
    month, target, audit = observed_target(data, vintages)
    inputs = pd.DataFrame(
        {"dollar_change": 1.0 if rising else -1.0, "real_yield_change": 0.0},
        index=pd.DatetimeIndex(decision_pairs(data.close.index)),
    )
    expected = build_from_inputs(data, inputs, read_json(ROOT / "config/dollar-risk-guard.json"))[
        CANDIDATE
    ].loc[month]
    pd.testing.assert_series_equal(pd.Series(target), expected, check_names=False)
    assert month == "2026-09-30"
    assert audit["source_month_is_not_actual_generation_time"]
    assert audit["dollar_rising"] == rising
    assert sum(target.values()) == pytest.approx(0.98)
    assert target["BIL"] == pytest.approx(0.49 if rising else 0.0)


def test_prices_and_releases_after_the_rule_month_cannot_change_its_initial_intent(market_factory):
    data = market_factory("2024-10-09", INITIAL, SYMBOLS)
    vintages = vintage_fixture(data)
    original = observed_target(data, vintages)
    closing, opening = data.close.copy(), data.open.copy()
    later = closing.index > pd.Timestamp(original[0])
    closing.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    future = vintages.iloc[-1:].copy()
    future["available_session"] = pd.Timestamp("2026-10-05")
    future["value"] = 500.0
    changed = replace(data, close=closing, open=opening, raw_close=closing.copy())
    assert observed_target(changed, pd.concat([vintages, future])) == original


def test_twelve_symbol_price_chain_ignores_adjusted_level_rebasing_but_not_raw_revisions(
    market_factory,
):
    first = market_factory("2024-10-09", INITIAL, SYMBOLS)
    second = market_factory("2024-10-09", "2026-10-12", SYMBOLS)
    before = chained_market({pd.Timestamp(INITIAL): first, pd.Timestamp("2026-10-12"): second})
    renormalized = replace(second, close=second.close * 0.4, open=second.open * 0.4)
    after = chained_market({pd.Timestamp(INITIAL): first, pd.Timestamp("2026-10-12"): renormalized})
    pd.testing.assert_frame_equal(before.close, after.close)
    pd.testing.assert_frame_equal(before.open, after.open)
    revised = second.raw_close.copy()
    revised.loc[INITIAL, "SOXX"] += 1
    with pytest.raises(QuantError, match="raw close was revised"):
        chained_market(
            {
                pd.Timestamp(INITIAL): first,
                pd.Timestamp("2026-10-12"): replace(second, raw_close=revised),
            }
        )
    gap = market_factory("2024-10-09", "2026-10-13", SYMBOLS)
    with pytest.raises(QuantError, match="missing observed"):
        chained_market({pd.Timestamp(INITIAL): first, pd.Timestamp("2026-10-13"): gap})


def test_preopen_registration_initial_costs_delays_and_independent_model_accounts(
    market_factory, profile
):
    snapshots = {
        pd.Timestamp(day): market_factory("2024-10-09", day, SYMBOLS)
        for day in (INITIAL, "2026-10-12", "2026-10-13")
    }
    _, weights, _ = observed_target(
        snapshots[pd.Timestamp(INITIAL)], vintage_fixture(snapshots[pd.Timestamp(INITIAL)])
    )
    registration = {"baseline_session": INITIAL, "registered_at": "2026-10-11T04:00:00+00:00"}
    instructions = [
        {
            "source_session": INITIAL,
            "generated_at": "2026-10-11T03:30:00+00:00",
            "weights": weights,
        }
    ]
    data = chained_market(snapshots)
    accounts, errors = calculate(data, instructions, registration, profile)
    assert max(errors.values()) < 1e-8
    for name, account in accounts.items():
        assert account.frame.loc[INITIAL, "equity"] == 10000
        assert account.frame.loc[INITIAL, "return"] == 0
        assert account.frame["cash"].min() >= 0
        if name.endswith("base"):
            assert account.frame.loc["2026-10-12", "orders"] > 0
            assert account.frame.loc["2026-10-12", "cost"] > 0
        else:
            assert account.frame.loc["2026-10-12", "orders"] == 0
            assert account.frame.loc["2026-10-12", "cost"] == 0
            assert account.frame.loc["2026-10-13", "cost"] > 0
    instructions[0]["generated_at"] = "2026-10-12T14:00:00+00:00"
    with pytest.raises(QuantError, match="after that open"):
        target_signals(data, instructions, registration, 1)


def release_document(day):
    dates = pd.bdate_range(end=day - pd.Timedelta(days=3), periods=5)
    headers = "".join(f'<th id="a{i + 3}">{date:%b. %d}</th>' for i, date in enumerate(dates))
    values = "".join(f"<td>{100 + day.month + i}</td>" for i in range(5))
    return (
        f"<div>Release Date: {day:%B %d, %Y}</div>"
        '<table class="statistics"><tr><th id="a1">COUNTRY</th>'
        f'<th id="a2">CURRENCY</th>{headers}</tr>'
        f"<tr><th>1) BROAD</th><td>JAN06=100</td>{values}</tr></table>"
    )


@pytest.fixture
def artifacts(tmp_path, market_factory, profile, monkeypatch):
    parent_policy = read_json(ROOT / "config/prospective-data.json")
    write_json(tmp_path / "config/prospective-data.json", parent_policy)
    registration = tmp_path / profile["source_registration"]["path"]
    registration.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / profile["source_registration"]["path"], registration)
    write_json(tmp_path / "config/prospective-dollar-guard.json", profile)
    monkeypatch.setattr(
        "us_quant.prospective_dollar_guard.source_reference",
        lambda config, root: ({"id": CANDIDATE}, read_json(ROOT / "config/dollar-risk-guard.json")),
    )
    now = [moment("2026-10-11T03:30:00+00:00")]
    monkeypatch.setattr("us_quant.prospective_dollar_guard.utc_now", lambda: now[0])
    parent = ProspectiveArchive(tmp_path / "data/prospective-market-v1", parent_policy)
    parent.initialize(moment("2026-10-10T08:00:00+00:00"))
    end = [INITIAL]
    calls = []

    def parent_fetch(policy, day, output):
        data = market_factory("2024-10-09", str(day.date()), tuple(policy["symbols"]))
        for name in ("open", "close", "raw_close", "volume"):
            write_text_atomic(
                output / f"{name}.csv", getattr(data, name).to_csv(float_format="%.17g")
            )
        write_text_atomic(
            output / "risk_free.csv", data.risk_free.to_frame().to_csv(float_format="%.17g")
        )
        return {
            "session": str(day.date()),
            "quote_sources": [
                {"source": name, "retrieved_at": now[0].isoformat()}
                for name in (*policy["symbols"], "^IRX")
            ],
            "macro_sources": [{"series": name} for name in ("T10Y3M", "DFII10")],
            "option_index_sources": [{"symbol": name} for name in ("VIX", "VIX3M")],
            "all_required_sources_verified": True,
            "strategy_returns_calculated": False,
        }

    parent.collect(parent_fetch, lambda: now[0])

    class Response:
        def __init__(self, symbol, url):
            self.url = url
            data = market_factory("2024-10-09", end[0], ("QQQ", "XLK", "SOXX"))
            self.payload = {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "meta": {"symbol": symbol, "currency": "USD", "instrumentType": "ETF"},
                            "timestamp": [
                                int((day.tz_localize("UTC") + pd.Timedelta(hours=16)).timestamp())
                                for day in data.close.index
                            ],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": data.open[symbol].tolist(),
                                        "close": data.close[symbol].tolist(),
                                        "high": (
                                            np.maximum(data.open[symbol], data.close[symbol]) + 1
                                        ).tolist(),
                                        "low": (
                                            np.minimum(data.open[symbol], data.close[symbol]) - 1
                                        ).tolist(),
                                        "volume": data.volume[symbol].tolist(),
                                    }
                                ],
                                "adjclose": [{"adjclose": data.close[symbol].tolist()}],
                            },
                        }
                    ],
                },
            }
            self.text = json.dumps(self.payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            symbol = url.rsplit("/", 1)[-1]
            calls.append(symbol)
            return Response(symbol, url)

    monkeypatch.setattr("us_quant.prospective_dollar_guard.requests.Session", Client)

    def public(url, destination):
        calls.append(url)
        if url.endswith("releaseDates.json"):
            body = [
                {
                    "yearValue": "2026",
                    "Months": [
                        {
                            "MonthValue": f"2026{month:02}",
                            "Dates": [f"2026{month:02}{day:02}" for day in dates],
                        }
                        for month, dates in (
                            (6, (22, 29)),
                            (7, (20, 27)),
                            (9, (21, 28)),
                            (10, (19, 26)),
                        )
                    ],
                }
            ]
            write_json(destination, body)
        else:
            write_text_atomic(
                destination, release_document(pd.Timestamp(url.rstrip("/").rsplit("/", 1)[-1]))
            )
        return {
            "url": url,
            "retrieved_at": now[0].isoformat(),
            "sha256": file_digest(destination),
        }

    monkeypatch.setattr("us_quant.prospective_dollar_guard.acquire_public", public)
    inputs = DollarInputArchive(Path(profile["inputs"]["directory"]), profile, root=tmp_path)
    inputs.initialize(now[0])

    def add(day, timestamp, *, target=None):
        end[0], now[0] = day, moment(timestamp)
        parent.collect(parent_fetch, lambda: now[0])
        inputs.collect(clock=lambda: now[0])
        if target:
            target.collect(clock=lambda: now[0])

    return parent, inputs, now, calls, add


def test_actual_adapter_inherits_parent_bytes_and_acquires_only_three_quotes_and_official_vintages(
    artifacts, profile
):
    parent, inputs, now, calls, add = artifacts
    before = parent.status()
    result = inputs.collect(clock=lambda: now[0])
    assert result["complete_snapshots"] == 1
    assert result["observations_after_baseline"] == 0
    assert calls[:3] == ["QQQ", "XLK", "SOXX"]
    assert len(calls) == 8
    assert parent.status() == before
    receipt = inputs.verify()[0]
    folder = inputs.directory / receipt["snapshot_path"]
    inherited = read_json(parent.directory / parent.verify()[0]["snapshot_path"] / "manifest.json")
    assert all(
        file_digest(folder / "parent" / name) == digest
        for name, digest in inherited["files"].items()
    )
    duplicate = inputs.collect(
        fetcher=lambda *args: pytest.fail("a duplicate must not fetch"), clock=lambda: now[0]
    )
    assert duplicate["receipt_chain_head"] == result["receipt_chain_head"]
    assert len(calls) == 8
    altered = folder / "h10/20260928.html"
    write_text_atomic(altered, "changed published source")
    with pytest.raises(QuantError, match="revised|hash"):
        inputs.status()


def test_new_target_is_not_backdated_and_old_month_is_not_retuned(artifacts, profile, tmp_path):
    parent, inputs, now, calls, add = artifacts
    inputs.collect(clock=lambda: now[0])
    targets = DollarTargetJournal(Path(profile["targets"]["directory"]), profile, root=tmp_path)
    targets.initialize(now[0])
    first = targets.collect(clock=lambda: now[0])
    assert first["source_candidate_id"] == CANDIDATE
    assert first["latest_rule_signal_session"] == "2026-09-30"
    record = targets.verify()[0]
    manifest = read_json(targets.directory / record["snapshot_path"] / "manifest.json")
    assert manifest["hypothetical_base_execution_session"] == "2026-10-12"
    assert manifest["hypothetical_stress_execution_session"] == "2026-10-13"
    initial = (targets.directory / record["snapshot_path"] / "target.json").read_bytes()
    add("2026-10-12", "2026-10-13T01:00:00+00:00")
    later = targets.collect(
        clock=lambda: now[0], generator=lambda *args: pytest.fail("old month must stay frozen")
    )
    last = targets.verify()[-1]
    assert later["new_target_observations"] == 1 and later["source_sessions_recorded"] == 2
    assert (targets.directory / last["snapshot_path"] / "target.json").read_bytes() == initial
    add("2026-10-30", "2026-10-31T01:00:00+00:00", target=targets)
    assert targets.status()["new_target_observations"] == 2
    assert targets.status()["missed_observation_sessions"]
    assert not (targets.directory / "receipts/2026-10-13.json").exists()


def initialized_model(artifacts, profile, tmp_path):
    parent, inputs, now, calls, add = artifacts
    inputs.collect(clock=lambda: now[0])
    targets = DollarTargetJournal(Path(profile["targets"]["directory"]), profile, root=tmp_path)
    targets.initialize(now[0])
    targets.collect(clock=lambda: now[0])
    model = DollarResearchAccounts(Path(profile["model"]["directory"]), profile, root=tmp_path)
    model.initialize(now[0])
    return parent, inputs, targets, model, now, add


def test_plausible_but_unrelated_targets_fail_before_a_receipt_is_committed(
    artifacts, profile, tmp_path
):
    parent, inputs, now, calls, add = artifacts
    inputs.collect(clock=lambda: now[0])
    targets = DollarTargetJournal(Path(profile["targets"]["directory"]), profile, root=tmp_path)
    targets.initialize(now[0])

    def wrong(data, vintages):
        month, weights, audit = observed_target(data, vintages)
        for symbol in ("MTUM", "VLUE", "QUAL", "USMV"):
            weights[symbol] += 0.005
        weights["SOXX"] += 0.02
        weights["GLD"] -= 0.04
        return month, weights, audit

    with pytest.raises(QuantError, match="frozen source rule"):
        targets.collect(clock=lambda: now[0], generator=wrong)
    assert targets.status()["source_sessions_recorded"] == 0
    assert not list((targets.directory / "receipts").glob("*.json"))


def test_separate_model_starts_with_null_returns_and_replays_only_actual_daily_receipts(
    artifacts, profile, tmp_path, monkeypatch
):
    parent, inputs, targets, model, now, add = initialized_model(artifacts, profile, tmp_path)
    before = parent.status()
    initial = model.advance(now[0])
    assert initial["observed_model_sessions"] == 0
    assert initial["source_candidate_id"] == CANDIDATE
    assert all(
        row["total_return"] is None and row["sharpe"] is None
        for row in initial["accounts"].values()
    )
    add("2026-10-12", "2026-10-13T01:00:00+00:00", target=targets)
    first = model.advance(now[0])
    assert first["observed_model_sessions"] == 1
    assert first["accounts"]["strategy_base"]["cost_paid_usd"] > 0
    assert first["accounts"]["strategy_stress"]["cost_paid_usd"] == 0
    add("2026-10-13", "2026-10-14T01:00:00+00:00", target=targets)
    second = model.advance(now[0])
    assert second["observed_model_sessions"] == 2
    assert second["accounts"]["strategy_stress"]["cost_paid_usd"] > 0
    monkeypatch.setattr(model, "replay", lambda *args: pytest.fail("duplicate must not replay"))
    assert model.advance(now[0])["receipt_chain_head"] == second["receipt_chain_head"]
    assert not second["actual_orders_or_broker_account"]
    assert not second["investment_objective_verified"]
    assert before["complete_snapshots"] == 1 and parent.status()["complete_snapshots"] == 3


@pytest.mark.parametrize("problem", ["missing_day", "missing_target", "future_time"])
def test_missing_or_future_observations_pause_new_model_instead_of_backfilling(
    artifacts, profile, tmp_path, problem
):
    parent, inputs, targets, model, now, add = initialized_model(artifacts, profile, tmp_path)
    if problem == "missing_day":
        add("2026-10-13", "2026-10-14T01:00:00+00:00", target=targets)
        processing = now[0]
    elif problem == "missing_target":
        add("2026-10-12", "2026-10-13T01:00:00+00:00")
        processing = now[0]
    else:
        add("2026-10-12", "2026-10-13T02:00:00+00:00", target=targets)
        processing = moment("2026-10-13T01:00:00+00:00")
    with pytest.raises(QuantError, match="Missing actual|before their actual acquisition"):
        model.advance(processing)
    assert model.status()["observed_model_sessions"] == 0
    failures = list((model.directory / "attempts").glob("*/failure.json"))
    assert len(failures) == 1
    assert not read_json(failures[0])["account_rows_committed"]


def test_future_acquisition_missed_deadline_and_nested_directories_are_refused(
    artifacts, profile, tmp_path
):
    parent, inputs, now, calls, add = artifacts
    now[0] = moment("2026-10-11T04:00:00+00:00")
    with pytest.raises(QuantError, match="outside its actual window"):
        inputs.collect(clock=lambda: moment("2026-10-11T03:30:00+00:00"))
    assert inputs.status()["complete_snapshots"] == 0
    late = inputs.collect(
        fetcher=lambda *args: pytest.fail("must not request after the deadline"),
        clock=lambda: moment("2026-10-12T15:00:00+00:00"),
    )
    assert late["action"] == "missed_preopen_deadline"
    with pytest.raises(QuantError, match="separate"):
        DollarInputArchive(parent.directory / "new", profile, root=tmp_path)
    with pytest.raises(QuantError, match="reset"):
        inputs.initialize(now[0])


def test_new_model_keeps_the_exact_63_session_reporting_boundary_without_promotion(
    artifacts, profile, tmp_path, monkeypatch
):
    parent, inputs, targets, model, now, add = initialized_model(artifacts, profile, tmp_path)
    dates = sessions("2026-10-12", "2027-01-20")[:63]
    returns = 0.0005 + 0.002 * np.sin(np.arange(len(dates)))
    wealth = 10000 * np.cumprod(1 + returns)
    rows = [
        {
            "session": str(day.date()),
            "accounts": {
                identifier: {
                    "equity": float(equity),
                    "return": float(value),
                    "cost": 0.0,
                    "orders": 0.0,
                    "risk_free": 0.00004,
                }
                for identifier in ("strategy_base", "strategy_stress", "spy_base", "spy_stress")
            },
        }
        for day, value, equity in zip(dates, returns, wealth, strict=True)
    ]
    monkeypatch.setattr(model, "verify", lambda: rows[:62])
    assert all(row["sharpe"] is None for row in model.status()["accounts"].values())
    monkeypatch.setattr(model, "verify", lambda: rows)
    status = model.status()
    assert all(row["sharpe"] is not None for row in status["accounts"].values())
    assert not status["research_champion_updated"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("inputs", "additional_symbols", ["QLD", "XLK", "SOXX"]),
        ("inputs", "maximum_dollar_observation_age_days", 28),
        ("inputs", "backfill_missed_observations", True),
        ("targets", "target_investment_budget", 1.0),
        ("targets", "retain_recorded_target_when_no_new_month", False),
        ("targets", "submit_orders", True),
        ("model", "minimum_observed_sessions_for_reported_sharpe", 20),
        ("model", "automatic_live_deployment", True),
        ("model", "backfill_missing_observations", True),
    ],
)
def test_profile_sources_monthly_freeze_evidence_and_authority_cannot_change(
    profile, section, key, value
):
    profile[section][key] = value
    with pytest.raises(QuantError):
        validate_profile(profile)
