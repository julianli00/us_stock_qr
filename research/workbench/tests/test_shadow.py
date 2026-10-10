from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import us_quant.shadow as shadow
from us_quant.backtest import simulate
from us_quant.calendar import market_calendar
from us_quant.config import QuantError
from us_quant.shadow import ShadowLedger
from us_quant.storage import write_json
from us_quant.strategy import buy_and_hold_signals


def through(data, end):
    return replace(
        data,
        open=data.open.loc[:end],
        close=data.close.loc[:end],
        raw_close=data.raw_close.loc[:end],
        volume=data.volume.loc[:end],
        risk_free=data.risk_free.loc[:end],
    )


def after_close(day):
    return market_calendar().session_close(pd.Timestamp(day)) + pd.Timedelta(minutes=45)


@pytest.fixture
def market(market_factory, config):
    return market_factory("2019-12-02", "2021-02-08", config.symbols)


@pytest.fixture
def fixed_weights(config, monkeypatch):
    allocation = pd.Series(0.0, index=config.symbols)
    allocation.loc["SPY"], allocation.loc["QQQ"] = 0.4, 0.3
    monkeypatch.setattr(shadow, "target_weights", lambda *args: allocation.copy())
    return allocation


@pytest.fixture
def initialized(tmp_path, config, market, fixed_weights):
    freeze = tmp_path / "frozen.json"
    frozen = {"selected_candidate_id": config.candidates[1].id}
    write_json(freeze, frozen)
    ledger = ShadowLedger(tmp_path / "shadow.sqlite3", create=True)
    ledger.initialize(
        config,
        frozen,
        freeze,
        through(market, "2021-01-29"),
        "0" * 64,
        after_close("2021-01-29"),
    )
    yield ledger, freeze
    ledger.close()


def test_registered_baseline_and_weekends_do_not_count_as_forward(initialized, config, market):
    ledger, freeze = initialized
    status = ledger.status(config)
    assert status["prospective_shadow_sessions"] == 0
    assert status["first_forward_session"] == "2021-02-01"
    assert status["annualized_metrics_withheld_for_short_history"] is True
    result = ledger.update(
        config,
        freeze,
        through(market, "2021-01-29"),
        "0" * 64,
        pd.Timestamp("2021-01-30 22:00Z"),
    )
    assert result["prospective_shadow_sessions"] == 0
    assert result["broker_orders_sent"] == 0 and not result["objective_verified"]


def test_online_accounting_matches_frozen_causal_backtest(
    initialized, config, market, fixed_weights
):
    ledger, freeze = initialized
    for day in market.close.loc["2021-02-01":"2021-02-05"].index:
        ledger.update(config, freeze, through(market, day), "0" * 64, after_close(day))
    observations = [
        shadow.json.loads(row[0])
        for row in ledger.connection.execute(
            "SELECT observation_json FROM observations WHERE session_date>'2021-01-29' "
            "ORDER BY session_date"
        )
    ]
    signals = pd.DataFrame(np.nan, index=market.close.index, columns=market.close.columns)
    signals.loc["2021-01-29"] = fixed_weights
    expected = simulate(
        market,
        signals,
        "2021-02-01",
        "2021-02-05",
        initial_capital=config.initial_capital,
        cost_bps=config.cost_bps_per_side,
        commission=config.commission_per_order,
    )
    np.testing.assert_allclose(
        [row["equity"] for row in observations], expected.frame["equity"], rtol=1e-12
    )
    np.testing.assert_allclose(
        [row["return"] for row in observations], expected.frame["return"], atol=1e-12
    )
    benchmark = simulate(
        market,
        buy_and_hold_signals(market.close, "SPY", "2021-02-01"),
        "2021-02-01",
        "2021-02-05",
        initial_capital=config.initial_capital,
        cost_bps=config.cost_bps_per_side,
        commission=config.commission_per_order,
    )
    np.testing.assert_allclose(
        [row["benchmarks"]["SPY"]["equity"] for row in observations],
        benchmark.frame["equity"],
        rtol=1e-12,
    )
    assert observations[0]["synthetic_order_tickets"] == 2
    assert all(row["broker_orders_sent"] == 0 for row in observations)
    assert ledger.status(config)["prospective_shadow_sessions"] == 5
    assert ledger.status(config)["annualized_metrics_withheld_for_short_history"] is True


def test_duplicate_update_is_idempotent(initialized, config, market):
    ledger, freeze = initialized
    sample = through(market, "2021-02-01")
    first = ledger.update(config, freeze, sample, "1" * 64, after_close("2021-02-01"))
    second = ledger.update(
        config, freeze, sample, "1" * 64, after_close("2021-02-01") + pd.Timedelta(minutes=1)
    )
    assert first == second
    assert second["prospective_shadow_sessions"] == 1
    assert ledger.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2


def test_skipped_session_is_not_backfilled(initialized, config, market):
    ledger, freeze = initialized
    with pytest.raises(QuantError, match="Missed forward session"):
        ledger.update(
            config, freeze, through(market, "2021-02-02"), "0" * 64, after_close("2021-02-02")
        )
    assert ledger.status(config)["prospective_shadow_sessions"] == 0


def test_decision_after_its_execution_open_is_rejected(initialized, config, market):
    ledger, freeze = initialized
    with ledger.connection:
        ledger.connection.execute(
            "UPDATE decisions SET created_at=?",
            ((market_calendar().session_open("2021-02-01") + pd.Timedelta(seconds=1)).isoformat(),),
        )
    with pytest.raises(QuantError, match="predate execution"):
        ledger.update(
            config, freeze, through(market, "2021-02-01"), "0" * 64, after_close("2021-02-01")
        )
    assert ledger.status(config)["prospective_shadow_sessions"] == 0


def test_stale_data_and_late_checkin_cannot_masquerade_as_prospective(initialized, config, market):
    ledger, freeze = initialized
    with pytest.raises(QuantError, match="latest fully completed"):
        ledger.update(
            config, freeze, through(market, "2021-01-29"), "0" * 64, after_close("2021-02-01")
        )
    with pytest.raises(QuantError, match="before the next open"):
        ledger.update(
            config,
            freeze,
            through(market, "2021-01-29"),
            "0" * 64,
            market_calendar().session_open("2021-02-01") + pd.Timedelta(minutes=1),
        )


def test_data_revision_or_code_change_requires_audit(initialized, config, market, monkeypatch):
    ledger, freeze = initialized
    data = through(market, "2021-02-01")
    raw = data.raw_close.copy()
    raw.loc["2021-01-29", "SPY"] *= 1.01
    with pytest.raises(QuantError, match="revised"):
        ledger.update(
            config, freeze, replace(data, raw_close=raw), "0" * 64, after_close("2021-02-01")
        )
    monkeypatch.setattr(shadow, "engine_fingerprint", lambda: "changed")
    with pytest.raises(QuantError, match="accounting code changed"):
        ledger.update(config, freeze, data, "0" * 64, after_close("2021-02-01"))


def test_existing_ledger_cannot_be_reinitialized(initialized, tmp_path):
    with pytest.raises(QuantError, match="replace an existing"):
        ShadowLedger(tmp_path / "shadow.sqlite3", create=True)


def test_prospective_metrics_never_grant_order_authority(
    tmp_path, config, market_factory, fixed_weights
):
    data = market_factory("2019-12-02", "2021-06-30", config.symbols)
    freeze = tmp_path / "frozen.json"
    frozen = {"selected_candidate_id": config.candidates[1].id}
    write_json(freeze, frozen)
    ledger = ShadowLedger(tmp_path / "reference.sqlite3", create=True)
    try:
        ledger.initialize(
            config,
            frozen,
            freeze,
            through(data, "2021-01-29"),
            "0" * 64,
            after_close("2021-01-29"),
        )
        for day in data.close.loc["2021-02-01":].index[:63]:
            ledger.update(config, freeze, through(data, day), "0" * 64, after_close(day))
        status = ledger.status(config)
        assert status["prospective_shadow_sessions"] == 63
        assert "descriptive_metrics_not_a_forecast" in status
        assert not status["objective_verified"] and not status["paper_submission_eligible"]
        assert status["forward_paper_sessions"] == 0
    finally:
        ledger.close()
