from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import us_quant.expanded as expanded
from us_quant.config import QuantError
from us_quant.expanded import (
    Features,
    audit_reused,
    build_intents,
    develop,
    load_expanded,
    register,
    rsi_ewm,
    scale_risk,
    scheduled,
    verify_registration,
)
from us_quant.storage import (
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)


@pytest.fixture
def protocol(config):
    value = load_expanded(Path(__file__).parents[1] / "config/expanded.json", config)
    return replace(
        value,
        data_start="2019-12-02",
        simulation_start="2020-01-02",
        first_test_year=2021,
        training_years=1,
        test_years=1,
        final_training_start="2021-01-01",
        as_of="2022-01-31",
        momentum_lookback=10,
        momentum_skip=2,
        trend_lookback=6,
        volatility_lookback=5,
    )


def test_round_is_bounded_and_original_protocol_is_unchanged(config):
    path = Path(__file__).parents[1] / "config/expanded.json"
    protocol = load_expanded(path, config)
    assert protocol.prior_candidate_trials + len(protocol.candidates) == 14
    assert protocol.symbols == ("SPY", "QQQ", *expanded.ORIGINAL_SECTORS)
    with pytest.raises(QuantError, match="complete original"):
        replace(protocol, sectors=protocol.sectors[:-1]).validate(config)
    with pytest.raises(QuantError, match="preceding"):
        replace(protocol, prior_candidate_trials=0).validate(config)
    with pytest.raises(QuantError, match="preceding"):
        replace(protocol, candidates=(protocol.candidates[-1],)).validate(config)


def test_registration_rejects_overwrite_and_parameter_changes(tmp_path, protocol):
    target = tmp_path / "registration.json"
    receipt = register(protocol, target)
    assert receipt["cumulative_candidate_trials"] == 14
    verify_registration(protocol, target)
    with pytest.raises(QuantError, match="overwrite"):
        register(protocol, target)
    with pytest.raises(QuantError, match="changed after registration"):
        verify_registration(replace(protocol, rsi_entry=15), target)


def test_rsi_defined_for_flat_rising_and_falling_prices():
    close = pd.DataFrame(
        {
            "flat": [100.0] * 10,
            "up": np.arange(100.0, 110.0),
            "down": np.arange(100.0, 90.0, -1),
        }
    )
    rsi = rsi_ewm(close, 2)
    assert rsi.iloc[-1].to_dict() == {"flat": 50.0, "up": 100.0, "down": 0.0}
    assert rsi.iloc[:2].isna().all().all()


def test_schedule_uses_exchange_week_end_not_calendar_friday():
    assert scheduled(pd.Timestamp("2026-04-02"), "weekly")
    assert scheduled(pd.Timestamp("2026-09-30"), "monthly")
    assert not scheduled(pd.Timestamp("2026-09-29"), "monthly")
    assert scheduled(pd.Timestamp("2026-09-29"), "daily")
    with pytest.raises(QuantError, match="cadence"):
        scheduled(pd.Timestamp("2026-09-29"), "unknown")


def test_risk_scaling_never_leverages_and_hits_requested_cap():
    weights = np.array([0.49, 0.49])
    covariance = np.array([[0.09, 0.045], [0.045, 0.09]])
    result = scale_risk(weights, covariance, 0.15)
    assert np.sqrt(result @ covariance @ result) == pytest.approx(0.15)
    assert (result <= weights).all()
    assert np.array_equal(scale_risk(weights, covariance, 0.8), weights)
    with pytest.raises(QuantError, match="covariance"):
        scale_risk(weights, covariance * np.nan, 0.15)


def test_feature_covariance_preserves_symbol_order(protocol, market_factory):
    data = market_factory(symbols=protocol.symbols)
    computed = expanded.features(data.close, protocol)
    returns = data.close.pct_change(fill_method=None)
    for index in (15, 50, len(data.close) - 1):
        trailing = returns.iloc[index - protocol.volatility_lookback + 1 : index + 1]
        np.testing.assert_allclose(computed.covariance[index], trailing.cov().to_numpy() * 252)
        np.testing.assert_allclose(
            computed.volatility[index], trailing.std(ddof=1).to_numpy() * np.sqrt(252)
        )


def test_all_expanded_signals_are_causal_bounded_and_prefix_invariant(protocol, market_factory):
    data = market_factory(symbols=protocol.symbols)
    original = build_intents(data, protocol)
    cut = pd.Timestamp("2021-06-15")
    selected = data.close.index <= cut
    prefix_data = replace(
        data,
        open=data.open.loc[selected],
        close=data.close.loc[selected],
        raw_close=data.raw_close.loc[selected],
        volume=data.volume.loc[selected],
        risk_free=data.risk_free.loc[selected],
    )
    prefix = build_intents(prefix_data, protocol)
    altered_close = data.close.copy()
    altered_close.loc[altered_close.index > cut, "XLK"] *= 1.1
    altered = build_intents(replace(data, close=altered_close), protocol)
    for candidate in protocol.candidates:
        intent = original[candidate.id]
        pd.testing.assert_frame_equal(intent.signals.loc[:cut], prefix[candidate.id].signals)
        pd.testing.assert_frame_equal(
            intent.signals.loc[:cut], altered[candidate.id].signals.loc[:cut]
        )
        assert (intent.targets >= 0).all().all()
        assert (intent.targets <= candidate.max_weight + 1e-10).all().all()
        assert (intent.targets.sum(axis=1) <= 1 - protocol.cash_reserve + 1e-10).all()
        assert (intent.targets.iloc[: protocol.momentum_lookback] == 0).all().all()


def test_pullback_holding_limit_and_no_same_day_reentry(protocol, market_factory, monkeypatch):
    data = market_factory("2020-01-02", "2020-02-28", protocol.symbols)
    shape = data.close.shape
    features = Features(
        trend=data.close.to_numpy() * 0.8,
        momentum=np.ones(shape),
        rsi=np.zeros(shape),
        short_average=data.close.to_numpy() * 1.2,
        volatility=np.full(shape, 0.2),
        covariance=np.tile(np.eye(shape[1])[None, :, :] * 0.04, (shape[0], 1, 1)),
        warmup=2,
    )
    monkeypatch.setattr(expanded, "features", lambda *args: features)
    candidate = next(item for item in protocol.candidates if item.id == "index_pullback_15")
    shortened = replace(protocol, candidates=(candidate,), maximum_holding_sessions=3)
    intent = build_intents(data, shortened)[candidate.id]
    assert intent.targets.iloc[2].sum() > 0
    assert intent.targets.iloc[4].sum() > 0
    assert intent.targets.iloc[5].sum() == 0
    assert intent.targets.iloc[6].sum() > 0


def fixture_snapshot(config, data, output, phase):
    start = config.data_start if phase == "development" else config.development_end
    end = config.development_end if phase == "development" else config.as_of
    files = {}
    for symbol in config.symbols:
        frame = pd.DataFrame(
            {
                "open": data.open[symbol],
                "close": data.raw_close[symbol],
                "adj_open": data.open[symbol],
                "adj_close": data.close[symbol],
                "volume": data.volume[symbol],
            }
        ).loc[start:end]
        frame.index.name = "date"
        target = output / f"{symbol}.csv"
        write_text_atomic(target, frame.to_csv(float_format="%.12g"))
        files[target.name] = file_digest(target)
    index = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=14), end, freq="B")
    rates = pd.DataFrame({"close": 1.0}, index=index)
    rates.index.name = "date"
    target = output / "IRX.csv"
    write_text_atomic(target, rates.to_csv())
    files[target.name] = file_digest(target)
    write_json(
        output / "manifest.json",
        {
            "phase": phase,
            "protocol_sha256": digest_json(config.to_dict()),
            "files": files,
            "retrieved_at": utc_now(),
            "start": start,
            "end": end,
            "research_round": 2,
            "evidence_role": "development"
            if phase == "development"
            else "reused_diagnostic_not_untouched_holdout",
            "independent_prospective_evidence": False,
        },
    )


def test_full_expanded_workflow_never_claims_reused_history_is_independent(
    tmp_path, protocol, config, market_factory, monkeypatch
):
    base = replace(
        config, stress=replace(config.stress, bootstrap_samples=100, bootstrap_block_sessions=5)
    )
    protocol = replace(protocol, base_protocol_sha256=digest_json(base.to_dict()))
    market = market_factory(symbols=protocol.symbols)
    registration = tmp_path / "registration.json"
    register(protocol, registration)
    development, reused, output = (tmp_path / key for key in ("development", "reused", "results"))
    fixture_snapshot(protocol.data_config(base), market, development, "development")
    development_result = develop(protocol, base, development, registration, output)
    assert development_result["cumulative_candidate_trials"] == 14
    assert len(development_result["candidate_diagnostics"]) == 8
    assert development_result["walk_forward"]["start"] == "2021-01-04"
    freeze = output / "frozen.json"
    frozen_hash = file_digest(freeze)
    fixture_snapshot(protocol.data_config(base), market, reused, "holdout")
    result = audit_reused(
        protocol, base, development, reused, registration, freeze, tmp_path / "audit"
    )
    assert result["stage"] == "reused_history_diagnostic"
    assert not result["independent_prospective_evidence"]
    assert not result["objective_verified"] and not result["paper_submission_eligible"]
    assert result["forward_paper_sessions"] == 0
    assert result["diagnostic"]["start"] == "2022-01-03"
    assert file_digest(freeze) == frozen_hash
    assert read_json(tmp_path / "audit/results.json")["stage"] == "reused_history_diagnostic"
    monkeypatch.setattr(expanded, "expanded_fingerprint", lambda: "changed")
    with pytest.raises(QuantError, match="no longer matches"):
        audit_reused(
            protocol, base, development, reused, registration, freeze, tmp_path / "changed"
        )
