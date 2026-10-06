from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError, load_config
from us_quant.regime import build_intent, load_protocol
from us_quant.storage import digest_json, write_json


@pytest.fixture
def base():
    return load_config(Path(__file__).parents[1] / "config/geared-etf.json")


@pytest.fixture
def protocol(base):
    return load_protocol(Path(__file__).parents[1] / "config/daily-regime.json", base)


@pytest.fixture
def fast_base(base):
    return replace(base, volatility_lookback=5)


def compact(protocol, candidate_id):
    candidate = next(item for item in protocol.candidates if item.id == candidate_id)
    candidate = replace(
        candidate,
        fast_average=3 if candidate.fast_average else 0,
        slow_average=6,
        momentum_lookback=5 if candidate.momentum_lookback else 0,
    )
    return replace(protocol, candidates=(candidate,)), candidate


def test_protocol_discloses_six_new_trials_and_keeps_risk_bounded(base, protocol):
    assert protocol.prior_disclosed_trials == 28
    assert len(protocol.candidates) == 6
    assert all(item.target_volatility <= 0.18 for item in protocol.candidates)
    assert all(item.max_weight <= 0.98 for item in protocol.candidates)
    with pytest.raises(QuantError):
        replace(protocol, prior_disclosed_trials=0).validate(base)
    with pytest.raises(QuantError):
        replace(
            protocol,
            candidates=(replace(protocol.candidates[0], target_volatility=0.3),),
        ).validate(base)


def test_daily_signal_is_prefix_invariant_and_future_prices_cannot_change_past(
    protocol, fast_base, market_factory
):
    protocol, candidate = compact(protocol, "qld_dual50_200_v15")
    data = market_factory("2020-01-02", "2020-06-30", fast_base.symbols)
    original = build_intent(data, candidate, protocol, fast_base)
    cutoff = pd.Timestamp("2020-04-30")
    mask = data.close.index <= cutoff
    prefix = replace(
        data,
        open=data.open.loc[mask],
        close=data.close.loc[mask],
        raw_close=data.raw_close.loc[mask],
        volume=data.volume.loc[mask],
        risk_free=data.risk_free.loc[mask],
    )
    shorter = build_intent(prefix, candidate, protocol, fast_base)
    altered = data.close.copy()
    future = altered.index > cutoff
    altered.loc[future, "QLD"] *= np.linspace(1.01, 1.50, future.sum())
    changed = build_intent(replace(data, close=altered), candidate, protocol, fast_base)
    pd.testing.assert_frame_equal(original.signals.loc[:cutoff], shorter.signals)
    pd.testing.assert_frame_equal(original.signals.loc[:cutoff], changed.signals.loc[:cutoff])


def test_all_targets_are_long_only_unlevered_and_reserve_cash(protocol, fast_base, market_factory):
    data = market_factory("2020-01-02", "2020-12-31", fast_base.symbols)
    for original in protocol.candidates:
        candidate = replace(
            original,
            fast_average=3 if original.fast_average else 0,
            slow_average=6,
            momentum_lookback=5 if original.momentum_lookback else 0,
        )
        local = replace(protocol, candidates=(candidate,))
        intent = build_intent(data, candidate, local, fast_base)
        assert (intent.targets >= 0).all().all()
        assert (intent.targets <= candidate.max_weight + 1e-10).all().all()
        assert (intent.targets.sum(axis=1) <= 1 - fast_base.cash_reserve + 1e-10).all()
        assert intent.signals.iloc[:6].isna().all().all()


def test_trend_break_generates_an_exit_for_next_open(protocol, fast_base, market_factory):
    protocol, candidate = compact(protocol, "qld_sma200_v15")
    data = market_factory("2020-01-02", "2020-03-31", fast_base.symbols)
    close = data.close.copy()
    close.loc[:, "QLD"] = np.linspace(100, 150, len(close))
    close.loc[close.index[-1], "QLD"] = 70
    data = replace(data, close=close, raw_close=close.copy())
    intent = build_intent(data, candidate, protocol, fast_base)
    assert intent.targets.iloc[-2]["QLD"] > 0
    assert intent.signals.iloc[-1]["QLD"] == 0
    assert intent.targets.iloc[-1]["QLD"] == 0


def test_rebalance_band_suppresses_micro_adjustments(protocol, fast_base, market_factory):
    protocol, candidate = compact(protocol, "qld_sma200_v15")
    data = market_factory("2020-01-02", "2020-12-31", fast_base.symbols)
    tight = build_intent(data, candidate, replace(protocol, rebalance_band=0.01), fast_base)
    loose = build_intent(data, candidate, replace(protocol, rebalance_band=0.10), fast_base)
    assert loose.signals.notna().all(axis=1).sum() <= tight.signals.notna().all(axis=1).sum()


def test_rotation_selects_only_the_highest_momentum_eligible_asset(
    protocol, fast_base, market_factory
):
    protocol, candidate = compact(protocol, "geared_rotation_dual_v15")
    data = market_factory("2020-01-02", "2020-06-30", fast_base.symbols)
    close = data.close.copy()
    close.loc[:, "QLD"] = np.linspace(100, 200, len(close))
    close.loc[:, "SSO"] = np.linspace(100, 160, len(close))
    close.loc[:, "GLD"] = np.linspace(100, 130, len(close))
    close.loc[:, "TLT"] = np.linspace(100, 110, len(close))
    data = replace(data, close=close, raw_close=close.copy())
    intent = build_intent(data, candidate, protocol, fast_base)
    active = intent.targets.iloc[-1][intent.targets.iloc[-1] > 0]
    assert active.index.tolist() == ["QLD"]


def test_registration_requires_all_candidates_and_exact_base_protocol(tmp_path, base, protocol):
    from us_quant.regime import verify_registration

    path = tmp_path / "registration.json"
    write_json(
        path,
        {
            "protocol_sha256": digest_json(
                {
                    **protocol.__dict__,
                    "candidates": [candidate.__dict__ for candidate in protocol.candidates],
                }
            ),
            "base_protocol_sha256": digest_json(base.to_dict()),
            "candidate_ids": [item.id for item in protocol.candidates],
        },
    )
    # asdict is used by the implementation and produces the same JSON structure.
    from dataclasses import asdict

    payload = {
        "protocol_sha256": digest_json(asdict(protocol)),
        "base_protocol_sha256": digest_json(base.to_dict()),
        "candidate_ids": [item.id for item in protocol.candidates],
    }
    write_json(path, payload)
    assert verify_registration(protocol, base, path)["candidate_ids"][0]
    payload["candidate_ids"].pop()
    write_json(path, payload)
    with pytest.raises(QuantError, match="omits"):
        verify_registration(protocol, base, path)
