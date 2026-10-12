from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.adaptive_factor_allocation import ASSETS, build_targets, target, validate_policy
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/adaptive-factor-allocation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2014-01-02", "2016-03-31", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def test_static_control_has_explicit_fixed_weights(market, policy):
    previous = pd.Series(0.0, index=market.close.columns)
    previous["BIL"] = 0.98
    weights = target(market.close, market.risk_free, previous, policy["candidates"][0], policy)
    np.testing.assert_allclose(weights.loc[list(FACTORS)], 0.98 * 0.70 / 4)
    assert weights["GLD"] == pytest.approx(0.294)
    assert weights.sum() == pytest.approx(0.98)


def test_adaptive_target_preserves_families_and_funding(market, policy):
    previous = pd.Series(0.0, index=market.close.columns)
    previous["BIL"] = 0.98
    weights = target(market.close, market.risk_free, previous, policy["candidates"][1], policy)
    assert weights.sum() == pytest.approx(0.98)
    assert weights.loc[list(FACTORS)].between(0.025 - 1e-10, 0.25 + 1e-10).all()
    assert 0 <= weights["GLD"] <= 0.5 + 1e-10
    assert 0 <= weights["BIL"] <= 0.88 + 1e-10
    assert weights[["SPY", "IEF"]].eq(0).all()


def test_online_state_and_monthly_weights_do_not_use_future_observations(market, policy):
    original = build_targets(market, policy)
    cutoff = pd.Timestamp("2015-09-30")
    close = market.close.copy()
    later = close.index > cutoff
    close.loc[later, "MTUM"] *= np.linspace(1, 1.4, later.sum())
    updated = replace(market, close=close, raw_close=close.copy(), open=close * 0.999)
    after = build_targets(updated, policy)
    for name, frame in original.items():
        pd.testing.assert_frame_equal(frame.loc[:cutoff], after[name].loc[:cutoff])
        assert all(is_month_end(day) for day in frame.dropna(how="all").index)
        assert np.allclose(frame.dropna(how="all").sum(axis=1), 0.98)
        assert not {"QLD", "TQQQ", "UPRO"} & set(ASSETS)


def test_optimizer_failure_cannot_be_presented_as_fallback_success(market, policy, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "us_quant.adaptive_factor_allocation.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=np.ones(6) / 6),
    )
    previous = pd.Series(0.0, index=market.close.columns)
    previous["BIL"] = 0.98
    with pytest.raises(QuantError, match="no silent fallback"):
        target(market.close, market.risk_free, previous, policy["candidates"][1], policy)


def test_parameters_are_frozen_not_refit_to_full_history(policy):
    validate_policy(policy)
    policy["risk_aversion"] = 2
    with pytest.raises(QuantError):
        validate_policy(policy)
