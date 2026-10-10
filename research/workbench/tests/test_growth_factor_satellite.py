from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.growth_factor_satellite import build_targets, composed_monthly_target, validate_policy
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/growth-factor-satellite.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *FACTORS, "TLT", "QQQ")
    )


def test_growth_does_not_replace_the_entire_multifactor_core(market, policy):
    control = composed_monthly_target(market.close, policy["candidates"][0])
    satellite = composed_monthly_target(market.close, policy["candidates"][1])
    assert control["QQQ"] == 0
    equity = satellite.loc[[*FACTORS, "QQQ"]].sum()
    assert satellite["QQQ"] == pytest.approx(equity / 2)
    np.testing.assert_allclose(satellite.loc[list(FACTORS)], equity / 8)
    assert satellite.sum() == pytest.approx(0.98)
    assert not {"QLD", "TQQQ"} & set(satellite.index)


def test_daily_projection_keeps_matched_controls_and_cash_funding(market, policy):
    outputs = build_targets(market, policy)
    for name, signal in outputs.items():
        active = signal.dropna(how="all")
        assert (active >= -1e-12).all().all()
        assert np.allclose(active.sum(axis=1), 0.98)
        assert active[["SPY", "IEF", "TLT"]].eq(0).all().all()
        if name.endswith("satellite50"):
            assert np.allclose(active["QQQ"], active.loc[:, list(FACTORS)].sum(axis=1))
        else:
            assert active["QQQ"].eq(0).all()


def test_future_growth_returns_do_not_change_past_targets(market, policy):
    original = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-01-31")
    close = market.close.copy()
    later = close.index > cutoff
    close.loc[later, "QQQ"] *= np.linspace(1, 1.5, later.sum())
    altered = replace(market, close=close, raw_close=close.copy(), open=close * 0.999)
    after = build_targets(altered, policy)
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], after[name].loc[:cutoff])


def test_growth_fraction_and_risk_target_are_not_retuned(policy):
    validate_policy(policy)
    policy["candidates"][1]["growth_share_of_equity"] = 0.75
    with pytest.raises(QuantError):
        validate_policy(policy)
