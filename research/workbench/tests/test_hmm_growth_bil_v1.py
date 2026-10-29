from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import hmm_growth_bil_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_probability_blend_retains_actual_stock_floor_and_cash_funding(probability):
    row = rules.allocation(probability)
    assert row.sum() == pytest.approx(0.98)
    assert row["GLD"] == 0
    assert (row >= 0).all()
    assert row.loc[list(rules.CORE)].min() >= 0.147 / 8
    assert row.loc[[*rules.CORE, "SOXX"]].sum() == pytest.approx(0.147 + 0.833 * probability)


@pytest.mark.parametrize("probability", [-0.01, 1.01, np.nan])
def test_invalid_probability_never_becomes_a_safe_portfolio(probability):
    with pytest.raises(QuantError):
        rules.allocation(probability)


def fixture():
    index = sessions("2015-08-10", "2016-12-30")
    rng = np.random.default_rng(87432)
    values = np.r_[rng.normal(0, 0.004, 200), rng.normal(0, 0.025, len(index) - 200)]
    prices = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    prices["SPY"] = 100 * np.exp(np.cumsum(values))
    return MarketData(
        prices.copy(), prices.copy(), prices.copy(), prices * 1000, pd.Series(0.0, index=index)
    )


def test_current_filtered_state_and_independent_target_agree():
    data = fixture()
    day = pd.Timestamp("2016-08-31")
    target, audit = rules.target_for_day(data, day)
    independent, probability, likelihood, pooled = rules.independent_row(data, day, audit)
    np.testing.assert_allclose(target, independent, rtol=0, atol=1e-10)
    assert probability[0] == pytest.approx(
        audit["model"]["low_variance_state_probability"], abs=1e-10
    )
    assert likelihood == pytest.approx(audit["model"]["log_likelihood"], abs=1e-8)
    assert pooled == pytest.approx(audit["model"]["pooled_daily_squared_return"], abs=1e-12)


def test_future_price_changes_do_not_change_an_earlier_fit():
    data = fixture()
    day = pd.Timestamp("2016-08-31")
    before, audit = rules.target_for_day(data, day)
    data.close.loc[data.close.index > day, "SPY"] *= 100
    after, later_audit = rules.target_for_day(data, day)
    pd.testing.assert_series_equal(before, after)
    assert audit == later_audit


@pytest.mark.parametrize(
    "key,value",
    [
        ("states", 3),
        ("lookback_return_sessions", 126),
        ("transition_dirichlet_alpha", 1.0),
        ("stock_target_minimum", 0.1),
    ],
)
def test_hypothesis_and_inherited_floor_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
