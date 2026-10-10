from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.macro_factor_tilt import align_observations, build_targets, validate
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/macro-factor-tilt.json")


def test_two_session_release_lag_and_negative_macro_values_are_preserved():
    observed = pd.Series(
        [-0.5, -0.25, 0.2, 0.4],
        index=pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ]
        ),
    )
    index = sessions("2020-01-06", "2020-01-09")
    values, audit = align_observations(observed, index)
    assert list(values) == [-0.5, -0.25, 0.2, 0.4]
    assert list(audit["observation_date"]) == list(observed.index)
    assert (audit["available_session"] <= audit.index).all()


def test_macro_missing_release_is_not_a_fabricated_new_observation():
    observed = pd.Series(
        [1.0, np.nan, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    )
    values, audit = align_observations(observed, sessions("2020-01-06", "2020-01-08"))
    assert list(values) == [1.0, 1.0, 2.0]
    assert audit.loc["2020-01-07", "observation_date"] == pd.Timestamp("2020-01-02")
    assert audit.loc["2020-01-07", "carried_after_availability"]
    with pytest.raises(QuantError, match="stale"):
        align_observations(observed, sessions("2020-01-20", "2020-01-24"))


def test_future_macro_releases_do_not_change_earlier_aligned_values():
    dates = sessions("2020-01-02", "2020-02-28")
    observations = pd.Series(np.arange(len(dates)) / 100, index=dates)
    index = dates[5:]
    prior, _ = align_observations(observations, index)
    changed = observations.copy()
    changed.loc["2020-02-03":] += 10
    after, _ = align_observations(changed, index)
    pd.testing.assert_series_equal(prior.loc[:"2020-02-04"], after.loc[:"2020-02-04"])


def test_macro_tilt_preserves_four_factor_and_gold_budgets(market_factory, policy):
    market = market_factory("2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *FACTORS))
    macro = pd.DataFrame(
        {"T10Y3M": -0.1, "DFII10": np.arange(len(market.close)) / 1000}, index=market.close.index
    )
    tilt = build_targets(market, macro, policy["candidates"][0], policy).dropna(how="all")
    defense = build_targets(market, macro, policy["candidates"][1], policy).dropna(how="all")
    shares = tilt.loc[:, list(FACTORS)].div(tilt.loc[:, list(FACTORS)].sum(axis=1), axis=0)
    assert np.allclose(shares, np.array([0.15, 0.35, 0.15, 0.35]))
    assert np.allclose(tilt.sum(axis=1), 0.98) and np.allclose(defense.sum(axis=1), 0.98)
    assert np.allclose(tilt["GLD"], defense["GLD"])
    assert np.allclose(defense.loc[:, list(FACTORS)], tilt.loc[:, list(FACTORS)] * 0.5)
    assert (defense["BIL"] > 0).all()


def test_published_lag_may_not_be_reduced_to_improve_backtest(policy):
    policy["publication_delay_sessions"] = 0
    with pytest.raises(QuantError):
        validate(policy)
