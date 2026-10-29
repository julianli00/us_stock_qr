from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import core_activity_momentum_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_constant_activity_matches_uniform_time():
    close = 100 * np.exp(np.cumsum(np.r_[0, np.linspace(-0.01, 0.01, 105)]))
    a = rules.score(close, np.full(105, 100.0), np.full(105, 1000.0), "uniform_time")
    b = rules.score(close, np.full(105, 100.0), np.full(105, 1000.0), "reported_price_volume")
    assert a == b


def test_high_activity_can_weight_a_negative_return():
    returns = np.zeros(105)
    returns[0] = -0.1
    close = 100 * np.exp(np.r_[0, np.cumsum(returns)])
    volume = np.ones(105)
    volume[0] = 100
    assert rules.score(close, np.ones(105), volume, "reported_price_volume") < rules.score(
        close, np.ones(105), volume, "uniform_time"
    )


def test_activity_scale_does_not_change_the_score():
    close = 100 * np.exp(np.cumsum(np.r_[0, np.linspace(-0.01, 0.02, 105)]))
    volume = np.arange(1, 106, dtype=float)
    a = rules.score(close, np.full(105, 100.0), volume, "reported_price_volume")
    b = rules.score(close, np.full(105, 200.0), volume * 100, "reported_price_volume")
    assert a == pytest.approx(b, abs=1e-15)


@pytest.mark.parametrize("bad", [0.0, np.nan, -1.0])
def test_bad_volume_is_not_a_uniform_fallback(bad):
    volume = np.ones(105)
    volume[3] = bad
    with pytest.raises(QuantError):
        rules.score(np.full(106, 100.0), np.full(105, 100.0), volume, "reported_price_volume")


def test_fixed_rank_resolution_retains_every_family_and_lexical_ties():
    scores = {"MTUM": 0.0, "VLUE": -0.01, "QUAL": 1e-14, "USMV": -0.01}
    assert rules.rank(scores) == ["MTUM", "QUAL", "USMV", "VLUE"]


def fixture():
    index = sessions("2015-08-10", "2016-06-30")
    price = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    for j, name in enumerate(rules.CORE, 1):
        price[name] *= np.exp(np.arange(len(index)) * j * 0.0001)
    data = MarketData(
        price.copy(), price.copy(), price.copy(), price * 1000, pd.Series(0.0, index=index)
    )
    desired = pd.DataFrame(np.nan, index=index, columns=price.columns)
    row = pd.Series(0.0, index=price.columns)
    row["SOXX"], row["GLD"], row["BIL"] = 0.1225, 0.245, 0.49
    row.loc[list(rules.CORE)] = 0.030625
    for i, day in enumerate(index):
        if i >= 63 and is_month_end(day):
            desired.loc[day] = row
    return data, desired


def test_warmup_exact_noncore_and_independent_ranking():
    data, desired = fixture()
    targets, audits = rules.build_targets(data, desired, read_json(rules.POLICY))
    for candidate in rules.CANDIDATES:
        target = targets[candidate["id"]]
        assert target.dropna().index[0] == pd.Timestamp("2016-02-29")
        assert target.loc[:"2016-01-29"].isna().all().all()
        for day in target.dropna().index:
            oracle, scores, order = rules.independent_row(
                data, desired.loc[day], day, candidate["score"]
            )
            np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
            assert order == rules.rank(scores)
            assert target.loc[day, target.columns.difference(rules.CORE)].equals(
                desired.loc[day, desired.columns.difference(rules.CORE)]
            )
        assert audits[candidate["id"]][0]["return_sessions"] == 105


def test_skipped_recent_volume_does_not_change_current_ranking():
    data, desired = fixture()
    end = pd.Timestamp("2016-05-31")
    early = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end],
    )
    before, _ = rules.build_targets(early, desired.loc[:end], read_json(rules.POLICY))
    early.volume.iloc[-21:, early.volume.columns.get_loc("MTUM")] *= 1e6
    after, _ = rules.build_targets(early, desired.loc[:end], read_json(rules.POLICY))
    for identifier in before:
        pd.testing.assert_series_equal(before[identifier].loc[end], after[identifier].loc[end])


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("skip_recent_sessions", 0),
        ("ranking_score_decimal_places", 8),
        ("ranked_factor_shares", [0.4, 0.3, 0.2, 0.1]),
    ],
)
def test_frozen_comparison_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
