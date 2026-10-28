from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import relative_equity_odds_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def fixture(semiconductor_rate=0.001):
    dates = sessions("2019-01-02", "2020-03-10")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        100 * np.exp(0.001 * t[:, None]) * np.ones((1, len(rules.UNIVERSE))),
        index=dates,
        columns=rules.UNIVERSE,
    )
    close["SOXX"] = 100 * np.exp(semiconductor_rate * t)
    desired = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    for day in dates:
        if is_month_end(day):
            desired.loc[day] = 0
            desired.loc[day, "SOXX"] = 0.196
            desired.loc[day, list(rules.CORE)] = 0.049
            desired.loc[day, "GLD"] = 0.294
            desired.loc[day, "BIL"] = 0.294
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    return data, desired


def test_equal_relative_wealth_recovers_half_sector_and_half_core():
    data, _ = fixture()
    day = data.close.index[160]
    share, proof = rules.relative_share(data.close, day)
    assert share == pytest.approx(0.5)
    assert proof["observed_return_sessions"] == 105
    assert proof["synthetic_daily_rebalanced_core_index_claimed"] is False


def test_relative_gross_wealth_formula_and_recent_skip():
    data, _ = fixture(0.002)
    day = data.close.index[160]
    share, proof = rules.relative_share(data.close, day)
    expected = np.exp(0.002 * 105) / (np.exp(0.002 * 105) + np.exp(0.001 * 105))
    assert share == pytest.approx(expected)
    altered = data.close.copy()
    cutoff = data.close.index[160 - 21]
    altered.loc[altered.index > cutoff, "SOXX"] *= 2
    assert rules.relative_share(altered, day) == (share, proof)


def test_source_equity_defense_and_factor_participation_preserved():
    data, desired = fixture(0.002)
    target_map, audits = rules.build_targets(data, desired, read_json(rules.POLICY))
    target = target_map[rules.CANDIDATES[0]["id"]]
    eligible = desired.dropna().index[
        [data.close.index.get_loc(day) >= 126 for day in desired.dropna().index]
    ]
    assert target.dropna().index.equals(eligible)
    assert target.loc[target.index < eligible[0]].isna().all().all()
    assert len(audits) == len(eligible)
    for day in eligible:
        np.testing.assert_allclose(target.loc[day, list(rules.STOCKS)].sum(), 0.392, atol=1e-12)
        assert target.loc[day, "GLD"] == desired.loc[day, "GLD"]
        assert target.loc[day, "BIL"] == desired.loc[day, "BIL"]
        assert (target.loc[day, list(rules.CORE)] > 0).all()
        oracle = rules.independent_row(data.close, desired.loc[day], day)
        np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)


def test_prefix_does_not_change_any_eligible_earlier_target():
    data, desired = fixture(0.002)
    whole, _ = rules.build_targets(data, desired, read_json(rules.POLICY))
    end = data.close.index[200]
    sliced = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(sliced, desired.loc[:end], read_json(rules.POLICY))
    pd.testing.assert_frame_equal(
        early[rules.CANDIDATES[0]["id"]], whole[rules.CANDIDATES[0]["id"]].loc[:end]
    )


def test_missing_history_or_bad_endpoint_is_not_neutral_fallback():
    data, _ = fixture()
    with pytest.raises(QuantError, match="prehistory"):
        rules.relative_share(data.close, data.close.index[125])
    damaged = data.close.copy()
    damaged.loc[damaged.index[0], "SOXX"] = np.nan
    with pytest.raises(QuantError, match="historical endpoints"):
        rules.relative_share(damaged, damaged.index[126])


def test_source_composition_corruption_is_rejected():
    _, desired = fixture()
    row = desired.dropna().iloc[0].copy()
    row["SOXX"] += 0.01
    row["MTUM"] -= 0.01
    with pytest.raises(QuantError, match="half-sector"):
        rules.source_equity(row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("skip_recent_sessions", 0),
        ("cash_reserve", 0.0),
        ("actual_weight_trade_band", 0.05),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_fixed_rule_does_not_allow_parameter_grid(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
