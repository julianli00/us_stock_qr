from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import vxn_relative_uncertainty_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_equal_underlying_normalized_uncertainty_recovers_half():
    share, audit = rules.growth_share(40.0, 20.0, 0.16, 0.04)
    assert share == pytest.approx(0.5)
    assert audit["NDX_implied_variance_proxy"] == pytest.approx(0.16)
    assert audit["SPX_implied_variance_proxy"] == pytest.approx(0.04)


def test_higher_relative_growth_uncertainty_reduces_growth_share_without_clipping():
    assert rules.growth_share(40.0, 20.0, 0.04, 0.04)[0] == pytest.approx(0.2)
    assert rules.growth_share(20.0, 40.0, 0.04, 0.04)[0] == pytest.approx(0.8)


def test_common_quote_or_observed_variation_scaling_preserves_odds():
    first = rules.growth_share(30.0, 20.0, 0.08, 0.04)[0]
    assert rules.growth_share(60.0, 40.0, 0.08, 0.04)[0] == pytest.approx(first)
    assert rules.growth_share(30.0, 20.0, 0.16, 0.08)[0] == pytest.approx(first)


def test_finite_input_overflow_is_an_explicit_error_not_a_cap():
    with pytest.raises(QuantError, match="conversion"):
        rules.growth_share(1e308, 20.0, 0.08, 0.04)


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_quote_or_variation_is_not_a_neutral_share(position, bad):
    values = [30.0, 20.0, 0.08, 0.04]
    values[position] = bad
    with pytest.raises(QuantError):
        rules.growth_share(*values)


def fixture(end="2016-11-30"):
    index = sessions("2015-08-10", end)
    close = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    opening = close.copy()
    for symbol, slope in (("SPY", 0.0001), ("QQQ", 0.0002)):
        close[symbol] *= np.exp(np.arange(len(index)) * slope)
        opening[symbol] = close[symbol].shift(1).fillna(close[symbol].iloc[0]) * np.exp(slope / 2)
    data = MarketData(opening, close, close.copy(), close * 1000, pd.Series(0.0, index=index))
    desired = pd.DataFrame(np.nan, index=index, columns=close.columns)
    row = pd.Series(0.0, index=close.columns)
    row["SOXX"], row["BIL"] = 0.35, 0.28
    row.loc[list(rules.CORE)] = 0.35 / 4
    for i, day in enumerate(index):
        if i >= 252 and is_month_end(day):
            desired.loc[day] = row
    information = pd.DataFrame({"VXN": 20.0, "VIX": 10.0}, index=index)
    return data, desired, information


def test_all_source_dates_nonstock_capital_and_equal_styles_remain_without_hmm(monkeypatch):
    from us_quant import hmm_growth_bil_v1, volatility_hmm_v1

    monkeypatch.setattr(volatility_hmm_v1, "fit", lambda *a, **k: pytest.fail("No HMM fit"))
    monkeypatch.setattr(
        hmm_growth_bil_v1, "build_targets", lambda *a, **k: pytest.fail("No parent generation")
    )
    data, desired, info = fixture()
    info["VXN"] = 30.0
    targets, audits = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    target = targets[rules.CANDIDATES[0]["id"]]
    assert target.dropna().index.equals(desired.dropna().index)
    nonstock = desired.columns.difference(rules.STOCKS)
    for day in target.dropna().index:
        oracle, share, measured = rules.independent_row(data, desired.loc[day], info.loc[day], day)
        np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
        assert target.loc[day, nonstock].equals(desired.loc[day, nonstock])
        assert target.loc[day, list(rules.STOCKS)].sum() == pytest.approx(0.7)
        assert share == pytest.approx(1 / 3.25)
        assert measured["QQQ"] == pytest.approx(4 * measured["SPY"], abs=1e-14)
    audit = audits[rules.CANDIDATES[0]["id"]][0]
    assert audit["observed_variation"]["QQQ"]["symbol"] == "QQQ"
    assert audit["hmm_refitted_or_states_recomputed"] is False


def test_equal_information_reproduces_source_targets():
    data, desired, info = fixture()
    targets, _ = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    np.testing.assert_allclose(
        targets[rules.CANDIDATES[0]["id"]], desired, rtol=0, atol=1e-12, equal_nan=True
    )


def test_missing_option_data_is_not_filled_or_mapped_to_cash():
    data, desired, info = fixture()
    info.loc[desired.dropna().index[0], "VXN"] = np.nan
    with pytest.raises(QuantError):
        rules.build_targets(data, desired, info, read_json(rules.POLICY))


def test_future_prices_and_option_quotes_cannot_change_earlier_targets():
    data, desired, info = fixture()
    before, _ = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    end = pd.Timestamp("2016-09-30")
    data.close.loc[data.close.index > end, "QQQ"] *= 1.1
    info.loc[info.index > end, "VXN"] *= 2
    after, _ = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    name = rules.CANDIDATES[0]["id"]
    pd.testing.assert_frame_equal(before[name].loc[:end], after[name].loc[:end])


def test_an_altered_parent_composition_is_not_accepted():
    data, desired, info = fixture()
    day = desired.dropna().index[0]
    desired.loc[day, "SOXX"] += 0.01
    desired.loc[day, "MTUM"] -= 0.01
    with pytest.raises(QuantError, match="half-sector"):
        rules.build_targets(data, desired, info, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("realized_return_sessions", 63),
        ("annualization_sessions", 504),
        ("actual_weight_trade_band", 0.05),
        ("additional_source_months_dropped", 1),
    ],
)
def test_the_frozen_information_rule_is_not_a_parameter_grid(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
