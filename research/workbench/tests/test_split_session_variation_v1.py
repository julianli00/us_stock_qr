from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import daily_variance_disagreement_v1 as old
from us_quant import split_session_variation_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.mark.parametrize("open_at", ["prior", "current"])
def test_zero_segment_recovers_close_only_proxy_and_original_equity_law(open_at):
    closes = 100 * np.exp(np.arange(22) * 0.01)
    opens = closes[:-1] if open_at == "prior" else closes[1:]
    value, audit = rules.measure(closes, opens)
    expected, _ = old.equity_share(15, 20, np.diff(np.log(closes)))
    assert value == pytest.approx(252 * 0.01**2)
    assert rules.equity_share(15, 20, value) == pytest.approx(expected)
    assert audit["signed_cross_term_annualized"] == pytest.approx(0.0)


def test_cancelled_day_has_positive_variation_and_negative_cross_term():
    closes, opens = np.full(22, 100.0), np.full(21, 100 * np.exp(0.01))
    value, audit = rules.measure(closes, opens)
    assert value == pytest.approx(252 * 2 * 0.01**2)
    assert audit["SPY_close_only_squared_log_variation_annualized"] == 0
    assert audit["signed_cross_term_annualized"] == pytest.approx(-value)
    assert audit["squared_variation_identity_error"] < 1e-14


def test_same_direction_segments_can_lower_proxy_and_raise_equity_budget():
    closes = 100 * np.exp(np.arange(22) * 0.02)
    opens = np.sqrt(closes[:-1] * closes[1:])
    value, audit = rules.measure(closes, opens)
    close_only = audit["SPY_close_only_squared_log_variation_annualized"]
    assert value == pytest.approx(close_only / 2)
    assert audit["signed_cross_term_annualized"] == pytest.approx(value)
    assert rules.equity_share(20, 20, value) > rules.equity_share(20, 20, close_only)


def test_price_scale_invariance_and_daily_annualization_not_double_counted():
    closes = 100 * np.exp(np.arange(22) * 0.01)
    opens = closes[:-1] * np.exp(0.003)
    value, _ = rules.measure(closes, opens)
    scaled, _ = rules.measure(0.01 * closes, 0.01 * opens)
    assert value == pytest.approx(252 * (0.003**2 + 0.007**2))
    assert scaled == pytest.approx(value)


@pytest.mark.parametrize("kind", ["short_close", "short_open", "nan", "negative", "zero"])
def test_bad_intervals_fail_without_cash_or_cap_defaults(kind):
    closes = 100 * np.exp(np.arange(22) * 0.01)
    opens = closes[:-1].copy()
    if kind == "short_close":
        closes = closes[:-1]
    elif kind == "short_open":
        opens = opens[:-1]
    elif kind == "nan":
        opens[-1] = np.nan
    elif kind == "negative":
        closes[2] = -1
    else:
        closes[:], opens[:] = 100, 100
    with pytest.raises(QuantError):
        rules.measure(closes, opens)


def fixture():
    dates = sessions("2019-01-02", "2020-03-10")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0002 * t + 0.008 * np.sin(t / (4 + i)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )
    open_ = close.mul(np.exp(0.003 * np.cos(t / 7)), axis=0)
    target = pd.DataFrame(np.nan, index=dates, columns=rules.UNIVERSE)
    monthly = [day for i, day in enumerate(dates) if i >= 63 and is_month_end(day)]
    options = pd.DataFrame({"GVZ": 20.0, "VIX": 20.0}, index=pd.DatetimeIndex(monthly))
    for i, day in enumerate(monthly):
        risky = 0.49 if i % 2 else 0.98
        target.loc[day] = 0.0
        target.loc[day, list(rules.CORE)] = risky / 16
        target.loc[day, ["SOXX", "GLD", "BIL"]] = [risky / 4, risky / 2, 0.98 - risky]
    data = MarketData(
        open_, close, close.copy(), close * 1000, pd.Series(0.0, index=dates, name="risk_free")
    )
    return data, target, options


def test_scalar_oracle_source_budget_and_prefix_invariance():
    data, desired, options = fixture()
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, desired, options, policy)
    target = targets[rules.CANDIDATES[0]["id"]]
    assert target.dropna().index.equals(desired.dropna().index)
    assert all(row["segments"] == 42 and row["sessions"] == 21 for row in audits)
    for day in options.index:
        oracle = rules.independent_row(data, desired.loc[day], options.loc[day], day)
        np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
        assert target.loc[day, "BIL"] == desired.loc[day, "BIL"]
        assert target.loc[day].sum() == pytest.approx(0.98)
        stock = target.loc[day, list(rules.STOCKS)].sum()
        assert target.loc[day, "SOXX"] == pytest.approx(stock / 2)
        np.testing.assert_allclose(target.loc[day, list(rules.CORE)], stock / 8)
    end = options.index[4]
    short = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(short, desired.loc[:end], options.loc[:end], policy)
    pd.testing.assert_frame_equal(early[rules.CANDIDATES[0]["id"]], target.loc[:end])
    assert target.iloc[-1].isna().all()


@pytest.mark.parametrize("kind", ["option_gap", "partial_row", "opening_alignment"])
def test_invalid_calendar_and_source_states_are_not_filled(kind):
    data, desired, options = fixture()
    if kind == "option_gap":
        options = options.iloc[1:]
    elif kind == "partial_row":
        desired.iloc[0, 0] = 0.1
    else:
        data = MarketData(
            data.open.iloc[1:], data.close, data.raw_close, data.volume, data.risk_free
        )
    with pytest.raises(QuantError):
        rules.build_targets(data, desired, options, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("segments_per_session", 3),
        ("annualization_sessions", 504),
        ("realized_return_sessions", 63),
        ("equity_share_max", 0.80),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_information_partition_and_parameters_frozen(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
