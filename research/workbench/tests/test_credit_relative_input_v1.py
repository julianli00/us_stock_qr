import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.credit_relative_input_v1 import LaggedQuarterCreditInput

INTRODUCED = "2026-11-04T21:15:00+00:00"


@pytest.fixture
def prices():
    index = sessions("2015-08-10", "2016-01-29")
    elapsed = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {"HYG": 17.0 * 1.001**elapsed, "LQD": 200.0 * 1.0005**elapsed}, index=index
    )


def test_exact_lag_and_quarter_endpoints(prices):
    observed = LaggedQuarterCreditInput(prices, INTRODUCED).observe(prices.index[64])
    assert observed.history_mature
    assert observed.source_start_session == prices.index[0]
    assert observed.source_end_session == prices.index[63]
    expected = (prices.iloc[63, 0] / prices.iloc[0, 0]) / (
        prices.iloc[63, 1] / prices.iloc[0, 1]
    ) - 1
    assert observed.relative_total_return == expected


def test_every_warmup_session_is_explicit(prices):
    consumer = LaggedQuarterCreditInput(prices, INTRODUCED)
    for day in prices.index[:64]:
        observed = consumer.observe(day)
        assert not observed.history_mature
        assert observed.relative_total_return is None
        assert observed.source_start_session is None
    assert consumer.observe(prices.index[0]).source_end_session is None


@pytest.mark.parametrize("position", [64, 80, 100])
def test_prefix_and_same_day_future_changes_do_not_change_input(prices, position):
    day = prices.index[position]
    whole = LaggedQuarterCreditInput(prices, INTRODUCED).observe(day)
    prefix = LaggedQuarterCreditInput(prices.loc[:day], INTRODUCED).observe(day)
    changed = prices.copy()
    changed.loc[day:, "HYG"] *= 1.1
    changed.loc[day:, "LQD"] *= 0.9
    future_changed = LaggedQuarterCreditInput(changed, INTRODUCED).observe(day)
    assert whole == prefix == future_changed


def test_independent_price_units_cancel_and_caller_cannot_mutate_state(prices):
    day = prices.index[100]
    consumer = LaggedQuarterCreditInput(prices, INTRODUCED)
    before = consumer.observe(day)
    scaled = prices * pd.Series({"HYG": 8.0, "LQD": 0.25})
    assert LaggedQuarterCreditInput(scaled, INTRODUCED).observe(day) == before
    prices.iloc[:, :] = 1.0
    assert consumer.observe(day) == before


def test_introduction_is_not_backdated_availability(prices):
    assert LaggedQuarterCreditInput(prices, INTRODUCED).observe(
        prices.index[100]
    ).predates_consumer_introduction
    assert not LaggedQuarterCreditInput(prices, "2014-01-01T00:00:00Z").observe(
        prices.index[100]
    ).predates_consumer_introduction


@pytest.mark.parametrize("introduced", ["2026-11-04", "NaT", "not-a-time"])
def test_introduction_requires_an_aware_actual_clock(prices, introduced):
    with pytest.raises(QuantError, match="aware timestamp"):
        LaggedQuarterCreditInput(prices, introduced)


@pytest.mark.parametrize(
    "malformation",
    ["columns", "missing_session", "duplicate", "reverse", "integer", "nan", "zero", "negative"],
)
def test_invalid_input_is_not_repaired(prices, malformation):
    bad = prices.copy()
    if malformation == "columns":
        bad = bad.rename(columns={"HYG": "SPY"})
    elif malformation == "missing_session":
        bad = bad.drop(bad.index[10])
    elif malformation == "duplicate":
        bad = pd.concat([bad.iloc[:1], bad])
    elif malformation == "reverse":
        bad = bad.iloc[::-1]
    elif malformation == "integer":
        bad = bad.astype("int64")
    else:
        bad.iloc[10, 0] = {"nan": np.nan, "zero": 0.0, "negative": -1.0}[malformation]
    with pytest.raises(QuantError):
        LaggedQuarterCreditInput(bad, INTRODUCED)


@pytest.mark.parametrize("day", ["2015-08-08", "2026-11-04", "2015-08-10T00:00:00Z", "NaT"])
def test_uncovered_or_ambiguous_dates_do_not_become_carry_forward(prices, day):
    with pytest.raises(QuantError, match="covered NYSE"):
        LaggedQuarterCreditInput(prices, INTRODUCED).observe(day)


def test_arithmetic_overflow_does_not_create_a_fallback_feature(prices):
    prices.iloc[0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(QuantError, match="arithmetic"):
        LaggedQuarterCreditInput(prices, INTRODUCED).observe(prices.index[64])
