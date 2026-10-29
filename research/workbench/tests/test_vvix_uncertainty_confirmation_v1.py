from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from us_quant import vvix_uncertainty_confirmation_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def fixtures(guarded=True):
    index = sessions("2015-08-10", "2016-02-29")
    prices = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    data = MarketData(
        prices.copy(),
        prices.copy(),
        prices.copy(),
        prices * 1000,
        pd.Series(0.0, index=index, name="risk_free"),
    )
    target = pd.DataFrame(np.nan, index=index, columns=rules.UNIVERSE)
    row = pd.Series(0.0, index=rules.UNIVERSE)
    budget = 0.49 if guarded else 0.98
    row["SOXX"] = budget / 4
    row.loc[list(rules.CORE)] = budget / 16
    row["GLD"] = budget / 2
    row["BIL"] = 0.49 if guarded else 0.0
    target.loc["2015-11-30"] = row
    target.loc["2015-12-31"] = row
    quotes = pd.Series(100.0, index=index, name="VVIX")
    return data, target, quotes


@pytest.mark.parametrize("current,retained", [(101.0, True), (100.0, False), (99.0, False)])
def test_strict_quarter_confirmation_preserves_or_releases_exact_budget(current, retained):
    data, old, quotes = fixtures()
    quotes.loc["2015-11-30"] = current
    result, audits = rules.build_targets(data, old, quotes, read_json(rules.POLICY))
    target = result[rules.CANDIDATES[0]["id"]]
    row = target.loc["2015-11-30"]
    assert row["BIL"] == (0.49 if retained else 0.0)
    assert row.sum() == pytest.approx(0.98)
    assert all(row[name] > 0 for name in rules.CORE)
    oracle = rules.independent_row(old.loc["2015-11-30"], Decimal(str(current)), Decimal("100"))
    pd.testing.assert_series_equal(row, oracle)
    position = data.close.index.get_loc(pd.Timestamp("2015-11-30"))
    assert audits[0]["VVIX_reference_session"] == str(data.close.index[position - 63].date())
    assert audits[0]["uncertainty_guard_confirmed"] is retained


def test_new_source_does_not_create_a_guard_where_donor_has_none():
    data, old, quotes = fixtures(guarded=False)
    quotes.loc["2015-11-30"] = 200
    result, audits = rules.build_targets(data, old, quotes, read_json(rules.POLICY))
    pd.testing.assert_frame_equal(result[rules.CANDIDATES[0]["id"]], old)
    assert audits[0]["original_dollar_guard_active"] is False
    assert audits[0]["uncertainty_guard_confirmed"] is False


def test_future_source_values_and_extra_market_rows_do_not_change_prefix():
    data, old, quotes = fixtures()
    end = pd.Timestamp("2015-11-30")
    quotes.loc[quotes.index > end] = 250
    full, _ = rules.build_targets(data, old, quotes, read_json(rules.POLICY))
    sliced = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(sliced, old.loc[:end], quotes.loc[:end], read_json(rules.POLICY))
    pd.testing.assert_frame_equal(
        early[rules.CANDIDATES[0]["id"]], full[rules.CANDIDATES[0]["id"]].loc[:end]
    )
    assert full[rules.CANDIDATES[0]["id"]].dropna().index.equals(old.dropna().index)


@pytest.mark.parametrize("damage", ["missing", "zero", "name", "date", "donor_budget"])
def test_required_information_never_becomes_a_safe_default(damage):
    data, old, quotes = fixtures()
    if damage == "missing":
        quotes.iloc[0] = np.nan
    elif damage == "zero":
        quotes.iloc[0] = 0
    elif damage == "name":
        quotes.name = "VIX"
    elif damage == "date":
        quotes = quotes.iloc[1:]
    else:
        old.loc["2015-11-30", "BIL"] = 0.3
    with pytest.raises(QuantError):
        rules.build_targets(data, old, quotes, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "name,value",
    [
        ("vvix_comparison_sessions", 21),
        ("dollar_risk_off_investment_scale", 0.25),
        ("actual_weight_trade_band", 0.05),
        ("decision_information_clock", "16_00_ET"),
        ("source_monthly_opportunities", 130),
    ],
)
def test_no_post_outcome_horizon_scale_band_or_clock_rescue(name, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[name] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
