from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from us_quant import overnight_range_product_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_range_only_path_is_not_misread_as_zero_movement():
    bars = np.tile([100.0, 110.0, 90.0, 100.0], (21, 1))
    total, parts = rules.measure(np.full(22, 100), np.full(21, 100), bars)
    assert total == pytest.approx(252 * (math.log(1.1) ** 2 + math.log(0.9) ** 2))
    assert parts["overnight_squared_component_annualized"] == 0


def test_opening_gap_remains_separate_from_intraday_range():
    total, parts = rules.measure(
        np.r_[100.0, np.full(21, 200)],
        np.full(21, 200),
        np.full((21, 4), 200.0),
    )
    assert total == pytest.approx(252 * math.log(2) ** 2 / 21)
    assert parts["intraday_range_product_component_annualized"] == 0


def test_raw_intraday_common_scale_cancels():
    bars = np.tile([100.0, 110.0, 90.0, 102.0], (21, 1))
    closes, opens = np.linspace(100, 104, 22), np.linspace(100, 104, 21)
    left, _ = rules.measure(closes, opens, bars)
    right, _ = rules.measure(closes, opens, bars * np.linspace(0.5, 2, 21)[:, None])
    assert left == pytest.approx(right, abs=1e-12)


@pytest.mark.parametrize("damage", ["flat", "monotone", "high", "low", "missing"])
def test_degenerate_or_invalid_input_is_not_a_risk_free_default(damage):
    bars = np.full((21, 4), 100.0)
    close, open_ = np.full(22, 100.0), np.full(21, 100.0)
    if damage == "monotone":
        close = 100 * 1.01 ** np.arange(22)
        open_ = close[:-1]
        bars = np.column_stack((open_, close[1:], open_, close[1:]))
    elif damage == "high":
        bars[0, 1] = 99
    elif damage == "low":
        bars[0, 2] = 101
    elif damage == "missing":
        bars[0, 1] = np.nan
    with pytest.raises(QuantError):
        rules.measure(close, open_, bars)


def fixture():
    index = sessions("2015-08-10", "2016-02-29")
    prices = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    data = MarketData(
        prices.copy(), prices.copy(), prices.copy(), prices * 1000, pd.Series(0.0, index=index)
    )
    desired = pd.DataFrame(np.nan, index=index, columns=rules.UNIVERSE)
    row = pd.Series(0.0, index=rules.UNIVERSE)
    row["SOXX"], row["GLD"], row["BIL"] = 0.1225, 0.245, 0.49
    row.loc[list(rules.CORE)] = 0.030625
    for day in ("2015-11-30", "2015-12-31"):
        desired.loc[day] = row
    bars = pd.DataFrame(
        np.tile([100, 102, 98, 100], (len(index), 1)), index=index, columns=rules.FIELDS
    )
    options = pd.DataFrame({"GVZ": [15.0, 15.0], "VIX": [20.0, 20.0]}, index=desired.dropna().index)
    return data, desired, {"bars": bars, "options": options}


def test_defense_budget_and_independent_target_match():
    data, desired, info = fixture()
    result, audits = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    target = result[rules.CANDIDATES[0]["id"]]
    for day in desired.dropna().index:
        oracle, components = rules.independent_row(data, desired.loc[day], info, day)
        np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
        assert components["combined_proxy_annualized"] > 0
        assert target.loc[day, "BIL"] == desired.loc[day, "BIL"]
        assert target.loc[day].sum() == pytest.approx(0.98)
        assert all(target.loc[day, name] > 0 for name in rules.CORE)
    assert len(audits) == 2


def test_future_extrema_do_not_change_past_decisions():
    data, desired, info = fixture()
    end = pd.Timestamp("2015-11-30")
    info["bars"].loc[info["bars"].index > end, "high"] = 150
    full, _ = rules.build_targets(data, desired, info, read_json(rules.POLICY))
    sliced = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(
        sliced,
        desired.loc[:end],
        {key: value.loc[:end] for key, value in info.items()},
        read_json(rules.POLICY),
    )
    pd.testing.assert_frame_equal(
        early[rules.CANDIDATES[0]["id"]], full[rules.CANDIDATES[0]["id"]].loc[:end]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("realized_return_sessions", 63),
        ("annualization_sessions", 504),
        ("equity_share_max", 0.8),
        ("actual_weight_trade_band", 0.05),
    ],
)
def test_fixed_measurement_and_financial_scope_cannot_drift(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
