from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import symmetric_variance_disagreement_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def constant_logs(volatility):
    return np.full(21, volatility / np.sqrt(252))


def test_both_unit_disagreement_ratios_recover_original_option_budget():
    share, audit = rules.equity_share(15, 20, constant_logs(0.20), constant_logs(0.15))
    assert share == pytest.approx(15 / 35)
    assert audit["equity_implied_over_backward_realized_variance_ratio"] == pytest.approx(1)
    assert audit["gold_implied_over_backward_realized_variance_ratio"] == pytest.approx(1)
    assert audit["true_variance_risk_premia_or_intraday_paper_claimed"] is False


def test_unit_gold_ratio_recovers_the_frozen_one_sided_rule():
    stock = constant_logs(0.18)
    expected = rules.previous.equity_share(20, 25, stock)[0]
    actual = rules.equity_share(20, 25, stock, constant_logs(0.20))[0]
    assert actual == pytest.approx(expected)


def test_gold_information_changes_allocation_without_changing_bounds():
    stock = constant_logs(0.20)
    assert rules.equity_share(20, 20, stock, constant_logs(0.10))[0] == 0.30
    assert rules.equity_share(20, 20, stock, constant_logs(0.40))[0] == 0.70


@pytest.mark.parametrize("damage", ["short", "nan", "zero_gold", "zero_stock", "bad_quote"])
def test_invalid_observations_are_not_a_standby_or_cap(damage):
    a, b = constant_logs(0.20), constant_logs(0.20)
    quote = 20.0
    if damage == "short":
        b = b[:-1]
    elif damage == "nan":
        b[0] = np.nan
    elif damage == "zero_gold":
        b[:] = 0
    elif damage == "zero_stock":
        a[:] = 0
    else:
        quote = 0
    with pytest.raises(QuantError):
        rules.equity_share(quote, 20, a, b)


def fixture():
    index = sessions("2019-01-02", "2020-03-10")
    x = np.arange(len(index))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0001 * x + 0.008 * np.sin(x / (5 + i)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=index,
    )
    source = pd.DataFrame(np.nan, index=index, columns=rules.UNIVERSE)
    monthly = [day for i, day in enumerate(index) if i >= 63 and is_month_end(day)]
    options = pd.DataFrame({"GVZ": 20.0, "VIX": 20.0}, index=pd.DatetimeIndex(monthly))
    for i, day in enumerate(monthly):
        risky = 0.49 if i % 2 else 0.98
        row = pd.Series(0.0, index=rules.UNIVERSE)
        row["SOXX"] = risky / 4
        row.loc[list(rules.CORE)] = risky / 16
        row["GLD"] = risky / 2
        row["BIL"] = 0.98 - risky
        source.loc[day] = row
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=index, name="risk_free"),
    )
    return data, source, options


def test_independent_formula_source_preservation_and_prefix():
    data, source, options = fixture()
    target_map, audits = rules.build_targets(data, source, options, read_json(rules.POLICY))
    target = target_map[rules.CANDIDATES[0]["id"]]
    assert target.dropna().index.equals(source.dropna().index)
    assert len(audits) == len(options)
    for day in options.index:
        expected = rules.independent_row(data.close, source.loc[day], options.loc[day], day)
        np.testing.assert_allclose(target.loc[day], expected, rtol=0, atol=1e-12)
        assert target.loc[day, "BIL"] == source.loc[day, "BIL"]
        assert target.loc[day].sum() == pytest.approx(0.98)
        assert target.loc[day, "SOXX"] == pytest.approx(target.loc[day, list(rules.CORE)].sum())
    end = options.index[3]
    prefix = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(
        prefix, source.loc[:end], options.loc[:end], read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(early[rules.CANDIDATES[0]["id"]], target.loc[:end])
    assert target.iloc[-1].isna().all()


def test_missing_monthly_gold_quote_is_fatal():
    data, source, options = fixture()
    options.loc[options.index[0], "GVZ"] = np.nan
    with pytest.raises(QuantError, match="filled or dropped"):
        rules.build_targets(data, source, options, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("realized_return_sessions", 63),
        ("equity_share_max", 0.8),
        ("cash_reserve", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_no_parameter_grid_or_scope_relaxation(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
