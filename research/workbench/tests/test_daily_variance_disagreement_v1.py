from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import daily_variance_disagreement_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def source_row(gvz=20.0, vix=20.0, risky=0.98):
    row = pd.Series(0.0, index=rules.UNIVERSE)
    share = float(np.clip(gvz / (gvz + vix), 0.30, 0.70))
    row["SOXX"] = risky * share / 2
    row.loc[list(rules.CORE)] = risky * share / 8
    row["GLD"] = risky * (1 - share)
    row["BIL"] = 0.98 - risky
    return row


def test_identity_when_daily_realized_equals_implied_variance():
    observations = np.full(21, 0.2 / np.sqrt(252))
    share, audit = rules.equity_share(15, 20, observations)
    assert share == pytest.approx(15 / 35)
    assert audit["realized_SPY_daily_log_variance_annualized"] == pytest.approx(0.04)
    assert audit["true_variance_risk_premium_estimated"] is False
    assert audit["intraday_paper_prediction_claimed"] is False


def test_disagreement_changes_budget_with_original_bounds():
    lower_realized = np.full(21, 0.1 / np.sqrt(252))
    higher_realized = np.full(21, 0.4 / np.sqrt(252))
    assert rules.equity_share(20, 20, lower_realized)[0] == 0.70
    assert rules.equity_share(20, 20, higher_realized)[0] == 0.30


@pytest.mark.parametrize("kind", ["short", "nan", "zero", "bad_quote"])
def test_invalid_proxy_inputs_fail_instead_of_becoming_a_safe_allocation(kind):
    values = np.full(21, 0.01)
    gvz = 20.0
    if kind == "short":
        values = values[:-1]
    elif kind == "nan":
        values[0] = np.nan
    elif kind == "zero":
        values[:] = 0
    else:
        gvz = 0.0
    with pytest.raises(QuantError):
        rules.equity_share(gvz, 20, values)


@pytest.mark.parametrize("risky", [0.49, 0.98])
def test_original_dollar_scale_BIL_and_relative_core_are_preserved(risky):
    old = source_row(risky=risky)
    amount, _ = rules.source_row(old, 20, 20)
    actual = rules.transform(old, 0.70, amount)
    assert actual["BIL"] == old["BIL"]
    assert actual.loc[list(rules.UNUSED)].equals(old.loc[list(rules.UNUSED)])
    assert actual.sum() == pytest.approx(0.98)
    assert actual["SOXX"] == pytest.approx(risky * 0.7 / 2)
    np.testing.assert_allclose(actual[list(rules.CORE)], risky * 0.7 / 8)
    assert actual["GLD"] == pytest.approx(risky * 0.3)
    old["SOXX"] += 0.01
    old["GLD"] -= 0.01
    with pytest.raises(QuantError):
        rules.source_row(old, 20, 20)


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
    target = pd.DataFrame(np.nan, index=dates, columns=rules.UNIVERSE)
    monthly = [day for i, day in enumerate(dates) if i >= 63 and is_month_end(day)]
    options = pd.DataFrame({"GVZ": 20.0, "VIX": 20.0}, index=pd.DatetimeIndex(monthly))
    for i, day in enumerate(monthly):
        target.loc[day] = source_row(risky=0.49 if i % 2 else 0.98)
    data = MarketData(
        close.copy(),
        close.copy(),
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    return data, target, options


def test_scalar_oracle_prefix_and_sparse_monthly_dates():
    data, old, options = fixture()
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, old, options, policy)
    target = targets[rules.CANDIDATES[0]["id"]]
    assert target.dropna().index.equals(old.dropna().index)
    assert len(audits) == len(options)
    for day in options.index:
        oracle = rules.independent_row(data.close, old.loc[day], options.loc[day], day)
        np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
    cut = options.index[4]
    prefix = MarketData(
        data.open.loc[:cut],
        data.close.loc[:cut],
        data.raw_close.loc[:cut],
        data.volume.loc[:cut],
        data.risk_free.loc[:cut],
    )
    early, _ = rules.build_targets(prefix, old.loc[:cut], options.loc[:cut], policy)
    pd.testing.assert_frame_equal(early[rules.CANDIDATES[0]["id"]], target.loc[:cut])
    assert target.iloc[-1].isna().all()


def test_missing_monthly_option_never_fills_or_pauses():
    data, old, options = fixture()
    with pytest.raises(QuantError, match="aligned"):
        rules.build_targets(data, old, options.iloc[1:], read_json(rules.POLICY))
    damaged = old.copy()
    damaged.iloc[0, 0] = 0.1
    with pytest.raises(QuantError, match="sparse monthly"):
        rules.build_targets(data, damaged, options, read_json(rules.POLICY))


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
def test_frozen_proxy_definition_and_scope_cannot_change(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
