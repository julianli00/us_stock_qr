from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_initial_monthly_execution_v1 as adapted
from us_quant import observed_weight_execution as original
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def synthetic_market():
    dates = sessions("2016-01-04", "2016-12-30")
    symbols = (
        "SPY",
        "IEF",
        "TLT",
        "GLD",
        "BIL",
        "MTUM",
        "VLUE",
        "QUAL",
        "USMV",
        "QQQ",
        "XLK",
        "SOXX",
        "IJR",
        "PKW",
    )
    phase = np.arange(len(dates))
    closing = pd.DataFrame(
        {
            s: 100 * np.exp(np.cumsum(0.0001 + 0.001 * np.sin(phase / (3 + i))))
            for i, s in enumerate(symbols)
        },
        index=dates,
    )
    opening = closing.shift(1) * np.exp(0.0002 * np.cos(phase))[:, None]
    opening.iloc[0] = closing.iloc[0]
    data = MarketData(
        opening,
        closing,
        closing.copy(),
        pd.DataFrame(1e6, index=dates, columns=symbols),
        pd.Series(0.00005, index=dates, name="risk_free"),
    )
    targets = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for day in dates:
        if day >= pd.Timestamp("2016-09-30") and is_month_end(day):
            targets.loc[day] = 0.0
            if day.month == 10:
                targets.loc[day, "BIL"] = 0.98
            else:
                targets.loc[day, "SOXX"] = 0.49
                targets.loc[day, ["MTUM", "VLUE", "QUAL", "USMV"]] = 0.1225
    return data, targets


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_monthly_execution_is_identical_for_valid_monthly_targets(cost, delay):
    data, targets = synthetic_market()
    policy = read_json(original.POLICY)
    args = (data, targets, original.CANDIDATES[0], policy, "2016-10-06", "2016-12-29", cost, delay)
    before, before_issued, before_decisions = original.run(*args)
    after, after_issued, after_decisions = adapted.run(*args)
    pd.testing.assert_frame_equal(before.frame, after.frame, check_exact=True)
    pd.testing.assert_frame_equal(before.weights, after.weights, check_exact=True)
    pd.testing.assert_frame_equal(before_issued, after_issued, check_exact=True)
    assert before_decisions == after_decisions


@pytest.mark.parametrize(
    "cost,delay,first_fill",
    [
        (5.0, 1, "2016-10-06"),
        (20.0, 2, "2016-10-07"),
    ],
)
def test_real_initial_target_is_not_backdated_to_a_month_end(cost, delay, first_fill):
    data, targets = synthetic_market()
    targets.loc["2016-10-05"] = targets.loc["2016-09-30"]
    targets.loc["2016-09-30"] = np.nan
    policy = read_json(original.POLICY)
    args = (data, targets, original.CANDIDATES[0], policy, "2016-10-06", "2016-12-29", cost, delay)
    with pytest.raises(QuantError, match="monthly"):
        original.run(*args)
    result, issued, decisions = adapted.run(*args)
    assert issued.loc[:"2016-10-04"].isna().all().all()
    assert decisions[0]["signal_session"] == "2016-10-05"
    assert decisions[0]["execution_session"] == first_fill
    assert str(result.frame.index[result.frame["orders"] > 0][0].date()) == first_fill
    assert result.frame.index.equals(sessions("2016-10-06", "2016-12-29"))
    assert result.frame["risk_free"].equals(data.risk_free.loc[result.frame.index])
    assert (result.frame["cash"] >= 0).all() and (result.frame["gross_exposure"] <= 1).all()
    feedback = adapted.original.verify_feedback(
        data,
        targets,
        result,
        issued,
        decisions,
        {"start": "2016-10-06", "end": "2016-12-29"},
        {"delay_sessions": delay},
    )
    assert feedback["all_actual_gaps_targets_and_delays_checked"] is True
    if delay == 2:
        assert result.frame.iloc[0]["equity"] == 10000
        assert result.frame.iloc[0]["orders"] == result.frame.iloc[0]["cost"] == 0


def test_initial_exception_does_not_allow_new_nonmonthly_trading_dates():
    data, targets = synthetic_market()
    targets.loc["2016-10-05"] = targets.loc["2016-09-30"]
    targets.loc["2016-09-30"] = np.nan
    targets.loc["2016-10-12"] = targets.loc["2016-10-05"]
    with pytest.raises(QuantError, match="Only one real initial seed"):
        adapted.run(
            data,
            targets,
            original.CANDIDATES[0],
            read_json(original.POLICY),
            "2016-10-06",
            "2016-12-29",
            5.0,
            1,
        )


def test_future_target_cannot_supply_missing_initial_capital_information():
    data, targets = synthetic_market()
    targets.loc["2016-09-30"] = np.nan
    with pytest.raises(QuantError, match="future recommendation"):
        adapted.run(
            data,
            targets,
            original.CANDIDATES[0],
            read_json(original.POLICY),
            "2016-10-06",
            "2016-12-29",
            5.0,
            1,
        )
