import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import calendar_seasonal_expert_v1 as law
from us_quant.calendar import is_month_end, market_calendar, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)


def market():
    dates = sessions("2018-01-02", "2020-12-31")
    values = 100.0 + np.arange(len(dates))[:, None] * 0.01
    return pd.DataFrame(np.repeat(values, len(law.UNIVERSE), axis=1), dates, law.UNIVERSE)


@pytest.mark.parametrize("month", range(1, 13))
def test_all_twelve_calendar_opinions_match_independent_signal_month_table(month):
    days = sessions(f"2019-{month:02d}-01", pd.Timestamp(2019, month, 1) + pd.offsets.MonthEnd())
    target, audit = law.decision(days[-1])
    selected = list(law.CORE) if month in (10, 11, 12, 1, 2, 3) else ["BIL"]
    expected = pd.Series(
        [0.98 / len(selected) if name in selected else 0.0 for name in law.UNIVERSE],
        index=law.UNIVERSE,
    )
    pd.testing.assert_series_equal(target, expected)
    assert audit["selected_assets"] == selected
    assert audit["intended_calendar_month"] == (1 if month == 12 else month + 1)
    assert abs(target.sum() - 0.98) < 1e-12 and (target >= 0).all()


def test_april_and_october_opinions_transition_before_original_delayed_execution():
    april, a = law.decision(pd.Timestamp("2019-04-30"))
    october, o = law.decision(pd.Timestamp("2019-10-31"))
    assert april["BIL"] == 0.98 and a["intended_calendar_month"] == 5
    assert october["BIL"] == 0 and o["intended_calendar_month"] == 11
    assert "execution_session" not in a and "execution_session" not in o


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "2019-04-30",
        pd.NaT,
        pd.Timestamp("2019-04-29"),
        pd.Timestamp("2020-05-31"),
        pd.Timestamp("2019-04-30 12:00:00"),
        pd.Timestamp("2019-04-30", tz="UTC"),
        pd.Timestamp("1990-01-31"),
        pd.Timestamp("2100-04-30"),
    ],
)
def test_missing_nonsession_intraday_and_timezone_inputs_are_not_defaulted(bad):
    with pytest.raises(QuantError):
        law.decision(bad)


def test_unknown_following_session_is_not_assumed_to_be_a_month_boundary():
    with pytest.raises(QuantError, match="covered NYSE month end"):
        law.decision(market_calendar().last_session)
    dates = sessions(market_calendar().last_session, market_calendar().last_session)
    close = pd.DataFrame(100.0, dates, law.UNIVERSE)
    with pytest.raises(QuantError, match="following session"):
        law.build_targets(close, read_json(law.POLICY))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("intended_equity_calendar_months", [10, 11, 12, 1, 2, 3]),
        ("source_price_warmup_sessions", False),
        ("prior_completed_configurations", 228),
        ("actual_weight_trade_band", 0.01),
        ("cash_asset", "cash"),
        ("parameter_search", True),
    ],
)
def test_calendar_or_financial_scope_drift_is_rejected(key, value):
    policy = copy.deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)


def test_no_month_end_or_native_prehistory_is_cropped_and_inputs_precede_resolution():
    close = market()
    records = []
    targets, audits = law.build_targets(close, read_json(law.POLICY), retain=records.append)
    month_ends = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    assert targets.index.equals(close.index) and targets.columns.equals(close.columns)
    assert targets.dropna(how="all").index.equals(month_ends)
    assert len(audits) == len(month_ends) and len(records) == 2 * len(audits)
    for i, day in enumerate(month_ends):
        before, after = records[2 * i : 2 * i + 2]
        assert before["record"] == "decision_input"
        assert before["decision_session"] == str(day.date())
        assert before["future_price_or_return_observed"] is False
        assert after["record"] == "decision_resolved" and after["audit"] == audits[i]
        assert after["complete_target"] == targets.loc[day].tolist()
        assert np.isfinite(targets.loc[day]).all()


def test_prefixes_future_perturbations_and_past_returns_never_choose_the_season():
    close = market()
    policy = read_json(law.POLICY)
    targets, audits = law.build_targets(close, policy)
    for position in (0, 20, 125, 252, len(close) - 1):
        day = close.index[position]
        prefix, earlier = law.build_targets(close.loc[:day], policy)
        assert prefix.equals(targets.loc[:day])
        assert earlier == [r for r in audits if pd.Timestamp(r["decision_session"]) <= day]
    different = close.iloc[::-1].to_numpy().copy()
    changed = pd.DataFrame(different, close.index, close.columns)
    other, different_audits = law.build_targets(changed, policy)
    assert other.equals(targets) and different_audits == audits


@pytest.mark.parametrize("failure", ["gap", "missing", "zero", "reorder", "boolean"])
def test_invalid_market_is_not_filled_or_reclassified_as_bill_safety(failure):
    close = market()
    if failure == "gap":
        close = close.drop(close.index[10])
    elif failure == "missing":
        close.iloc[10, 0] = np.nan
    elif failure == "zero":
        close.iloc[10, 0] = 0.0
    elif failure == "reorder":
        close = close.loc[:, list(reversed(close.columns))]
    else:
        close = close > 0
    with pytest.raises(QuantError):
        law.build_targets(close, read_json(law.POLICY))


def test_existing_expert_interface_has_real_monthly_and_dense_adapters():
    close = market()
    calendar_targets, _ = law.build_targets(close, read_json(law.POLICY))
    dense_targets = pd.DataFrame(0.0, close.index, close.columns)
    dense_targets.loc[:, list(law.CORE)] = 0.98 / len(law.CORE)
    shadow = pd.DataFrame(
        {
            "equity": 10000.0,
            "return": 0.0,
            "risk_free": 0.0,
            "cash": 10000.0,
            "gross_exposure": 0.0,
            "cost": 0.0,
            "orders": 0.0,
        },
        index=close.index,
    )
    monthly = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, "calendar_seasonality", "2026-11-05T02:21:44Z"),
        calendar_targets,
        shadow,
        law.UNIVERSE,
    )
    dense = ArchivedDenseExpert(
        ExpertIdentity("synthetic_daily_core", "synthetic_fixture", "2026-11-05T02:21:44Z"),
        dense_targets,
        shadow,
        law.UNIVERSE,
    )
    before = monthly.observe(close.index[0], 126)
    assert before.target is None and not before.score_mature
    day = pd.Timestamp("2019-05-15")
    opinion = monthly.observe(day, 126)
    current = dense.observe(day, 126)
    assert opinion.target_known_at == pd.Timestamp("2019-04-30")
    assert opinion.target["BIL"] == 0.98 and current.target["BIL"] == 0
    assert opinion.target_known_at < day and current.target_known_at == day
    cold = monthly.observe(close.index[124], 126)
    mature = monthly.observe(close.index[125], 126)
    assert cold.target is not None and not cold.score_mature
    assert len(cold.net_returns) == 125
    assert mature.score_mature and len(mature.net_returns) == 126
