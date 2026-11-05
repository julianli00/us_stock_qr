from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import fixed_four_sleeve_expert_v1 as law
from us_quant.calendar import is_month_end, market_calendar, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ArchivedMonthlyExpert, ExpertIdentity


def market():
    dates = sessions("2018-01-02", "2020-12-31")
    values = 100.0 + np.arange(len(dates))[:, None] * np.linspace(0.01, 0.14, 14)
    return pd.DataFrame(values, dates, law.UNIVERSE)


@pytest.mark.parametrize("month", range(1, 13))
def test_each_month_matches_independent_fractional_quarter_capital_oracle(month):
    dates = sessions(f"2019-{month:02d}-01", pd.Timestamp(2019, month, 1) + pd.offsets.MonthEnd())
    target, audit = law.decision(dates[-1])
    expected = pd.Series(
        [
            float(Fraction(49, 1200))
            if name in law.CORE
            else float(Fraction(49, 200))
            if name in ("TLT", "GLD", "BIL")
            else 0.0
            for name in law.UNIVERSE
        ],
        index=law.UNIVERSE,
    )
    pd.testing.assert_series_equal(target, expected)
    assert abs(target.sum() - 0.98) < 1e-12 and (target >= 0).all()
    assert target["BIL"] == 0.245 and audit["idle_cash_reserve"] == 0.02
    assert audit["BIL_is_actual_ETF_not_idle_cash"] is True
    assert audit["capital_buckets_claimed_independent_strategy_experts"] is False


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
def test_invalid_or_nonmonthend_dates_are_not_defaulted(bad):
    with pytest.raises(QuantError):
        law.decision(bad)


def test_unknown_following_session_is_not_guessed_to_be_month_end():
    with pytest.raises(QuantError):
        law.decision(market_calendar().last_session)
    dates = sessions(market_calendar().last_session, market_calendar().last_session)
    with pytest.raises(QuantError, match="following session"):
        law.build_targets(pd.DataFrame(100.0, dates, law.UNIVERSE), read_json(law.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("bucket_weights_within_ETF_budget", [0.30, 0.20, 0.25, 0.25]),
        ("source_price_warmup_sessions", False),
        ("source_price_warmup_sessions", 21),
        ("prior_completed_configurations", 228),
        ("prior_completed_configurations", 259),
        ("actual_weight_trade_band", 0.01),
        ("cash_asset", "cash"),
        ("price_return_covariance_or_forecast_used_to_choose_weights", True),
        ("parameter_search", True),
    ],
)
def test_no_weight_band_clock_scope_or_authoritative_prior_rescue(key, value):
    policy = deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)


@pytest.mark.parametrize("failure", ["gap", "missing", "zero", "reorder", "boolean"])
def test_bad_market_is_not_a_successful_fixed_or_bill_fallback(failure):
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


def test_no_price_warmup_is_not_a_mature_NET_assessment():
    day = pd.Timestamp("2019-01-31")
    close = pd.DataFrame(100.0, pd.DatetimeIndex([day]), law.UNIVERSE)
    targets, audits = law.build_targets(close, read_json(law.POLICY))
    assert targets.dropna(how="all").index.equals(close.index)
    assert len(audits) == 1
    assert audits[0]["mature_NET_score_inferred_from_fixed_target_availability"] is False


def test_all_native_dates_and_input_before_resolution_are_retained():
    close = market()
    records, guarded = [], []
    targets, audits = law.build_targets(
        close,
        read_json(law.POLICY),
        before_decision=lambda: guarded.append(True),
        retain=records.append,
    )
    expected = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    assert targets.index.equals(close.index) and targets.columns.equals(close.columns)
    assert targets.dropna(how="all").index.equals(expected)
    assert len(audits) == len(guarded) == len(expected)
    assert len(records) == 2 * len(audits)
    for index, day in enumerate(expected):
        before, after = records[2 * index : 2 * index + 2]
        assert before["record"] == "decision_input"
        assert before["decision_session"] == str(day.date())
        assert before["price_or_future_return_observed_to_choose_weight"] is False
        assert after["record"] == "decision_resolved" and after["audit"] == audits[index]
        assert after["complete_target"] == targets.loc[day].tolist()


def test_prefixes_and_arbitrary_valid_price_changes_do_not_choose_allocations():
    close = market()
    policy = read_json(law.POLICY)
    targets, audits = law.build_targets(close, policy)
    for position in (0, 20, 125, 252, len(close) - 1):
        day = close.index[position]
        prefix, earlier = law.build_targets(close.loc[:day], policy)
        pd.testing.assert_frame_equal(prefix, targets.loc[:day])
        assert earlier == [row for row in audits if pd.Timestamp(row["decision_session"]) <= day]
    changed = pd.DataFrame(close.iloc[::-1].to_numpy(), close.index, close.columns)
    other, alternative = law.build_targets(changed, policy)
    pd.testing.assert_frame_equal(other, targets)
    assert alternative == audits


def test_existing_monthly_and_dense_adapters_keep_policy_and_NET_clocks_separate():
    close = market()
    targets, _ = law.build_targets(close, read_json(law.POLICY))
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
        index=close.index[252:],
    )
    monthly = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, "fixed_cross_asset_capital_rebalancing", "2026-11-05T07:39:44Z"),
        targets,
        shadow,
        law.UNIVERSE,
    )
    dense = ArchivedDenseExpert(
        ExpertIdentity("synthetic_daily_core", "synthetic_fixture", "2026-11-05T07:39:44Z"),
        dense_targets,
        shadow,
        law.UNIVERSE,
    )
    assert monthly.observe(close.index[0], 126).target is None
    initial = monthly.observe(close.index[251], 126)
    assert initial.target is not None and len(initial.net_returns) == 0
    assert not initial.score_mature
    cold, mature = [monthly.observe(shadow.index[i], 126) for i in (124, 125)]
    assert len(cold.net_returns) == 125 and not cold.score_mature
    assert len(mature.net_returns) == 126 and mature.score_mature
    day = pd.Timestamp("2019-05-15")
    opinion = monthly.observe(day, 126)
    assert opinion.target_known_at == pd.Timestamp("2019-04-30")
    assert opinion.target["BIL"] == 0.245
    assert dense.observe(day, 126).target["BIL"] == 0
    assert dense.observe(day, 126).target_known_at == day
