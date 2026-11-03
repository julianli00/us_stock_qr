from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import conditional_hedge_expert_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json


def market(*, rising_stocks=False, flat=False):
    dates = sessions("2015-01-02", "2017-12-29")
    t = np.arange(len(dates))
    data = pd.DataFrame(
        {symbol: 100 * np.exp(0.0003 * t) for symbol in rules.UNIVERSE}, index=dates
    )
    data["SPY"] = 200 * np.exp((0.001 if rising_stocks else -0.001) * t + 0.003 * np.sin(t / 3))
    data["TLT"] = 100 * np.exp(0.0007 * t - 0.003 * np.sin(t / 3))
    data["IEF"] = 100 * np.exp(0.0004 * t - 0.001 * np.sin(t / 3))
    data["GLD"] = 100 * np.exp(0.0005 * t + 0.003 * np.sin(t / 3))
    return data * 0 + 100 if flat else data


def test_positive_own_momentum_does_not_make_a_positively_coupled_asset_a_hedge():
    close = market()
    target, audits = rules.build_targets(close, read_json(rules.POLICY))
    assert audits
    for row in audits:
        assert row["stock_year_context_positive"] is False
        assert row["selected_assets"] == ["TLT", "IEF"]
        gold = next(a for a in row["defensive_assessments"] if a["symbol"] == "GLD")
        assert gold["own_momentum_positive"] and gold["observed_stock_covariance_sign"] == 1
        assert gold["eligible"] is False
    active = target.dropna(how="all")
    assert np.allclose(active["TLT"], 0.49) and np.allclose(active["IEF"], 0.49)


def test_stock_context_keeps_one_complete_six_family_policy():
    target, audits = rules.build_targets(market(rising_stocks=True), read_json(rules.POLICY))
    assert all(row["state"] == "six_factor_core" for row in audits)
    assert np.allclose(target.dropna(how="all").loc[:, list(rules.CORE)], 0.98 / 6)
    assert all(row["selected_assets"] == list(rules.CORE) for row in audits)


def test_flat_observed_covariance_is_not_an_undefined_or_successful_risk_score():
    assert rules.covariance_numerator(np.zeros(63), np.ones(63)) == 0
    target, audits = rules.build_targets(market(flat=True), read_json(rules.POLICY))
    assert all(row["state"] == "explicit_BIL_no_eligible_hedge" for row in audits)
    assert np.allclose(target.dropna(how="all")["BIL"], 0.98)
    assert all(
        not row["source_missingness_or_undefined_NET_score_used_as_safe_success"] for row in audits
    )


def test_fraction_and_independent_integer_oracles_match_every_synthetic_decision():
    for close in (market(), market(rising_stocks=True), market(flat=True)):
        targets, audits = rules.build_targets(close, read_json(rules.POLICY))
        for audit in audits:
            day = pd.Timestamp(audit["decision_session"])
            expected, selected = rules.independent_decision(close, close.index.get_loc(day))
            pd.testing.assert_series_equal(expected, targets.loc[day], check_names=False)
            assert selected == audit["selected_assets"]


def test_genuine_month_ends_warmup_full_calendar_and_prefix_causality():
    close = market()
    target, audits = rules.build_targets(close, read_json(rules.POLICY))
    active = target.dropna(how="all")
    assert active.index[0] >= close.index[252]
    assert all(is_month_end(day) for day in active.index)
    assert target.index.equals(close.index)
    cutoff = active.index[8]
    early, first = rules.build_targets(close.loc[:cutoff], read_json(rules.POLICY))
    pd.testing.assert_frame_equal(early, target.loc[:cutoff])
    assert first == audits[:9]
    changed = close.copy()
    changed.loc[changed.index > cutoff, "SPY"] *= 1.8
    later, after = rules.build_targets(changed, read_json(rules.POLICY))
    pd.testing.assert_frame_equal(later.loc[:cutoff], target.loc[:cutoff])
    assert [row for row in after if row["decision_session"] <= str(cutoff.date())] == first


def test_source_inputs_are_retained_before_a_failed_new_decision(monkeypatch):
    records = []

    def failed(*args):
        raise QuantError("Synthetic new expert decision failure")

    monkeypatch.setattr(rules, "decision", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(market(), read_json(rules.POLICY), retain=records.append)
    assert len(records) == 1 and records[0]["record"] == "decision_input"
    assert all(
        len(values) == 64 for values in records[0]["used_source_prices_binary64_hex"].values()
    )


@pytest.mark.parametrize("kind", ["missing", "negative", "nan", "wrong_order", "gap"])
def test_invalid_or_incomplete_admitted_source_never_becomes_a_safe_default(kind):
    close = market()
    if kind == "missing":
        close = close.drop(columns=["GLD"])
    elif kind == "negative":
        close.iloc[300, 1] = -1
    elif kind == "nan":
        close.iloc[300, 1] = np.nan
    elif kind == "wrong_order":
        close = close.loc[:, close.columns[::-1]]
    else:
        close = close.drop(index=close.index[300])
    with pytest.raises(QuantError):
        rules.build_targets(close, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("stock_context_sessions", 200),
        ("hedge_observation_sessions", 126),
        ("invested_target_budget", 0.90),
        ("actual_weight_trade_band", 0.01),
        ("decision_information_clock", "at_16_00_ET"),
        ("candidate_count_in_this_study", True),
    ],
)
def test_no_window_budget_band_clock_or_quota_reinterpretation(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError, match="Freeze one complete"):
        rules.validate(policy)
