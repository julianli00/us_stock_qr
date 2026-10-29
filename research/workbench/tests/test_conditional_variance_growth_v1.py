from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import conditional_variance_growth_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.mark.parametrize("ratio,stock", [(0.5, 0.98), (1.0, 0.98), (2.0, 0.49), (10.0, 0.098)])
def test_relative_variance_budget_has_no_hidden_hmm_floor(ratio, stock):
    row = rules.allocation(0.0001, ratio * 0.0001)
    assert row.sum() == pytest.approx(0.98)
    assert row.loc[list(rules.STOCKS)].sum() == pytest.approx(stock)
    assert row["SOXX"] == pytest.approx(stock / 2)
    assert row.loc[list(rules.CORE)].tolist() == pytest.approx([stock / 8] * 4)
    assert row["BIL"] == pytest.approx(0.98 - stock)
    assert (row >= 0).all() and row["GLD"] == 0


@pytest.mark.parametrize("pooled,forecast", [(0.0, 1.0), (1.0, 0.0), (1.0, np.nan), (np.inf, 1.0)])
def test_invalid_variance_never_becomes_a_safe_allocation(pooled, forecast):
    with pytest.raises(QuantError):
        rules.allocation(pooled, forecast)


def fixture():
    index = sessions("2015-08-10", "2016-12-30")
    rng = np.random.default_rng(952732)
    prices = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    common = rng.normal(0, 0.009, len(index))
    for i, name in enumerate(rules.STOCKS):
        values = common + rng.normal(0, 0.003 + i * 0.001, len(index))
        prices[name] = 100 * np.exp(np.cumsum(values))
    return MarketData(
        prices.copy(), prices.copy(), prices.copy(), prices * 1000, pd.Series(0.0, index=index)
    )


def test_proxy_is_weighted_gross_then_log_not_a_weighted_sum_of_logs():
    data = fixture()
    day = pd.Timestamp("2016-08-31")
    values, information = rules.observed_returns(data, day)
    position = data.close.index.get_loc(day)
    prices = data.close.loc[:, list(rules.STOCKS)].iloc[position - 252 : position + 1]
    gross = prices.to_numpy()[1:] / prices.to_numpy()[:-1]
    expected = np.log(gross @ np.array([0.5, 0.125, 0.125, 0.125, 0.125]))
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-15)
    wrong = np.log(gross) @ np.array([0.5, 0.125, 0.125, 0.125, 0.125])
    assert float(np.max(abs(values - wrong))) > 1e-6
    assert information["observed_returns"] == 252


@pytest.mark.parametrize("candidate", rules.CANDIDATES)
def test_saved_fit_reconstruction_and_future_independence_without_refitting(candidate, monkeypatch):
    data = fixture()
    day = pd.Timestamp("2016-08-31")
    target, audit = rules.target_for_day(data, day, candidate)
    independent, oracle = rules.independent_row(data, day, audit)
    np.testing.assert_allclose(target, independent, rtol=0, atol=1e-10)
    assert audit["model"]["mean_negative_log_likelihood_daily"] == pytest.approx(
        oracle["mean_negative_log_likelihood"], abs=1e-8
    )
    before, information = rules.observed_returns(data, day)
    data.close.loc[data.close.index > day, list(rules.STOCKS)] *= 100
    after, later_information = rules.observed_returns(data, day)
    np.testing.assert_array_equal(before, after)
    assert information == later_information
    monkeypatch.setattr(rules.model, "fit", lambda *args: pytest.fail("No repeated completed fit"))
    restored, restored_audit = rules.target_for_day(data, day, candidate, audit)
    pd.testing.assert_series_equal(target, restored)
    assert restored_audit == audit


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_return_sessions", 126),
        ("forecast_sessions", 5),
        ("growth_share_of_equity", 0.7),
        ("optimizer_initializations_per_fit", 2),
        ("optimizer_max_iterations", 2000),
        ("normalized_intercept_bounds", [1e-10, 10.0]),
    ],
)
def test_rule_optimizer_and_numerical_domain_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def one_opportunity():
    data = fixture()
    calendar = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    calendar.loc[pd.Timestamp("2016-08-31")] = 0.0
    return data, calendar


def test_interrupted_preparation_reuses_each_saved_fit_without_optimizing(monkeypatch):
    data, calendar = one_opportunity()
    fitted = []

    def fit(values, asymmetric):
        fitted.append(asymmetric)
        return {
            "asymmetric": asymmetric,
            "pooled_daily_squared_return": 0.0001,
            "mean_forecast_variance_daily": 0.0002,
        }

    monkeypatch.setattr(rules.model, "fit", fit)
    saved = {}
    starts = []

    def retain(audit):
        saved[audit["candidate_id"], audit["information"]["decision_session"]] = audit

    def before(identifier, day, reuse):
        starts.append((identifier, str(day.date()), reuse))

    targets, audits = rules.build_targets(
        data, calendar, read_json(rules.POLICY), retain_fit=retain, before_fit=before
    )
    assert fitted == [False, True]
    monkeypatch.setattr(rules.model, "fit", lambda *args: pytest.fail("No repeated optimization"))
    restored, restored_audits = rules.build_targets(
        data,
        calendar,
        read_json(rules.POLICY),
        retain_fit=retain,
        before_fit=before,
        load_fit=lambda identifier, day: saved[identifier, str(day.date())],
    )
    assert [row[2] for row in starts] == [False, False, True, True]
    assert restored_audits == audits
    for identifier in targets:
        pd.testing.assert_frame_equal(targets[identifier], restored[identifier])


def test_failed_fit_is_retained_and_not_replaced_by_a_safe_target(monkeypatch):
    data, calendar = one_opportunity()
    retained = []

    def fail(*args):
        raise rules.model.FitError("synthetic nonconvergence", {"optimizer_success": False})

    monkeypatch.setattr(rules.model, "fit", fail)
    with pytest.raises(QuantError, match="nonconvergence"):
        rules.build_targets(data, calendar, read_json(rules.POLICY), retain_fit=retained.append)
    assert len(retained) == 1
    assert retained[0]["status"] == "failed"
    assert retained[0]["information"]["decision_session"] == "2016-08-31"
    assert retained[0]["optimizer_receipt"]["optimizer_success"] is False
