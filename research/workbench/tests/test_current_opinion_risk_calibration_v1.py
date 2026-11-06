from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import current_opinion_risk_calibration_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError


def fixture():
    dates = sessions("2020-01-02", "2020-12-31")[:147]
    x = np.arange(147)
    assets = pd.DataFrame(
        {s: 0.001 * np.sin(x / (3 + i)) for i, s in enumerate(rules.ASSETS)},
        index=dates,
    )
    opinions = np.zeros((4, 14))
    for i in range(4):
        opinions[i, i] = 0.98
    net = pd.DataFrame(
        assets.to_numpy() @ opinions.T - 0.00001,
        index=dates,
        columns=rules.EXPERT_IDS,
    )
    rf = pd.Series(0.00002, index=dates)
    weights = {"equal": np.full(4, 0.25), "recorded_rolling": np.array([0.6, 0, 0.4, 0])}
    return dates, assets, net, rf, opinions, weights


def prediction():
    dates, assets, net, rf, opinions, weights = fixture()
    return rules.forecast(
        dates[125], assets.iloc[:126], net.iloc[:126], rf.iloc[:126], opinions, weights
    )


def test_synthetic_forecast_and_separate_future_labels():
    dates, _, net, rf, _, _ = fixture()
    pred = prediction()
    rows = rules.assess(pred, net.iloc[126:147], rf.iloc[126:147])
    assert len(rows) == 4
    assert all(r["label_start"] == str(dates[126].date()) for r in rows)
    assert all(not r["label_is_main_account_or_tradable_portfolio"] for r in rows)
    assert not pred["current_opinion_projection_includes_fees_or_future_policy_changes"]
    assert not pred["new_allocation_or_model_selection"]
    assert pred["maximum_scalar_matrix_variance_difference"] <= 1e-12


def test_fraction_variance_and_covariance_reference():
    data = np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]], dtype=float) / 8
    means = [sum(Fraction(float(x)) for x in data[:, i]) / 4 for i in range(4)]
    expected = np.array(
        [
            [
                float(
                    sum(
                        (Fraction(float(row[i])) - means[i]) * (Fraction(float(row[j])) - means[j])
                        for row in data
                    )
                    / 3
                )
                for j in range(4)
            ]
            for i in range(4)
        ]
    )
    np.testing.assert_array_equal(rules.covariance(data), expected)
    assert rules.variance(data[:, 0]) == expected[0, 0]


def test_future_mutation_cannot_change_forecast():
    dates, assets, net, rf, opinions, weights = fixture()
    expected = prediction()
    assets.iloc[126:] = 0.99
    net.iloc[126:] = -0.99
    assert (
        rules.forecast(
            dates[125], assets.iloc[:126], net.iloc[:126], rf.iloc[:126], opinions, weights
        )
        == expected
    )


@pytest.mark.parametrize("bad", ["short", "future", "missing", "order", "nan", "minus_one"])
def test_bad_history_is_error_not_unscored_success(bad):
    dates, assets, net, rf, opinions, weights = fixture()
    a, n, r = assets.iloc[:126].copy(), net.iloc[:126].copy(), rf.iloc[:126].copy()
    if bad == "short":
        a = a.iloc[1:]
    elif bad == "future":
        a = assets.iloc[1:127]
    elif bad == "missing":
        a.index = a.index[:-1].append(pd.DatetimeIndex([dates[126]]))
    elif bad == "order":
        a = a.iloc[:, ::-1]
    elif bad == "nan":
        a.iloc[3, 0] = np.nan
    else:
        a.iloc[3, 0] = -1
    with pytest.raises(QuantError):
        rules.forecast(dates[125], a, n, r, opinions, weights)


@pytest.mark.parametrize("bad", ["budget", "negative", "shape", "weight", "views"])
def test_bad_opinion_or_allocation_is_error(bad):
    dates, assets, net, rf, opinions, weights = fixture()
    if bad == "budget":
        opinions[0, 0] = 1
    elif bad == "negative":
        opinions[0, 0], opinions[0, 1] = -0.1, 1.08
    elif bad == "shape":
        opinions = opinions[:3]
    elif bad == "weight":
        weights["equal"] = np.ones(4)
    else:
        weights = {"equal": weights["equal"]}
    with pytest.raises(QuantError):
        rules.forecast(
            dates[125], assets.iloc[:126], net.iloc[:126], rf.iloc[:126], opinions, weights
        )


@pytest.mark.parametrize("bad", ["short", "past", "crop", "rf"])
def test_bad_future_label_cannot_enter_scoring(bad):
    _, _, net, rf, _, _ = fixture()
    n, r = net.iloc[126:147].copy(), rf.iloc[126:147].copy()
    if bad == "short":
        n = n.iloc[:-1]
    elif bad == "past":
        n, r = net.iloc[125:146], rf.iloc[125:146]
    elif bad == "crop":
        n.index = n.index + pd.Timedelta(days=1)
        r.index = n.index
    else:
        r = r.iloc[::-1]
    with pytest.raises(QuantError):
        rules.assess(prediction(), n, r)


def test_zero_realized_variance_is_nullable_not_epsilon_filled():
    _, _, net, rf, _, _ = fixture()
    net.iloc[126:147] = 0
    rf.iloc[126:147] = 0
    rows = rules.assess(prediction(), net.iloc[126:147], rf.iloc[126:147])
    assert all(row["forecast_to_realized_variance_ratio"] is None for row in rows)


def test_covariance_rejects_nonfinite_or_wrong_dimension():
    with pytest.raises(QuantError):
        rules.covariance(np.ones((126, 3)))
    with pytest.raises(QuantError):
        rules.covariance(np.full((126, 4), np.nan))


def test_saved_input_adapter_preserves_forecast_and_unscored_origin():
    from us_quant import ssd_current_opinion_risk_calibration_v1 as driver

    dates, assets, net, rf, opinions, weights = fixture()
    day = str(dates[125].date())
    observations = [
        {
            "id": identifier,
            "as_of": day,
            "target_known_at": day,
            "asset_order": list(rules.ASSETS),
            "target": opinions[i].tolist(),
            "return_dates": [str(d.date()) for d in dates[:126]],
            "NET_returns": net[identifier].iloc[:126].tolist(),
            "RF": rf.iloc[:126].tolist(),
            "score_mature": True,
        }
        for i, identifier in enumerate(rules.EXPERT_IDS)
    ]
    before = {"as_of": day, "observations": observations}
    after = {
        "audit": {
            "decision_session": day,
            "equal_expert_weights": weights["equal"].tolist(),
            "rolling_expert_weights": weights["recorded_rolling"].tolist(),
        }
    }
    assert driver.prediction(before, after, assets)["forecast"] == prediction()
    for row in observations:
        row["score_mature"] = False
        row["NET_returns"] = []
    assert driver.prediction(before, after, assets)["status"] == "unscored_immature_source_history"


def test_completed_actual_calibration_cannot_be_replayed(monkeypatch, tmp_path):
    from us_quant import ssd_current_opinion_risk_calibration_v1 as driver

    result = tmp_path / "existing-result.json"
    result.write_text("{}")
    monkeypatch.setattr(driver, "RESULTS", result)
    monkeypatch.setattr(driver.lifecycle, "runtime_guard", lambda: None)
    with pytest.raises(QuantError, match="do not retry"):
        driver.calibrate()


def test_fixed_source_and_calibration_contract():
    from us_quant import ssd_current_opinion_risk_calibration_v1 as driver

    policy = driver.policy()
    assert policy["history_sessions"] == 126 and policy["future_label_sessions"] == 21
    assert policy["sample_covariance_ddof"] == 1
    assert policy["new_financial_trials"] == 0
    assert not policy["new_allocation_or_model_selection"]
