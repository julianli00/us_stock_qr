from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol
from us_quant.learned_allocation import (
    MODEL_PARAMETERS,
    NUMERIC_FEATURES,
    allocated_signals,
    feature_rows,
    model_pipeline,
    register,
    training_rows,
    validate_policy,
    verify_registration,
    walk_forward_forecasts,
)
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/learned-allocation.json")


@pytest.fixture
def comparison():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


@pytest.fixture
def prices(market_factory):
    return market_factory("2010-01-04", "2015-06-30", ("SPY", "QQQ", "QLD", "IEF", "TIP", "BIL"))


def trim(data, cutoff):
    return replace(
        data,
        open=data.open.loc[:cutoff],
        close=data.close.loc[:cutoff],
        raw_close=data.raw_close.loc[:cutoff],
        volume=data.volume.loc[:cutoff],
        risk_free=data.risk_free.loc[:cutoff],
    )


def test_labels_are_next_open_to_next_open_and_only_available_after_exit(prices, policy):
    features = feature_rows(prices)
    day = pd.Timestamp("2014-07-31")
    row = features.loc[(features["decision_date"] == day) & (features["symbol"] == "QQQ")].iloc[0]
    assert row["entry_session"] == pd.Timestamp("2014-08-01")
    assert row["label_available_session"] == pd.Timestamp("2014-09-02")
    expected = (
        prices.open.at["2014-09-02", "QQQ"] / prices.open.at["2014-08-01", "QQQ"]
        - prices.open.at["2014-09-02", "BIL"] / prices.open.at["2014-08-01", "BIL"]
    )
    assert row["next_month_excess"] == pytest.approx(expected)
    before_exit = training_rows(features, pd.Timestamp("2014-08-29"), policy)
    assert before_exit["decision_date"].max() == pd.Timestamp("2014-06-30")
    assert (before_exit["label_available_session"] <= pd.Timestamp("2014-08-29")).all()
    later = training_rows(features, pd.Timestamp("2014-09-30"), policy)
    assert later["decision_date"].max() == pd.Timestamp("2014-07-31")
    assert all(is_month_end(value) for value in features["decision_date"])


def test_unfinished_month_target_is_never_filled_with_current_close(prices):
    features = feature_rows(prices)
    last = features.loc[features["decision_date"] == pd.Timestamp("2015-06-30")]
    assert last["next_month_excess"].isna().all()
    assert last["label_available_session"].isna().all()
    assert (last["entry_session"] == pd.Timestamp("2015-07-01")).all()


def test_real_models_are_prefix_causal_and_future_values_cannot_change_forecasts(prices, policy):
    original, full_audit = walk_forward_forecasts(feature_rows(prices), policy)
    cutoff = pd.Timestamp("2015-03-31")
    prefix, prefix_audit = walk_forward_forecasts(feature_rows(trim(prices, cutoff)), policy)
    altered_close, altered_open = prices.close.copy(), prices.open.copy()
    future = prices.close.index > cutoff
    multiplier = np.linspace(1.01, 1.30, future.sum())
    altered_close.loc[future, "QQQ"] *= multiplier
    altered_open.loc[future, "QQQ"] *= multiplier
    altered = replace(
        prices, open=altered_open, close=altered_close, raw_close=altered_close.copy()
    )
    changed, changed_audit = walk_forward_forecasts(feature_rows(altered), policy)
    for model in original:
        pd.testing.assert_frame_equal(
            original[model].loc[:cutoff], prefix[model], atol=1e-12, rtol=1e-12
        )
        pd.testing.assert_frame_equal(
            original[model].loc[:cutoff], changed[model].loc[:cutoff], atol=1e-12, rtol=1e-12
        )
    known = [row for row in full_audit if row["decision_date"] <= str(cutoff.date())]
    assert prefix_audit == known
    assert [row for row in changed_audit if row["decision_date"] <= str(cutoff.date())] == known
    assert all(row["latest_label_available"] <= row["decision_date"] for row in full_audit)
    assert all(row["unique_training_months"] >= 36 for row in full_audit)


def test_standard_scaler_and_tree_never_fit_the_prediction_rows(prices, policy, monkeypatch):
    import us_quant.learned_allocation as learned

    seen = []

    class FakeLearner:
        def fit(self, x, y):
            self.count = len(x)
            seen.append((len(x), len(y)))

        def predict(self, x):
            assert len(x) == len(prices.close.columns)
            assert self.count > len(x)
            return np.zeros(len(x))

    monkeypatch.setattr(learned, "model_pipeline", lambda *args: FakeLearner())
    forecasts, audit = learned.walk_forward_forecasts(feature_rows(prices), policy)
    assert len(seen) == 2 * len(audit)
    assert all(x == y for x, y in seen)
    assert all((frame == 0).all().all() for frame in forecasts.values())


def test_feature_rescaling_preserves_returns_and_does_not_invent_price_level_signal(prices):
    first = feature_rows(prices)
    scale = pd.Series([2, 3, 4, 5, 6, 7], index=prices.close.columns)
    data = replace(
        prices,
        open=prices.open * scale,
        close=prices.close * scale,
        raw_close=prices.raw_close * scale,
    )
    second = feature_rows(data)
    np.testing.assert_allclose(
        first.loc[:, NUMERIC_FEATURES], second.loc[:, NUMERIC_FEATURES], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        first["next_month_excess"],
        second["next_month_excess"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_same_fixed_models_do_not_run_hyperparameter_search(policy, comparison):
    validate_policy(policy, comparison)
    assert policy["model_parameters"] == MODEL_PARAMETERS
    assert model_pipeline("ridge", policy).steps[-1][1].alpha == 10
    tree = model_pipeline("hist_gradient_boosting", policy).steps[-1][1]
    assert tree.early_stopping is False and tree.random_state == 20261006
    assert tree.max_depth == 3 and tree.max_iter == 150


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_months", 12),
        ("minimum_training_months", 1),
        ("capital_usd", 1000000),
        ("hyperparameter_search", True),
        ("random_train_test_split", True),
        ("auto_order_submission", True),
        ("prior_disclosed_trials", 0),
        ("cash_reserve", 0),
        ("cost_bps_per_side", 0),
    ],
)
def test_invalid_or_leaking_learning_policies_are_rejected(policy, comparison, field, value):
    with pytest.raises(QuantError):
        validate_policy({**policy, field: value}, comparison)


def test_candidate_risk_limits_and_identifiers_are_validated(policy, comparison):
    changed = deepcopy(policy)
    changed["candidates"][0]["id"] = "../escape"
    with pytest.raises(QuantError):
        validate_policy(changed, comparison)
    changed = deepcopy(policy)
    changed["candidates"][0]["target_volatility"] = 0.5
    with pytest.raises(QuantError):
        validate_policy(changed, comparison)


def test_allocation_uses_only_positive_predicted_edge_and_explicit_bill_fallback(prices, policy):
    date = pd.Timestamp("2015-05-29")
    negative = pd.DataFrame(-0.01, index=[date], columns=prices.close.columns)
    for candidate in policy["candidates"]:
        cash = allocated_signals(prices, negative, candidate, policy).loc[date]
        assert cash["BIL"] == pytest.approx(0.98) and (cash > 0).sum() == 1
        positive = negative.copy()
        positive.loc[date, ["SPY", "QQQ", "QLD"]] = [0.02, 0.03, 0.04]
        allocated = allocated_signals(prices, positive, candidate, policy).loc[date]
        assert (allocated >= 0).all() and allocated.sum() == pytest.approx(0.98)
        assert allocated.drop("BIL").max() <= candidate["max_weight"] + 1e-10
        if candidate["top_k"] == 1:
            assert allocated["QLD"] > 0 and allocated["QQQ"] == 0


def test_registration_precedes_models_and_prevents_parameter_or_data_changes(
    policy, comparison, tmp_path
):
    source = {"snapshot": "fixed-test-data"}
    path = tmp_path / "registration.json"
    result = register(policy, comparison, source, path)
    assert result["prior_disclosed_trials"] == 42 and result["total_trials_after_round"] == 46
    assert len(result["candidate_ids"]) == 4 and not result["order_authority"]
    verify_registration(policy, comparison, source, path)
    with pytest.raises(QuantError, match="overwrite"):
        register(policy, comparison, source, path)
    with pytest.raises(QuantError, match="evidence"):
        verify_registration(policy, comparison, {"snapshot": "changed"}, path)
