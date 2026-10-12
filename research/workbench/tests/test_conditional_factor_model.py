from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.conditional_factor_model import (
    allocation,
    build_from_inputs,
    feature_rows,
    forecasts,
    validate_policy,
)
from us_quant.config import QuantError
from us_quant.learned_allocation import training_rows
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/conditional-factor-model.json")


@pytest.fixture
def inputs(market_factory):
    data = market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FACTORS))
    t = np.arange(len(data.close))
    macro = pd.DataFrame(
        {"T10Y3M": 1 + np.sin(t / 80), "DFII10": 0.2 + np.cos(t / 65)}, index=data.close.index
    )
    terms = pd.DataFrame(
        {"VIX": 15 + np.sin(t / 10), "VIX3M": 20 + np.cos(t / 15)}, index=data.close.index
    )
    return data, macro, terms


def test_labels_are_matured_before_training_and_not_the_current_month(inputs, policy):
    data, macro, terms = inputs
    features = feature_rows(data, macro, terms)
    train = training_rows(features, pd.Timestamp("2016-09-30"), policy)
    assert train["decision_date"].nunique() == 9
    assert train["label_available_session"].max() == pd.Timestamp("2016-09-01")
    assert (train["entry_session"] > train["decision_date"]).all()
    assert (train["label_available_session"] > train["entry_session"]).all()
    current = features.loc[features["decision_date"] == pd.Timestamp("2016-08-31")]
    assert current["label_available_session"].eq(pd.Timestamp("2016-10-03")).all()
    assert not set(current.index) & set(train.index)


def test_training_only_scaling_and_future_label_changes_preserve_past_predictions(inputs, policy):
    data, macro, terms = inputs
    cutoff = pd.Timestamp("2017-06-30")
    before_features = feature_rows(data, macro, terms)
    before, before_audit = forecasts(before_features, policy, True)
    closing, opening = data.close.copy(), data.open.copy()
    later = closing.index > cutoff
    closing.loc[later, "GLD"] *= np.linspace(1, 1.4, later.sum())
    opening.loc[later, "GLD"] *= np.linspace(1, 1.4, later.sum())
    changed = replace(data, close=closing, open=opening, raw_close=closing.copy())
    modified_macro = macro.copy()
    modified_macro.loc[later] += 100
    modified_terms = terms.copy()
    modified_terms.loc[later] *= 2
    after, after_audit = forecasts(
        feature_rows(changed, modified_macro, modified_terms), policy, True
    )
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    assert [row for row in before_audit if row["decision_date"] <= str(cutoff.date())] == [
        row for row in after_audit if row["decision_date"] <= str(cutoff.date())
    ]


def test_price_only_control_does_not_use_macro_or_option_predictors(inputs, policy):
    data, macro, terms = inputs
    before, _ = forecasts(feature_rows(data, macro, terms), policy, False)
    after, _ = forecasts(feature_rows(data, macro * 5, terms * 3), policy, False)
    pd.testing.assert_frame_equal(before, after)


def test_forecast_allocation_is_cash_funded_and_retains_four_factor_shares(inputs, policy):
    data, _, _ = inputs
    day = data.close.index[-1]
    forecasts_positive = pd.Series(
        {"MTUM": 0.04, "VLUE": 0.03, "QUAL": 0.02, "USMV": 0.01, "GLD": 0.03}
    )
    weights = allocation(data, day, forecasts_positive, policy)
    eq = weights.loc[list(FACTORS)]
    np.testing.assert_allclose(eq / eq.sum(), [0.35, 0.30, 0.20, 0.15])
    assert weights.sum() == pytest.approx(0.98)
    assert weights["GLD"] > 0
    cash = allocation(data, day, forecasts_positive * 0, policy)
    assert cash["BIL"] == pytest.approx(0.98)
    assert cash.loc[list(FACTORS)].sum() == 0


def test_full_target_build_has_sufficient_pre_window_training(inputs, policy):
    data, macro, terms = inputs
    targets, audit = build_from_inputs(data, macro, terms, policy)
    for name, frame in targets.items():
        assert audit[name][0]["decision_date"] == "2016-09-30"
        assert all(row["latest_label_available"] <= row["decision_date"] for row in audit[name])
        assert all(row["unique_training_months"] >= 9 for row in audit[name])
        active = frame.dropna(how="all")
        assert np.allclose(active.sum(axis=1), 0.98)
        assert (active >= 0).all().all()
        assert active[["SPY", "IEF"]].eq(0).all().all()


def test_no_hidden_hyperparameter_or_hurdle_tuning(policy):
    policy["model_parameters"]["ridge"]["alpha"] = 0.01
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_zero_cash_residue_is_not_emitted_as_a_short_position(inputs, policy):
    data, _, _ = inputs
    prediction = pd.Series({"MTUM": 0.04, "VLUE": 0.03, "QUAL": 0.02, "USMV": 0.01, "GLD": 0.03})
    for day in data.close.index[63:]:
        weights = allocation(data, day, prediction, policy)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(0.98, abs=1e-12)


def test_revision_does_not_change_economic_parameters(policy):
    before = read_json(
        Path(__file__).parents[1] / "evidence/conditional_factor_model_20261011_registration.json"
    )["study"]
    after = dict(policy)
    after.pop("implementation_revision")
    after.pop("technical_correction")
    after["candidates"] = [
        {**row, "id": row["id"].removesuffix("_v2")} for row in after["candidates"]
    ]
    assert after == before
