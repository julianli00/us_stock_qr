from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import simplex_wealth_average_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_uniform_prior_and_exact_first_update():
    state = rules.SimplexMoments()
    np.testing.assert_allclose(state.weights(), [1 / 3] * 3, rtol=0, atol=1e-15)
    state.update(np.array([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(state.weights(), [0.375, 0.3125, 0.3125], rtol=0, atol=1e-15)
    assert state.degree == 1


def test_common_growth_scaling_and_order_do_not_choose_a_hindsight_winner():
    observations = np.array([[1.1, 0.9, 1.02], [0.94, 1.08, 1.0], [1.01, 1.04, 0.98]])
    values = []
    for rows in (
        observations,
        observations[::-1],
        observations * np.array([2.0, 0.5, 3.0])[:, None],
    ):
        state = rules.SimplexMoments()
        for row in rows:
            state.update(row)
        values.append(state.weights())
    np.testing.assert_allclose(values, np.tile(values[0], (3, 1)), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        values[0], rules.independent_weights(observations), rtol=0, atol=1e-14
    )
    assert (values[0] > 0).all()


def test_long_synthetic_sequence_matches_independent_quadrature():
    t = np.arange(140)
    observations = np.column_stack(
        (1.01 + 0.05 * np.sin(t), 1.02 + 0.08 * np.cos(t / 3), 1.005 + 0.03 * np.sin(t / 7))
    )
    state = rules.SimplexMoments()
    for row in observations:
        state.update(row)
    np.testing.assert_allclose(
        state.weights(), rules.independent_weights(observations), rtol=0, atol=1e-12
    )
    assert state.mass.shape == (141, 141)


@pytest.mark.parametrize("value", [[1, 1], [1, np.nan, 1], [1, 0, 1], [1, -0.5, 1]])
def test_missing_or_nonpositive_interval_never_defaults(value):
    with pytest.raises(QuantError):
        rules.SimplexMoments().update(np.array(value))


def test_core_relative_is_equal_starting_monthly_capital():
    dates = pd.to_datetime(["2019-01-31", "2019-02-28"])
    close = pd.DataFrame(100.0, index=dates, columns=rules.UNIVERSE)
    close.loc[dates[1], rules.CORE[0]] = 200
    close.loc[dates[1], "GLD"] = 110
    actual = rules.month_relatives(close).iloc[0]
    np.testing.assert_allclose(actual, [1.25, 1.0, 1.1], rtol=0, atol=1e-14)


def test_nullable_parameter_must_be_explicitly_frozen():
    policy = read_json(rules.POLICY)
    del policy["forgetting_or_temperature_parameter"]
    with pytest.raises(QuantError):
        rules.validate(policy)


def fixture():
    dates = sessions("2019-01-02", "2020-03-10")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0002 * (i + 1) * t + 0.01 * np.sin(t / (4 + i)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )
    desired = pd.DataFrame(np.nan, index=dates, columns=rules.UNIVERSE)
    monthly = [day for i, day in enumerate(dates) if i >= 63 and is_month_end(day)]
    for i, day in enumerate(monthly):
        budget = 0.49 if i % 2 else 0.98
        desired.loc[day] = 0
        desired.loc[day, list(rules.CORE)] = budget / 16
        desired.loc[day, ["SOXX", "GLD", "BIL"]] = [budget / 4, budget / 2, 0.98 - budget]
    return MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    ), desired


def test_saved_source_budget_all_families_independent_proof_and_prefix():
    data, desired = fixture()
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(data, desired, policy)
    target = targets[rules.CANDIDATES[0]["id"]]
    assert target.dropna().index.equals(desired.dropna().index)
    for day in target.dropna().index:
        expected = rules.independent_row(data.close, desired.loc[day], day)
        np.testing.assert_allclose(target.loc[day], expected, rtol=0, atol=1e-12)
        assert target.loc[day, "BIL"] == desired.loc[day, "BIL"]
        assert (target.loc[day, list(rules.CORE)] > 0).all()
        assert target.loc[day].sum() == pytest.approx(0.98)
    assert audits[0]["completed_learning_periods"] > 0
    assert audits[-1]["last_observed_score_close"] < str(data.close.index[-1].date())
    end = pd.Timestamp("2019-09-10")
    short = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(short, desired.loc[:end], policy)
    pd.testing.assert_frame_equal(early[rules.CANDIDATES[0]["id"]], target.loc[:end])
    assert target.loc[short.close.index[-1]].isna().all()


def test_first_partial_month_cannot_invent_a_learning_interval():
    data, desired = fixture()
    first_month = next(day for day in data.close.index if is_month_end(day))
    desired.loc[first_month] = desired.dropna().iloc[0]
    with pytest.raises(QuantError, match="matured"):
        rules.build_targets(data, desired, read_json(rules.POLICY))


@pytest.mark.parametrize("kind", ["partial", "non_month_end", "guard_budget"])
def test_source_contract_corruption_fails(kind):
    data, desired = fixture()
    if kind == "partial":
        desired.iloc[0, 0] = 0
    elif kind == "non_month_end":
        desired.iloc[-1] = desired.dropna().iloc[-1]
    else:
        day = desired.dropna().index[0]
        desired.loc[day, "BIL"] += 0.1
        desired.loc[day, "GLD"] -= 0.1
    with pytest.raises(QuantError):
        rules.build_targets(data, desired, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("dirichlet_prior", [2, 1, 1]),
        ("learning_window_reset", True),
        ("forgetting_or_temperature_parameter", 0.9),
        ("actual_weight_trade_band", 0.01),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_frozen_learning_architecture_and_execution_contract(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
