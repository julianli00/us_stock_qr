from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import joint_tail_construction_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def problem():
    x = np.tile(np.array([-0.02, 0.01, 0.005, -0.01]), 63)
    scenarios = np.column_stack([x, 2 * x, -x, np.zeros((252, 3))])
    scores = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    return scenarios, scores


def test_fractional_expected_shortfall_and_negative_tail_are_not_clipped():
    assert rules.tail_loss(np.array([4.0, 3.0, 1.0, -2.0]), 0.375) == pytest.approx(11 / 3)
    assert rules.tail_loss(np.array([-4.0, -3.0, -1.0, -2.0]), 0.25) == -1
    assert rules.tail_loss(np.array([4.0, 3.0, 1.0, -2.0]), 1.0) == 1.5
    with pytest.raises(QuantError):
        rules.tail_loss(np.array([np.nan]))


def test_analytical_feasible_bound_and_unique_equity_corner():
    matrix, score = problem()
    assert rules.maximum_opportunity(score) == 2
    weights, record = rules.allocate(matrix, score, 2)
    np.testing.assert_allclose(weights, [0.5, 0.5, 0, 0, 0, 0], atol=1e-7)
    assert record["empirical_joint_tail_loss"] == pytest.approx(0.03)
    assert record["reason"] == "joint_tail_optimum"
    rules.verify_plan(weights, record, matrix, score, 2)


def test_joint_scenario_hedge_and_secondary_opportunity_have_known_solution():
    matrix, score = problem()
    weights, record = rules.allocate(matrix, score, 0.5)
    np.testing.assert_allclose(weights, [0.2, 0.2, 0.6, 0, 0, 0], atol=1e-6)
    assert abs(record["empirical_joint_tail_loss"]) <= rules.FEAS_TOL
    assert record["achieved_observed_opportunity"] == pytest.approx(0.8, abs=1e-6)
    assert rules.tail_loss(-matrix[:, 0]) > 0
    assert rules.tail_loss(-matrix[:, 2]) > 0
    rules.verify_plan(weights, record, matrix, score, 0.5)


@pytest.mark.parametrize(
    "hurdle,reason",
    [
        (-0.1, "nonpositive_benchmark_opportunity"),
        (0.0, "nonpositive_benchmark_opportunity"),
        (2.1, "observed_benchmark_opportunity_unavailable"),
    ],
)
def test_explicit_economic_standby_does_not_call_a_solver(hurdle, reason, monkeypatch):
    matrix, score = problem()
    monkeypatch.setattr(rules, "linprog", lambda *args, **kwargs: pytest.fail("No solver needed"))
    weights, record = rules.allocate(matrix, score, hurdle)
    assert record["reason"] == reason and record["solver_called"] is False
    np.testing.assert_array_equal(weights, [0, 0, 0, 0, 0, 1])
    rules.verify_plan(weights, record, matrix, score, hurdle)


def test_solver_failure_never_becomes_cash(monkeypatch):
    matrix, score = problem()
    monkeypatch.setattr(
        rules,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, status=2, message="synthetic failure"
        ),
    )
    with pytest.raises(QuantError, match="solver failed"):
        rules.allocate(matrix, score, 0.5)


@pytest.mark.parametrize("damage", ["nan", "short", "score", "cash_score"])
def test_invalid_inputs_cannot_become_economic_standby(damage):
    matrix, score = problem()
    if damage == "nan":
        matrix[0, 0] = np.nan
    elif damage == "short":
        matrix = matrix[:-1]
    elif damage == "score":
        score[0] = np.inf
    else:
        score[-1] = 0.01
    with pytest.raises(QuantError):
        rules.allocate(matrix, score, -1)


def test_corrupt_primal_dual_or_economic_reason_is_rejected():
    matrix, score = problem()
    weights, record = rules.allocate(matrix, score, 0.5)
    for damage in ("primal", "dual", "reason", "score"):
        changed = deepcopy(record)
        if damage == "primal":
            changed["primary_solution"]["primal"][0] += 0.1
        elif damage == "dual":
            changed["secondary_solution"]["inequality_dual"][0] += 0.1
        elif damage == "reason":
            changed["reason"] = "nonpositive_benchmark_opportunity"
        else:
            changed["benchmark_hurdle"] += 0.1
        with pytest.raises(QuantError):
            rules.verify_plan(weights, changed, matrix, score, 0.5)


def prices():
    dates = sessions("2019-01-02", "2020-03-10")
    step = np.arange(len(dates))
    value = pd.DataFrame(index=dates, columns=rules.UNIVERSE, dtype=float)
    for i, name in enumerate(value):
        value[name] = 100 * np.exp((0.0002 + i * 0.00003) * step + 0.003 * np.sin(step / (5 + i)))
    return value


def test_observed_and_independent_price_problems_use_only_past_data():
    close = prices()
    day = close.index[270]
    first = rules.observed_problem(close, day)
    independent = rules.independent_problem(close, day)
    for a, b in zip(first, independent, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    changed = close.copy()
    changed.loc[changed.index > day] *= 7
    for a, b in zip(first, rules.observed_problem(changed, day), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    with pytest.raises(QuantError, match="253"):
        rules.observed_problem(close, close.index[251])
    changed = close.copy()
    changed.loc[day, "IJR"] = np.nan
    with pytest.raises(QuantError):
        rules.observed_problem(changed, day)


def test_month_end_only_warmup_and_actual_fund_scatter_without_account_simulation(monkeypatch):
    close = prices()
    data = MarketData(
        close.copy(),
        close.copy(),
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=close.index, name="risk_free"),
    )
    calls = []

    def fixed_plan(matrix, scores, hurdle):
        calls.append(len(matrix))
        return np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.2]), {"reason": "synthetic"}

    monkeypatch.setattr(rules, "allocate", fixed_plan)
    target_map, audits = rules.build_targets(data, read_json(rules.POLICY))
    target = target_map[rules.CANDIDATES[0]["id"]]
    expected = [day for i, day in enumerate(close.index) if i >= 252 and is_month_end(day)]
    assert len(calls) == len(expected)
    assert [row["decision_session"] for row in audits] == [str(day.date()) for day in expected]
    assert all(count == 252 for count in calls)
    assert target.loc[target.index < expected[0]].isna().all().all()
    assert target.loc[expected[0] :].notna().all().all()
    first = target.loc[expected[0]]
    np.testing.assert_allclose(first[list(rules.CORE)], 0.98 * 0.3 / 6)
    assert first["SOXX"] == pytest.approx(0.98 * 0.2)
    assert first.sum() == pytest.approx(0.98)
    assert not is_month_end(close.index[-1])
    assert target.iloc[-1].equals(target.iloc[-2])


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 126),
        ("tail_probability", 0.1),
        ("core_at_least_SOXX", False),
        ("cash_reserve", 0.0),
        ("weight_band", 0.01),
        ("candidate_count_in_this_study", True),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_frozen_rule_or_scope_change_is_not_silent(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
