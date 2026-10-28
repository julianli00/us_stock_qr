from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import benchmark_tail_budget_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def problem():
    x = np.tile(np.array([-0.02, 0.01, 0.005, -0.01]), 63)
    scenarios = np.column_stack([x, 2 * x, -x, np.zeros((252, 3))])
    return scenarios, np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0]), np.column_stack([x, 2 * x])


def test_benchmark_anchor_and_growth_first_solution_have_closed_form():
    matrix, scores, benchmarks = problem()
    weights, record = rules.allocate(matrix, scores, benchmarks)
    assert record["risk_budget"] == pytest.approx(0.02)
    assert record["observed_tail_losses"] == pytest.approx({"SPY": 0.02, "QQQ": 0.04, "BIL": 0})
    np.testing.assert_allclose(weights, [0.4, 0.4, 0.2, 0, 0, 0], atol=1e-6)
    assert record["achieved_observed_opportunity"] == pytest.approx(1.6, abs=1e-6)
    assert record["empirical_joint_tail_loss"] == pytest.approx(0.02, abs=1e-8)
    rules.verify_plan(weights, record, matrix, scores, benchmarks)


def test_lower_benchmark_tail_budget_is_not_a_fitted_multiplier():
    matrix, scores, benchmarks = problem()
    benchmarks[:, 0] *= 0.5
    weights, record = rules.allocate(matrix, scores, benchmarks)
    assert record["risk_budget"] == pytest.approx(0.01)
    np.testing.assert_allclose(weights, [0.3, 0.3, 0.4, 0, 0, 0], atol=1e-6)
    rules.verify_plan(weights, record, matrix, scores, benchmarks)


def test_standby_uses_actual_bil_risk_floor_not_synthetic_zero():
    matrix, scores, benchmarks = problem()
    matrix[:, -1] = -0.03
    scores[:] = 0
    weights, record = rules.allocate(matrix, scores, benchmarks)
    assert record["risk_budget"] == pytest.approx(0.03)
    assert record["reason"] == "no_positive_asset_opportunity"
    assert record["solver_called"] is False
    np.testing.assert_array_equal(weights, [0, 0, 0, 0, 0, 1])
    rules.verify_plan(weights, record, matrix, scores, benchmarks)


def test_certified_zero_risk_feasible_opportunity_has_explicit_standby():
    matrix = np.zeros((252, 6))
    matrix[:, :2] = [-0.02, -0.04]
    scores = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    benchmarks = np.full((252, 2), 0.001)
    weights, record = rules.allocate(matrix, scores, benchmarks)
    assert record["risk_budget"] == 0
    assert record["reason"] == "no_positive_risk_feasible_opportunity"
    assert record["solver_called"] is True
    np.testing.assert_array_equal(weights, [0, 0, 0, 0, 0, 1])
    rules.verify_plan(weights, record, matrix, scores, benchmarks)


def test_solver_failure_is_not_a_standby_condition(monkeypatch):
    matrix, scores, benchmarks = problem()
    monkeypatch.setattr(
        rules,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, status=2, message="synthetic"),
    )
    with pytest.raises(QuantError, match="solver failed"):
        rules.allocate(matrix, scores, benchmarks)


@pytest.mark.parametrize("damage", ["shape", "nan", "scores", "bil"])
def test_missing_or_invalid_inputs_do_not_create_cash(damage):
    matrix, scores, benchmarks = problem()
    if damage == "shape":
        benchmarks = benchmarks[:-1]
    elif damage == "nan":
        benchmarks[0, 0] = np.nan
    elif damage == "scores":
        scores[0] = np.inf
    else:
        scores[-1] = 0.1
    with pytest.raises(QuantError):
        rules.allocate(matrix, scores, benchmarks)


def test_retained_budget_primal_and_dual_evidence_cannot_be_changed():
    matrix, scores, benchmarks = problem()
    weights, record = rules.allocate(matrix, scores, benchmarks)
    for damage in ("budget", "primal", "dual", "reason"):
        changed = deepcopy(record)
        if damage == "budget":
            changed["risk_budget"] += 0.01
        elif damage == "primal":
            changed["primary_solution"]["primal"][0] += 0.1
        elif damage == "dual":
            changed["secondary_solution"]["inequality_dual"][0] += 0.1
        else:
            changed["reason"] = "no_positive_asset_opportunity"
        with pytest.raises(QuantError):
            rules.verify_plan(weights, changed, matrix, scores, benchmarks)


def prices():
    dates = sessions("2019-01-02", "2020-03-10")
    positions = np.arange(len(dates))
    return pd.DataFrame(
        {
            name: 100
            * np.exp((0.0002 + i * 0.00002) * positions + 0.003 * np.sin(positions / (5 + i)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )


def test_independent_inputs_and_future_invariance():
    close = prices()
    day = close.index[270]
    original = rules.observed_problem(close, day)
    independent = rules.independent_problem(close, day)
    for a, b in zip(original, independent, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    changed = close.copy()
    changed.loc[changed.index > day] *= 10
    for a, b in zip(original, rules.observed_problem(changed, day), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_month_ends_only_and_initial_warmup_do_not_depend_on_final_row(monkeypatch):
    close = prices()
    data = MarketData(
        close.copy(),
        close.copy(),
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=close.index, name="risk_free"),
    )
    calls = []

    def fixed_plan(matrix, scores, benchmarks):
        calls.append((matrix.shape, benchmarks.shape))
        return np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.2]), {"reason": "synthetic"}

    monkeypatch.setattr(rules, "allocate", fixed_plan)
    target_map, audits = rules.build_targets(data, read_json(rules.POLICY))
    expected = [day for i, day in enumerate(close.index) if i >= 252 and is_month_end(day)]
    assert [row["decision_session"] for row in audits] == [str(day.date()) for day in expected]
    assert len(calls) == len(expected) and all(pair == ((252, 6), (252, 2)) for pair in calls)
    target = target_map[rules.CANDIDATES[0]["id"]]
    assert target.loc[target.index < expected[0]].isna().all().all()
    assert target.loc[expected[0] :].notna().all().all()
    assert target.iloc[-1].equals(target.iloc[-2])
    np.testing.assert_allclose(target.loc[expected[0], list(rules.CORE)], 0.98 * 0.3 / 6)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 126),
        ("tail_probability", 0.1),
        ("core_at_least_SOXX", False),
        ("risk_budget_multiplier", 0.5),
        ("cash_reserve", 0.0),
        ("weight_band", 0.01),
        ("candidate_count_in_this_study", True),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_one_fixed_policy_cannot_become_a_parameter_grid(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
