from __future__ import annotations

import ast
import copy
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from us_quant import ordered_path_growth_v1 as previous
from us_quant import ordered_path_growth_v2 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def path_from_returns(values):
    stock = np.r_[1.0, np.cumprod(values)]
    return np.column_stack([stock, np.ones(len(stock)), np.ones(len(stock))])


def test_peak_drawdown_is_not_initial_capital_floor():
    paths = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.5, 1.0, 1.0]])
    weights = np.array([0.98, 0.0, 0.0])
    _, a, b, _, _, _ = rules.problem(paths)
    wealth = 0.02 + paths @ weights
    assert wealth.min() >= 0.85
    assert rules.drawdown(paths, weights)[0] > 0.24
    assert (a @ weights > b).any()


def test_same_daily_multiset_and_endpoint_have_different_ordered_risk_budgets():
    alternate = path_from_returns([1.3, 0.8, 1.3, 0.8])
    clustered = path_from_returns([1.3, 1.3, 0.8, 0.8])
    assert alternate[-1, 0] == pytest.approx(clustered[-1, 0])
    first, a = rules.allocate(alternate)
    second, b = rules.allocate(clustered)
    assert first[0] > second[0] + 0.3
    for weights, record, paths in ((first, a, alternate), (second, b, clustered)):
        rules.verify_plan(weights, record, paths)
        assert rules.drawdown(paths, weights)[0] <= 0.15 + rules.FEAS_TOL
        assert weights.sum() == pytest.approx(0.98)
        assert weights[0] >= 0.147 - rules.FEAS_TOL


def test_flat_tie_prefers_bills_with_genuine_stock_floor():
    paths = np.ones((6, 3))
    weights, record = rules.allocate(paths)
    np.testing.assert_allclose(weights, [0.147, 0, 0.833], atol=1e-8)
    assert all(rules.scatter(weights)[name] > 0 for name in rules.CORE)
    rules.verify_plan(weights, record, paths)


def test_infeasible_family_floor_stops_without_cash_fallback():
    paths = np.array([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(QuantError, match="without fallback"):
        rules.allocate(paths)


def test_sparse_dual_corruption_is_detected():
    paths = path_from_returns([1.3, 1.3, 0.8, 0.8])
    weights, record = rules.allocate(paths)
    corrupted = copy.deepcopy(record)
    sparse = corrupted["primary_solution"]["nonzero_inequality_dual"]
    assert sparse
    sparse[0]["value"] += 0.1
    with pytest.raises(QuantError):
        rules.verify_plan(weights, corrupted, paths)


@pytest.mark.parametrize("kind", ["nan", "zero", "wrong_initial", "wrong_width"])
def test_bad_path_data_never_produces_a_portfolio(kind):
    paths = np.ones((5, 3))
    if kind == "nan":
        paths[2, 0] = np.nan
    elif kind == "zero":
        paths[2, 0] = 0
    elif kind == "wrong_initial":
        paths[0, 0] = 1.1
    else:
        paths = paths[:, :2]
    with pytest.raises(QuantError):
        rules.allocate(paths)


def test_constant_units_and_prefix_invariance_preserve_full_warmup():
    dates = sessions("2015-08-10", "2016-10-05")
    t = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(0.0002 * (i + 1) * t + 0.002 * np.sin(t / (i + 4)))
            for i, name in enumerate(rules.UNIVERSE)
        },
        index=dates,
    )
    data = MarketData(
        close * 0.9999,
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(data, policy)
    target = full[rules.CANDIDATES[0]["id"]]
    assert target.iloc[:252].isna().all().all()
    assert target.dropna().index.strftime("%Y-%m-%d").tolist() == ["2016-08-31", "2016-09-30"]
    assert all(row["ordered_peak_trough_constraints"] == 31878 for row in audits)
    end = pd.Timestamp("2016-08-31")
    sliced = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    prefix, _ = rules.build_targets(sliced, policy)
    pd.testing.assert_frame_equal(prefix[rules.CANDIDATES[0]["id"]], target.loc[:end])
    np.testing.assert_allclose(
        rules.observed_paths(close, end), rules.independent_paths(close, end), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_return_sessions", 126),
        ("training_maximum_drawdown", 0.2),
        ("stock_target_minimum", 0.01),
        ("cash_reserve", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("solver_tolerance", 1e-9),
        ("certificate_tolerance", 1e-6),
        ("weight_roundoff_tolerance", 1e-9),
        ("decision_feasibility_tolerance", 1e-7),
        ("primary_optimum_slack", 1e-8),
        ("new_independent_economic_hypotheses", 1),
    ],
)
def test_fixed_research_contract_is_not_retuned(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_policy_diff_contains_only_precision_and_version_identity():
    old, new = read_json(previous.POLICY), read_json(rules.POLICY)
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    assert changed == {
        "study_id", "prior_journal_sequence", "prior_journal_head_sha256",
        "solver_tolerance", "candidates", "revision_of_candidate", "revision_kind",
        "preserved_preparation_failure", "new_independent_economic_hypotheses",
    }
    assert old["solver_tolerance"] == 1e-9
    assert new["solver_tolerance"] == new["weight_roundoff_tolerance"] == 1e-10
    assert new["revision_of_candidate"] == previous.CANDIDATES[0]["id"]
    assert new["new_independent_economic_hypotheses"] == 0
    assert new["preserved_preparation_failure"] == rules.PREVIOUS_FAILURE


def test_frozen_helpers_and_allocation_bodies_are_reused_without_economic_edits():
    for name in (
        "problem", "record_solution", "verify_plan", "drawdown", "scatter",
        "observed_paths", "independent_paths",
    ):
        assert getattr(rules, name) is getattr(previous, name)
    for name in ("allocate", "build_targets"):
        old = ast.parse(inspect.getsource(getattr(previous, name)))
        new = ast.parse(inspect.getsource(getattr(rules, name)))
        assert ast.dump(new) == ast.dump(old)
    for name in (
        "BASKET", "BUDGET", "CASH", "CERT_TOL", "CORE", "DRAWDOWN", "FACTOR_IDS",
        "FEAS_TOL", "LOOKBACK", "ROUND_TOL", "SLACK", "SLEEVES", "STOCK_MIN", "UNIVERSE",
    ):
        assert getattr(rules, name) == getattr(previous, name)


def test_both_solver_calls_use_stricter_supported_tolerances_without_warnings(monkeypatch):
    actual, calls = rules.linprog, []

    def checked(*args, **kwargs):
        assert kwargs["method"] == "highs-ds"
        assert kwargs["options"] == {
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = actual(*args, **kwargs)
        assert not caught
        calls.append(True)
        return result

    monkeypatch.setattr(rules, "linprog", checked)
    paths = np.ones((6, 3))
    weights, record = rules.allocate(paths)
    rules.verify_plan(weights, record, paths)
    assert calls == [True, True]
    assert previous.SOLVER_TOL == 1e-9
    assert previous.linprog is actual
    assert previous.CANDIDATES[0]["id"] == "ordered_path_growth_budget_v1"


def test_negative_bound_violation_still_stops_instead_of_widening_cleanup(monkeypatch):
    actual, calls = rules.linprog, []

    def tiny_negative(*args, **kwargs):
        result = actual(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            result.x[1] -= 2.9006786461321336e-10
            result.x[2] += 2.9006786461321336e-10
        return result

    monkeypatch.setattr(rules, "linprog", tiny_negative)
    with pytest.raises(QuantError, match="nonnegative funded budget"):
        rules.allocate(np.ones((6, 3)))
    assert calls == [True, True]
