from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import factor_covariance_risk as risk
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import file_digest, read_json, write_json
from us_quant.strategy_replay import registered_generator, registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-covariance-risk.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *risk.FACTORS))


@pytest.mark.parametrize("method", [row["method"] for row in risk.CANDIDATES])
def test_diagonal_covariance_matches_analytic_inverse_volatility(method):
    deviation = np.array([0.12, 0.14, 0.13, 0.11, 0.12, 0.10])
    actual, audit = risk.allocation(np.diag(deviation**2), method)
    expected = 1 / deviation
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
    if method == "risk_budget":
        np.testing.assert_allclose(audit["signed_variance_contribution_shares"], 1 / 6, atol=2e-7)
    assert (risk.constraint_matrix() @ actual >= -1e-10).all()


def test_maximum_diversification_matches_unconstrained_covariance_solution_when_bounds_inactive():
    deviation = np.array([0.12, 0.14, 0.13, 0.11, 0.12, 0.10])
    correlation = np.full((6, 6), 0.15)
    np.fill_diagonal(correlation, 1)
    covariance = correlation * np.outer(deviation, deviation)
    actual, _ = risk.allocation(covariance, "maximum_diversification")
    expected = np.linalg.solve(covariance, deviation)
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("method", [row["method"] for row in risk.CANDIDATES])
def test_allocation_is_covariance_scale_invariant_and_keeps_binding_groups(method):
    covariance = np.diag([0.25, 0.20, 0.18, 0.16, 0.09, 0.0025])
    actual, audit = risk.allocation(covariance, method)
    scaled, _ = risk.allocation(covariance * 100, method)
    np.testing.assert_allclose(actual, scaled, rtol=0, atol=1e-8)
    equity, defense = actual[:4].sum(), actual[4:].sum()
    assert 0.30 - 1e-9 <= equity <= 0.70 + 1e-9
    assert np.all((actual[:4] / equity >= 0.15 - 1e-9) & (actual[:4] / equity <= 0.35 + 1e-9))
    assert np.all((actual[4:] / defense >= 0.25 - 1e-9) & (actual[4:] / defense <= 0.75 + 1e-9))
    assert audit["group_constraints_binding_count"] > 0
    assert audit["equal_risk_contributions_guaranteed_by_constraints"] is False


@pytest.mark.parametrize("problem", ["shape", "nan", "asymmetric", "singular", "negative", "unknown"])
def test_invalid_risk_models_are_errors_not_fallback_allocations(problem):
    covariance, method = np.eye(6), "risk_budget"
    if problem == "shape":
        covariance = np.eye(5)
    elif problem == "nan":
        covariance[0, 0] = np.nan
    elif problem == "asymmetric":
        covariance[0, 1] = 0.5
    elif problem == "singular":
        covariance[:] = 1
    elif problem == "negative":
        covariance[0, 0] = -1
    else:
        method = "selected_after_return"
    with pytest.raises(QuantError):
        risk.allocation(covariance, method)


def test_failed_solver_does_not_return_equal_weight_success(monkeypatch):
    monkeypatch.setattr(
        risk, "minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="synthetic failure"),
    )
    with pytest.raises(QuantError, match="optimizer failed"):
        risk.allocation(np.eye(6), "risk_budget")


def test_targets_keep_four_positive_styles_two_defenses_monthly_causality_and_cash(market, policy):
    frames, diagnostics = risk.build_targets(market, policy)
    assert len(frames) == 2
    for frame in frames.values():
        active = frame.dropna(how="all")
        assert all(is_month_end(day) for day in active.index)
        assert (active[list(risk.ASSETS)] > 0).all().all()
        assert active[["SPY", "BIL"]].eq(0).all().all()
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert all(row["observed_returns_count"] == 252 for row in diagnostics)
    assert all(row["observed_returns_last_session"] == row["decision_session"] for row in diagnostics)


def test_future_prices_cannot_change_previously_issued_covariance_targets(market, policy):
    frames, _ = risk.build_targets(market, policy)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multipliers = np.linspace(1.0, 1.3, later.sum())
    close.loc[later, "GLD"] *= multipliers
    opening.loc[later, "GLD"] *= multipliers
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, _ = risk.build_targets(changed, policy)
    for identifier in frames:
        pd.testing.assert_frame_equal(frames[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("scenario,cost,delay", [("base", 5.0, 1), ("stress", 20.0, 2)])
def test_each_seeded_window_uses_funded_v2_and_independent_bt(market, policy, scenario, cost, delay):
    frames, _ = risk.build_targets(market, policy)
    for frame in frames.values():
        targets = seed_window(frame, "2016-10-06")
        own = simulate(
            market, targets, "2016-10-06", "2017-06-30",
            initial_capital=10000, cost_bps=cost, commission=1, delay=delay,
        )
        independent = independent_equity(
            market, targets, "2016-10-06", "2017-06-30",
            capital=10000, cost_bps=cost, commission=1, delay=delay,
        )
        np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0
        if scenario == "stress":
            assert own.frame.iloc[0]["orders"] == 0
            assert own.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize(
    "field,value",
    [
        ("covariance_sessions", 63),
        ("covariance_diagonal_shrinkage", 0.5),
        ("equity_share_max", 0.98),
        ("factor_share_of_equity_min", 0),
        ("each_defensive_share_min", 0),
        ("risk_assets", ["SOXX"]),
        ("new_configurations", 10),
    ],
)
def test_parameters_cannot_be_changed_to_hit_a_return_goal(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        risk.validate(policy)


def test_registered_replay_requires_exact_new_source_policy_and_configuration(
    tmp_path, policy, market,
):
    paths = {
        "config/factor-covariance-risk.json": policy,
        "evidence/risk_only_covariance_20261011_source_context.json": {"synthetic": True},
        "evidence/factor_gold_risk_20261010_registration.json": {"synthetic": True},
    }
    for name, body in paths.items():
        write_json(tmp_path / name, body)
    code = tmp_path / "src/us_quant/factor_covariance_risk.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(Path(risk.__file__).read_bytes())
    spec = {
        "id": risk.CANDIDATES[0]["id"], "configuration": risk.CANDIDATES[0],
        "frozen_files": {
            name: file_digest(tmp_path / name) for name in (*paths, "src/us_quant/factor_covariance_risk.py")
        },
    }
    assert registered_generator(spec, tmp_path)[0] == "factor_covariance_risk"
    actual = registered_targets(
        spec, market, "2016-10-06", "2017-06-30", 5.0, 1, tmp_path, {},
    )
    expected = seed_window(risk.build_targets(market, policy)[0][spec["id"]], "2016-10-06")
    pd.testing.assert_frame_equal(actual, expected)
    spec["configuration"] = {**risk.CANDIDATES[0], "method": "fit_after_return"}
    with pytest.raises(QuantError, match="differs from its policy"):
        registered_generator(spec, tmp_path)
