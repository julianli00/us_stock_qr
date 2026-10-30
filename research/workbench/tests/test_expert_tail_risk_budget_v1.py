from __future__ import annotations

import copy
import runpy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_tail_risk_budget_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic():
    helper = runpy.run_path(str(rules.ROOT / "tests/test_expert_dual_relative_selection_v1.py"))
    experts, calendar, universe, benchmarks = helper["synthetic"]()
    extended = (*universe, "BIL")
    adapted = []
    for expert in experts:
        targets = expert._targets.copy()
        targets["BIL"] = np.nan
        targets.loc[expert._targets.notna().all(axis=1), "BIL"] = 0.0
        adapted.append(type(expert)(expert.identity, targets, expert._shadow, extended))
    return tuple(adapted), calendar, extended, benchmarks


def problem_inputs():
    t = np.arange(126)
    excess = np.column_stack(
        [0.001 - i * 0.0001 + 0.002 * np.sin(t * (0.13 + i * 0.02)) for i in range(6)]
    )
    benchmark = np.column_stack([0.03 * np.sin(t * 0.21), 0.04 * np.sin(t * 0.19)])
    targets = np.eye(6) * 0.98
    previous = np.full(6, 0.98 / 6)
    return excess, benchmark, targets, previous


def test_joint_tail_budget_has_no_old_asset_group_constraints():
    values = problem_inputs()
    problem = rules.linear_problem(*values)
    weights, proof = rules.solve(problem)
    assert proof["mode"] == "expert_mix"
    assert np.isclose(weights.sum(), 1) and (weights >= 0).all()
    assert weights[0] > 0.7
    assert proof["primary_certificate"]["primal_dual_gap"] < rules.lp.CERT_TOL
    assert len(proof["periods"]) == 2


def test_negative_finite_opportunity_is_not_a_cash_cutoff():
    excess, benchmark, targets, previous = problem_inputs()
    excess -= 0.005
    weights, proof = rules.solve(rules.linear_problem(excess, benchmark, targets, previous))
    assert proof["mode"] == "expert_mix" and weights.sum() == pytest.approx(1)
    assert min(np.mean(excess[:84] @ weights), np.mean(excess[84:] @ weights)) < 0


def test_infeasibility_requires_positive_phase_one_dual_certificate():
    excess, _, targets, previous = problem_inputs()
    excess[:] = -0.02
    benchmarks = np.zeros((126, 2))
    weights, proof = rules.solve(rules.linear_problem(excess, benchmarks, targets, previous))
    assert np.array_equal(weights, np.zeros(6))
    assert proof["mode"] == "funding_standby"
    assert proof["phase_one_certificate"]["dual_objective"] > 0.019
    assert proof["allocation_risk_feasibility_or_safety_claimed"] is False


def test_uncertified_solver_infeasibility_is_not_funding_standby(monkeypatch):
    real = rules.linprog
    called = []

    def false_infeasible(*args, **kwargs):
        called.append(True)
        if len(called) == 1:
            return SimpleNamespace(status=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(rules, "linprog", false_infeasible)
    with pytest.raises(QuantError, match="positive independent PhaseI"):
        rules.solve(rules.linear_problem(*problem_inputs()))


def test_joint_tail_and_euler_contributions_are_not_average_individual_risk():
    values = np.column_stack(
        [
            np.tile([0.02, -0.02], 63),
            np.tile([-0.02, 0.02], 63),
            np.zeros((126, 4)),
        ]
    )
    weights = np.array([0.5, 0.5, 0, 0, 0, 0])
    marginal, contributions = rules.tail_contributions(values, weights)
    joint = rules.lp.tail_loss(-values @ weights)
    individual_average = sum(weights[i] * rules.lp.tail_loss(-values[:, i]) for i in range(6))
    assert joint == pytest.approx(0) and individual_average > 0.019
    assert contributions.sum() == pytest.approx(joint)
    assert np.isfinite(marginal).all()


def test_unmatured_history_retains_equal_control_and_initial_capital_anchor():
    experts, calendar, universe, benchmarks = synthetic()
    targets, audits = rules.build_targets(
        experts, calendar, universe, read_json(rules.POLICY), benchmarks
    )
    assert targets.index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["allocation"]["mode"] == "unscored_equal"
    assert audits[0]["allocation"]["risk_problem_mature"] is False
    assert audits[0]["expert_weights"] == [1 / 6] * 6
    assert audits[0]["previous_nominal_meta_target"] is None
    assert all(abs(row.sum() - 0.98) < 1e-12 for _, row in targets.dropna().iterrows())


def test_certified_infeasible_funding_standby_is_not_a_seventh_expert():
    experts, calendar, universe, _ = synthetic()
    day = calendar[-1]
    observed = [e.observe(day, 126) for e in experts]
    observed = [replace(o, net_returns=o.risk_free - 0.02) for o in observed]
    rf = observed[0].risk_free
    benchmark = pd.DataFrame({"SPY": rf, "QQQ": rf})
    target, weights, proof = rules.choose_allocation(observed, None, benchmark)
    assert target["BIL"] == 0.98 and target.drop("BIL").sum() == 0
    assert weights.tolist() == [0] * 6
    assert proof["mode"] == "funding_standby"
    assert not proof["funding_standby_is_strategy_expert"]
    assert not proof["allocation_risk_feasibility_or_safety_claimed"]


def test_future_source_and_benchmark_changes_preserve_prior_targets():
    experts, calendar, universe, benchmarks = synthetic()
    policy = read_json(rules.POLICY)
    cutoff = rules.decision_sessions(calendar, policy)[-2]
    prefix = calendar[calendar <= cutoff]
    before = rules.build_targets(experts, prefix, universe, policy, benchmarks)
    changed = copy.deepcopy(experts)
    for expert in changed:
        expert._shadow.loc[expert._shadow.index > cutoff, "return"] += 0.15
    altered = benchmarks.copy()
    altered.loc[altered.index > cutoff] += 0.2
    after = rules.build_targets(changed, prefix, universe, policy, altered)
    assert before[0].equals(after[0]) and before[1] == after[1]


@pytest.mark.parametrize("kind", ("date", "RF", "identity", "nonfinite", "benchmark"))
def test_missing_or_changed_history_cannot_be_a_successful_tail_assessment(kind):
    experts, calendar, _, benchmarks = synthetic()
    day = calendar[-1]
    observed = [e.observe(day, 126) for e in experts]
    if kind == "date":
        observed[0] = replace(observed[0], net_returns=observed[0].net_returns.iloc[1:])
    elif kind == "RF":
        observed[0] = replace(observed[0], risk_free=observed[0].risk_free + 0.001)
    elif kind == "identity":
        observed.reverse()
    elif kind == "nonfinite":
        v = observed[0].net_returns.copy()
        v.iloc[-1] = np.nan
        observed[0] = replace(observed[0], net_returns=v)
    else:
        benchmarks = benchmarks.drop(observed[0].net_returns.index[-1])
    with pytest.raises(QuantError):
        rules.choose_allocation(observed, None, benchmarks)
