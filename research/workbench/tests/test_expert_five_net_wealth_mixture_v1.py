from __future__ import annotations

import copy
import math
import runpy

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_five_net_wealth_mixture_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.mark.parametrize("degree", range(9))
def test_packed_compositions_and_ranks_cover_exactly_the_valid_simplex(degree):
    layout = rules.CompositionLayout(degree)
    actual = np.concatenate([counts for _, _, counts in layout.chunks()])
    expected = []
    for a in range(degree + 1):
        for b in range(degree - a + 1):
            for c in range(degree - a - b + 1):
                for e in range(degree - a - b - c + 1):
                    expected.append((a, b, c, e, degree - a - b - c - e))
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == math.comb(degree + 4, 4)
    ranks = rules.packed_rank(degree, *[actual[:, i] for i in range(4)])
    np.testing.assert_array_equal(ranks, np.arange(len(actual)))
    assert layout.nbytes == math.comb(degree + 3, 3) * 14 + 8
    for policy in range(5):
        shifted = actual.copy()
        shifted[:, policy] += 1
        destinations = rules.packed_rank(degree + 1, *[shifted[:, i] for i in range(4)])
        assert len(np.unique(destinations)) == len(actual)
        assert destinations.min() >= 0 and destinations.max() < math.comb(degree + 5, 4)


def test_one_period_matches_analytic_five_policy_dirichlet_integral():
    relative = np.array([1.08, 0.97, 1.02, 1.01, 1.03])
    expected = (relative.sum() + relative) / (6 * relative.sum())
    state = rules.PackedFiveMoments()
    np.testing.assert_allclose(state.weights(), 0.2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(state.update(relative), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        rules.independent_prefix_weights(relative[None, :])[1], expected, rtol=0, atol=1e-15
    )


def test_packed_recursion_matches_independent_streamed_integration_at_every_prefix():
    generator = np.random.default_rng(20261030)
    relatives = np.exp(generator.normal(0.003, 0.05, size=(9, 5)))
    expected = rules.independent_prefix_weights(relatives)
    state = rules.PackedFiveMoments()
    for i, row in enumerate(relatives):
        np.testing.assert_allclose(state.update(row), expected[i + 1], rtol=0, atol=1e-12)
    assert (state.weights() > 0).all() and state.weights().sum() == pytest.approx(1)


def test_permuting_policies_permutes_weights_and_common_scaling_is_irrelevant():
    relatives = np.array([[1.1, 0.9, 1.0, 1.04, 1.02], [0.96, 1.02, 1.01, 1.03, 0.99]])
    permutation = [4, 0, 3, 2, 1]
    original, changed = rules.PackedFiveMoments(), rules.PackedFiveMoments()
    for row in relatives:
        original.update(row)
        changed.update(row[permutation] * 100)
    np.testing.assert_allclose(changed.weights(), original.weights()[permutation], atol=1e-12)


def test_identical_streams_keep_uniform_prior_at_full_sparse_interval_and_memory_bound():
    state = rules.PackedFiveMoments()
    for i in range(119):
        state.update(np.full(5, 1 + 0.001 * np.sin(i)))
    np.testing.assert_allclose(state.weights(), 0.2, rtol=0, atol=1e-12)
    assert state.mass.ndim == 1 and state.mass.size == 9078630
    assert state.mass.nbytes == 72629040
    assert state.maximum_coefficient_buffer_bytes == 142896160
    assert state.layout.nbytes == math.comb(122, 3) * 14 + 8
    with pytest.raises(QuantError, match="interval bound"):
        state.update(np.ones(5))


@pytest.mark.parametrize(
    "relative",
    [
        np.ones(4),
        np.array([1, 1, 1, 1, 0]),
        np.full(5, np.nan),
    ],
)
def test_invalid_net_relative_never_becomes_a_neutral_score(relative):
    with pytest.raises(QuantError, match="five positive"):
        rules.PackedFiveMoments().update(relative)


def archived_experts(changed=False):
    helper = runpy.run_path(str(rules.ROOT / "tests/test_strategy_expert_pool_v3.py"))
    return helper["archived_experts"](changed_after=pd.Timestamp("2017-04-28") if changed else None)


@pytest.fixture(scope="module")
def prepared():
    experts, calendar, assets = archived_experts()
    policy = read_json(rules.POLICY)
    target, audits, periods = rules.build_targets(experts, calendar, assets, policy)
    return experts, calendar, assets, policy, target, audits, periods


def test_true_initial_anchor_and_prequential_validation_precede_assimilation(prepared):
    _, _, _, _, target, audits, periods = prepared
    assert [a["decision_session"] for a in audits[:3]] == [
        "2016-10-05",
        "2016-10-31",
        "2016-11-30",
    ]
    assert [a["complete_learning_intervals"] for a in audits[:3]] == [0, 0, 1]
    assert [a["score_mature"] for a in audits[:3]] == [False, False, True]
    assert periods[0]["validation_weights_before_assimilation"] == [0.2] * 5
    relative = np.array(periods[0]["net_shadow_relatives"])
    assert periods[0]["prequential_net_shadow_score_relative"] == pytest.approx(relative.mean())
    np.testing.assert_allclose(
        audits[2]["expert_weights"], (relative.sum() + relative) / (6 * relative.sum())
    )
    assert target.loc[:"2016-10-04"].isna().all().all()
    assert audits[2]["adapter_requested_observation_window_filled"] == [False] * 5
    assert all(a["array_bytes_are_total_process_RAM"] is False for a in audits)


def test_every_allocation_matches_independent_integration_and_capital_interface(prepared):
    experts, calendar, assets, policy, target, audits, periods = prepared
    proof = rules.verify_targets(experts, calendar, assets, policy, target, audits, periods)
    assert proof["every_decision_independently_integrated"] is True
    assert proof["maximum_weight_difference"] < 1e-12
    assert proof["maximum_target_difference"] < 1e-12
    assert np.allclose(target.dropna().sum(axis=1), 0.98)
    assert (target.dropna() >= 0).all().all()


def test_prefix_and_future_perturbations_leave_past_targets_identical(prepared):
    experts, calendar, assets, policy, target, audits, periods = prepared
    end = pd.Timestamp("2017-04-28")
    shorter, a, p = rules.build_targets(experts, calendar[calendar <= end], assets, policy)
    pd.testing.assert_frame_equal(shorter, target.loc[:end], check_exact=True)
    assert a == [r for r in audits if r["decision_session"] <= str(end.date())]
    assert p == [r for r in periods if r["end"] <= str(end.date())]
    changed, _, _ = archived_experts(changed=True)
    after, aa, pp = rules.build_targets(changed, calendar, assets, policy)
    pd.testing.assert_frame_equal(after.loc[:end], target.loc[:end], check_exact=True)
    assert [r for r in aa if r["decision_session"] <= str(end.date())] == a
    assert [r for r in pp if r["end"] <= str(end.date())] == p


@pytest.mark.parametrize(
    "damage",
    [
        "weights",
        "nan_weights",
        "validation",
        "future_clock",
        "risk",
        "proxy",
        "memory",
        "capital",
    ],
)
def test_verifier_rejects_rewritten_scores_risk_cost_clocks_or_capital(prepared, damage):
    experts, calendar, assets, policy, target, audits, periods = prepared
    changed = copy.deepcopy(audits)
    row = changed[3]
    if damage == "weights":
        row["expert_weights"][0] += 0.01
    elif damage == "nan_weights":
        row["expert_weights"][0] = np.nan
    elif damage == "validation":
        row["prequential_net_shadow_score_relative"] += 0.01
    elif damage == "future_clock":
        row["expert_target_known_at"][0] = "2026-10-05"
    elif damage == "risk":
        row["risk_and_cost"]["net_excess_covariance"][0][0] += 0.01
    elif damage == "proxy":
        row["risk_and_cost"]["incremental_weight_switch_proxy_at_10000_usd"] += 1
    elif damage == "memory":
        row["learner_maximum_coefficient_buffer_bytes"] = 1658880000
    else:
        row["shadow_wealth_added_to_main_capital"] = True
    with pytest.raises(QuantError):
        rules.verify_targets(experts, calendar, assets, policy, target, changed, periods)


def test_existing_period_score_and_risk_cost_law_is_reused():
    assert rules.period_relative is rules.original.period_relative
    assert rules.risk_cost_audit is rules.original.risk_cost_audit
    policy = read_json(rules.POLICY)
    policy["methodology"]["score_is_actual_main_net_wealth"] = True
    with pytest.raises(QuantError, match="funded wealth"):
        rules.validate(policy)
