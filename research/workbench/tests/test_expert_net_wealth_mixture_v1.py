from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_net_wealth_mixture_v1 as rules
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ArchivedMonthlyExpert, ExpertIdentity

ASSETS = (
    "SPY",
    "IEF",
    "TLT",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)


def archived_experts(future_change=False):
    calendar = sessions("2016-08-01", "2017-09-29")
    history = sessions("2016-10-06", "2017-09-29")
    phase = np.arange(len(history))
    result = []
    allocations = (
        {"SOXX": 0.49, "MTUM": 0.245, "BIL": 0.245},
        {"GLD": 0.3, "IEF": 0.3, "QUAL": 0.2, "PKW": 0.18},
        {"SOXX": 0.5, "VLUE": 0.24, "BIL": 0.24},
        {"SOXX": 0.2, "USMV": 0.2, "GLD": 0.2, "BIL": 0.38},
    )
    for i, identifier in enumerate(rules.EXPERT_IDS):
        target = pd.DataFrame(0.0, index=calendar, columns=ASSETS)
        for symbol, weight in allocations[i].items():
            target[symbol] = weight
        if i == 2:
            target.loc[[d for d in calendar if not is_month_end(d)]] = np.nan
        elif i == 3:
            target.loc[:"2016-10-04"] = np.nan
        returns = pd.Series(0.0002 * (i + 1) + 0.004 * np.sin(phase / (2 + i)), index=history)
        if future_change:
            returns.loc["2017-04-03":] += 0.02 * (i + 1)
            changed_dates = target.loc["2017-04-03":].dropna(how="all").index
            target.loc[changed_dates] = 0.0
            target.loc[changed_dates, "BIL"] = 0.98
        nav = 10000 * (1 + returns).cumprod()
        shadow = pd.DataFrame(
            {
                "equity": nav,
                "return": returns,
                "risk_free": 0.00005,
                "cash": nav * 0.02,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0,
            },
            index=history,
        )
        adapter = ArchivedMonthlyExpert if i == 2 else ArchivedDenseExpert
        result.append(
            adapter(
                ExpertIdentity(identifier, rules.MECHANISMS[i], "2026-10-29T22:00:00+00:00"),
                target,
                shadow,
                ASSETS,
            )
        )
    return result, calendar


def test_one_period_matches_analytic_uniform_dirichlet_integral():
    relative = np.array([1.08, 0.97, 1.02, 1.01])
    state = rules.TetrahedralMoments()
    expected = (relative.sum() + relative) / (5 * relative.sum())
    np.testing.assert_allclose(state.weights(), 0.25, rtol=0, atol=1e-15)
    np.testing.assert_allclose(state.update(relative), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(rules.independent_weights(relative[None, :]), expected, atol=1e-15)


def test_dense_recursion_matches_independent_polynomial_exact_integration():
    generator = np.random.default_rng(20261029)
    relatives = np.exp(generator.normal(0.003, 0.05, size=(9, 4)))
    state = rules.TetrahedralMoments()
    for i, row in enumerate(relatives):
        actual = state.update(row)
        np.testing.assert_allclose(
            actual, rules.independent_weights(relatives[: i + 1]), rtol=0, atol=1e-12
        )
    assert (actual > 0).all() and actual.sum() == pytest.approx(1.0)


def test_permuting_complete_policies_permutes_weights_not_asset_sleeves():
    relatives = np.array([[1.1, 0.9, 1.0, 1.04], [0.96, 1.02, 1.01, 1.03]])
    permutation = [3, 0, 2, 1]
    original, changed = rules.TetrahedralMoments(), rules.TetrahedralMoments()
    for row in relatives:
        original.update(row)
        changed.update(row[permutation])
    np.testing.assert_allclose(changed.weights(), original.weights()[permutation], atol=1e-12)


def test_identical_streams_keep_uniform_prior_at_full_bounded_history():
    state = rules.TetrahedralMoments()
    for i in range(119):
        state.update(np.full(4, 1 + 0.001 * np.sin(i)))
    np.testing.assert_allclose(state.weights(), np.full(4, 0.25), rtol=0, atol=1e-12)
    assert state.mass.shape == (120, 120, 120)
    assert state.mass.nbytes == 13824000
    with pytest.raises(QuantError, match="interval bound"):
        state.update(np.ones(4))


@pytest.mark.parametrize("relative", [np.ones(3), np.array([1, 1, 1, 0]), np.full(4, np.nan)])
def test_invalid_net_relative_never_becomes_a_neutral_score(relative):
    with pytest.raises(QuantError, match="four positive"):
        rules.TetrahedralMoments().update(relative)


@pytest.fixture(scope="module")
def prepared():
    experts, calendar = archived_experts()
    policy = read_json(rules.POLICY)
    target, audits, periods = rules.build_targets(experts, calendar, ASSETS, policy)
    return experts, calendar, policy, target, audits, periods


def test_true_initial_anchor_and_prequential_validation_precede_assimilation(prepared):
    _, _, _, target, audits, periods = prepared
    assert [a["decision_session"] for a in audits[:3]] == [
        "2016-10-05",
        "2016-10-31",
        "2016-11-30",
    ]
    assert [a["complete_learning_intervals"] for a in audits[:3]] == [0, 0, 1]
    assert [a["score_mature"] for a in audits[:3]] == [False, False, True]
    assert periods[0]["start_after"] == "2016-10-31"
    assert periods[0]["end"] == "2016-11-30"
    assert periods[0]["validation_weights_before_assimilation"] == [0.25] * 4
    relative = np.array(periods[0]["net_shadow_relatives"])
    assert periods[0]["prequential_net_shadow_score_relative"] == pytest.approx(relative.mean())
    np.testing.assert_allclose(
        audits[2]["expert_weights"], (relative.sum() + relative) / (5 * relative.sum())
    )
    assert target.loc[:"2016-10-04"].isna().all().all()
    assert audits[2]["adapter_requested_observation_window_filled"] == [False] * 4
    assert audits[2]["risk_and_cost"]["risk_status"] == "observed"
    assert all(a["shadow_wealth_added_to_main_capital"] is False for a in audits)


def test_every_allocation_matches_independent_integration_and_capital_interface(prepared):
    experts, calendar, policy, target, audits, periods = prepared
    proof = rules.verify_targets(experts, calendar, ASSETS, policy, target, audits, periods)
    assert proof["every_decision_independently_integrated"] is True
    assert proof["maximum_weight_difference"] < 1e-12
    assert proof["maximum_target_difference"] < 1e-12
    assert proof["candidate_accounts_or_metrics_computed"] is False
    assert np.allclose(target.dropna().sum(axis=1), 0.98)
    assert (target.dropna() >= 0).all().all()


def test_prefix_and_future_perturbations_leave_past_targets_identical(prepared):
    experts, calendar, policy, target, audits, periods = prepared
    end = pd.Timestamp("2017-03-31")
    shorter, a, p = rules.build_targets(experts, calendar[calendar <= end], ASSETS, policy)
    pd.testing.assert_frame_equal(shorter, target.loc[:end], check_exact=True)
    assert a == [r for r in audits if r["decision_session"] <= str(end.date())]
    assert p == [r for r in periods if r["end"] <= str(end.date())]
    changed, _ = archived_experts(future_change=True)
    after, aa, pp = rules.build_targets(changed, calendar, ASSETS, policy)
    pd.testing.assert_frame_equal(after.loc[:end], target.loc[:end], check_exact=True)
    assert [r for r in aa if r["decision_session"] <= str(end.date())] == a
    assert [r for r in pp if r["end"] <= str(end.date())] == p


@pytest.mark.parametrize(
    "damage",
    [
        "weights",
        "nan_weights",
        "validation",
        "nan_validation",
        "future_clock",
        "risk",
        "proxy",
        "memory",
        "capital",
    ],
)
def test_scalar_proof_rejects_rewritten_scores_risk_cost_clocks_or_capital(prepared, damage):
    experts, calendar, policy, target, audits, periods = prepared
    changed = copy.deepcopy(audits)
    row = changed[3]
    if damage == "weights":
        row["expert_weights"][0] += 0.01
    elif damage == "nan_weights":
        row["expert_weights"][0] = np.nan
    elif damage == "validation":
        row["prequential_net_shadow_score_relative"] += 0.01
    elif damage == "nan_validation":
        row["prequential_net_shadow_score_relative"] = np.nan
    elif damage == "future_clock":
        row["expert_target_known_at"][0] = "2026-10-05"
    elif damage == "risk":
        row["risk_and_cost"]["net_excess_covariance"][0][0] += 0.01
    elif damage == "proxy":
        row["risk_and_cost"]["incremental_weight_switch_proxy_at_10000_usd"] += 1
    elif damage == "memory":
        row["learner_mass_array_bytes"] = 999999999
    else:
        row["shadow_wealth_added_to_main_capital"] = True
    with pytest.raises(QuantError):
        rules.verify_targets(experts, calendar, ASSETS, policy, target, changed, periods)


def test_unavailable_or_future_expert_observation_is_not_filled():
    experts, calendar = archived_experts()

    class FutureExpert:
        def observe(self, day, lookback):
            return replace(experts[0].observe(day, lookback), target_known_at=next_session(day))

    with pytest.raises(QuantError, match="observations cannot be filled"):
        rules.build_targets(
            [FutureExpert(), *experts[1:]], calendar, ASSETS, read_json(rules.POLICY)
        )


def test_fixed_policy_cannot_claim_net_wealth_or_virtual_trial_guarantees():
    policy = read_json(rules.POLICY)
    policy["methodology"]["score_is_actual_main_net_wealth"] = True
    with pytest.raises(QuantError, match="funded wealth"):
        rules.validate(policy)
