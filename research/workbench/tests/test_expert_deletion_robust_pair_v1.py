from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_expert_distinct_opinion_pair_v1 import examples, observations

from us_quant import expert_deletion_robust_pair_v1 as rule
from us_quant.config import QuantError
from us_quant.storage import read_json


def numpy_score(values, cost):
    array = np.asarray(values)
    if np.ptp(array) == 0:
        return None
    return float(np.sqrt(252) * (np.mean(array) - cost / 21) / np.std(array, ddof=1))


def test_all_original_pairs_and_independent_lower_envelope_determine_the_winner():
    observed, previous = observations(), np.full(9, 1 / 9)
    _, original = rule.score_catalogue(observed, previous)
    weights, audit = rule.choose_pair(observed, previous)
    assert len(audit["pairs"]) == 36
    for old, row in zip(original["pairs"], audit["pairs"], strict=True):
        assert all(row[k] == v for k, v in old.items())
        if "unscorable_reason" in old:
            assert row["conditional_deletion_admissible"] is False
            continue
        series = np.asarray(
            [
                sum(
                    w * float((o.net_returns - o.risk_free).iloc[t])
                    for w, o in zip(old["expert_weights"], observed, strict=True)
                )
                for t in range(126)
            ]
        )
        cost = old["estimated_incremental_switch_cost_fraction"]
        train, valid = series[:84], series[84:]
        expected = [numpy_score(train, cost), numpy_score(valid, cost)]
        expected += [numpy_score(np.delete(train, i), cost) for i in range(84)]
        expected += [numpy_score(np.delete(valid, i), cost) for i in range(42)]
        assert all(v is not None for v in expected)
        assert abs(row["conditional_profile"]["conditional_lower_envelope"] - min(expected)) < 1e-8
        assert len(row["conditional_profile"]["training_one_return_deleted_scores"]) == 84
        assert len(row["conditional_profile"]["validation_one_return_deleted_scores"]) == 42
    best = min(
        audit["pairs"],
        key=lambda p: (-p["conditional_profile"]["conditional_lower_envelope_key"], p["pair"]),
    )
    assert audit["selected_pair"] == best["pair"]
    assert np.array_equal(weights, best["expert_weights"])
    assert audit["pair_covariance_and_weights_refitted_after_deletion"] is False


def test_identical_current_opinions_are_not_excluded_and_lexical_ties_are_preserved():
    observed = observations()
    common = observed[0].net_returns.copy()
    observed = [
        replace(o, target=observed[0].target.copy(), net_returns=common.copy()) for o in observed
    ]
    _, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert audit["conditionally_admissible_pairs"] == 36
    assert audit["selected_pair"] == min(row["pair"] for row in audit["pairs"])
    assert audit["current_opinion_exclusion_used"] is False


def test_a_single_return_deletion_can_make_every_pair_explicitly_unscorable():
    observed = observations()
    series = observed[0].net_returns.copy()
    series.iloc[:] = 0.0
    series.iloc[10] = 0.01
    series.iloc[84:] = np.sin(np.arange(42)) * 0.01
    observed = [replace(o, net_returns=series.copy()) for o in observed]
    weights, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert len(audit["pairs"]) == 36 and audit["conditionally_admissible_pairs"] == 0
    assert all(p["conditional_profile"] is not None for p in audit["pairs"])
    assert np.array_equal(weights, np.full(9, 1 / 9))
    assert audit["NET_history_mature"] is True and audit["scores_available"] is False
    assert audit["reason"] == "no_conditionally_admissible_pair_explicit_equal_weights"
    assert audit["selected_pair"] is None and audit["safety_or_success_claimed"] is False


def test_originally_unscorable_pairs_are_not_rescued():
    observed = observations()
    observed = [replace(o, net_returns=o.net_returns * 0) for o in observed]
    _, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert all(p["unscorable_reason"] == "zero_training_variance" for p in audit["pairs"])
    assert all(p["conditional_profile"] is None for p in audit["pairs"])
    assert audit["selected_pair"] is None


def test_cold_start_keeps_original_unscored_state():
    weights, audit = rule.choose_pair(observations(mature=False), np.full(9, 1 / 9))
    assert np.array_equal(weights, np.full(9, 1 / 9))
    assert audit["pairs"] == [] and audit["NET_history_mature"] is False
    assert audit["scores_available"] is False


def test_exact_rational_deleted_sample_variance_matches_the_profile():
    values = [float(0.001 + 0.01 * np.sin(i)) for i in range(126)]
    profile = rule._envelope(values, 0.0003)
    sample = [Fraction.from_float(v) for i, v in enumerate(values[:84]) if i != 10]
    mean = sum(sample) / len(sample)
    variance = sum((v - mean) ** 2 for v in sample) / (len(sample) - 1)
    expected = (
        np.sqrt(252) * float(mean - Fraction.from_float(0.0003) / 21) / np.sqrt(float(variance))
    )
    assert abs(profile["training_one_return_deleted_scores"][10] - expected) < 1e-8


@pytest.mark.parametrize("kind", ["missing", "identity", "NaN", "negative", "budget", "asset"])
def test_invalid_or_unavailable_opinions_cannot_receive_fallback(kind):
    observed = observations()
    if kind == "missing":
        observed[0] = replace(observed[0], target=None)
    elif kind == "identity":
        observed.reverse()
    else:
        target = observed[0].target.copy()
        if kind == "NaN":
            target.iloc[0] = np.nan
        elif kind == "negative":
            target.iloc[0], target.iloc[1] = 1.0, -0.02
        elif kind == "budget":
            target.iloc[0] = 1.0
        else:
            target = target.iloc[::-1]
        observed[0] = replace(observed[0], target=target)
    with pytest.raises(QuantError):
        rule.choose_pair(observed, np.full(9, 1 / 9))


@pytest.mark.parametrize(
    "previous", [np.full(8, 1 / 8), np.full(9, float("nan")), np.full(9, -1.0)]
)
def test_invalid_previous_statistical_state_fails(previous):
    with pytest.raises(QuantError):
        rule.choose_pair(observations(), previous)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("covariance_refitted_after_deletion", True),
        ("exact_or_approximate_opinion_exclusion", True),
        ("prior_completed_configurations", 267),
        ("global_pool_changed", True),
    ],
)
def test_no_window_membership_menu_or_prior_rescue(key, value):
    policy = deepcopy(read_json(rule.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rule.validate(policy)


def test_single_policy_prefix_future_and_rational_combination_are_exact():
    experts, calendar = examples()
    changed, _ = examples(pd.Timestamp("2017-04-28"))
    policy = read_json(rule.POLICY)
    records = []
    old_selector = rule.template.choose_pair
    full, audits = rule.build_targets(
        experts, calendar, rule.UNIVERSE, policy, retain=records.append
    )
    later, after = rule.build_targets(changed, calendar, rule.UNIVERSE, policy)
    early, before = rule.build_targets(
        experts, calendar[calendar <= "2017-04-28"], rule.UNIVERSE, policy
    )
    assert set(full) == {rule.ID} and rule.template.choose_pair is old_selector
    assert early[rule.ID].equals(full[rule.ID].loc[:"2017-04-28"])
    assert early[rule.ID].equals(later[rule.ID].loc[:"2017-04-28"])
    assert before == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert before == [a for a in after if a["decision_session"] <= "2017-04-28"]
    assert len(records) == 2 * len(audits)
    for audit in audits:
        observed = [e.observe(pd.Timestamp(audit["decision_session"]), 126) for e in experts]
        expected = [
            float(
                sum(
                    Fraction.from_float(float(w)) * Fraction.from_float(float(o.target.iloc[j]))
                    for w, o in zip(audit["rolling_expert_weights"], observed, strict=True)
                )
            )
            for j in range(len(rule.UNIVERSE))
        ]
        assert np.allclose(
            full[rule.ID].loc[audit["decision_session"]], expected, rtol=0, atol=1e-12
        )
        assert audit["shadow_wealth_added_to_main_capital"] is False
        assert audit["source_feedback_floor_transferred_to_main_account"] is False
