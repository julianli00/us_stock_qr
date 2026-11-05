import itertools
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_distinct_opinion_pair_v1 as rule
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity, ExpertObservation


def observations(mature=True):
    dates = sessions("2016-10-06", "2017-04-06")
    values = []
    for i, name in enumerate(rule.EXPERT_IDS):
        target = pd.Series(0.0, index=rule.UNIVERSE)
        target.iloc[i] = 0.98
        returns = pd.Series(0.001 + 0.005 * np.sin(np.arange(len(dates)) / (4 + i) + i), dates)
        values.append(
            ExpertObservation(
                ExpertIdentity(name, rule.MECHANISMS[i], "2026-11-05T00:00:00Z"),
                dates[-1],
                target,
                dates[-1],
                returns if mature else returns.iloc[:0],
                pd.Series(0.0, dates) if mature else pd.Series(dtype=float, index=dates[:0]),
                mature,
            )
        )
    return values


def catalogue(observed, previous):
    pairs = []
    for index, (a, b) in enumerate(itertools.combinations(range(9), 2)):
        weights = np.zeros(9)
        weights[[a, b]] = 0.5
        pairs.append(
            {
                "pair": sorted((rule.EXPERT_IDS[a], rule.EXPERT_IDS[b])),
                "expert_weights": weights.tolist(),
                "selection_score_key": float(100 - index),
                "selection_score": float(100 - index),
            }
        )
    return np.asarray(pairs[0]["expert_weights"]), {
        "reason": "synthetic_original_catalogue",
        "selected_pair": pairs[0]["pair"],
        "pairs": pairs,
        "scores_available": True,
        "safety_or_success_claimed": False,
    }


def test_unchanged_choice_when_original_winner_has_distinct_current_opinions(monkeypatch):
    monkeypatch.setattr(rule, "score_catalogue", catalogue)
    observed, previous = observations(), np.full(9, 1 / 9)
    expected, old = catalogue(observed, previous)
    weights, audit = rule.choose_pair(observed, previous)
    assert np.array_equal(weights, expected)
    assert audit["selected_pair"] == old["selected_pair"]
    assert len(audit["pairs"]) == 36
    assert audit["eligible_scorable_pairs"] == 36


def test_only_identical_current_pair_is_ineligible_and_all_scores_remain(monkeypatch):
    monkeypatch.setattr(rule, "score_catalogue", catalogue)
    observed = observations()
    observed[1] = replace(observed[1], target=observed[0].target.copy())
    weights, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert set(audit["selected_pair"]) == {rule.EXPERT_IDS[0], rule.EXPERT_IDS[2]}
    assert set(audit["same_state_original_catalogue_selected_pair"]) == set(rule.EXPERT_IDS[:2])
    assert len(audit["pairs"]) == 36
    assert audit["pairs"][0]["selection_score_key"] == 100
    assert audit["pairs"][0]["eligible_by_current_opinion"] is False
    assert audit["eligible_scorable_pairs"] == 35
    assert np.array_equal(weights, np.array([0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]))


def test_all_equal_current_opinions_have_explicit_mature_equal_fallback(monkeypatch):
    monkeypatch.setattr(rule, "score_catalogue", catalogue)
    observed = observations()
    observed = [replace(o, target=observed[0].target.copy()) for o in observed]
    weights, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert np.array_equal(weights, np.full(9, 1 / 9))
    assert audit["NET_history_mature"] is True
    assert audit["scores_available"] is False
    assert audit["same_state_original_catalogue_scores_available"] is True
    assert audit["reason"] == "no_distinct_current_opinion_pair_equal_weights"
    assert audit["selected_pair"] is None
    assert audit["safety_or_success_claimed"] is False
    assert len(audit["pairs"]) == 36


def test_distinct_but_unscorable_menu_is_not_missing_history_or_success(monkeypatch):
    def unscorable(observed, previous):
        weights, audit = catalogue(observed, previous)
        audit["pairs"] = [
            {"pair": row["pair"], "unscorable_reason": "zero_training_variance"}
            for row in audit["pairs"]
        ]
        return weights, {**audit, "selected_pair": None, "scores_available": False}

    monkeypatch.setattr(rule, "score_catalogue", unscorable)
    weights, audit = rule.choose_pair(observations(), np.full(9, 1 / 9))
    assert np.array_equal(weights, np.full(9, 1 / 9))
    assert audit["reason"] == "no_scorable_distinct_current_opinion_pair_equal_weights"
    assert all(row["unscorable_reason"] == "zero_training_variance" for row in audit["pairs"])


def test_near_equal_is_not_collapsed_by_the_geometric_diagnostic_tolerance(monkeypatch):
    monkeypatch.setattr(rule, "score_catalogue", catalogue)
    observed = observations()
    changed = observed[0].target.copy()
    changed.iloc[0] -= 1e-14
    changed.iloc[1] += 1e-14
    observed[1] = replace(observed[1], target=changed)
    _, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert audit["pairs"][0]["eligible_by_current_opinion"] is True
    assert set(audit["selected_pair"]) == set(rule.EXPERT_IDS[:2])


def test_original_cold_start_and_scorer_namespace_are_unchanged():
    observed = observations(mature=False)
    old_function = rule.source.choose_pair
    weights, audit = rule.choose_pair(observed, np.full(9, 1 / 9))
    assert np.array_equal(weights, np.full(9, 1 / 9))
    assert audit["reason"] == "unscored_initial_equal_weights"
    assert audit["NET_history_mature"] is False
    assert audit["pairs"] == []
    assert len(audit["current_opinion_pair_eligibility"]) == 36
    assert rule.source.choose_pair is old_function


def test_real_unchanged_scorer_matches_when_no_current_pair_collides():
    observed, previous = observations(), np.full(9, 1 / 9)
    expected, old = rule.source.choose_pair(observed, previous)
    weights, audit = rule.choose_pair(observed, previous)
    assert np.array_equal(weights, expected)
    assert audit["selected_pair"] == old["selected_pair"]
    for before, after in zip(old["pairs"], audit["pairs"], strict=True):
        assert all(after[key] == value for key, value in before.items())


def test_missing_original_pair_catalogue_record_fails(monkeypatch):
    def incomplete(observed, previous):
        weights, audit = catalogue(observed, previous)
        audit["pairs"].pop()
        return weights, audit

    monkeypatch.setattr(rule, "score_catalogue", incomplete)
    with pytest.raises(QuantError):
        rule.choose_pair(observations(), np.full(9, 1 / 9))


@pytest.mark.parametrize("failure", ["missing", "NaN", "negative", "budget", "identity", "asset"])
def test_invalid_or_unavailable_opinions_cannot_receive_fallback(failure):
    observed = observations()
    if failure == "missing":
        observed[0] = replace(observed[0], target=None)
    elif failure == "identity":
        observed.reverse()
    else:
        changed = observed[0].target.copy()
        if failure == "NaN":
            changed.iloc[0] = np.nan
        elif failure == "negative":
            changed.iloc[0], changed.iloc[1] = 1.0, -0.02
        elif failure == "budget":
            changed.iloc[0] = 1.0
        else:
            changed = changed.iloc[::-1]
        observed[0] = replace(observed[0], target=changed)
    with pytest.raises(QuantError):
        rule.choose_pair(observed, np.full(9, 1 / 9))


def examples(changed_after=None):
    calendar = sessions("2016-09-30", "2017-08-31")
    days = calendar[calendar >= "2016-10-06"]
    experts = []
    for i, name in enumerate(rule.EXPERT_IDS):
        values = 0.0006 + 0.006 * np.sin(np.arange(len(days)) / (4 + i) + 0.7 * i)
        if changed_after is not None:
            values = values.copy()
            values[days > changed_after] += 0.01
        wealth = 10000 * np.cumprod(1 + values)
        shadow = pd.DataFrame(
            {
                "equity": wealth,
                "return": values,
                "risk_free": 0.0,
                "cash": 200.0,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0.0,
            },
            index=days,
        )
        targets = pd.DataFrame(np.nan, calendar, rule.UNIVERSE)
        for number, day in enumerate(calendar):
            if is_month_end(day):
                targets.loc[day] = 0.0
                asset = (i // 2 + number % 2) % len(rule.UNIVERSE)
                if changed_after is not None and day > changed_after:
                    asset = (asset + 3) % len(rule.UNIVERSE)
                targets.loc[day, rule.UNIVERSE[asset]] = 0.98
        experts.append(
            ArchivedMonthlyExpert(
                ExpertIdentity(name, rule.MECHANISMS[i], "2026-11-05T00:00:00Z"),
                targets,
                shadow,
                rule.UNIVERSE,
            )
        )
    return tuple(experts), calendar


def test_new_single_policy_prefix_future_and_rational_combination_are_exact():
    experts, calendar = examples()
    changed, _ = examples(pd.Timestamp("2017-04-28"))
    policy = read_json(rule.POLICY)
    records = []
    full, audits = rule.build_targets(
        experts, calendar, rule.UNIVERSE, policy, retain=records.append
    )
    later, after = rule.build_targets(changed, calendar, rule.UNIVERSE, policy)
    early, before = rule.build_targets(
        experts, calendar[calendar <= "2017-04-28"], rule.UNIVERSE, policy
    )
    assert set(full) == {rule.ID}
    assert early[rule.ID].equals(full[rule.ID].loc[:"2017-04-28"])
    assert early[rule.ID].equals(later[rule.ID].loc[:"2017-04-28"])
    assert before == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert before == [a for a in after if a["decision_session"] <= "2017-04-28"]
    assert len(records) == 2 * len(audits)
    assert all(a["selection"]["scores_available"] is False for a in audits[:7])
    for audit in audits:
        observed = [e.observe(pd.Timestamp(audit["decision_session"]), 126) for e in experts]
        weights = audit["rolling_expert_weights"]
        expected = [
            float(
                sum(
                    Fraction.from_float(float(w)) * Fraction.from_float(float(o.target.iloc[j]))
                    for w, o in zip(weights, observed, strict=True)
                )
            )
            for j in range(len(rule.UNIVERSE))
        ]
        assert np.allclose(
            full[rule.ID].loc[audit["decision_session"]], expected, rtol=0, atol=1e-12
        )
        assert len(audit["selection"]["current_opinion_pair_eligibility"]) == 36


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("approximate_geometry_or_rank_threshold_used", True),
        ("all_pair_catalogue_size", 28),
        ("global_pool_changed", True),
        ("prior_completed_configurations", 265),
    ],
)
def test_no_window_membership_threshold_or_prior_rescue(key, value):
    policy = deepcopy(read_json(rule.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rule.validate(policy)
