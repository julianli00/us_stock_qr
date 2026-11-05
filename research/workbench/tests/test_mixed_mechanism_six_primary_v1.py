from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_expert_distinct_opinion_pair_v1 import examples as prior_examples

from us_quant import mixed_mechanism_six_primary_v1 as law
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    prior, calendar = prior_examples(changed_after)
    original = prior[:8]
    extras = tuple(
        ArchivedMonthlyExpert(
            ExpertIdentity(law.EXPERT_IDS[i + 3], law.MECHANISMS[i + 3], "2026-11-05T00:00:00Z"),
            prior[j]._targets.copy(),
            prior[j]._shadow.copy(),
            law.UNIVERSE,
        )
        for i, j in enumerate((1, 2, 4))
    )
    return law.assemble(original, *extras), calendar, original, extras


def test_real_assembly_seam_selects_six_complete_policies_not_asset_sleeves():
    experts, _, original, extras = examples()
    assert tuple(e.identity.identifier for e in experts) == law.EXPERT_IDS
    assert tuple(e.identity.mechanism_group for e in experts) == law.MECHANISMS
    assert experts[:3] == (original[0], original[3], original[7])
    assert experts[3:] == extras
    assert len(set(e.identity.identifier for e in experts)) == 6


@pytest.mark.parametrize(
    "kind", ["original_missing", "original_reordered", "extra_missing", "duplicate", "wrong_group"]
)
def test_admissible_but_nonselected_or_missing_sources_are_not_silently_substituted(kind):
    _, _, original, extras = examples()
    if kind == "original_missing":
        original = original[:-1]
    elif kind == "original_reordered":
        original = original[::-1]
    elif kind == "extra_missing":
        extras = (original[1], *extras[1:])
    elif kind == "duplicate":
        extras = (extras[1], extras[1], extras[2])
    else:
        changed = deepcopy(extras[0])
        changed.identity = replace(changed.identity, mechanism_group="not source")
        extras = (changed, *extras[1:])
    with pytest.raises(QuantError):
        law.assemble(original, *extras)


def test_unchanged_original_point_scorer_and_all_fifteen_pairs_are_preserved():
    experts, _, _, _ = examples()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    previous = np.full(6, 1 / 6)
    expected, old = law.scorer.choose_pair(observed, previous)
    actual, record = law.choose_pair(observed, previous)
    assert np.array_equal(expected, actual)
    assert all(record[k] == value for k, value in old.items())
    assert len(record["pairs"]) == 15
    assert record["mechanism_labels_are_independent_risks"] is False
    assert record["current_opinion_exclusion_used"] is False
    assert record["conditional_deletion_lower_envelope_used"] is False


def test_cold_origins_and_mature_unscorable_catalogues_are_distinct():
    experts, _, _, _ = examples()
    cold = [e.observe(pd.Timestamp("2016-10-05"), 126) for e in experts]
    weights, early = law.choose_pair(cold, np.full(6, 1 / 6))
    assert np.array_equal(weights, np.full(6, 1 / 6))
    assert early["NET_history_mature"] is False and early["scores_available"] is False
    assert early["pairs"] == []
    for expert in experts:
        expert._shadow.loc[:, "return"] = 0.0
        expert._shadow.loc[:, "equity"] = 10000.0
    mature = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    _, later = law.choose_pair(mature, np.full(6, 1 / 6))
    assert later["NET_history_mature"] is True and later["scores_available"] is False
    assert len(later["pairs"]) == 15
    assert later["safety_or_success_claimed"] is False


@pytest.mark.parametrize(
    "kind", ["missing", "NaN", "negative", "budget", "asset", "identity", "mechanism"]
)
def test_incomplete_or_invalid_opinions_never_receive_successful_fallback(kind):
    experts, _, _, _ = examples()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    if kind == "missing":
        observed[0] = replace(observed[0], target=None)
    elif kind == "identity":
        observed.reverse()
    elif kind == "mechanism":
        observed[0] = replace(
            observed[0], identity=replace(observed[0].identity, mechanism_group="bad")
        )
    else:
        target = observed[0].target.copy()
        if kind == "NaN":
            target.iloc[0] = np.nan
        elif kind == "negative":
            target.iloc[0], target.iloc[1] = 1.0, -0.02
        elif kind == "budget":
            target.iloc[:] = 0.0
        else:
            target = target.iloc[::-1]
        observed[0] = replace(observed[0], target=target)
    with pytest.raises(QuantError):
        law.choose_pair(observed, np.full(6, 1 / 6))


def test_missing_mature_pair_is_not_hidden_by_cardinality_or_a_neutral_score(monkeypatch):
    experts, _, _, _ = examples()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    original = law.score_catalogue

    def incomplete(*args):
        weights, record = original(*args)
        record["pairs"].pop()
        return weights, record

    monkeypatch.setattr(law, "score_catalogue", incomplete)
    with pytest.raises(QuantError):
        law.choose_pair(observed, np.full(6, 1 / 6))


@pytest.mark.parametrize(
    "previous",
    [
        np.full(9, 1 / 9),
        np.full(6, float("nan")),
        np.full(6, -1.0),
    ],
)
def test_invalid_previous_statistical_state_fails(previous):
    experts, _, _, _ = examples()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    with pytest.raises(QuantError):
        law.choose_pair(observed, previous)


def test_two_fresh_targets_prefix_future_and_rational_capital_separation():
    experts, calendar, _, _ = examples()
    changed, _, _, _ = examples(pd.Timestamp("2017-04-28"))
    policy = read_json(law.POLICY)
    rows = []
    full, audits = law.build_targets(experts, calendar, law.UNIVERSE, policy, retain=rows.append)
    later, after = law.build_targets(changed, calendar, law.UNIVERSE, policy)
    early, before = law.build_targets(
        experts, calendar[calendar <= "2017-04-28"], law.UNIVERSE, policy
    )
    assert set(full) == set(law.IDS) and len(rows) == 2 * len(audits)
    for identifier in law.IDS:
        assert early[identifier].equals(full[identifier].loc[:"2017-04-28"])
        assert early[identifier].equals(later[identifier].loc[:"2017-04-28"])
    assert before == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert before == [a for a in after if a["decision_session"] <= "2017-04-28"]
    for audit in audits:
        observed = [e.observe(pd.Timestamp(audit["decision_session"]), 126) for e in experts]
        for identifier, key in zip(
            law.IDS, ("equal_expert_weights", "rolling_expert_weights"), strict=True
        ):
            expected = [
                float(
                    sum(
                        Fraction.from_float(float(w)) * Fraction.from_float(float(o.target.iloc[j]))
                        for w, o in zip(audit[key], observed, strict=True)
                    )
                )
                for j in range(len(law.UNIVERSE))
            ]
            assert np.allclose(
                full[identifier].loc[audit["decision_session"]], expected, rtol=0, atol=1e-12
            )
        assert audit["shadow_wealth_added_to_main_capital"] is False
        assert audit["source_feedback_floor_transferred_to_main_account"] is False
        if audit["selection"]["NET_history_mature"]:
            assert len(audit["selection"]["pairs"]) == 15


@pytest.mark.parametrize(
    "key,value",
    [
        ("expert_count", 9),
        ("selected_experts", 3),
        ("lookback_sessions", 252),
        ("mature_pair_catalogue_size", 36),
        ("conditional_deletion_lower_envelope", True),
        ("exact_or_approximate_opinion_exclusion", True),
        ("global_pool_changed", True),
        ("prior_completed_configurations", 268),
    ],
)
def test_no_roster_window_scoring_or_prior_rescue(key, value):
    policy = deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)
