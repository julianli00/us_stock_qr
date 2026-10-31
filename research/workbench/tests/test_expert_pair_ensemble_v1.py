import copy

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_v5 import archived_experts

from us_quant import expert_pair_ensemble_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def observed():
    experts, _, _ = archived_experts()
    return [expert.observe(pd.Timestamp("2017-04-28"), 126) for expert in experts]


def profile(monkeypatch, keys):
    names = [
        tuple(sorted(rules.EXPERT_IDS[:2])),
        tuple(sorted(rules.EXPERT_IDS[2:4])),
        tuple(sorted(rules.EXPERT_IDS[4:6])),
    ]
    pairs = []
    for pair, key in zip(names, keys, strict=True):
        weights = np.zeros(7)
        weights[[rules.EXPERT_IDS.index(name) for name in pair]] = [0.4, 0.6]
        pairs.append(
            {
                "pair": list(pair),
                "expert_weights": weights.tolist(),
                "selection_score_key": key,
            }
        )
    ranked = sorted(pairs, key=lambda row: (-row["selection_score_key"], row["pair"]))
    record = {
        "pairs": pairs,
        "selected_pair": ranked[0]["pair"],
        "scores_available": True,
    }
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda items, previous: (np.asarray(ranked[0]["expert_weights"]), copy.deepcopy(record)),
    )
    return pairs


def test_uniform_and_ordinal_aggregate_all_scorable_pairs(observed):
    uniform, equal = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[0])
    ordinal, ranked = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[1])
    for weights, evidence in ((uniform, equal), (ordinal, ranked)):
        assert len(evidence["pair_masses"]) == evidence["scorable_pair_count"] == 21
        assert all(row["pair_mass"] > 0 for row in evidence["pair_masses"])
        assert sum(row["pair_mass"] for row in evidence["pair_masses"]) == pytest.approx(1)
        assert (weights > 0).sum() == 7 and weights.sum() == pytest.approx(1)
        assert evidence["overlapping_pairs_are_independent_experts_or_capital"] is False
        assert evidence["confidence_interval_or_significance_claimed"] is False
        assert rules.combine(observed, weights).sum() == pytest.approx(0.98)
    assert not np.allclose(uniform, ordinal)
    instant, _ = rules.source.choose_pair(observed, np.full(7, 1 / 7))
    assert (instant > 0).sum() == 2


def test_tied_rank_points_are_averaged_without_lexical_weight_bias(monkeypatch, observed):
    profile(monkeypatch, [3.0, 3.0, 1.0])
    _, evidence = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[1])
    points = [row["aggregation_points"] for row in evidence["pair_masses"]]
    assert points == [2.5, 2.5, 1.0]
    assert evidence["pair_masses"][0]["pair_mass"] == evidence["pair_masses"][1]["pair_mass"]
    profile(monkeypatch, [1.0, 1.0, 1.0])
    uniform, _ = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[0])
    ordinal, evidence = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[1])
    np.testing.assert_array_equal(uniform, ordinal)
    assert [row["aggregation_points"] for row in evidence["pair_masses"]] == [2.0] * 3


def test_negative_valid_scores_keep_positive_full_support(monkeypatch, observed):
    profile(monkeypatch, [-3.0, -1.0, -2.0])
    for scheme in rules.SCHEMES:
        weights, evidence = rules.select(observed, np.full(7, 1 / 7), scheme)
        assert weights.sum() == pytest.approx(1)
        assert evidence["scorable_pair_count"] == 3
        assert all(row["pair_mass"] > 0 for row in evidence["pair_masses"])
    assert evidence["aggregation_point_total"] == 6


@pytest.mark.parametrize(
    "reason", ["unscored_initial_equal_weights", "unscored_degenerate_pair_equal_weights"]
)
def test_original_initial_and_degenerate_states_are_unscored(monkeypatch, observed, reason):
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda items, previous: (
            np.full(7, 1 / 7),
            {"scores_available": False, "reason": reason},
        ),
    )
    for scheme in rules.SCHEMES:
        weights, evidence = rules.select(observed, np.full(7, 1 / 7), scheme)
        np.testing.assert_array_equal(weights, np.full(7, 1 / 7))
        assert evidence["reason"] == reason and evidence["pair_masses"] == []
        assert evidence["aggregation_point_total"] is None
        assert evidence["scorable_pair_count"] == 0


def test_actual_inputs_survive_a_failing_new_ensemble(monkeypatch):
    experts, calendar, assets = archived_experts()
    records = []

    def blocked(*args):
        raise QuantError("Synthetic ensemble assessment failure.")

    monkeypatch.setattr(rules.source, "choose_pair", blocked)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1 and records[0]["record"] == "decision_input"
    assert len(records[0]["observations"]) == 7
    assert records[0]["previous_weights_binary64_hex"] == [float(1 / 7).hex()] * 7
    assert records[0]["binary64_values_truncated_to_fixed_decimals"] is False


def test_prefix_causality_and_separate_ensemble_states():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert set(full) == {candidate["id"] for candidate in rules.CANDIDATES}
    for rows in audits.values():
        assert rows[0]["decision_session"] == "2016-10-05"
        assert sum(row["pair_ensemble"]["scorable_pair_count"] == 0 for row in rows) == 7
        assert all(row["decision_information_clock"] == rules.CLOCK for row in rows)
        assert all(row["pair_ensemble"]["scorable_pair_count"] == 21 for row in rows[7:])
        assert all(sum(weight > 0 for weight in row["expert_weights"]) == 7 for row in rows)
    assert (
        audits[next(iter(full))][8]["previous_expert_weights"]
        != audits[list(full)[1]][8]["previous_expert_weights"]
    )
    for end in ("2016-10-05", "2017-04-28", "2017-10-18"):
        early, before = rules.build_targets(
            experts, calendar[calendar <= end], assets, read_json(rules.POLICY)
        )
        for name in full:
            pd.testing.assert_frame_equal(early[name], full[name].loc[:end])
            assert before[name] == [row for row in audits[name] if row["decision_session"] <= end]
    changed, _, _ = archived_experts(changed_after=pd.Timestamp("2017-04-28"))
    later, changed_audits = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    for name in full:
        pd.testing.assert_frame_equal(full[name].loc[:"2017-04-28"], later[name].loc[:"2017-04-28"])
        assert [row for row in audits[name] if row["decision_session"] <= "2017-04-28"] == [
            row for row in changed_audits[name] if row["decision_session"] <= "2017-04-28"
        ]


def test_unscorable_pair_receives_no_successful_mass(monkeypatch, observed):
    pairs = profile(monkeypatch, [3.0, 2.0, 1.0])
    pairs[1] = {"pair": pairs[1]["pair"], "unscorable_reason": "zero_training_variance"}
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda items, previous: (
            np.asarray(pairs[0]["expert_weights"]),
            {"pairs": pairs, "selected_pair": pairs[0]["pair"], "scores_available": True},
        ),
    )
    _, evidence = rules.select(observed, np.full(7, 1 / 7), rules.SCHEMES[1])
    assert evidence["scorable_pair_count"] == 2
    assert [row["pair_mass"] for row in evidence["pair_masses"]] == pytest.approx([2 / 3, 1 / 3])
    assert len(evidence["pair_assessments"]["pairs"]) == 3


def test_missing_expert_is_not_dropped(observed):
    with pytest.raises(QuantError, match="full-support"):
        rules.select(observed[:-1], np.full(6, 1 / 6), rules.SCHEMES[0])


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("decision_information_clock", "at_16_00_ET"),
    ],
)
def test_ensemble_cannot_retune_original_assessment_execution_or_clock(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="unchanged"):
        rules.validate(policy)
