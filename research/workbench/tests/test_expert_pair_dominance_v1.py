import copy
import runpy

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_pair_dominance_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json

INCUMBENT = tuple(sorted(rules.EXPERT_IDS[:2]))
CHALLENGER = tuple(sorted(rules.EXPERT_IDS[2:4]))


def assessment_profile(monkeypatch, incumbent_scores=(1.0, 0.5), challenger_scores=(1.2, 0.7)):
    def pair(names, scores):
        weights = [0.0] * 7
        weights[rules.EXPERT_IDS.index(names[0])] = 0.4
        weights[rules.EXPERT_IDS.index(names[1])] = 0.6
        return {
            "pair": list(names),
            "expert_weights": weights,
            "train_net_risk_score": scores[0],
            "validation_net_risk_score": scores[1],
            "selection_score_key": round(min(scores), 12),
        }

    pairs = [pair(INCUMBENT, incumbent_scores), pair(CHALLENGER, challenger_scores)]
    best = sorted(pairs, key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
    record = {"pairs": pairs, "selected_pair": best["pair"], "scores_available": True}
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda observed, previous: (np.asarray(best["expert_weights"]), copy.deepcopy(record)),
    )
    return pairs, record


def test_pareto_and_empirical_envelope_gates_are_distinct(monkeypatch):
    assessment_profile(monkeypatch)
    _, pair, evidence = rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[0])
    assert pair == CHALLENGER and evidence["reason"] == "dominant_challenger_selected"
    _, pair, evidence = rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[1])
    assert pair == INCUMBENT and evidence["reason"] == "no_dominance_retain_incumbent"
    assessment_profile(monkeypatch, challenger_scores=(2.0, 1.5))
    for gate in rules.GATES:
        assert rules.select([], np.full(7, 1 / 7), INCUMBENT, gate)[1] == CHALLENGER


def test_strict_ties_do_not_switch_or_claim_confidence(monkeypatch):
    assessment_profile(
        monkeypatch,
        incumbent_scores=(1.0, 1.0),
        challenger_scores=(1.0000000000001, 1.0000000000001),
    )
    for gate in rules.GATES:
        _, pair, evidence = rules.select([], np.full(7, 1 / 7), INCUMBENT, gate)
        assert pair == INCUMBENT and evidence["qualified_challengers"] == []
        assert evidence["confidence_interval_or_significance_claimed"] is False


def test_incumbent_updates_ERC_without_freezing_recommendations(monkeypatch):
    pairs, record = assessment_profile(monkeypatch, challenger_scores=(0.9, 0.4))
    previous = np.asarray(pairs[0]["expert_weights"])
    changed = copy.deepcopy(pairs[0])
    a, b = [rules.EXPERT_IDS.index(name) for name in INCUMBENT]
    changed["expert_weights"][a], changed["expert_weights"][b] = 0.3, 0.7
    record["pairs"][0] = changed
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda observed, prev: (np.asarray(changed["expert_weights"]), record),
    )
    weights, pair, evidence = rules.select([], previous, INCUMBENT, rules.GATES[0])
    assert pair == INCUMBENT and not np.array_equal(weights, previous)
    assert evidence["reason"] == "no_dominance_retain_incumbent"


def test_initial_reset_and_unscorable_states_are_explicit(monkeypatch):
    pairs, record = assessment_profile(monkeypatch)
    assert rules.select([], np.full(7, 1 / 7), None, rules.GATES[0])[1] == CHALLENGER
    record["pairs"][0] = {"pair": list(INCUMBENT), "unscorable_reason": "zero_training_variance"}
    monkeypatch.setattr(
        rules.source,
        "choose_pair",
        lambda observed, previous: (np.asarray(pairs[1]["expert_weights"]), record),
    )
    _, pair, evidence = rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[0])
    assert pair == CHALLENGER and evidence["reason"] == "incumbent_unscorable_select_original_best"
    for reason in ("unscored_initial_equal_weights", "unscored_degenerate_pair_equal_weights"):
        monkeypatch.setattr(
            rules.source,
            "choose_pair",
            lambda observed, previous, reason=reason: (
                np.full(7, 1 / 7),
                {"scores_available": False, "reason": reason},
            ),
        )
        weights, pair, evidence = rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[0])
        assert pair is None and np.array_equal(weights, np.full(7, 1 / 7))
        assert evidence["reason"] == reason


def test_negative_valid_assessments_are_not_a_cash_cutoff(monkeypatch):
    assessment_profile(monkeypatch, incumbent_scores=(-1.0, -2.0), challenger_scores=(-0.9, -1.8))
    assert rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[0])[1] == CHALLENGER
    assert rules.select([], np.full(7, 1 / 7), INCUMBENT, rules.GATES[1])[1] == INCUMBENT


def archived_experts(changed_after=None, missing_initial=False):
    namespace = runpy.run_path(str(rules.ROOT / "tests/test_strategy_expert_pool_v5.py"))
    return namespace["archived_experts"](
        changed_after=changed_after, missing_initial=missing_initial
    )


def test_actual_inputs_survive_a_failing_new_assessment(monkeypatch):
    experts, calendar, assets = archived_experts()
    records = []

    def blocked(*args):
        raise QuantError("Synthetic new assessment failure.")

    monkeypatch.setattr(rules.source, "choose_pair", blocked)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1 and records[0]["record"] == "decision_input"
    assert records[0]["incumbent_pair"] is None and len(records[0]["observations"]) == 7
    assert records[0]["previous_weights_binary64_hex"] == [float(1 / 7).hex()] * 7


def test_prefix_causality_and_separate_rule_states():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert set(full) == {candidate["id"] for candidate in rules.CANDIDATES}
    for rows in audits.values():
        assert rows[0]["decision_session"] == "2016-10-05"
        assert (
            sum(
                row["switch_evidence"]["reason"] == "unscored_initial_equal_weights" for row in rows
            )
            == 7
        )
        assert rows[7]["switch_evidence"]["reason"] == "no_incumbent_select_original_best"
        assert all(row["decision_information_clock"] == rules.CLOCK for row in rows)
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


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("decision_information_clock", "16_00_ET"),
    ],
)
def test_retention_cannot_retune_original_assessment_execution_or_clock(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="unchanged"):
        rules.validate(policy)
