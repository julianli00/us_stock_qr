from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_v5 import archived_experts

from us_quant import expert_target_consensus_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def observations():
    experts, _, _ = archived_experts()
    return [expert.observe(pd.Timestamp("2017-04-28"), 126) for expert in experts]


def with_rows(items, rows):
    return [
        replace(item, target=pd.Series(row, index=item.target.index))
        for item, row in zip(items, rows, strict=True)
    ]


def test_complete_policy_is_frozen_without_reopening_source_or_financial_gates():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["candidate_count_in_this_study"] == 1
    assert policy["primary_control"] == "expert_pool_seven_rolling_pair_v1"
    assert policy["secondary_context"] == "expert_pool_seven_equal_weight_v1"
    assert policy["new_source_requests"] == policy["new_shadow_account_runs"] == 0


def test_medoid_is_a_complete_representative_not_a_coordinatewise_sleeve():
    items = observations()
    center = [0.49, 0.49, 0.0, 0.0]
    rows = [
        center,
        [0.48, 0.50, 0.0, 0.0],
        [0.50, 0.48, 0.0, 0.0],
        center,
        [0.47, 0.51, 0.0, 0.0],
        [0.51, 0.47, 0.0, 0.0],
        [0.0, 0.0, 0.49, 0.49],
    ]
    items = with_rows(items, rows)
    weights, evidence = rules.select(items)
    assert evidence["selected_expert_ids"] == [rules.EXPERT_IDS[0], rules.EXPERT_IDS[3]]
    np.testing.assert_array_equal(weights, [0.5, 0, 0, 0.5, 0, 0, 0])
    combined = rules.source.combine(items, weights)
    np.testing.assert_array_equal(combined, center)
    independent, selected = rules.independent_target(items)
    np.testing.assert_array_equal(independent, combined)
    assert selected == evidence["selected_expert_ids"]
    assert evidence["target_distance_is_actual_main_turnover_or_risk_forecast"] is False


def test_identical_recommendations_share_all_votes_without_lexical_tie_break():
    items = observations()
    items = with_rows(items, [[0.49, 0.245, 0.1225, 0.1225]] * 7)
    weights, evidence = rules.select(items)
    np.testing.assert_array_equal(weights, np.full(7, 1 / 7))
    assert evidence["selected_expert_ids"] == list(rules.EXPERT_IDS)
    assert all(value["numerator"] == "0" for value in evidence["objective_exact_fractions"])


def test_exact_fraction_and_integer_oracles_agree_without_a_tolerance_scan():
    rng = np.random.default_rng(42)
    original = observations()
    for _ in range(12):
        rows = rng.dirichlet(np.ones(4), size=7) * 0.98
        items = with_rows(original, rows)
        weights, evidence = rules.select(items)
        expected, chosen = rules.independent_target(items)
        assert chosen == evidence["selected_expert_ids"]
        np.testing.assert_allclose(
            rules.source.combine(items, weights), expected, rtol=0, atol=1e-12
        )
        assert weights.sum() == pytest.approx(1)
        assert (expected >= 0).all() and expected.sum() == pytest.approx(0.98)


def test_current_target_law_never_ranks_or_fills_source_net_performance():
    items = observations()
    before, evidence = rules.select(items)
    changed = [
        replace(item, net_returns=item.net_returns + (index - 3) * 0.02)
        for index, item in enumerate(items)
    ]
    after, later = rules.select(changed)
    np.testing.assert_array_equal(before, after)
    assert later == evidence
    assert evidence["return_scores_used"] is False
    assert evidence["source_NET_or_RF_values_ranked_or_filled"] is False


def test_common_source_calendar_gaps_are_not_waived_for_a_return_free_rule():
    items = observations()
    items = [
        replace(
            item,
            net_returns=item.net_returns.drop(item.net_returns.index[10]),
            risk_free=item.risk_free.drop(item.risk_free.index[10]),
            score_mature=False,
        )
        for item in items
    ]
    with pytest.raises(QuantError, match="complete source coordinates"):
        rules.select(items)


def test_original_initial_anchor_does_not_invent_mature_return_scores():
    experts, calendar, assets = archived_experts()
    targets, audits = rules.build_targets(
        experts, calendar[calendar <= "2016-10-05"], assets, read_json(rules.POLICY)
    )
    first = audits[rules.CANDIDATES[0]["id"]][0]
    assert first["decision_session"] == "2016-10-05"
    assert first["consensus"]["source_score_mature"] == [False] * 7
    assert first["consensus"]["return_scores_used"] is False
    assert targets[rules.CANDIDATES[0]["id"]].dropna(how="all").shape[0] == 1


def test_prefix_and_future_target_or_net_changes_do_not_revise_prior_decisions():
    experts, calendar, assets = archived_experts()
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(experts, calendar, assets, policy)
    name = rules.CANDIDATES[0]["id"]
    for end in ("2016-10-05", "2017-04-28", "2017-10-18"):
        early, earlier = rules.build_targets(experts, calendar[calendar <= end], assets, policy)
        pd.testing.assert_frame_equal(early[name], full[name].loc[:end])
        assert earlier[name] == [r for r in audits[name] if r["decision_session"] <= end]
    changed, _, _ = archived_experts(changed_after=pd.Timestamp("2017-04-28"))
    later, changed_audits = rules.build_targets(changed, calendar, assets, policy)
    pd.testing.assert_frame_equal(later[name].loc[:"2017-04-28"], full[name].loc[:"2017-04-28"])
    assert [r for r in changed_audits[name] if r["decision_session"] <= "2017-04-28"] == [
        r for r in audits[name] if r["decision_session"] <= "2017-04-28"
    ]


def test_actual_targets_and_binary64_inputs_survive_a_failing_decision(monkeypatch):
    experts, calendar, assets = archived_experts()
    records = []

    def failed(*args):
        raise QuantError("Synthetic new consensus failure")

    monkeypatch.setattr(rules, "select", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1 and records[0]["record"] == "decision_input"
    assert len(records[0]["observations"]) == 7
    for record in records[0]["observations"]:
        assert [float.fromhex(value) for value in record["target_binary64_hex"]] == record["target"]


def test_unavailable_expert_is_never_dropped_or_filled():
    experts, calendar, assets = archived_experts(missing_initial=True)
    with pytest.raises(QuantError, match="all seven valid complete"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    with pytest.raises(QuantError, match="all seven valid complete"):
        rules.select(observations()[:-1])


@pytest.mark.parametrize("kind", ["future", "RF", "negative", "nan", "partial", "coordinates"])
def test_invalid_clock_RF_or_recommendation_stops_the_rule(kind):
    items = observations()
    item = items[-1]
    if kind == "future":
        items[-1] = replace(item, target_known_at=item.as_of + pd.Timedelta(days=1))
    elif kind == "RF":
        items[-1] = replace(item, risk_free=item.risk_free + 1e-8)
    elif kind == "coordinates":
        items[-1] = replace(item, target=item.target.iloc[::-1])
    else:
        row = item.target.copy()
        row.iloc[0] = {"negative": -0.01, "nan": np.nan, "partial": 0.0}[kind]
        items[-1] = replace(item, target=row)
    with pytest.raises(QuantError):
        rules.select(items)


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_decision_session", "2016-09-30"),
        ("aggregation", "L2_medoid"),
        ("tie_rule", "lexical_first"),
        ("include_residual_idle_cash_coordinate", False),
        ("return_ranking_or_score_maturity_drives_allocation", True),
        ("actual_weight_trade_band", 0.01),
        ("source_observation_lookback_sessions", 252),
        ("decision_information_clock", "at_16_00_ET"),
        ("candidate_count_in_this_study", True),
    ],
)
def test_no_metric_maturity_norm_band_or_clock_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError, match="exact target-consensus law"):
        rules.validate(policy)
