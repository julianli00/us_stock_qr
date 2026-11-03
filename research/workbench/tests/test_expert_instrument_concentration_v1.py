import copy
import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_expert_native_arrival_equal_v1 import ASSETS, examples

from us_quant import expert_instrument_concentration_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_duplicate_instrument_opinions_do_not_count_as_diversification():
    matrix = np.zeros((8, 2))
    matrix[:, 0] = 0.98
    matrix[1] = [0, 0.98]
    weights, audit = rules.choose_pair(matrix, None)
    assert weights[1] == 0.5
    assert audit["selected_concentration_key_12dp"] == 0.5
    assert len(audit["pair_records"]) == 28
    assert any(row["instrument_concentration"] == 1 for row in audit["pair_records"])
    assert all(row["switch_cost_proxy_usd"] == 0 for row in audit["pair_records"])


def test_all_pairs_match_separate_rational_combination_and_concentration():
    matrix = np.eye(8) * 0.98
    _, audit = rules.choose_pair(matrix, None)
    for row in audit["pair_records"]:
        names = row["pair"]
        positions = [rules.EXPERT_IDS.index(name) for name in names]
        exact = [
            (Fraction(matrix[positions[0], j]) + Fraction(matrix[positions[1], j])) / 2
            for j in range(8)
        ]
        concentration = float(sum((x / Fraction(0.98)) ** 2 for x in exact))
        assert row["instrument_concentration"] == pytest.approx(concentration, abs=1e-15)
        assert row["concentration_key_12dp"] == round(concentration, 12)
        assert [float.fromhex(x) for x in row["combined_target_binary64_hex"]] == [
            float(x) for x in exact
        ]
    assert {tuple(row["pair"]) for row in audit["pair_records"]} == set(
        itertools.combinations(sorted(rules.EXPERT_IDS), 2)
    )


def test_cost_is_only_a_tie_break_and_a_real_prior_pair_can_persist():
    matrix = np.eye(8) * 0.98
    previous = np.zeros(8)
    previous[[2, 5]] = 0.5
    weights, audit = rules.choose_pair(matrix, previous)
    assert np.array_equal(weights, previous)
    assert audit["selected_switch_cost_key_12dp"] == 0
    matrix = np.zeros((8, 2))
    matrix[:, 0] = 0.98
    matrix[1] = [0, 0.98]
    previous[:] = 0
    previous[[0, 2]] = 0.5
    weights, audit = rules.choose_pair(matrix, previous)
    assert weights[1] == 0.5
    assert audit["selected_concentration_key_12dp"] == 0.5
    assert audit["selected_switch_cost_key_12dp"] > 0


def test_asset_coordinate_permutation_and_structural_zeros_preserve_selection():
    matrix = np.eye(8) * 0.98
    previous = np.zeros(8)
    previous[[2, 5]] = 0.5
    before, audit = rules.choose_pair(matrix, previous)
    after, permuted = rules.choose_pair(matrix[:, ::-1], previous)
    extended, zeros = rules.choose_pair(np.column_stack([matrix, np.zeros((8, 2))]), previous)
    assert np.array_equal(before, after) and np.array_equal(before, extended)
    assert audit["selected_pair"] == permuted["selected_pair"] == zeros["selected_pair"]


def test_complete_identical_opinions_keep_explicit_lexical_ties():
    matrix = np.tile([0.98, 0.0], (8, 1))
    _, audit = rules.choose_pair(matrix, None)
    assert audit["selected_pair"] == sorted(rules.EXPERT_IDS)[:2]
    assert audit["selected_concentration_key_12dp"] == 1
    assert all(row["initial_no_previous_allocation"] for row in audit["pair_records"])


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((8, 2)),
        np.full((8, 2), np.nan),
        np.ones((7, 2)),
        np.full((8, 2), -0.49),
    ],
)
def test_invalid_opinions_cannot_be_normalized_into_safe_success(matrix):
    with pytest.raises(QuantError):
        rules.choose_pair(matrix, None)


def test_original_monthly_decision_clock_and_raw_input_retention():
    calendar, experts = examples()
    rows = []
    target, audits = rules.build_targets(
        experts, calendar, ASSETS, read_json(rules.POLICY), retain=rows.append
    )
    assert [row["decision_session"] for row in audits] == [
        "2016-10-05",
        "2016-10-31",
        "2016-11-30",
        "2016-12-30",
    ]
    assert [row["record"] for row in rows] == ["decision_input", "decision_resolved"] * 4
    assert rows[0]["previous_expert_weights_binary64_hex"] is None
    assert rows[0]["source_opinions"][0]["source_query_as_of"] == "2016-09-30"
    assert all(len(row["selection"]["pair_records"]) == 28 for row in audits)
    assert target.dropna().sum(axis=1).sub(0.98).abs().max() < 1e-12
    assert target.loc["2017-01-03":].isna().all().all()


def test_prefix_future_opinion_poison_and_source_return_scores():
    calendar, experts = examples()
    target, audits = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    for stamp in ("2016-10-05", "2016-10-31", "2016-11-30", "2016-12-30"):
        day = pd.Timestamp(stamp)
        early, earlier = rules.build_targets(
            experts, calendar[calendar <= day], ASSETS, read_json(rules.POLICY)
        )
        assert early.equals(target.loc[:day])
        assert earlier == [row for row in audits if row["decision_session"] <= stamp]
    for expert in experts:
        expert.return_level = 5.0
    same, unchanged = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    assert same.equals(target) and unchanged == audits
    experts[0].future_start = pd.Timestamp("2016-12-30")
    later, _ = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    assert later.loc[:"2016-12-29"].equals(target.loc[:"2016-12-29"])


def test_policy_cannot_add_return_score_weight_scan_or_risk_claim():
    policy = read_json(rules.POLICY)
    for key, value in (
        ("pair_fractions", [0.25, 0.75]),
        ("concentration_is_stock_overlap_independent_risk_or_drawdown_bound", True),
        ("source_return_RF_or_covariance_scores_used", True),
    ):
        changed = copy.deepcopy(policy)
        changed[key] = value
        with pytest.raises(QuantError):
            rules.validate(changed)
