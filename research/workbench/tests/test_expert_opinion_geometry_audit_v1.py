from copy import deepcopy

import numpy as np
import pytest

from us_quant import expert_opinion_geometry_audit_v1 as audit
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_identical_opinions_are_not_multiple_exposure_directions():
    result = audit.inspect_opinions([[0.49, 0.49]] * 3, (0, 2))
    assert result["distinct_current_opinions"] == 1
    assert result["exact_binary64_affine_rank"] == result["numerical_affine_rank"] == 0
    assert result["saved_selected_pair_geometry"]["exact_same_opinion"] is True
    assert result["saved_selected_pair_geometry"]["normalized_target_overlap"] == 1


def test_affine_combination_is_distinct_without_an_added_direction():
    result = audit.inspect_opinions([[0.98, 0], [0, 0.98], [0.49, 0.49]], (0, 2))
    assert result["distinct_current_opinions"] == 3
    assert result["exact_binary64_affine_rank"] == result["numerical_affine_rank"] == 1
    assert result["extra_source_adds_numerical_direction"] is False
    assert result["saved_selected_pair_geometry"]["L1_target_distance"] == 0.98
    assert result["saved_selected_pair_geometry"]["normalized_target_overlap"] == 0.5


def test_new_vertex_adds_one_direction_without_an_independence_claim():
    result = audit.inspect_opinions(np.eye(3) * 0.98)
    assert result["numerical_affine_rank"] == result["exact_binary64_affine_rank"] == 2
    assert result["first_sources_numerical_affine_rank"] == 1
    assert result["extra_source_adds_numerical_direction"] is True
    assert result["saved_selected_pair_geometry"] is None
    assert result["rank_or_overlap_is_independent_economic_risk"] is False


def test_exact_binary64_noise_rank_is_not_a_new_numerical_direction():
    result = audit.inspect_opinions([[0.98, 0, 0], [0, 0.98, 0], [0.49, 0.49 - 1e-14, 1e-14]])
    assert result["exact_binary64_affine_rank"] == 2
    assert result["numerical_affine_rank"] == 1
    assert result["common_numerical_rank_tolerance"] == 1e-12


def test_asset_permutation_keeps_ranks_distances_and_overlaps():
    matrix = np.array([[0.6, 0.2, 0.18], [0.1, 0.5, 0.38], [0.2, 0.3, 0.48]])
    first, after = [audit.inspect_opinions(m, (0, 2)) for m in (matrix, matrix[:, ::-1])]
    assert first["numerical_affine_rank"] == after["numerical_affine_rank"]
    assert first["exact_binary64_affine_rank"] == after["exact_binary64_affine_rank"]
    assert first["all_current_opinion_pairs"] == after["all_current_opinion_pairs"]


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.98, 0]],
        [[True, False], [True, False]],
        [[np.nan, 0], [0.49, 0.49]],
        [[np.inf, 0], [0.49, 0.49]],
        [[-0.01, 0.99], [0.49, 0.49]],
        [[0.5, 0.5], [0.49, 0.49]],
    ],
)
def test_bad_opinions_are_explicit_errors(matrix):
    with pytest.raises(QuantError):
        audit.inspect_opinions(matrix)


@pytest.mark.parametrize("pair", [(0, 0), (0, 3), (False, 1), [0, 1]])
def test_selected_pair_is_saved_two_source_indices_or_unavailable(pair):
    with pytest.raises(QuantError):
        audit.inspect_opinions([[0.98, 0], [0.49, 0.49]], pair)


def fixture():
    ids, mechanisms, assets = ("a", "b", "c"), ("x", "y", "z"), ("one", "two")
    day = "2020-01-31"
    observed = [
        {
            "id": name,
            "mechanism_group": mechanisms[i],
            "introduced_at": "2026-11-05T00:00:00Z",
            "as_of": day,
            "target_known_at": day,
            "target": [0.49, 0.49],
            "asset_order": list(assets),
            "score_mature": False,
        }
        for i, name in enumerate(ids)
    ]
    saved = {
        "decision_session": day,
        "decision_information_clock": audit.pool.CLOCK,
        "preserved_archive_identity_introduced_at": [row["introduced_at"] for row in observed],
        "expert_target_known_at": [day] * 3,
        "selection": {"scores_available": False, "selected_pair": None, "pairs": []},
    }
    before = {
        "record": "decision_input",
        "decision_index": 0,
        "as_of": day,
        "information_clock": audit.pool.CLOCK,
        "observations": observed,
        "binary64_values_truncated_to_fixed_decimals": False,
    }
    after = {
        "record": "decision_resolved",
        "decision_index": 0,
        "audit": saved,
        "asset_order": list(assets),
    }
    return before, after, saved, ids, mechanisms, assets


def test_adapter_retains_unscored_availability_without_backdating_introduction():
    before, after, saved, ids, mechanisms, assets = fixture()
    result = audit.retained_case(before, after, saved, 0, "2020-01-31", ids, mechanisms, assets)
    assert result["saved_selected_pair_indices"] is None
    assert result["saved_scores_available"] is False
    assert len(result["opinion_binary64_hex"]) == 3


def test_adapter_consumes_the_saved_mature_pair_without_reranking():
    before, after, saved, ids, mechanisms, assets = fixture()
    for row in before["observations"]:
        row["score_mature"] = True
    saved["selection"] = {
        "scores_available": True,
        "selected_pair": ["a", "c"],
        "pairs": [{"pair": ["a", "b"]}, {"pair": ["a", "c"]}, {"pair": ["b", "c"]}],
    }
    result = audit.retained_case(before, after, saved, 0, "2020-01-31", ids, mechanisms, assets)
    assert result["saved_selected_pair_indices"] == [0, 2]
    saved["selection"]["pairs"].pop()
    with pytest.raises(QuantError):
        audit.retained_case(before, after, saved, 0, "2020-01-31", ids, mechanisms, assets)


@pytest.mark.parametrize("failure", ["future", "identity", "asset", "index", "maturity", "missing"])
def test_adapter_rejects_source_clock_identity_scope_and_missing_history(failure):
    before, after, saved, ids, mechanisms, assets = fixture()
    if failure == "future":
        before["observations"][0]["target_known_at"] = "2020-02-03"
    elif failure == "identity":
        before["observations"][0]["id"] = "replacement"
    elif failure == "asset":
        before["observations"][0]["asset_order"].reverse()
    elif failure == "index":
        after["decision_index"] = 1
    elif failure == "maturity":
        saved["selection"]["scores_available"] = True
    else:
        before["observations"].pop()
    with pytest.raises(QuantError):
        audit.retained_case(before, after, saved, 0, "2020-01-31", ids, mechanisms, assets)


def test_a_future_case_change_does_not_change_earlier_geometry():
    before, after, saved, ids, mechanisms, assets = fixture()
    first = audit.retained_case(before, after, saved, 0, "2020-01-31", ids, mechanisms, assets)
    changed = deepcopy(first)
    changed["opinion_binary64_hex"][-1] = [(0.98).hex(), (0.0).hex()]
    original = audit.inspect_opinions(
        [[float.fromhex(v) for v in row] for row in first["opinion_binary64_hex"]]
    )
    audit.inspect_opinions(
        [[float.fromhex(v) for v in row] for row in changed["opinion_binary64_hex"]]
    )
    assert original == audit.inspect_opinions(
        [[float.fromhex(v) for v in row] for row in first["opinion_binary64_hex"]]
    )


def test_policy_is_descriptive_and_cannot_turn_into_selection_or_tolerance_scan():
    policy = read_json(audit.POLICY)
    audit.validate(policy)
    for key, value in (
        ("numerical_rank_tolerance_floor", 1e-6),
        ("new_membership_selection_or_financial_gate", True),
        ("new_financial_evaluations", False),
        ("source_prior_combined_completed", 265),
    ):
        changed = deepcopy(policy)
        changed[key] = value
        with pytest.raises(QuantError):
            audit.validate(changed)
