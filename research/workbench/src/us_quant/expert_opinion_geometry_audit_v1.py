from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import lagged_spy_price_impact_expert_inclusion_v1 as pool
from us_quant.conditional_hedge_expert_v1 import UNIVERSE
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-opinion-geometry-audit-v1.json"
ID = "ssd_expert_opinion_geometry_audit_20261105_v1"
BUDGET, TOLERANCE = 0.98, 1e-12


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": ID,
        "source_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "lagged-spy-price-impact-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "c2229fe31184f00bbb2d5d31f1b0130afef8ac0d6f1e8de398c7b486038572ab",
        },
        "source_decisions": 121,
        "source_input_resolved_records": 242,
        "source_experts": 9,
        "source_assets": 14,
        "universe": list(UNIVERSE),
        "source_prior_journal_sequence": 1062,
        "source_prior_combined_completed": 267,
        "invested_target_budget": BUDGET,
        "target_numeric_tolerance": TOLERANCE,
        "numerical_rank_tolerance_floor": TOLERANCE,
        "relative_rank_tolerance": (
            "max_rows_columns_times_binary64_epsilon_times_full_centered_largest_singular_value"
        ),
        "same_date_rank_comparison": "common_tolerance_first_eight_vs_all_nine",
        "exact_rank": "rational_elimination_of_saved_binary64_coordinate_differences",
        "identical_opinion": "exact_numeric_binary64_coordinates_signed_zero_equal",
        "pair_distance": "asset_target_L1_not_executed_turnover",
        "pair_overlap": "sum_min_ETF_weights_divided_by_0.98",
        "unscored_origins": "retain_geometry_selected_pair_not_applicable",
        "future_returns_or_financial_outcomes_used_as_labels": False,
        "rank_is_independent_economic_risk_or_future_diversification": False,
        "new_membership_selection_or_financial_gate": False,
        "source_scores_targets_accounts_or_benchmarks_replayed": False,
        "new_financial_evaluations": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[k]) is not type(v) for k, v in expected.items())
    ):
        raise QuantError(
            "Freeze one descriptive opinion-geometry convention, not a selection rule."
        )


def _exact_affine_rank(rows):
    matrix = [[v - origin for v, origin in zip(row, rows[0], strict=True)] for row in rows[1:]]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for i in range(rank + 1, len(matrix)):
            multiplier = matrix[i][column]
            if multiplier:
                matrix[i] = [
                    value - multiplier * anchor
                    for value, anchor in zip(matrix[i], matrix[rank], strict=True)
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _checked_opinions(opinions):
    matrix = np.asarray(opinions)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 1
        or matrix.dtype.kind not in "fi"
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or any(abs(math.fsum(float(v) for v in row) - BUDGET) > TOLERANCE for row in matrix)
    ):
        raise QuantError("Geometry needs complete finite long-only funded current opinions.")
    return matrix.astype(float, copy=False)


def inspect_opinions(opinions, selected_pair=None):
    matrix = _checked_opinions(opinions)
    count = len(matrix)
    if selected_pair is not None and (
        not isinstance(selected_pair, tuple)
        or len(selected_pair) != 2
        or any(type(i) is not int or not 0 <= i < count for i in selected_pair)
        or len(set(selected_pair)) != 2
    ):
        raise QuantError("A selected pair is two actual source indices, or explicitly unavailable.")
    rational = [[Fraction.from_float(float(v)) for v in row] for row in matrix]
    try:
        singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        first = matrix[:-1]
        prior_singular = np.linalg.svd(first - first.mean(axis=0), compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise QuantError("Failed numerical rank cannot receive a successful default.") from error
    tolerance = max(TOLERANCE, max(matrix.shape) * np.finfo(float).eps * float(singular[0]))
    full_rank = int((singular > tolerance).sum())
    prior_rank = int((prior_singular > tolerance).sum())
    exact_full, exact_prior = _exact_affine_rank(rational), _exact_affine_rank(rational[:-1])
    if (
        not np.isfinite(singular).all()
        or not np.isfinite(prior_singular).all()
        or not 0 <= prior_rank <= full_rank <= exact_full <= count - 1
        or prior_rank > exact_prior
        or full_rank - prior_rank not in (0, 1)
        or exact_full - exact_prior not in (0, 1)
    ):
        raise QuantError("Numerical and exact rank bounds disagree; do not change the tolerance.")
    groups = {}
    for index, row in enumerate(matrix):
        groups.setdefault(tuple(float(v) for v in row), []).append(index)
    pairs, max_gap = [], 0.0
    for a, b in itertools.combinations(range(count), 2):
        distance = math.fsum(
            abs(float(x) - float(y)) for x, y in zip(matrix[a], matrix[b], strict=True)
        )
        overlap = (
            math.fsum(min(float(x), float(y)) for x, y in zip(matrix[a], matrix[b], strict=True))
            / BUDGET
        )
        exact_distance = float(
            sum(abs(x - y) for x, y in zip(rational[a], rational[b], strict=True))
        )
        exact_overlap = float(
            sum(min(x, y) for x, y in zip(rational[a], rational[b], strict=True))
            / Fraction.from_float(BUDGET)
        )
        gap = max(abs(distance - exact_distance), abs(overlap - exact_overlap))
        if gap > TOLERANCE or not 0 <= distance <= 2 * BUDGET + TOLERANCE:
            raise QuantError("Independent scalar opinion distance/overlap does not reconcile.")
        if not -TOLERANCE <= overlap <= 1 + TOLERANCE:
            raise QuantError("A normalized opinion overlap is outside its funded bounds.")
        max_gap = max(max_gap, gap)
        pairs.append(
            {
                "pair": [a, b],
                "L1_target_distance": distance,
                "normalized_target_overlap": overlap,
                "exact_same_opinion": rational[a] == rational[b],
            }
        )
    selected = None
    if selected_pair is not None:
        selected = next(row for row in pairs if row["pair"] == sorted(selected_pair))
    return {
        "source_opinion_count": count,
        "distinct_current_opinions": len(groups),
        "identical_opinion_groups": [values for values in groups.values() if len(values) > 1],
        "first_sources_distinct_opinions": len(
            {tuple(float(v) for v in row) for row in matrix[:-1]}
        ),
        "exact_binary64_affine_rank": exact_full,
        "first_sources_exact_binary64_affine_rank": exact_prior,
        "numerical_affine_rank": full_rank,
        "first_sources_numerical_affine_rank": prior_rank,
        "common_numerical_rank_tolerance": tolerance,
        "centered_singular_values": singular.tolist(),
        "first_sources_centered_singular_values": prior_singular.tolist(),
        "extra_source_adds_numerical_direction": full_rank > prior_rank,
        "extra_source_adds_exact_binary64_direction": exact_full > exact_prior,
        "extra_source_exact_matches": [i for i in range(count - 1) if rational[i] == rational[-1]],
        "all_current_opinion_pairs": pairs,
        "saved_selected_pair_geometry": selected,
        "maximum_independent_scalar_gap": max_gap,
        "rank_or_overlap_is_independent_economic_risk": False,
        "distance_is_executed_turnover_or_fee": False,
    }


def retained_case(before, after, audit, index, day, ids, mechanisms, assets):
    if (
        before.get("record") != "decision_input"
        or after.get("record") != "decision_resolved"
        or before.get("decision_index") != index
        or after.get("decision_index") != index
        or before.get("as_of") != day
        or before.get("information_clock") != pool.CLOCK
        or after.get("audit") != audit
        or audit.get("decision_session") != day
        or audit.get("decision_information_clock") != pool.CLOCK
        or tuple(after.get("asset_order", ())) != assets
        or before.get("binary64_values_truncated_to_fixed_decimals") is not False
    ):
        raise QuantError("Retain exact alternating current-input/resolved source records.")
    observations = before.get("observations")
    if not isinstance(observations, list) or len(observations) != len(ids):
        raise QuantError("Do not drop an unavailable or inconvenient source opinion.")
    opinions, known_at = [], []
    for i, observation in enumerate(observations):
        known = pd.Timestamp(observation["target_known_at"])
        introduced = pd.Timestamp(observation["introduced_at"])
        if (
            observation["id"] != ids[i]
            or observation["mechanism_group"] != mechanisms[i]
            or observation["as_of"] != day
            or tuple(observation["asset_order"]) != assets
            or pd.isna(known)
            or known.tzinfo is not None
            or known > pd.Timestamp(day)
            or pd.isna(introduced)
            or introduced.tzinfo is None
            or observation["introduced_at"] != audit["preserved_archive_identity_introduced_at"][i]
            or observation["target_known_at"] != audit["expert_target_known_at"][i]
            or type(observation["score_mature"]) is not bool
        ):
            raise QuantError("Source identity, known-at clock, asset scope or maturity changed.")
        opinions.append(observation["target"])
        known_at.append(observation["target_known_at"])
    matrix = _checked_opinions(opinions)
    if matrix.shape != (len(ids), len(assets)):
        raise QuantError("Every retained opinion needs the whole declared asset order.")
    selection = audit["selection"]
    mature = all(row["score_mature"] for row in observations)
    if type(selection["scores_available"]) is not bool or selection["scores_available"] != mature:
        raise QuantError("Retain unscored origins, not invented successful pair choices.")
    selected = selection["selected_pair"]
    if mature:
        menu = [tuple(row["pair"]) for row in selection["pairs"]]
        expected = {tuple(sorted(pair)) for pair in itertools.combinations(ids, 2)}
        if (
            len(menu) != len(expected)
            or set(menu) != expected
            or not isinstance(selected, list)
            or len(selected) != 2
            or tuple(selected) not in expected
        ):
            raise QuantError("Preserve all original pair records and the saved selected pair.")
        chosen = tuple(ids.index(name) for name in selected)
    else:
        if selected is not None or selection["pairs"]:
            raise QuantError("An unscored source decision cannot acquire a selected pair.")
        chosen = None
    return {
        "as_of": day,
        "source_decision_index": index,
        "opinion_binary64_hex": [[float(v).hex() for v in row] for row in matrix],
        "source_ids": list(ids),
        "asset_order": list(assets),
        "target_known_at": known_at,
        "saved_scores_available": mature,
        "saved_selected_pair_indices": list(chosen) if chosen is not None else None,
    }
