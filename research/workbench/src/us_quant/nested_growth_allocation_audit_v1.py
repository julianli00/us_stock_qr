from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from us_quant import growth_macro_defense_hedge_v1 as outer
from us_quant import strategy_expert_pool_eight_v1 as canonical
from us_quant.config import QuantError
from us_quant.expert_pair_menu_extension_v1 import _stats
from us_quant.expert_pair_menu_extension_v1 import _weights as saved_weights
from us_quant.saved_inclusion_execution_v1 import _choice, _target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/nested-growth-allocation-audit-v1.json"
ASSETS = outer.admitted.ASSETS
OUTER_IDS, CANONICAL_IDS = outer.EXPERT_IDS, canonical.EXPERT_IDS
DIRECT_SLOTS = outer.prior.SLOTS
IDS = tuple(c["id"] for c in outer.CANDIDATES)
CHILD_ID = outer.GROWTH_ID.removesuffix("_v2") + "_v1"
CLOCK = outer.CLOCK
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_nested_growth_allocation_audit_20261106_v1",
        "actual_native_event_seq": 179416,
        "prior_journal_sequence": 1320,
        "prior_journal_head_sha256": (
            "ef00d3856e2cd2fa0d18f9d8614a0d7e2ece8f8e64c2872cb94e0e09214ba41d"
        ),
        "outer_expert_ids": list(OUTER_IDS),
        "canonical_expert_ids": list(CANONICAL_IDS),
        "canonical_direct_slots": list(DIRECT_SLOTS),
        "outer_candidate_ids": list(IDS),
        "child_tape_candidate_id": CHILD_ID,
        "decision_information_clock": CLOCK,
        "unit_mass_and_target_tolerance": TOLERANCE,
        "support_uses_strict_positive_mass": True,
        "effective_number_is_independent_risk_count": False,
        "new_rankings_targets_shadow_accounts_or_financial_trials": 0,
        "source_algorithms_or_main_returns_recomputed": False,
        "allocation_mass_is_main_capital_PNL_or_causal_drawdown_attribution": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        isinstance(policy, dict)
        and set(policy) == set(fixed) | {"source_inputs"}
        and all(type(policy.get(k)) is type(v) and policy.get(k) == v for k, v in fixed.items()),
        "Freeze exact source nesting, original numeric limits and non-return interpretation.",
    )


def _weights(value, size):
    require(
        np.asarray(value).dtype.kind in "fiu",
        "Recorded allocation fractions cannot be coerced from strings or booleans.",
    )
    return saved_weights(value, size)


def _inputs(row):
    ob, oa = row["outer_before"], row["outer_after"]
    cb, ca = row["child_before"], row["child_after"]
    source = cb["source_before"]
    audit, child_audit = oa["audit"], ca["audit"]
    day, index = ob["as_of"], ob["decision_index"]
    require(
        ob["record"] == "decision_input"
        and oa["record"] == "decision_resolved"
        and cb["record"] == "saved_forecast_allocation_input"
        and ca["record"] == "saved_forecast_allocation_resolved"
        and oa["decision_index"]
        == source["decision_index"]
        == child_audit["decision_index"]
        == index
        and audit["decision_session"] == source["as_of"] == child_audit["decision_session"] == day
        and ob["information_clock"]
        == source["information_clock"]
        == audit["decision_information_clock"]
        == child_audit["decision_information_clock"]
        == CLOCK
        and ob["binary64_values_truncated_to_fixed_decimals"] is False
        and source["binary64_values_truncated_to_fixed_decimals"] is False
        and ob["previous_weights"] == audit["previous_rolling_expert_weights"]
        and ob["previous_weights_binary64_hex"] == [float(v).hex() for v in ob["previous_weights"]]
        and tuple(oa["asset_order"]) == tuple(ASSETS)
        and tuple(o["id"] for o in ob["observations"]) == OUTER_IDS
        and tuple(o["id"] for o in source["observations"]) == CANONICAL_IDS
        and audit["shadow_wealth_added_to_main_capital"] is False
        and audit["source_feedback_floor_transferred_to_main_account"] is False
        and child_audit["source_forecaster_refitted"] is False
        and child_audit["source_risk_weights_retrained"] is False
        and child_audit["source_shadow_wealth_or_floor_transferred"] is False,
        "Join the exact same-clock outer/child records, not another source state or capital.",
    )
    _weights(ob["previous_weights"], 4)
    parents, children = ob["observations"], source["observations"]
    for observations in (parents, children):
        require(
            all(
                o["as_of"] == day
                and o["target_known_at"] is not None
                and o["target_known_at"] <= day
                and tuple(o["asset_order"]) == tuple(ASSETS)
                and (not o["return_dates"] or o["return_dates"][-1] <= day)
                for o in observations
            ),
            "No unavailable or future source opinion may enter a look-through identity.",
        )
        for observed in observations:
            net, rf = np.asarray(observed["NET_returns"]), np.asarray(observed["RF"])
            count = len(observed["return_dates"])
            require(
                net.dtype.kind in "fiu"
                and rf.dtype.kind in "fiu"
                and net.shape == rf.shape == (count,)
                and count <= 126
                and np.isfinite(net).all()
                and np.isfinite(rf).all()
                and type(observed["score_mature"]) is bool
                and observed["score_mature"] is (count == 126),
                "Keep finite original NET/RF rows and explicit recorded score maturity.",
            )
    for position, slot in enumerate(DIRECT_SLOTS):
        require(
            parents[position] == children[slot],
            "A direct source must retain its exact opinion, clock, NET and RF observations.",
        )
    require(
        CHILD_ID in ca["targets"] and CHILD_ID in child_audit["candidates"],
        "Use the exact original V1 tape identity attested by the successful V2 source.",
    )
    source_targets = np.stack([_target(o["target"], len(ASSETS)) for o in children])
    child_target = _target(ca["targets"][CHILD_ID], len(ASSETS))
    require(
        np.array_equal(
            _target(parents[-1]["target"], len(ASSETS)).view(np.uint64),
            child_target.view(np.uint64),
        )
        and parents[-1]["target_known_at"] == day,
        "The recorded complete growth opinion must equal its attested child target bit-exactly.",
    )
    child = child_audit["candidates"][CHILD_ID]
    child_weights = _weights(child["expert_weights"], 8)
    require(type(child["scores_available"]) is bool, "Source score maturity must stay explicit.")
    if child["scores_available"]:
        pair = child["selected_pair"]
        require(
            isinstance(pair, list)
            and len(pair) == len(set(pair)) == 2
            and {CANONICAL_IDS[i] for i, v in enumerate(child_weights) if v > 0} == set(pair),
            "Preserve the recorded child's positive source pair; never rerank it.",
        )
    else:
        require(
            child["selected_pair"] is None and np.array_equal(child_weights, np.full(8, 0.125)),
            "An unscored child is its recorded uniform state, not a successful selected pair.",
        )
    equal = _weights(audit["equal_expert_weights"], 4)
    require(np.array_equal(equal, np.full(4, 0.25)), "Keep the original equal-quarter control.")
    rolling = _weights(audit["rolling_expert_weights"], 4)
    _choice(audit, OUTER_IDS)
    require(
        set(oa["combined_targets"]) == set(IDS),
        "Retain both original complete outer laws, not a chosen passing source.",
    )
    return source_targets, child_weights, (equal, rolling), audit, child


def _summary(weights):
    hhi = math.fsum(float(v) ** 2 for v in weights)
    return {
        "strict_positive_support": int(np.count_nonzero(weights > 0)),
        "allocation_HHI": hhi,
        "effective_allocation_number": 1 / hhi,
        "maximum_canonical_fraction": float(max(weights)),
    }


def decompose(row, policy):
    validate(policy)
    targets, child, outer_weights, audit, child_audit = _inputs(row)
    result = {
        "decision_session": audit["decision_session"],
        "decision_index": row["outer_before"]["decision_index"],
        "outer_scores_available": audit["selection"]["scores_available"],
        "child_scores_available": child_audit["scores_available"],
        "direct_source_observations_exactly_matched": 3,
        "growth_source_target_bit_exact": True,
        "candidates": {},
        "new_rankings_targets_or_financial_trials": 0,
        "main_PNL_or_causal_drawdown_attribution": False,
        "effective_number_is_independent_risk_count": False,
    }
    for name, weights in zip(IDS, outer_weights, strict=True):
        direct = np.zeros(8)
        direct[list(DIRECT_SLOTS)] = weights[:3]
        indirect = weights[-1] * child
        expanded = np.array([math.fsum((d, b)) for d, b in zip(direct, indirect, strict=True)])
        saved = _target(row["outer_after"]["combined_targets"][name], len(ASSETS))
        reconstructed = np.array(
            [
                math.fsum(float(expanded[i]) * float(targets[i, j]) for i in range(8))
                for j in range(len(ASSETS))
            ]
        )
        residual = float(max(abs(reconstructed - saved)))
        require(
            abs(math.fsum(expanded) - 1) <= TOLERANCE and residual <= TOLERANCE,
            "Look-through source mass and the retained ETF target must preserve original limits.",
        )
        result["candidates"][name] = {
            "outer_recorded_weights": weights.tolist(),
            "direct_canonical_mass": direct.tolist(),
            "growth_branch_canonical_mass": indirect.tolist(),
            "canonical_allocation_mass": expanded.tolist(),
            **_summary(expanded),
            "outer_strict_positive_support": int(np.count_nonzero(weights > 0)),
            "growth_branch_fraction": float(weights[-1]),
            "growth_branch_mass_on_trio_members": math.fsum(indirect[list(DIRECT_SLOTS)]),
            "canonical_members_with_both_channels_positive": int(
                np.count_nonzero((direct > 0) & (indirect > 0))
            ),
            "shared_channel_minimum_mass": math.fsum(np.minimum(direct, indirect)),
            "saved_underlying_target_max_residual": residual,
            "canonical_mass_is_statistical_intention_not_main_capital": True,
        }
    return result


def verify(row, result, policy):
    validate(policy)
    targets, child, allocations, _, _ = _inputs(row)
    require(
        result["decision_session"] == row["outer_before"]["as_of"]
        and result["decision_index"] == row["outer_before"]["decision_index"]
        and set(result["candidates"]) == set(IDS)
        and result["effective_number_is_independent_risk_count"] is False
        and result["main_PNL_or_causal_drawdown_attribution"] is False
        and result["new_rankings_targets_or_financial_trials"] == 0,
        "Keep both recorded laws and the explicitly non-capital interpretation.",
    )
    maximum_weight = maximum_target = 0.0
    for name, weights in zip(IDS, allocations, strict=True):
        f = Fraction.from_float
        direct = [f(0.0) for _ in range(8)]
        for i, slot in enumerate(DIRECT_SLOTS):
            direct[slot] = f(float(weights[i]))
        branch = [f(float(weights[-1])) * f(float(v)) for v in child]
        exact = [d + b for d, b in zip(direct, branch, strict=True)]
        recorded = result["candidates"][name]
        actual = _weights(recorded["canonical_allocation_mass"], 8)
        maximum_weight = max(
            maximum_weight,
            max(abs(float(x) - float(y)) for x, y in zip(actual, exact, strict=True)),
        )
        for i in range(8):
            require(
                abs(recorded["direct_canonical_mass"][i] - float(direct[i])) <= TOLERANCE
                and abs(recorded["growth_branch_canonical_mass"][i] - float(branch[i]))
                <= TOLERANCE,
                "The direct and nested channels cannot exchange or invent source mass.",
            )
        saved = _target(row["outer_after"]["combined_targets"][name], len(ASSETS))
        for j in range(len(ASSETS)):
            flat = sum(exact[i] * f(float(targets[i, j])) for i in range(8))
            nested = sum(
                f(float(weights[i])) * f(float(targets[slot, j]))
                for i, slot in enumerate(DIRECT_SLOTS)
            ) + f(float(weights[-1])) * sum(
                f(float(child[i])) * f(float(targets[i, j])) for i in range(8)
            )
            require(flat == nested, "Independent exact nested/flat allocation algebra disagrees.")
            maximum_target = max(maximum_target, abs(float(flat) - float(saved[j])))
        expected_hhi = float(sum(v * v for v in exact))
        require(
            recorded["strict_positive_support"] == sum(v > 0 for v in exact)
            and recorded["outer_recorded_weights"] == weights.tolist()
            and recorded["outer_strict_positive_support"] == int(np.count_nonzero(weights > 0))
            and recorded["growth_branch_fraction"] == float(weights[-1])
            and abs(
                recorded["growth_branch_mass_on_trio_members"]
                - float(sum(branch[i] for i in DIRECT_SLOTS))
            )
            <= TOLERANCE
            and abs(recorded["allocation_HHI"] - expected_hhi) <= TOLERANCE
            and abs(recorded["effective_allocation_number"] - 1 / expected_hhi) <= TOLERANCE
            and abs(recorded["maximum_canonical_fraction"] - float(max(exact))) <= TOLERANCE
            and recorded["canonical_members_with_both_channels_positive"]
            == sum(d > 0 and b > 0 for d, b in zip(direct, branch, strict=True))
            and abs(
                recorded["shared_channel_minimum_mass"]
                - float(sum(min(d, b) for d, b in zip(direct, branch, strict=True)))
            )
            <= TOLERANCE,
            "Support, overlap and concentration must match the independent exact source mass.",
        )
    require(
        maximum_weight <= TOLERANCE and maximum_target <= TOLERANCE,
        "Independent Fraction look-through exceeds the unchanged unit/target limits.",
    )
    return {
        "maximum_independent_canonical_weight_difference": maximum_weight,
        "maximum_independent_underlying_target_difference": maximum_target,
        "exact_symbolic_nested_flat_identity": True,
    }


def summarize(rows, checks):
    def summary(selected, name):
        values = [r["candidates"][name] for r in selected]
        return {
            "origins": len(values),
            "positive_canonical_support_counts": dict(
                sorted(Counter(str(v["strict_positive_support"]) for v in values).items())
            ),
            "allocation_HHI": _stats([v["allocation_HHI"] for v in values]),
            "effective_allocation_number": _stats(
                [v["effective_allocation_number"] for v in values]
            ),
            "maximum_canonical_fraction": _stats([v["maximum_canonical_fraction"] for v in values]),
            "shared_channel_minimum_mass": _stats(
                [v["shared_channel_minimum_mass"] for v in values]
            ),
            "growth_branch_mass_on_trio_members": _stats(
                [v["growth_branch_mass_on_trio_members"] for v in values]
            ),
            "origins_with_positive_shared_channels": sum(
                v["canonical_members_with_both_channels_positive"] > 0 for v in values
            ),
            "growth_branch_positive_origins": sum(v["growth_branch_fraction"] > 0 for v in values),
            "two_outer_labels_more_than_two_canonical_members": sum(
                v["outer_strict_positive_support"] == 2 and v["strict_positive_support"] > 2
                for v in values
            ),
        }

    mature = [r for r in rows if r["outer_scores_available"]]
    return {
        "matched_decisions": len(rows),
        "mature_outer_decisions": len(mature),
        "unscored_outer_decisions": len(rows) - len(mature),
        "exact_direct_source_observation_matches": 3 * len(rows),
        "exact_growth_target_matches": len(rows),
        "candidates": {
            name: {"all_origins": summary(rows, name), "mature_origins": summary(mature, name)}
            for name in IDS
        },
        "maximum_recorded_target_reconstruction_residual": max(
            r["candidates"][n]["saved_underlying_target_max_residual"] for r in rows for n in IDS
        ),
        "maximum_independent_canonical_weight_difference": max(
            c["maximum_independent_canonical_weight_difference"] for c in checks
        ),
        "maximum_independent_underlying_target_difference": max(
            c["maximum_independent_underlying_target_difference"] for c in checks
        ),
        "all_exact_symbolic_nested_flat_identities": all(
            c["exact_symbolic_nested_flat_identity"] for c in checks
        ),
        "effective_number_is_independent_risk_count": False,
        "allocation_mass_is_main_PNL_or_causal_drawdown_attribution": False,
        "alternative_rankings_targets_or_financial_trials": 0,
    }
