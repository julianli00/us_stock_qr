from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.saved_inclusion_execution_v1 import _choice, _target

TOLERANCE = 1e-12


def inspect_intervention(base_pair, changed_pair, spec):
    before, after = base_pair
    changed, resolved = changed_pair
    assets = spec["assets"]
    old_ids, new_ids = tuple(spec["base"]["expert_ids"]), tuple(spec["current"]["expert_ids"])
    slot = spec["slot"]
    core = [assets.index(name) for name in spec["core_assets"]]
    noncore = [j for j in range(len(assets)) if j not in core]
    day, index = before["as_of"], before["decision_index"]
    if (
        len(old_ids) != 8
        or len(set(old_ids)) != 8
        or len(set(new_ids)) != 8
        or len(new_ids) != 8
        or slot != 2
        or old_ids[slot] == new_ids[slot]
        or any(old_ids[i] != new_ids[i] for i in range(8) if i != slot)
        or len(assets) != 14
        or len(set(assets)) != 14
        or len(core) != 4
        or before["record"] != changed["record"]
        or before["record"] != "decision_input"
        or after["record"] != resolved["record"]
        or after["record"] != "decision_resolved"
        or type(index) is not int
        or any(r["decision_index"] != index for r in (after, changed, resolved))
        or changed["as_of"] != day
        or after["asset_order"] != assets
        or resolved["asset_order"] != assets
        or any(
            r["binary64_values_truncated_to_fixed_decimals"] is not False for r in (before, changed)
        )
        or before["information_clock"] != changed["information_clock"]
        or len(before["observations"]) != 8
        or len(changed["observations"]) != 8
    ):
        raise QuantError("Use exact aligned complete eight-source retained decision pairs.")
    for pair, ids in ((base_pair, old_ids), (changed_pair, new_ids)):
        source, result = pair
        audit = result["audit"]
        if (
            audit["decision_session"] != day
            or audit["decision_information_clock"] != source["information_clock"]
            or source["previous_weights"] != audit["previous_rolling_expert_weights"]
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or np.asarray(audit["rolling_expert_weights"]).dtype.kind not in "fiu"
            or tuple(o["id"] for o in source["observations"]) != ids
        ):
            raise QuantError("Keep actual clocks, statistical state and capital separation.")
        for observation in source["observations"]:
            known = pd.Timestamp(observation["target_known_at"])
            dates = pd.DatetimeIndex(observation["return_dates"])
            if (
                observation["as_of"] != day
                or observation["asset_order"] != assets
                or pd.isna(known)
                or known.tzinfo is not None
                or known > pd.Timestamp(day)
                or type(observation["score_mature"]) is not bool
                or not dates.is_unique
                or not dates.is_monotonic_increasing
                or dates.tz is not None
                or (len(dates) and dates[-1] > pd.Timestamp(day))
                or (observation["score_mature"] and len(dates) != 126)
                or (audit["selection"]["scores_available"] and not observation["score_mature"])
            ):
                raise QuantError("A saved opinion must be available; scoring requires maturity.")
    left, right = before["observations"], changed["observations"]
    if (
        any(left[i] != right[i] for i in range(8) if i != slot)
        or left[slot]["mechanism_group"] != "conditional_variance_growth"
        or right[slot]["mechanism_group"] != left[slot]["mechanism_group"]
        or any(
            left[slot][key] != right[slot][key]
            for key in ("RF", "return_dates", "score_mature", "target_known_at")
        )
    ):
        raise QuantError("Only one related source changes; seven sources and native RF stay exact.")
    old_opinions = np.stack([_target(o["target"], len(assets)) for o in left])
    new_opinions = np.stack([_target(o["target"], len(assets)) for o in right])
    if not np.array_equal(old_opinions[slot, noncore], new_opinions[slot, noncore]):
        raise QuantError("The declared intervention cannot alter noncore source coordinates.")
    old_roll, old_pair, old_scored = _choice(after["audit"], old_ids)
    new_roll, new_pair, new_scored = _choice(resolved["audit"], new_ids)
    old_candidates = spec["base"]["candidate_ids"]
    new_candidates = spec["current"]["candidate_ids"]
    if (
        len(old_candidates) != 2
        or len(new_candidates) != 2
        or set(after["combined_targets"]) != set(old_candidates)
        or set(resolved["combined_targets"]) != set(new_candidates)
        or after["audit"]["equal_expert_weights"] != [0.125] * 8
        or resolved["audit"]["equal_expert_weights"] != [0.125] * 8
    ):
        raise QuantError("Retain both original equal/rolling rules and exact equal eighths.")
    mapped_old = None if old_pair is None else sorted(old_ids.index(name) for name in old_pair)
    mapped_new = None if new_pair is None else sorted(new_ids.index(name) for name in new_pair)
    results = {}

    def fraction(value):
        return Fraction.from_float(float(value))

    for offset, law in enumerate(("equal", "rolling")):
        old_weights = np.full(8, 0.125) if offset == 0 else old_roll
        new_weights = np.full(8, 0.125) if offset == 0 else new_roll
        old_target = _target(after["combined_targets"][old_candidates[offset]], len(assets))
        new_target = _target(resolved["combined_targets"][new_candidates[offset]], len(assets))
        channels = {name: [] for name in ("direct_opinion", "allocation_revision", "interaction")}
        max_saved = max_numpy = 0.0
        for j in range(len(assets)):
            delta_opinion = fraction(new_opinions[slot, j]) - fraction(old_opinions[slot, j])
            delta_weight = fraction(new_weights[slot]) - fraction(old_weights[slot])
            direct = fraction(old_weights[slot]) * delta_opinion
            allocation = sum(
                (fraction(new_weights[i]) - fraction(old_weights[i])) * fraction(old_opinions[i, j])
                for i in range(8)
            )
            interaction = delta_weight * delta_opinion
            old_implied = sum(
                fraction(old_weights[i]) * fraction(old_opinions[i, j]) for i in range(8)
            )
            new_implied = sum(
                fraction(new_weights[i]) * fraction(new_opinions[i, j]) for i in range(8)
            )
            if direct + allocation + interaction != new_implied - old_implied:
                raise QuantError("The exact rational saved-opinion intervention identity failed.")
            max_saved = max(
                max_saved,
                abs(float(old_implied - fraction(old_target[j]))),
                abs(float(new_implied - fraction(new_target[j]))),
                abs(
                    float(direct + allocation + interaction) - float(new_target[j] - old_target[j])
                ),
            )
            numpy_terms = (
                old_weights[slot] * (new_opinions[slot, j] - old_opinions[slot, j]),
                (new_weights - old_weights) @ old_opinions[:, j],
                (new_weights[slot] - old_weights[slot])
                * (new_opinions[slot, j] - old_opinions[slot, j]),
            )
            for name, value, independent in zip(
                channels, (direct, allocation, interaction), numpy_terms, strict=True
            ):
                channels[name].append(float(value))
                max_numpy = max(max_numpy, abs(float(value) - float(independent)))
        if max(max_saved, max_numpy) > TOLERANCE:
            raise QuantError("Saved targets and independent binary64 channel arithmetic diverge.")
        results[law] = {
            "channels": {
                name: {
                    "vector": vector,
                    "L1": math.fsum(abs(x) for x in vector),
                    "noncore_L1": math.fsum(abs(vector[j]) for j in noncore),
                }
                for name, vector in channels.items()
            },
            "saved_target_L1": math.fsum(abs(new_target - old_target)),
            "saved_targets_exactly_equal": bool(np.array_equal(new_target, old_target)),
            "noncore_saved_targets_exactly_equal": bool(
                np.array_equal(new_target[noncore], old_target[noncore])
            ),
            "statistical_weights_exactly_equal": bool(np.array_equal(old_weights, new_weights)),
            "old_slot2_weight": float(old_weights[slot]),
            "new_slot2_weight": float(new_weights[slot]),
            "maximum_saved_target_reconciliation_difference": max_saved,
            "maximum_numpy_channel_difference": max_numpy,
            "exact_rational_identity_passed": True,
            "target_L1_is_executed_turnover": False,
        }
    return {
        "decision_index": index,
        "decision_session": day,
        "old_scored": old_scored,
        "new_scored": new_scored,
        "mapped_selected_pairs_equal": mapped_old == mapped_new,
        "source_core_opinion_exactly_equal": bool(
            np.array_equal(old_opinions[slot], new_opinions[slot])
        ),
        "other_seven_observations_exact": True,
        "source_noncore_coordinates_exact": True,
        "laws": results,
        "causal_alpha_or_main_PnL_attributed": False,
    }
