from __future__ import annotations

import numpy as np

from us_quant import equal_role_projection_v1 as allocation
from us_quant import protective_six_adaptive_hedge_v1 as source
from us_quant import protective_source_mass_audit_v1 as coordinates
from us_quant.nested_growth_allocation_audit_v1 import require
from us_quant.ssd_forecast_opportunity_gate_v1 import target_proof
from us_quant.storage import read_json


def verify_cached(targets, records, frozen):
    dates = source.decision_sessions(targets.index, read_json(source.POLICY))
    require(
        tuple(targets.columns) == coordinates.ASSETS
        and targets.dropna(how="all").index.equals(dates)
        and len(dates) == 121,
        "Verify all cached formal coordinates,not a regenerated or cropped target frame.",
    )
    maximum_weight = maximum_target = 0.0
    previous_source = None
    count = 0
    projected, cached_weights = [], []
    for index, pair in enumerate(records):
        require(index < len(dates), "An extra cached source pair cannot be ignored.")
        before, after = pair
        require(
            before["phase"] == "before_role_projection"
            and after["phase"] == "after_role_projection",
            "Use the original retained projection input and resolution pair.",
        )
        old_before, old_after = before["source_before"], before["source_after"]
        old = old_after["audit"]
        day = str(dates[index].date())
        require(
            old_before["decision_session"]
            == old_after["decision_session"]
            == old["decision_session"]
            == after["audit"]["decision_session"]
            == day
            and old["score_mature"] is (index >= 2)
            and old_before["state"]["complete_intervals"] == max(0, index - 2)
            and old["adaptive_state"]["complete_intervals"] == max(0, index - 1)
            and all(value <= day for value in old_before["known_target_dates"]),
            "Preserve actual source clocks and the two unscored origins.",
        )
        if previous_source is not None:
            require(
                old_before["state"] == previous_source["adaptive_state"]
                and old_before["previous_weights_hex"]
                == [float(v).hex() for v in previous_source["expert_weights"]],
                "Cached learning history cannot reset or acquire a new source update.",
            )
        previous_source = old
        opinions = np.stack(
            [
                coordinates._hex(row, len(coordinates.ASSETS))
                for row in old_before["known_targets_hex"]
            ]
        )
        require(opinions.shape == (6, 14), "Keep six complete source opinions.")
        for row in opinions:
            coordinates._target(row, 14)
        weights, _ = allocation.project(old["expert_weights"])
        require(
            [float(v).hex() for v in weights]
            == [float(v).hex() for v in after["audit"]["projected_expert_weights"]],
            "The archived new allocation must equal the unchanged frozen role rule.",
        )
        saved = coordinates._hex(after["unified_target_hex"], 14)
        actual = targets.loc[dates[index]].to_numpy()
        require(
            np.array_equal(saved.view(np.uint64), actual.view(np.uint64)),
            "The full stored target file must match every retained binary64 resolution.",
        )
        checked = allocation.independent(old["expert_weights"], weights, opinions, actual)
        maximum_weight = max(maximum_weight, checked["maximum_independent_weight_difference"])
        maximum_target = max(maximum_target, checked["maximum_independent_target_difference"])
        projected.append([float(v).hex() for v in weights])
        cached_weights.append(old["expert_weights"].copy())
        count += 1
    require(count == 121, "A partial cached projection cannot receive a completed proof.")
    prefixes = []
    for position in (0, 2, 12, 60, 120):
        for index, values in enumerate(cached_weights[: position + 1]):
            early, _ = allocation.project(values)
            require(
                [float(v).hex() for v in early] == projected[index],
                "Future cached inputs changed an earlier verified role projection.",
            )
        prefixes.append(str(dates[position].date()))
    return target_proof(
        frozen,
        {
            "decision_count": count,
            "actual_input_and_resolved_record_count": 2 * count,
            "initial_unscored_source_decisions": 2,
            "cached_mature_source_updates": 119,
            "prefix_truncations": prefixes,
            "maximum_independent_weight_difference": maximum_weight,
            "maximum_independent_target_difference": maximum_target,
            "all_exact_half_role_masses": True,
            "cached_partial_targets_verified_not_regenerated": True,
            "source_learning_state_reexecuted": False,
            "source_rate_loss_memory_or_labels_changed": False,
            "failed_V1_identity_reopened": False,
            "new_source_or_shadow_account_runs": 0,
        },
    )
