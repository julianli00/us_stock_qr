from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as original
from us_quant.config import QuantError
from us_quant.saved_inclusion_execution_v1 import _choice, _target
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/retained-pair-equal-allocation-v1.json"
ID = "expert_pool_eight_retained_pair_equal_v1"
CANDIDATES = ({"id": ID, "weighting": "equal_halves_of_retained_original_causal_pair"},)
EXPERT_IDS = original.EXPERT_IDS
MECHANISMS = original.MECHANISMS
CLOCK = original.CLOCK
ASSETS = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)
PRIOR_HEAD = "23d74d3b1159ef455704df871d022433739b2e3fbd9940d586ac34434971a2dc"


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_retained_pair_equal_allocation_20261105_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 1211,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 280,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "retained-pair-equal-allocation-20261105-v1/source-input-catalogue-v1.json"
            ),
            "sha256": "f874d8776138a897ee38a8e01cd9ab26a786ae80daea96165a0b8e10fba0f5ae",
        },
        "advisor_candidate_id": original.CANDIDATES[1]["id"],
        "source_equal_candidate_id": original.CANDIDATES[0]["id"],
        "expert_ids": list(EXPERT_IDS),
        "asset_order": list(ASSETS),
        "initial_decision_session": "2016-10-05",
        "decision_information_clock": CLOCK,
        "source_lookback_sessions": 126,
        "source_training_sessions": 84,
        "source_validation_sessions": 42,
        "selected_pair_fraction": 0.5,
        "unscored_allocation": "unchanged_cached_source_equal8",
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "independent_target_tolerance": 1e-12,
        "source_advisor_state_recomputed_with_new_allocation": False,
        "advisor_cost_proxy_is_new_allocation_actual_cost": False,
        "source_selector_or_shadow_replayed": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Freeze one symmetric allocation law, not another score or parameter scan."
        )


def allocate_retained_pair(before, resolved, policy):
    validate(policy)
    day, index = before["as_of"], before["decision_index"]
    audit = resolved["audit"]
    observations = before["observations"]
    if (
        before["record"] != "decision_input"
        or resolved["record"] != "decision_resolved"
        or type(index) is not int
        or resolved["decision_index"] != index
        or audit["decision_session"] != day
        or before["information_clock"] != CLOCK
        or audit["decision_information_clock"] != CLOCK
        or before["binary64_values_truncated_to_fixed_decimals"] is not False
        or before["previous_weights"] != audit["previous_rolling_expert_weights"]
        or before["previous_weights_binary64_hex"]
        != [float(x).hex() for x in before["previous_weights"]]
        or tuple(o["id"] for o in observations) != EXPERT_IDS
        or tuple(o["mechanism_group"] for o in observations) != MECHANISMS
        or resolved["asset_order"] != list(ASSETS)
        or audit["equal_expert_weights"] != [0.125] * 8
        or audit["shadow_wealth_added_to_main_capital"] is not False
        or audit["source_feedback_floor_transferred_to_main_account"] is not False
        or audit["selection"]["safety_or_success_claimed"] is not False
        or np.asarray(audit["rolling_expert_weights"]).dtype.kind not in "fiu"
    ):
        raise QuantError("Keep the full original eight-source advisory trace and clock.")
    source_weights, pair, scored = _choice(audit, EXPERT_IDS)
    common_dates = observations[0]["return_dates"]
    common_rf = observations[0]["RF"]
    values_checked = 0
    for observation in observations:
        known = pd.Timestamp(observation["target_known_at"])
        dates = pd.DatetimeIndex(observation["return_dates"])
        net, rf = np.asarray(observation["NET_returns"]), np.asarray(observation["RF"])
        if (
            observation["as_of"] != day
            or observation["asset_order"] != list(ASSETS)
            or pd.isna(known)
            or known.tzinfo is not None
            or known > pd.Timestamp(day)
            or type(observation["score_mature"]) is not bool
            or observation["return_dates"] != common_dates
            or observation["RF"] != common_rf
            or not dates.is_unique
            or not dates.is_monotonic_increasing
            or dates.tz is not None
            or (len(dates) and dates[-1] > pd.Timestamp(day))
            or net.dtype.kind not in "fiu"
            or rf.dtype.kind not in "fiu"
            or net.shape != rf.shape
            or net.shape != (len(dates),)
            or not np.isfinite(net).all()
            or not np.isfinite(rf).all()
            or (scored and (not observation["score_mature"] or len(dates) != 126))
        ):
            raise QuantError("No future, missing, mismatched RF or unmatured score may be filled.")
        values_checked += net.size + rf.size
    opinions = np.stack([_target(o["target"], len(ASSETS)) for o in observations])
    if scored:
        slots = [EXPERT_IDS.index(name) for name in pair]
        weights = np.zeros(8)
        weights[slots] = 0.5
        target = (opinions[slots[0]] + opinions[slots[1]]) * 0.5
        reason = "equal_halves_of_original_mature_advisor_pair"
    else:
        weights = np.full(8, 0.125)
        target = _target(
            resolved["combined_targets"][policy["source_equal_candidate_id"]], len(ASSETS)
        ).copy()
        reason = "original_unscored_equal8_intention_not_a_successful_score"
    scalar = [
        float(
            sum(
                Fraction.from_float(float(w)) * Fraction.from_float(float(t))
                for w, t in zip(weights, opinions[:, j], strict=True)
            )
        )
        for j in range(len(ASSETS))
    ]
    gap = max(abs(float(a) - b) for a, b in zip(target, scalar, strict=True))
    if gap > policy["independent_target_tolerance"] or abs(math.fsum(target) - 0.98) > 1e-12:
        raise QuantError("Independent symmetric target identity or original cash budget failed.")
    return target, {
        "decision_index": index,
        "decision_session": day,
        "decision_information_clock": CLOCK,
        "source_advisor_id": policy["advisor_candidate_id"],
        "source_scores_available": scored,
        "source_selected_pair": pair,
        "source_previous_statistical_weights": before["previous_weights"],
        "source_training_ERC_weights": source_weights.tolist(),
        "new_execution_expert_weights": weights.tolist(),
        "allocation_reason": reason,
        "source_advisor_state_changed": False,
        "source_selector_replayed": False,
        "source_proxy_is_new_allocation_actual_cost": False,
        "NET_RF_binary64_values_validated": values_checked,
        "maximum_independent_rational_target_difference": gap,
        "source_shadow_wealth_or_floor_transferred": False,
        "independent_forward_validation": False,
        "order_authority": False,
    }


def decision_sessions(calendar):
    return original.decision_sessions(calendar, read_json(original.POLICY))
