from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from us_quant import protective_six_adaptive_hedge_v1 as prior
from us_quant.nested_growth_allocation_audit_v1 import _weights, require

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/equal-role-projection-v1.json"
ID = "expert_pool_six_equal_role_projection_v1"
CANDIDATES = ({"id": ID, "weighting": "equal_economic_roles_cached_AdaHedge_ratios"},)
EXPERT_IDS, MECHANISMS, CLOCK = prior.EXPERT_IDS, prior.MECHANISMS, prior.CLOCK
GROUPS = ((0, 3, 4), (1, 2, 5))
TOLERANCE = 1e-12


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_equal_role_projection_20261106_v1",
        "actual_native_event_seq": 191685,
        "prior_journal_sequence": 1464,
        "prior_journal_head_sha256": (
            "0c971a26bbfebacf207e68e598b01c86af7d38680b30c21c2fc05647efb2d555"
        ),
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/equal-role-projection-20261106-v1/"
                "source-inputs-v1.json"
            ),
            "sha256": "48dfed6ab9181e1b918f380f9829a62c517bbf74b9ba9aa36d942d9607abfd68",
        },
        "growth_positions": list(GROUPS[0]),
        "response_positions": list(GROUPS[1]),
        "role_budgets": [0.5, 0.5],
        "zero_recorded_role_mass": "terminal_technical_stop_no_fallback",
        "source_learning_state_reexecuted": False,
        "source_rate_loss_memory_or_labels_changed": False,
        "within_role_recorded_belief_ratios_preserved": True,
        "role_budget_is_user_cap_or_fitted_ratio": False,
        "role_labels_are_independent_risks_or_protection_guarantees": False,
        "independent_weight_and_target_tolerance": TOLERANCE,
        "invested_target_budget": 0.98,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "global_pool_changed": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze exactly one equal-role capital rule and unchanged source learning.",
    )


def project(values):
    require(
        isinstance(values, list) and all(type(v) in (float, int) for v in values),
        "Recorded source beliefs cannot be coerced from strings or booleans.",
    )
    weights = _weights(values, 6)
    result, masses = np.zeros(6), []
    for members in GROUPS:
        mass = math.fsum(float(weights[i]) for i in members)
        require(
            math.isfinite(mass) and mass > 0,
            "A zero recorded role mass is a technical stop,not a safe fallback.",
        )
        masses.append(mass)
        for i in members:
            result[i] = 0.5 * float(weights[i]) / mass
    require(
        np.isfinite(result).all()
        and (result >= 0).all()
        and abs(math.fsum(result) - 1) <= TOLERANCE
        and all(abs(math.fsum(result[list(g)]) - 0.5) <= TOLERANCE for g in GROUPS),
        "Keep both half-capital roles on the original six-source simplex.",
    )
    return result, {
        "recorded_role_masses": masses,
        "projected_role_masses": [math.fsum(result[list(g)]) for g in GROUPS],
    }


def independent(values, projected, opinions, target):
    f = Fraction.from_float
    exact = [Fraction(0)] * 6
    for group in GROUPS:
        total = sum(f(float(values[i])) for i in group)
        require(total > 0, "Independent role normalization needs positive recorded mass.")
        for i in group:
            exact[i] = f(float(values[i])) / (2 * total)
    require(
        sum(exact) == 1 and all(sum(exact[i] for i in g) == Fraction(1, 2) for g in GROUPS),
        "Exact role budgets disagree.",
    )
    weight_error = max(abs(float(value) - float(projected[i])) for i, value in enumerate(exact))
    target_error = max(
        abs(float(sum(exact[i] * f(float(opinions[i, j])) for i in range(6))) - float(target[j]))
        for j in range(opinions.shape[1])
    )
    require(
        weight_error <= TOLERANCE and target_error <= TOLERANCE,
        "Independent role weights or unified target exceed original1e-12limits.",
    )
    return {
        "maximum_independent_weight_difference": weight_error,
        "maximum_independent_target_difference": target_error,
        "exact_half_role_masses": True,
    }
