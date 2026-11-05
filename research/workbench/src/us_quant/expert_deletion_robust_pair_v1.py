from __future__ import annotations

import itertools
import math
from types import FunctionType

import numpy as np

from us_quant import expert_distinct_opinion_pair_v1 as template
from us_quant.config import QuantError
from us_quant.ssd_expert_meta_research_v1 import scalar_score

ROOT = template.ROOT
POLICY = ROOT / "config/expert-deletion-robust-pair-v1.json"
ID = "expert_pool_nine_deletion_robust_pair_v1"
source = template.source
EXPERT_IDS, MECHANISMS, CLOCK, UNIVERSE = (
    template.EXPERT_IDS,
    template.MECHANISMS,
    template.CLOCK,
    template.UNIVERSE,
)
score_catalogue, combine = template.score_catalogue, template.combine
OWN = (
    "config/expert-deletion-robust-pair-v1.json",
    "src/us_quant/expert_deletion_robust_pair_v1.py",
    "tests/test_expert_deletion_robust_pair_v1.py",
    "EXPERT_DELETION_ROBUST_PAIR_V1.md",
)
FIXTURE_INPUTS = ("tests/test_expert_distinct_opinion_pair_v1.py",)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_expert_deletion_robust_pair_primary_20261105_v1",
        "candidate_id": ID,
        "source_pool_policy": {
            "path": "config/lagged-spy-price-impact-expert-inclusion-v1.json",
            "sha256": "d089bb061fc164dc328c00caa716d07bfb08ccf07d4b6620c817d815006f0118",
        },
        "source_cohort_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "lagged-spy-price-impact-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "c2229fe31184f00bbb2d5d31f1b0130afef8ac0d6f1e8de398c7b486038572ab",
        },
        "source_influence_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "expert-score-deletion-influence-20261105-v1/completion-v1.json"
            ),
            "sha256": "d515f1bf9b4ebb9122b658cd3e5486f4313e51973465adeed23efbc5fbe9c86c",
        },
        "prior_journal_sequence": 1074,
        "prior_journal_head_sha256": (
            "7a2074649982220a74e62d2339527d5353eb7138f671992a58d3755114f3b050"
        ),
        "prior_completed_configurations": 268,
        "expert_count": 9,
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "all_pair_catalogue_size": 36,
        "scorer": "lower_envelope_full_and_all_single_return_deleted_NET_excess_risk_scores",
        "pair_weights": "unchanged_full_training_ERC",
        "covariance_refitted_after_deletion": False,
        "nominal_switch_proxy_fixed_across_deletions": True,
        "switch_cost_amortization_sessions": 21,
        "score_decimal_places": 12,
        "exact_or_approximate_opinion_exclusion": False,
        "cold_start": "original_unscored_equal_nine",
        "no_admissible_pair": "explicit_equal_nine_known_opinions_not_safety_or_success",
        "constant_full_or_deleted_sample": "conditional_pair_unscorable_no_neutral_score",
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "decision_information_clock": CLOCK,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "global_pool_changed": False,
        "actual_registered_at_preparation": False,
        "first_real_policy_derivation_requires_actual_registration": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[k]) is not type(v) for k, v in expected.items())
    ):
        raise QuantError("Freeze one conditional robust ranking law without a parameter rescue.")


def _envelope(values, cost):
    if (
        len(values) != 126
        or not all(math.isfinite(v) for v in values)
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise QuantError("The robust score requires complete finite NET-excess operands.")

    def score(sample):
        return None if min(sample) == max(sample) else scalar_score(sample, cost)

    train, valid = values[:84], values[84:]
    first, second = score(train), score(valid)
    train_deleted = [score(train[:i] + train[i + 1 :]) for i in range(84)]
    valid_deleted = [score(valid[:i] + valid[i + 1 :]) for i in range(42)]
    all_scores = [first, second, *train_deleted, *valid_deleted]
    undefined = [i for i, value in enumerate(all_scores) if value is None]
    if any(v is not None and not math.isfinite(v) for v in all_scores):
        raise QuantError("A nonconstant conditional sample produced a nonfinite score.")
    lower = None if undefined else min(all_scores)
    return {
        "full_training_score": first,
        "full_validation_score": second,
        "training_one_return_deleted_scores": train_deleted,
        "validation_one_return_deleted_scores": valid_deleted,
        "undefined_sample_positions_full_train_valid_then_train_valid_deletions": undefined,
        "conditional_lower_envelope": lower,
        "conditional_lower_envelope_key": None if lower is None else round(lower, 12),
    }


def choose_pair(observed, previous):
    template._opinions(observed)
    if (
        not isinstance(previous, np.ndarray)
        or previous.shape != (9,)
        or previous.dtype.kind not in "fi"
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(math.fsum(float(v) for v in previous) - 1) > 1e-12
    ):
        raise QuantError("Keep the actual previous statistical weights, not main capital.")
    original_weights, original = score_catalogue(observed, previous)
    mature = all(o.score_mature for o in observed)
    added = {
        "NET_history_mature": mature,
        "same_state_original_catalogue_selected_pair": original["selected_pair"],
        "same_state_original_catalogue_scores_available": original["scores_available"],
        "pair_covariance_and_weights_refitted_after_deletion": False,
        "current_opinion_exclusion_used": False,
        "conditional_envelope_is_probability_or_risk_guarantee": False,
    }
    if not mature:
        return original_weights, {
            **original,
            **added,
            "conditionally_admissible_pairs": 0,
        }
    pairs = original["pairs"]
    expected = {
        tuple(sorted((EXPERT_IDS[i], EXPERT_IDS[j])))
        for i, j in itertools.combinations(range(9), 2)
    }
    if len(pairs) != 36 or {tuple(row["pair"]) for row in pairs} != expected:
        raise QuantError("Preserve every original scored or unscorable pair record.")
    values = [(o.net_returns - o.risk_free).tolist() for o in observed]
    records = []
    for pair in pairs:
        if "unscorable_reason" in pair:
            records.append(
                {
                    **pair,
                    "conditional_deletion_admissible": False,
                    "conditional_deletion_unscorable_reason": "original_"
                    + pair["unscorable_reason"],
                    "conditional_profile": None,
                }
            )
            continue
        weights = pair["expert_weights"]
        if (
            len(weights) != 9
            or not all(math.isfinite(v) and v >= 0 for v in weights)
            or abs(math.fsum(weights) - 1) > 1e-12
            or any(len(row) != 126 for row in values)
        ):
            raise QuantError("Keep original complete126-observation pair allocations.")
        portfolio = [math.fsum(weights[k] * values[k][t] for k in range(9)) for t in range(126)]
        profile = _envelope(portfolio, pair["estimated_incremental_switch_cost_fraction"])
        for actual, key in [
            (profile["full_training_score"], "train_net_risk_score"),
            (profile["full_validation_score"], "validation_net_risk_score"),
        ]:
            if actual is not None and abs(actual - pair[key]) > 1e-8:
                raise QuantError(
                    "The conditional full-sample scalar score differs from the original."
                )
        admissible = profile["conditional_lower_envelope"] is not None
        records.append(
            {
                **pair,
                "conditional_deletion_admissible": admissible,
                "conditional_deletion_unscorable_reason": (
                    None if admissible else "constant_full_or_single_return_deleted_sample"
                ),
                "conditional_profile": profile,
            }
        )
    usable = [p for p in records if p["conditional_deletion_admissible"]]
    result = {**original, **added, "pairs": records, "conditionally_admissible_pairs": len(usable)}
    if not usable:
        return np.full(9, 1 / 9), {
            **result,
            "reason": "no_conditionally_admissible_pair_explicit_equal_weights",
            "selected_pair": None,
            "scores_available": False,
            "safety_or_success_claimed": False,
        }
    chosen = min(
        usable,
        key=lambda row: (
            -row["conditional_profile"]["conditional_lower_envelope_key"],
            row["pair"],
        ),
    )
    return np.asarray(chosen["expert_weights"]), {
        **result,
        "reason": "conditional_single_return_deletion_lower_envelope",
        "selected_pair": chosen["pair"],
        "scores_available": True,
        "safety_or_success_claimed": False,
    }


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    # Bind the frozen one-policy builder locally; never mutate the old selector's namespace.
    builder = FunctionType(
        template.build_targets.__code__,
        {**template.build_targets.__globals__, **globals()},
        argdefs=template.build_targets.__defaults__,
    )
    return builder(experts, calendar, universe, policy, before_decision, retain)
