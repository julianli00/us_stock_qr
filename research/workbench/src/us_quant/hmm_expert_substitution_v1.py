from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/hmm-expert-substitution-v1.json"
PRIOR_HEAD = "b7603b8870278a610275d11c6c5cf989bf22f5db8fa1f4825d545522ddf7bde5"
REVIEW_SHA = "cf714bb2f98270524c8d568f332d72b55d27b6f67c1602413acfee529c1abdaf"
EXPERT_IDS = (*prior.EXPERT_IDS[:2], "causal_two_state_growth_bil_v1", *prior.EXPERT_IDS[3:])
MECHANISMS, CLOCK = prior.MECHANISMS, prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_hmm_substitution_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_hmm_substitution_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    source = read_json(prior.POLICY)
    prior.validate(source)
    unchanged = (
        "protocol_id",
        "initial_decision_session",
        "lookback_sessions",
        "training_sessions",
        "validation_sessions",
        "selected_experts",
        "selection_score_decimal_places",
        "reference_capital_usd",
        "switch_cost_bps_per_side",
        "switch_commission_per_changed_asset_usd",
        "switch_cost_amortization_sessions",
        "changed_asset_numeric_tolerance",
        "shadow_reference_path",
        "invested_target_budget",
        "cash_reserve",
        "actual_weight_trade_band",
        "minimum_economic_factor_families",
        "decision_information_clock",
    )
    expected = {
        **{key: source[key] for key in unchanged},
        "schema_version": 1,
        "study_id": "ssd_hmm_representative_substitution_20261104_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 761,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "hmm_eligibility_completion_sha256": REVIEW_SHA,
        "replaced_slot": 2,
        "replaced_expert_id": prior.EXPERT_IDS[2],
        "replacement_expert_id": EXPERT_IDS[2],
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_model_or_target_runs": 0,
        "new_economic_factor_definitions": 0,
        "original_active_seven_or_available_eight_representatives_changed": False,
        "source_variants_are_independent_economic_factors": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Only the frozen HMM family-slot substitution may change.")


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(prior.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 8 or len(universe) != len(set(universe)):
        raise QuantError("Keep exactly eight complete policies and the original asset lattice.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
