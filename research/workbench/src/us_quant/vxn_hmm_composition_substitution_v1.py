from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import hmm_expert_substitution_v1 as control
from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/vxn-hmm-composition-substitution-v1.json"
PRIOR_HEAD = "4d9979d6234629fcb9e342c87118431c4236d316082988d4ff30967d99ab0bde"
REVIEW_SHA = "9802990b58a874a8729a10262ec8c4697b5fc2df7f37529b6c16c6d29a18f788"
CONTROL_SHA = "e7921917cd7dc7efb667c15e8c451e6baa0dfe9bae51b8d7c9a4b341b19c346d"
EXPERT_IDS = (*control.EXPERT_IDS[:2], "hmm_vxn_relative_uncertainty_v1", *control.EXPERT_IDS[3:])
MECHANISMS, CLOCK = control.MECHANISMS, control.CLOCK
CANDIDATES = (
    {"id": "expert_pool_hmm_vxn_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_hmm_vxn_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    source = read_json(control.POLICY)
    control.validate(source)
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
        "study_id": "ssd_vxn_hmm_composition_substitution_20261104_v1",
        "source_expert_pool_policy_path": control.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 852,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "vxn_eligibility_completion_sha256": REVIEW_SHA,
        "matched_HMM_results": {
            "path": "evidence/hmm_expert_substitution_20261104_results_v1.json",
            "sha256": CONTROL_SHA,
        },
        "replaced_slot": 2,
        "replaced_expert_id": control.EXPERT_IDS[2],
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
        raise QuantError("Only the fixed HMM-family VXN composition source may change.")


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(prior.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if (
        len(experts) != 8
        or len(universe) != len(set(universe))
        or tuple(e.identity.identifier for e in experts) != EXPERT_IDS
    ):
        raise QuantError("Keep the exact eight-source substitution, not an appended variant.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
