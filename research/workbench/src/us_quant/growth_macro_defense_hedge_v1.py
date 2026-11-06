from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import funded_meta_experts_v1 as admitted
from us_quant import macro_defense_hedge_trio_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/growth-macro-defense-hedge-v1.json"
GROWTH_ID = admitted.IDS[0]
EXPERT_IDS = (*prior.EXPERT_IDS, GROWTH_ID)
MECHANISMS = (*prior.MECHANISMS, admitted.GROUP)
CLOCK = prior.CLOCK
PRIOR_HEAD = "689bed6d71de7e4851b13380482bd5ae1f42c420ef74eb33876630dbe0555d73"
CANDIDATES = (
    {"id": "expert_pool_growth_macro_defense_hedge_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_growth_macro_defense_hedge_rolling_pair_v1",
        "weighting": "train_erc_validate_net",
    },
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    source = read_json(prior.POLICY)
    inherited = (
        "protocol_id",
        "initial_decision_session",
        "lookback_sessions",
        "training_sessions",
        "validation_sessions",
        "selected_experts",
        "selection_score_decimal_places",
        "shadow_reference_path",
        "reference_capital_usd",
        "switch_cost_bps_per_side",
        "switch_commission_per_changed_asset_usd",
        "switch_cost_amortization_sessions",
        "changed_asset_numeric_tolerance",
        "invested_target_budget",
        "cash_reserve",
        "actual_weight_trade_band",
        "minimum_economic_factor_families",
        "decision_information_clock",
    )
    expected = {
        **{k: source[k] for k in inherited},
        "schema_version": 1,
        "study_id": "ssd_growth_macro_defense_hedge_20261106_v1",
        "actual_native_event_seq": 178787,
        "prior_journal_sequence": 1297,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 287,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/growth-macro-defense-hedge-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "55cb59532952e14f3faae98cc11efccb170817259a821048b97f9440e4f139c1",
        },
        "original_trio_source_slots": list(prior.SLOTS),
        "growth_expert_id": GROWTH_ID,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "mature_pair_count": 6,
        "admitted_legacy_defensive_views_reused_unchanged": True,
        "all_source_native_RF_claimed_equal_without_view": False,
        "growth_nested_source_overlap_disclosed": True,
        "mechanism_labels_are_independent_risks": False,
        "opportunity_gate_threshold_or_participation_state_tuned": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "candidate_count_in_this_study": 2,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
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
            "Freeze one complete growth representative and the unchanged six-pair law."
        )


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return bound(prior.original.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 4 or len(universe) != len(set(universe)):
        raise QuantError("Use exactly four complete declared policies and unique coordinates.")
    return bound(prior.original.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
