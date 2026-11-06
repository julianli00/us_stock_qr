from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import growth_macro_defense_hedge_v1 as prior
from us_quant import strategy_expert_pool_eight_v1 as admitted
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/commitment-macro-defense-hedge-v1.json"
SLOTS = (4, 5, 7, 0)
EXPERT_IDS = tuple(admitted.EXPERT_IDS[i] for i in SLOTS)
MECHANISMS = tuple(admitted.MECHANISMS[i] for i in SLOTS)
CLOCK = admitted.CLOCK
PRIOR_HEAD = "ef00d3856e2cd2fa0d18f9d8614a0d7e2ece8f8e64c2872cb94e0e09214ba41d"
CANDIDATES = (
    {"id": "expert_pool_commitment_macro_defense_hedge_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_commitment_macro_defense_hedge_rolling_pair_v1",
        "weighting": "train_erc_validate_net",
    },
)
choose_pair, combine = admitted.choose_pair, admitted.combine


def expected_policy():
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
        "mature_pair_count",
        "admitted_legacy_defensive_views_reused_unchanged",
        "all_source_native_RF_claimed_equal_without_view",
        "mechanism_labels_are_independent_risks",
        "new_source_requests",
        "new_shadow_account_runs",
        "new_source_RF_views",
        "candidate_count_in_this_study",
        "study_size_is_a_total_research_quota",
        "global_pool_changed",
        "independent_forward_validation",
        "investment_objective_verified",
        "order_authority",
    )
    return {
        **{k: source[k] for k in inherited},
        "schema_version": 1,
        "study_id": "ssd_commitment_macro_defense_hedge_20261106_v1",
        "actual_native_event_seq": 181713,
        "prior_journal_sequence": 1320,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 289,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/commitment-macro-defense-hedge-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "dd4a9171d8c563c48d28754030558930cede67ed1e3dbfd39851f2fb5af93c2f",
        },
        "original_source_slots": list(SLOTS),
        "direct_commitment_source_slot": 0,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "nested_growth_controller_included": False,
        "current_opinion_risk_forecast_promoted": False,
        "ranking_or_weighting_parameters_tuned": False,
        "candidates": list(CANDIDATES),
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Change only the predeclared direct growth-role representation.")


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return bound(prior.prior.original.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 4 or len(universe) != len(set(universe)):
        raise QuantError("Keep the four complete admitted policies and unique coordinates.")
    return bound(prior.prior.original.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
