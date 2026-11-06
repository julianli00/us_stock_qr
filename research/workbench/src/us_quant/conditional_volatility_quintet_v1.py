from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import commitment_macro_defense_hedge_v1 as prior
from us_quant import strategy_expert_pool_eight_v1 as admitted
from us_quant import strategy_expert_pool_v5 as calendar_source
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/conditional-volatility-quintet-v1.json"
SLOTS = (4, 5, 7, 0, 2)
EXPERT_IDS = tuple(admitted.EXPERT_IDS[i] for i in SLOTS)
MECHANISMS = tuple(admitted.MECHANISMS[i] for i in SLOTS)
CLOCK = admitted.CLOCK
PRIOR_HEAD = "fbc3b6d48b4e7aa508bf1a81c523114709870b104b558381ade7dc0990c3af4d"
CANDIDATES = (
    {"id": "expert_pool_conditional_volatility_quintet_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_conditional_volatility_quintet_rolling_pair_v1",
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
        "admitted_legacy_defensive_views_reused_unchanged",
        "all_source_native_RF_claimed_equal_without_view",
        "mechanism_labels_are_independent_risks",
        "nested_growth_controller_included",
        "current_opinion_risk_forecast_promoted",
        "ranking_or_weighting_parameters_tuned",
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
        "study_id": "ssd_conditional_volatility_quintet_20261106_v1",
        "actual_native_event_seq": 182832,
        "prior_journal_sequence": 1346,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 291,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/conditional-volatility-quintet-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "8d824fb5fecc9db1d65e3e7f369ca0701e92d9cfe29ffaede27e3c97a33694d3",
        },
        "original_source_slots": list(SLOTS),
        "volatility_source_slot": 2,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "mature_pair_count": 10,
        "total_target_cost_reference_promoted": False,
        "candidates": list(CANDIDATES),
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Append only the declared conditional-variance policy and ten-pair menu.")


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return bound(calendar_source.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 5 or len(universe) != len(set(universe)):
        raise QuantError("Use exactly five complete admitted policies and unique coordinates.")
    return bound(calendar_source.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
