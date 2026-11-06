from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import strategy_expert_pool_eight_v1 as original
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/macro-defense-hedge-trio-v1.json"
SLOTS = (4, 5, 7)
EXPERT_IDS = tuple(original.EXPERT_IDS[i] for i in SLOTS)
MECHANISMS = tuple(original.MECHANISMS[i] for i in SLOTS)
CLOCK = original.CLOCK
PRIOR_HEAD = "ba6b0523adcb75af4f78b23dcfd80b02bc56a8bf6d065e32efbae2a0e3c6ddd6"
CANDIDATES = (
    {"id": "expert_pool_macro_defense_hedge_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_macro_defense_hedge_rolling_pair_v1",
        "weighting": "train_erc_validate_net",
    },
)
choose_pair, combine = original.choose_pair, original.combine


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_macro_defense_hedge_trio_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 1223,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 281,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "macro-defense-hedge-trio-20261106-v1/source-input-catalogue-v1.json"
            ),
            "sha256": "b9a51251e8f945cfae50e1e2f75262df7be3a2f8c6e322cc2bc019953f8e8545",
        },
        "original_source_slots": list(SLOTS),
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "mature_pair_count": 3,
        "selection_score_decimal_places": 12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "switch_cost_amortization_sessions": 21,
        "changed_asset_numeric_tolerance": 1e-12,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "decision_information_clock": CLOCK,
        "admitted_legacy_defensive_views_reused_unchanged": True,
        "all_source_native_RF_claimed_equal_without_view": False,
        "macro_proxy_is_company_EPS_or_new_independent_factor": False,
        "mechanism_labels_are_independent_risks": False,
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
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Only declared three-role membership changes; keep the original law.")


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return bound(original.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 3 or len(universe) != len(set(universe)):
        raise QuantError("Use exactly three complete admitted experts and unique coordinates.")
    return bound(original.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
