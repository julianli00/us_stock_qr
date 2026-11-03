from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import strategy_expert_pool_v5 as prior
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/strategy-expert-pool-eight-v1.json"
PRIOR_HEAD = "e7149cd97cb2a34d2317262b8dab764d4ab8a55d6341bfbbfe4c9865a2579374"
CLOCK = prior.CLOCK
EXPERT_IDS = (*prior.EXPERT_IDS, "conditional_covariance_hedge_expert_v1")
MECHANISMS = (*prior.MECHANISMS, "conditional_cross_asset_hedge_suitability")
CANDIDATES = (
    {"id": "expert_pool_eight_equal_weight_v1", "weighting": "equal"},
    {"id": "expert_pool_eight_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    source = read_json(prior.POLICY)
    if (
        file_digest(prior.POLICY)
        != "f27b1ac96253faef7ce57520682dac83ef426272057dfa2c277c61fcc319416f"
    ):
        raise QuantError("The original seven policy is immutable.")
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
        "study_id": "ssd_eight_expert_inclusion_20261103_v1",
        "prior_journal_sequence": 699,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_registered": 66,
        "prior_completed": 58,
        "prior_technical_failures": 8,
        "prior_funded_paths": 232,
        "prior_combined_completed": 236,
        "source_seven_policy": {
            "path": "config/strategy-expert-pool-seven-v1.json",
            "sha256": "f27b1ac96253faef7ce57520682dac83ef426272057dfa2c277c61fcc319416f",
        },
        "source_seven_results": {
            "path": "reports/ssd-continuous-research-v1/seven-expert-meta-20261031-v1/results.json",
            "sha256": "871f0b649ace186d0f2f53a0cbe71fc01dfe4dc3ca22349ff9ecc1c984af8223",
        },
        "new_expert_admission": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "conditional-hedge-expert-20261103-v1/technical-admission-v1.json"
            ),
            "sha256": "8160e46c0e92389abb6e3a380887223f2d2963eebd5f144f9a5290b815d91962",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ):
        raise QuantError("Only declared eight-source membership changes; keep every original law.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Study inclusion is not portfolio qualification or trading authority.")


def decision_sessions(calendar, policy):
    return FunctionType(
        prior.decision_sessions.__code__,
        {**prior.decision_sessions.__globals__, **globals()},
        argdefs=prior.decision_sessions.__defaults__,
    )(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 8 or len(universe) != len(set(universe)):
        raise QuantError("Use exactly the eight declared complete experts and unique coordinates.")
    return FunctionType(
        prior.build_targets.__code__,
        {**prior.build_targets.__globals__, **globals()},
        argdefs=prior.build_targets.__defaults__,
    )(experts, calendar, universe, policy, before_decision, retain)
