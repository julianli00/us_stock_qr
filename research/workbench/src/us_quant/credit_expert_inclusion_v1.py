from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/credit-expert-inclusion-v1.json"
PRIOR_HEAD = "eb5bb7a2666237b47896315f6730dcc48188a796210bc42de92ce817b8445af7"
ADMISSION_SHA = "b54e416acffad52f22c95a2c71da6a8dafcf8e0c965d9ee2e7c25cf7ea2bb2f9"
COMPLETION_SHA = "c90820ae207cd536d4e46d6df8d62c4b6103ba7d30575803132f665812feb50d"
EXPERT_IDS = (*prior.EXPERT_IDS, "credit_quarter_style_defense_v1")
MECHANISMS = (*prior.MECHANISMS, "relative_credit_market_condition")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_credit_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_credit_rolling_pair_v1", "weighting": "train_erc_validate_net"},
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
        "study_id": "ssd_credit_expert_inclusion_20261104_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 887,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 252,
        "new_expert_admission": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "credit-quarter-complete-policy-20261104-v1/technical-admission-v1.json"
            ),
            "sha256": ADMISSION_SHA,
        },
        "admission_completion_sha256": COMPLETION_SHA,
        "matched_original_eight_results": {
            "path": "evidence/eight_expert_inclusion_20261103_results_v1.json",
            "sha256": "d3bce69f0e921ae6265218bbde6beeed58f8435caa20eaffb11ba9ba74808e7a",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "all_pairs_per_mature_decision": 36,
        "candidates": list(CANDIDATES),
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_model_or_target_runs": 0,
        "new_economic_factor_definitions": 0,
        "original_active_seven_or_selected_eight_rosters_changed": False,
        "nine_study_sources_claimed_independent_economic_risks": False,
        "credit_financial_rejection_preserved": True,
        "new_RF_assessment_view_or_source_Adapter_created": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze the deliberate credit inclusion and every original meta law.")


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(prior.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if (
        tuple(e.identity.identifier for e in experts) != EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in experts) != MECHANISMS
        or len(universe) != len(set(universe))
    ):
        raise QuantError("Append only the complete admitted credit policy to the original eight.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
