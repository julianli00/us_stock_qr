from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/calendar-expert-inclusion-v1.json"
PRIOR_HEAD = "6989ac630b715b35a219f5a31e5b4bef9cbae62940119e638d001200d8bed8e7"
EXPERT_IDS = (*prior.EXPERT_IDS, "calendar_seasonal_six_core_bil_v1")
MECHANISMS = (*prior.MECHANISMS, "fixed_calendar_seasonal_equity_participation")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_calendar_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_calendar_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    original = read_json(prior.POLICY)
    prior.validate(original)
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
        **{key: original[key] for key in unchanged},
        "schema_version": 1,
        "study_id": "ssd_calendar_expert_inclusion_20261105_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 934,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 256,
        "calendar_source": {
            "schema_version": 1,
            "id": EXPERT_IDS[-1],
            "mechanism_group": MECHANISMS[-1],
            "introduced_at": "2026-11-05T03:33:31.698701+00:00",
            "registration": {
                "path": "evidence/calendar_seasonal_financial_20261105_registration_v1.json",
                "sha256": "be79bd82660e8db44bcf3ddbe6aacafbac5cc56dfe25a65cd84cf0394794c915",
            },
            "results": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "calendar-seasonal-financial-20261105-v1/results.json"
                ),
                "sha256": "28b821d38563c49a4a890e9b5ebd96348adab4cca2924d95832ad328b20f5b28",
            },
            "admission": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "calendar-seasonal-financial-20261105-v1/technical-admission-v1.json"
                ),
                "sha256": "31935ee32e2040e5d36d40e173f1d4a53e57c9a42cafd6ce4f650b1a1610e59c",
            },
            "completion": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "calendar-seasonal-financial-20261105-v1/completion-v1.json"
                ),
                "sha256": "201751479dc6572d882963a98936f4bdfbc017d36e52ff33edc814039c26dc7c",
            },
        },
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
        "calendar_financial_rejection_preserved": True,
        "new_RF_assessment_view_or_source_scope_waiver": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze the controlled calendar inclusion without changing any old law.")


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
        raise QuantError("Keep exactly the original eight plus the admitted calendar expert.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
