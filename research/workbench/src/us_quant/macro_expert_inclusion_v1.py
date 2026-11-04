from __future__ import annotations

from pathlib import Path
from types import FunctionType

import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/macro-expert-inclusion-v1.json"
PRIOR_HEAD = "db82a9f525196460802554ea422ef73c09f23251d4009b4efdf782e7eaed1ea9"
ADMISSION_SHA = "edded165cfe279802a22270f0d18a517f76f0afde43da4f33fdeb37b166b0839"
COMPLETION_SHA = "6e7f49ddc2f78e3c1141a85da2f7df031265150fd40afdc67c7b80015b587560"
EXPERT_IDS = (*prior.EXPERT_IDS, "macro_real_yield_factor_tilt")
MECHANISMS = (*prior.MECHANISMS, "real_discount_rate_factor_composition")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_macro_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_macro_rolling_pair_v1", "weighting": "train_erc_validate_net"},
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
        "study_id": "ssd_macro_expert_inclusion_20261104_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 829,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "new_expert_admission": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "macro-compatible-expert-admission-20261104-v1/admission-v1.json"
            ),
            "sha256": ADMISSION_SHA,
        },
        "admission_completion_sha256": COMPLETION_SHA,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "macro_information_start": "2016-10-05",
        "macro_information_end": "2026-10-05",
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
        "macro_native_RF_claimed_exact_common": False,
        "new_macro_assessment_view_or_source_Adapter_created": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze the deliberate macro inclusion and every original law/scope.")


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(prior.prior.decision_sessions)(calendar, policy)


def macro_context(experts, day):
    if (
        len(experts) != 9
        or tuple(e.identity.identifier for e in experts) != EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in experts) != MECHANISMS
        or not pd.Timestamp("2016-10-05") <= day <= pd.Timestamp("2026-10-05")
    ):
        raise QuantError("Keep the exact nine-source pool within admitted macro information dates.")
    context = experts[-1].event_context(day)
    if (
        context["as_of"] != str(day.date())
        or context["observation_is_a_new_rebalance_request"] is not False
        or context["opinion_copied_from_actual_source_positions"] is not False
        or context["seed_removed_or_retimed_to_month_end"] is not False
        or context["all_native_seed_and_request_rows_retained"] is not True
        or context["original_policy_opinion_known_at"] is None
        or pd.Timestamp(context["original_policy_opinion_known_at"]) > day
    ):
        raise QuantError(
            "Do not turn policy observation or native capital seeds into source orders."
        )
    return context


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    dates = decision_sessions(calendar, policy)
    contexts = {str(day.date()): macro_context(experts, day) for day in dates}

    def keep(row):
        if row["record"] == "decision_input":
            row = {**row, "macro_policy_event_context": contexts[row["as_of"]]}
        retain(row)

    return _bound(prior.prior.build_targets)(
        experts,
        calendar,
        universe,
        policy,
        before_decision,
        keep if retain is not None else None,
    )
