from __future__ import annotations

from pathlib import Path
from types import FunctionType, SimpleNamespace

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/hmm-session-core-substitution-v1.json"
PRIOR_HEAD = "2ea7a7d9409b173a35896780a1e69644240dc6f0390df104a274e6eb52d5ce3b"
VARIANTS = ("close", "overnight", "intraday")
SOURCE_IDS = (
    "hmm_core_close_momentum_control_v1",
    "hmm_core_overnight_momentum_v1",
    "hmm_core_intraday_momentum_v1",
)
EXPERT_IDS = {
    variant: (*prior.EXPERT_IDS[:2], identifier, *prior.EXPERT_IDS[3:])
    for variant, identifier in zip(VARIANTS, SOURCE_IDS, strict=True)
}
MECHANISMS, CLOCK = prior.MECHANISMS, prior.CLOCK
CANDIDATES = tuple(
    {
        "id": f"expert_pool_hmm_session_{variant}_{label}_v1",
        "weighting": weight,
        "source_variant": variant,
    }
    for variant in VARIANTS
    for label, weight in (("equal", "equal"), ("rolling_pair", "train_erc_validate_net"))
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
        **{k: source[k] for k in unchanged},
        "schema_version": 1,
        "study_id": "ssd_hmm_session_core_substitution_20261105_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 1144,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 274,
        "source_eligibility": {
            "path": "evidence/archived_hmm_session_expert_eligibility_20261105_v1.json",
            "sha256": "cbf50ade4eb09361caff85cebc74f983fffd1560cff96063814b241462358a20",
        },
        "source_eligibility_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "archived-hmm-session-expert-eligibility-20261105-v1/completion-v1.json"
            ),
            "sha256": "2a5f4e876ff4c0c8ea642c222851da214b9c712771c1b7e0df0e292f036eff29",
        },
        "matched_cached_HMM_results": {
            "path": "evidence/hmm_expert_substitution_20261104_results_v1.json",
            "sha256": "e7921917cd7dc7efb667c15e8c451e6baa0dfe9bae51b8d7c9a4b341b19c346d",
        },
        "replaced_slot": 2,
        "replaced_expert_id": prior.EXPERT_IDS[2],
        "variant_source_ids": dict(zip(VARIANTS, SOURCE_IDS, strict=True)),
        "all_pairs_per_mature_decision_each_pool": 28,
        "candidates": list(CANDIDATES),
        "candidate_count_in_this_study": 6,
        "study_size_is_a_total_research_quota": False,
        "pools": 3,
        "complete_experts_each_pool": 8,
        "one_related_source_only_per_pool": True,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_model_or_target_runs": 0,
        "new_economic_factor_definitions": 0,
        "original_active_seven_or_selected_eight_rosters_changed": False,
        "source_variants_are_independent_economic_factors": False,
        "source_financial_rejections_preserved": True,
        "new_RF_assessment_view_or_source_scope_waiver": False,
        "full_history_winner_or_representative_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Freeze one three-sibling slot substitution with unchanged original laws.")


def decision_sessions(calendar, policy):
    function = prior.prior.decision_sessions
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )(calendar, policy)


def ids(variant):
    if variant not in VARIANTS:
        raise QuantError("Only the three declared source siblings belong to this cohort.")
    return tuple(c["id"] for c in CANDIDATES if c["source_variant"] == variant)


def oracle_context(variant):
    return SimpleNamespace(
        EXPERT_IDS=EXPERT_IDS[variant],
        CLOCK=CLOCK,
        POLICY=POLICY,
        decision_sessions=decision_sessions,
    )


def build_targets(pools, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if set(pools) != set(VARIANTS):
        raise QuantError("All three predeclared source pools are required.")
    all_targets, by_variant = {}, {}
    for variant in VARIANTS:
        experts = pools[variant]
        if (
            tuple(e.identity.identifier for e in experts) != EXPERT_IDS[variant]
            or tuple(e.identity.mechanism_group for e in experts) != MECHANISMS
        ):
            raise QuantError("Replace only slot2; siblings never coexist in one expert pool.")

        def keep(row, key=variant):
            if retain is not None:
                retain({"source_variant": key, **row})

        function = prior.prior.build_targets
        derived, audits = FunctionType(
            function.__code__,
            {
                **function.__globals__,
                **globals(),
                "EXPERT_IDS": EXPERT_IDS[variant],
                "CANDIDATES": tuple(c for c in CANDIDATES if c["source_variant"] == variant),
            },
            argdefs=function.__defaults__,
        )(
            experts,
            calendar,
            universe,
            policy,
            before_decision,
            keep if retain is not None else None,
        )
        all_targets.update(derived)
        by_variant[variant] = audits
    first = by_variant[VARIANTS[0]]
    if any(
        [a["decision_session"] for a in by_variant[key]] != [a["decision_session"] for a in first]
        for key in VARIANTS[1:]
    ):
        raise QuantError("Separate source pools must retain identical original decision clocks.")
    return all_targets, [
        {
            "decision_session": row["decision_session"],
            "variants": {key: by_variant[key][position] for key in VARIANTS},
        }
        for position, row in enumerate(first)
    ]
