from __future__ import annotations

from pathlib import Path
from types import FunctionType, SimpleNamespace

from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/paired-activity-information-v1.json"
PRIOR_HEAD = "171671760e7faf64c365fda0cbde4b47a4badaec9df70751ab69e82dca07dc01"
COMPATIBILITY_SHA = "43e5c3a6fbdc9af4fe343794037488d05f23a823ed889447ef394e966b0f5dff"
VARIANTS = ("price", "activity")
SOURCE_IDS = (
    "two_segment_core_price_momentum_control_v1",
    "two_segment_core_activity_momentum_v1",
)
EXPERT_IDS = {
    key: (*prior.EXPERT_IDS[:5], name, *prior.EXPERT_IDS[6:])
    for key, name in zip(VARIANTS, SOURCE_IDS, strict=True)
}
MECHANISMS, CLOCK = prior.MECHANISMS, prior.CLOCK
CANDIDATES = tuple(
    {"id": f"expert_pool_style_{key}_{label}_v1", "weighting": weight, "source_variant": key}
    for key in VARIANTS
    for label, weight in (("equal", "equal"), ("rolling", "train_erc_validate_net"))
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
        "study_id": "ssd_paired_activity_information_20261104_v1",
        "source_expert_pool_policy_path": prior.POLICY.relative_to(ROOT).as_posix(),
        "prior_journal_sequence": 784,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_pair_compatibility": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "paired-activity-information-20261104-v1/source-pair-compatibility-v1.json"
            ),
            "sha256": COMPATIBILITY_SHA,
        },
        "replaced_slot": 5,
        "replaced_expert_id": prior.EXPERT_IDS[5],
        "expert_ids_by_variant": {key: list(value) for key, value in EXPERT_IDS.items()},
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "candidate_count_in_this_study": 4,
        "study_size_is_a_total_research_quota": False,
        "study_role_group_mapping_only": True,
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
        raise QuantError("Freeze the complete paired field ablation, not a winner or retuned law.")


def ids(variant):
    if variant not in VARIANTS:
        raise QuantError("Only the two declared sibling source variants are allowed.")
    return tuple(c["id"] for c in CANDIDATES if c["source_variant"] == variant)


def decision_sessions(calendar, policy):
    return FunctionType(
        prior.prior.decision_sessions.__code__,
        {**prior.prior.decision_sessions.__globals__, **globals()},
        argdefs=prior.prior.decision_sessions.__defaults__,
    )(calendar, policy)


def oracle_context(variant):
    if variant not in VARIANTS:
        raise QuantError("Only the two declared sibling source variants are allowed.")
    return SimpleNamespace(
        EXPERT_IDS=EXPERT_IDS[variant],
        CLOCK=CLOCK,
        POLICY=POLICY,
        decision_sessions=decision_sessions,
    )


def build_targets(pools, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if set(pools) != set(VARIANTS):
        raise QuantError("Both predeclared source pools are required.")
    all_targets, by_variant = {}, {}
    for key in VARIANTS:
        experts = pools[key]
        if (
            tuple(e.identity.identifier for e in experts) != EXPERT_IDS[key]
            or tuple(e.identity.mechanism_group for e in experts) != MECHANISMS
        ):
            raise QuantError("Replace only the one style role; sibling variants never coexist.")

        def keep(row, variant=key):
            if retain is not None:
                retain({"source_variant": variant, **row})

        function = prior.prior.build_targets
        derived, audits = FunctionType(
            function.__code__,
            {
                **function.__globals__,
                **globals(),
                "EXPERT_IDS": EXPERT_IDS[key],
                "CANDIDATES": tuple(c for c in CANDIDATES if c["source_variant"] == key),
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
        by_variant[key] = audits
    if [a["decision_session"] for a in by_variant["price"]] != [
        a["decision_session"] for a in by_variant["activity"]
    ]:
        raise QuantError("The paired information ablation cannot compare different clocks.")
    return all_targets, [
        {
            "decision_session": row["decision_session"],
            "variants": {key: by_variant[key][position] for key in VARIANTS},
        }
        for position, row in enumerate(by_variant["price"])
    ]
