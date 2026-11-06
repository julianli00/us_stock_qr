from __future__ import annotations

import math

import pandas as pd

from us_quant import expert_active_forecast_pair_v1 as original
from us_quant.config import QuantError

ROOT = original.ROOT
POLICY = ROOT / "config/expert-active-forecast-pair-v2.json"
PRIOR_HEAD = "6f6b9b39ce7ff56564ea10d8e463fe9b95df7db96abdd0f8be8297cd440fd025"
CANDIDATES = (
    {
        "id": "expert_pool_eight_unconditional_forecast_pair_v2",
        "weighting": "same_V1_unconditional_saved_targets_explicit_proof_adapter",
    },
    {
        "id": "expert_pool_eight_contextual_forecast_pair_v2",
        "weighting": "same_V1_contextual_saved_targets_explicit_proof_adapter",
    },
)
IDS = tuple(c["id"] for c in CANDIDATES)
SOURCE_IDS = original.IDS


def validate(policy):
    fixed = {
        "schema_version": 2,
        "study_id": "ssd_expert_active_forecast_pair_20261106_v2",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 175938,
        "prior_journal_sequence": 1251,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 283,
        "adapter_role": (
            "canonical_no_source_signal_recompute_attestation_over_immutable_V1_targets"
        ),
        "economic_law_or_parameters_changed": False,
        "source_V1_terminal_failures_preserved": True,
        "source_V1_target_derivation_precedes_V2_registration": True,
        "new_target_derivations": 0,
        "new_Ridge_fits": 0,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "new_shadow_account_runs": 0,
        "candidate_count_in_this_study": 2,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_economic_research_questions_added_by_adapter": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in fixed.items()
    ):
        raise QuantError("Only a separate metadata proof Adapter may continue the saved targets.")
    if set(policy) != set(fixed) | {
        "source_economic_policy",
        "source_implementation_freeze",
        "source_prepared_targets",
        "source_terminal_closure",
        "source_public_failure",
        "cached_original_eight_results",
    }:
        raise QuantError("Do not add another numeric law, source view or acceptance waiver.")


def attest(source, registered_frozen_sha256, checked_at):
    proof = source["proof"]
    old_registered = pd.Timestamp(source["registered_at"])
    derived = pd.Timestamp(proof["checked_at"])
    attested = pd.Timestamp(checked_at)
    if (
        any(t.tzinfo is None or pd.isna(t) for t in (old_registered, derived, attested))
        or not old_registered <= derived <= attested
        or proof["frozen_specification_sha256"] != source["registered_frozen_sha256"]
        or proof["candidate_accounts_or_metrics_computed"] is not False
        or proof["source_models_selectors_forecasters_or_accounts_replayed"] is not False
        or (
            "original_source_signal_recomputed" in proof
            and proof["original_source_signal_recomputed"] is not False
        )
        or type(proof["new_Ridge_fits"]) is not int
        or proof["new_Ridge_fits"] != 0
        or type(proof["new_source_requests"]) is not int
        or proof["new_source_requests"] != 0
        or proof["decision_count"] != 121
        or proof["all_pairs_per_mature_decision"] != 28
        or proof["actual_input_resolved_record_count"] != 242
        or proof["source_NET_RF_binary64_values_checked"] != 236512
        or proof["independent_forward_validation"] is not False
        or len(proof["prefix_truncations"]) != 5
        or len(proof["formal_factor_coverage"]) != 2
        or any(len(row["actual_factor_symbols"]) < 3 for row in proof["formal_factor_coverage"])
    ):
        raise QuantError("A metadata Adapter cannot create missing source or numerical evidence.")
    tolerances = {
        "maximum_independent_target_difference": 1e-12,
        "maximum_independent_weight_difference": 1e-12,
        "maximum_independent_risk_difference": 1e-12,
        "maximum_independent_score_difference": 1e-10,
    }
    for key, limit in tolerances.items():
        value = proof[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise QuantError("Retain the original exact tolerances; never repair a failed metric.")
    return {
        **proof,
        "checked_at": checked_at,
        "frozen_specification_sha256": registered_frozen_sha256,
        "original_source_signal_recomputed": False,
        "source_target_derivation_checked_at": proof["checked_at"],
        "source_registered_at": source["registered_at"],
        "source_registered_event_sha256": source["registered_event_sha256"],
        "source_registered_frozen_sha256": source["registered_frozen_sha256"],
        "source_target_proof": source["proof_reference"],
        "reused_target_artifact": source["target_reference"],
        "V1_target_derivation_precedes_V2_registration": True,
        "new_V2_target_derivation_performed": False,
        "economic_law_inputs_or_parameters_changed": False,
        "terminal_V1_failures_erased_or_reclassified": False,
    }
