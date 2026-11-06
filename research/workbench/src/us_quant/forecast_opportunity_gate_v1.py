from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from us_quant import expert_active_forecast_pair_v1 as original
from us_quant.config import QuantError
from us_quant.saved_inclusion_execution_v1 import _target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/forecast-opportunity-gate-v1.json"
PRIOR_HEAD = "2056f5365d92c8da4d89d19ae71cc53cd326a1a2fa559f05853a3fd3868b3254"
ASSETS, EXPERT_IDS, MECHANISMS, CLOCK = (
    original.ASSETS,
    original.EXPERT_IDS,
    original.MECHANISMS,
    original.CLOCK,
)
SOURCE_IDS = original.IDS
CONTROL_IDS = (
    "expert_pool_eight_unconditional_forecast_pair_v2",
    "expert_pool_eight_contextual_forecast_pair_v2",
)
MODELS = original.MODELS
CANDIDATES = (
    {
        "id": "expert_pool_unconditional_forecast_opportunity_gate_v1",
        "weighting": "retained_unconditional_positive_opportunity_else_BIL_standby",
    },
    {
        "id": "expert_pool_contextual_forecast_opportunity_gate_v1",
        "weighting": "retained_contextual_positive_opportunity_else_BIL_standby",
    },
)
IDS = tuple(c["id"] for c in CANDIDATES)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_forecast_opportunity_gate_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 177422,
        "prior_journal_sequence": 1274,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 285,
        "initial_decision_session": "2016-10-05",
        "decision_information_clock": CLOCK,
        "strict_positive_post_cost_active_numerator_threshold": 0.0,
        "gate_uses_rounded_score_or_fitted_threshold": False,
        "positive_mode": "copy_exact_retained_advisor_target",
        "nonpositive_mode": "explicit_BIL_funding_standby",
        "unscored_mode": "explicit_unscored_BIL_funding_standby",
        "missing_mature_certificate_is_standby": False,
        "funding_standby_symbol": "BIL",
        "funding_standby_is_strategy_expert": False,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "source_advisor_state_recomputed_with_gate": False,
        "source_proxy_is_new_allocation_actual_cost_forecast": False,
        "source_post_cost_numerator_is_future_main_net_return": False,
        "new_Ridge_fits": 0,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "new_shadow_account_runs": 0,
        "candidate_count_in_this_study": 2,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in fixed.items()
    ):
        raise QuantError(
            "Freeze one zero-hurdle participation law, not another score or parameter scan."
        )
    if set(policy) != set(fixed) | {
        "source_allocation_tape",
        "source_sign_certificate_tape",
        "source_sign_implementation_freeze",
        "source_sign_result",
        "source_sign_cycle",
        "source_advisor_results",
        "source_advisor_cycle",
    }:
        raise QuantError(
            "Only declared immutable advisor and sign-certificate references are admitted."
        )


def standby():
    result = np.zeros(len(ASSETS))
    result[ASSETS.index("BIL")] = 0.98
    return result


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise QuantError("Only recorded finite numeric opportunity inputs are admissible.")
    return float(value)


def checked_join(row):
    before, after = row["allocation_before"], row["allocation_resolved"]
    certificate_before, certificate_after = row["certificate_before"], row["certificate_resolved"]
    audit = after["audit"]
    classified = certificate_after["result"]
    day, index = audit["decision_session"], audit["decision_index"]
    if (
        before["record"] != "saved_forecast_allocation_input"
        or after["record"] != "saved_forecast_allocation_resolved"
        or certificate_before["record"] != "normalizer_input"
        or certificate_after["record"] != "normalizer_resolved"
        or certificate_before["index"] != index
        or certificate_after["index"] != index
        or before["source_before"]["decision_index"] != index
        or before["source_before"]["as_of"] != day
        or certificate_before["source_audit"] != audit
        or classified["decision_session"] != day
        or classified["index"] != index
        or audit["decision_information_clock"] != CLOCK
        or set(audit["candidates"]) != set(SOURCE_IDS)
        or set(after["targets"]) != set(SOURCE_IDS)
        or set(classified["models"]) != set(MODELS)
        or audit["source_forecaster_refitted"] is not False
        or audit["source_risk_weights_retrained"] is not False
        or audit["source_shadow_wealth_or_floor_transferred"] is not False
    ):
        raise QuantError(
            "Join the exact same-clock advisor and certificate, never a future forecast."
        )
    for target in after["targets"].values():
        _target(target, len(ASSETS))
    return before, after, audit, classified


def gate(row, policy):
    validate(policy)
    before, after, source, classified = checked_join(row)
    targets, candidates = {}, {}
    for name, old_id, control, model in zip(IDS, SOURCE_IDS, CONTROL_IDS, MODELS, strict=True):
        advisor, certificate = source["candidates"][old_id], classified["models"][model]
        if advisor["model"] != model or advisor["safety_or_success_claimed"] is not False:
            raise QuantError(
                "Both original forecasters and explicit non-success declarations remain."
            )
        if advisor["scores_available"]:
            if certificate["status"] != "recorded_scored_origin_verified":
                raise QuantError("A missing mature certificate cannot become funding standby.")
            selected = [p for p in certificate["pairs"] if p["was_recorded_selected"]]
            if (
                len(selected) != 1
                or selected[0]["pair"] != advisor["selected_pair"]
                or selected[0]["source_NumPy_and_scalar_scaled_key_bit_exact"] is not True
                or selected[0]["recorded_source_score_or_key_rewritten"] is not False
                or selected[0]["status"] != "recorded_score_identity_verified"
            ):
                raise QuantError("Only the actual certified selected pair may supply the gate.")
            selected = selected[0]
            numerator = float.fromhex(selected["used_binary64_post_cost_numerator_hex"])
            if not math.isfinite(numerator):
                raise QuantError("An undefined opportunity is not zero or a safe funding state.")
            sign = "positive" if numerator > 0 else "negative" if numerator < 0 else "zero"
            if sign != selected["numerator_sign"] or sign != certificate["selected_numerator_sign"]:
                raise QuantError(
                    "The exact signed certificate is inconsistent with its saved value."
                )
            participate = sign == "positive"
            target = (
                _target(after["targets"][old_id], len(ASSETS)).copy() if participate else standby()
            )
            reason = (
                "positive_retained_opportunity_copy_exact_advisor"
                if participate
                else "nonpositive_retained_opportunity_explicit_standby"
            )
        else:
            if (
                certificate["status"] != "original_unscored_origin_retained"
                or advisor["selected_pair"] is not None
            ):
                raise QuantError(
                    "Unscored sources stay explicitly unscored, not a missing-input fallback."
                )
            forecast_available = before["forecast_resolved"]["audit"]["forecast_available"]
            if (
                forecast_available
                and (
                    len(advisor["pairs"]) != 28
                    or any("unscorable_reason" not in p for p in advisor["pairs"])
                )
                or not forecast_available
                and (advisor["pairs"] or certificate["pairs"])
            ):
                raise QuantError(
                    "Only a declared unmatured or wholly unscorable source may stand by."
                )
            numerator, sign, participate, target = None, None, False, standby()
            reason = "unscored_source_explicit_standby_no_success_claim"
        targets[name] = target
        candidates[name] = {
            "source_advisor_id": old_id,
            "matched_control_program_id": control,
            "model": model,
            "source_scores_available": advisor["scores_available"],
            "source_selected_pair": advisor["selected_pair"],
            "source_post_cost_active_numerator_hex": float(numerator).hex()
            if numerator is not None
            else None,
            "source_numerator_sign": sign,
            "participation": participate,
            "mode": "retained_advisor_target" if participate else "funding_standby",
            "reason": reason,
            "funding_standby_is_strategy_expert": False,
            "source_advisor_state_changed": False,
            "source_proxy_is_new_allocation_actual_cost_forecast": False,
            "safety_or_success_claimed": False,
        }
    return targets, {
        "decision_session": source["decision_session"],
        "decision_index": source["decision_index"],
        "decision_information_clock": CLOCK,
        "candidates": candidates,
        "new_Ridge_fits": 0,
        "source_ranking_or_statistical_state_replayed": False,
        "independent_forward_validation": False,
        "order_authority": False,
    }


def verify(row, targets, audit, policy):
    validate(policy)
    _, after, source, _ = checked_join(row)
    if audit["decision_session"] != source["decision_session"] or set(targets) != set(IDS):
        raise QuantError("Verify both new participation targets on the exact source date.")
    for name, old_id in zip(IDS, SOURCE_IDS, strict=True):
        advisor = source["candidates"][old_id]
        issued = audit["candidates"][name]
        if advisor["scores_available"]:
            chosen = [p for p in advisor["pairs"] if p["pair"] == advisor["selected_pair"]]
            if len(chosen) != 1:
                raise QuantError(
                    "The retained advisor's selected pair must be uniquely recoverable."
                )
            chosen = chosen[0]
            cost = number(chosen["estimated_incremental_switch_cost_fraction"])
            means = chosen["forecast_mean_daily_active"]
            if cost < 0 or not math.isfinite(cost) or set(means) != {"SPY", "QQQ"}:
                raise QuantError(
                    "The declared nominal source hurdle requires complete finite inputs."
                )
            numerators = [math.fsum([number(means[b]), -(cost / 21)]) for b in ("SPY", "QQQ")]
            if not all(math.isfinite(v) for v in numerators):
                raise QuantError(
                    "No nonfinite source opportunity may be silently parked as valid cash."
                )
            value = min(numerators)
            expected = value > 0
            sign = "positive" if value > 0 else "negative" if value < 0 else "zero"
            if issued["source_post_cost_active_numerator_hex"] != float(value).hex():
                raise QuantError("Independent raw-source binary64 opportunity sign differs.")
            if issued["source_numerator_sign"] != sign:
                raise QuantError("The reported opportunity sign must match raw-source arithmetic.")
        else:
            expected = False
            if (
                issued["source_post_cost_active_numerator_hex"] is not None
                or issued["source_numerator_sign"] is not None
            ):
                raise QuantError(
                    "Unmatured history cannot acquire a numeric successful assessment."
                )
        actual = _target(targets[name], len(ASSETS))
        if expected:
            if not np.array_equal(actual, _target(after["targets"][old_id], len(ASSETS))):
                raise QuantError("A positive participation must copy the exact original intention.")
        elif not np.array_equal(actual, standby()):
            raise QuantError(
                "Funding standby is exactly actual BIL, not another expert or idle RF."
            )
        if (
            issued["participation"] is not expected
            or issued["source_scores_available"] is not advisor["scores_available"]
            or issued["mode"] != ("retained_advisor_target" if expected else "funding_standby")
            or issued["funding_standby_is_strategy_expert"] is not False
            or issued["source_advisor_state_changed"] is not False
            or issued["source_proxy_is_new_allocation_actual_cost_forecast"] is not False
            or issued["safety_or_success_claimed"] is not False
        ):
            raise QuantError("Keep the declared mode, source state and no-guarantee distinctions.")
    return {
        "independent_gate_and_target_checks": len(IDS),
        "target_copy_or_BIL_identity_exact": True,
    }


def decision_sessions(calendar):
    return original.decision_sessions(calendar)
