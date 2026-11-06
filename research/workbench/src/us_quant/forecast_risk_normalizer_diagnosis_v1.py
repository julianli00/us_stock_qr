from __future__ import annotations

import itertools
import math
from decimal import Decimal, localcontext
from pathlib import Path

from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/forecast-risk-normalizer-diagnosis-v1.json"
IDS = (
    "expert_pool_eight_unconditional_forecast_pair_v1",
    "expert_pool_eight_contextual_forecast_pair_v1",
)
MODELS = ("unconditional_unit_ridge", "contextual_unit_ridge")
BENCHMARKS = ("SPY", "QQQ")
TOLERANCE = 1e-10


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_forecast_risk_normalizer_diagnosis_20261106_v1",
        "actual_native_event_seq": 176855,
        "prior_journal_sequence": 1274,
        "prior_journal_head_sha256": (
            "2056f5365d92c8da4d89d19ae71cc53cd326a1a2fa559f05853a3fd3868b3254"
        ),
        "prior_combined_completed": 285,
        "source_benchmarks": list(BENCHMARKS),
        "source_risk_windows": ["training", "validation"],
        "source_risk_basis": "original_RF_excess_pair_risk",
        "annualization_sessions": 252,
        "cost_amortization_sessions": 21,
        "recorded_score_decimal_places": 12,
        "numerator_sign_basis": "actual_binary64_post_cost_numerator_not_a_tolerance_band",
        "derivative_scope": "continuous_unrounded_score_with_numerator_fixed",
        "independent_decimal_precision": 50,
        "independent_score_tolerance": TOLERANCE,
        "all_models_all_pairs_and_unscored_origins_retained": True,
        "alternative_pair_ranking_or_selection_generated": False,
        "main_drawdown_loss_or_causal_attribution_generated": False,
        "main_holdings_are_inferred_from_source_scores": False,
        "new_forecast_fit_risk_fit_targets_or_accounts": 0,
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in fixed.items()
    ):
        raise QuantError(
            "Freeze the signed score identity, not a new allocation or performance test."
        )
    if set(policy) != set(fixed) | {
        "source_allocation_tape",
        "source_allocation_freeze",
        "source_economic_policy",
        "source_completed_financial_result",
        "source_completed_cycle",
    }:
        raise QuantError(
            "Keep only the declared saved-source inputs and diagnostic interpretation."
        )


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise QuantError("A missing or nonfinite score input cannot become neutral or successful.")
    return float(value)


def case(pair, selected):
    names = pair["pair"]
    if (
        len(names) != 2
        or names != sorted(names)
        or len(set(names)) != 2
        or not set(names).issubset(pool.EXPERT_IDS)
        or type(selected) is not bool
    ):
        raise QuantError("Diagnose an actual recorded complete-expert pair, not a new support.")
    if "unscorable_reason" in pair:
        return {
            "pair": names,
            "was_recorded_selected": selected,
            "status": "source_pair_unscorable_not_evaluated",
            "original_reason": pair["unscorable_reason"],
        }
    means = pair["forecast_mean_daily_active"]
    if set(means) != set(BENCHMARKS) or pair["cost_proxy_is_actual_main_account_cost"] is not False:
        raise QuantError(
            "Keep both stored active-rate means and the nominal-only cost declaration."
        )
    cost = number(pair["estimated_incremental_switch_cost_fraction"])
    risks = {
        "training": number(pair["training_RF_excess_pair_risk"]),
        "validation": number(pair["validation_RF_excess_pair_risk"]),
    }
    if cost < 0 or min(risks.values()) <= 0:
        raise QuantError("Do not repair undefined risk or a negative cost with a floor.")
    numerators = {b: number(means[b]) - cost / 21 for b in BENCHMARKS}
    worst = min(numerators.values())
    sign = "positive" if worst > 0 else "negative" if worst < 0 else "zero"
    binding = min(risks.values()) if worst < 0 else max(risks.values())
    windows = list(risks) if worst == 0 else [w for w, v in risks.items() if v == binding]
    score = math.sqrt(252) * worst / binding
    recorded = number(pair["selection_score"])
    if pair["selection_score_key"] != round(recorded, 12):
        raise QuantError("Keep the recorded12decimal key, not another rounding rule or winner.")
    four = pair["forecast_risk_scores"]
    expected_keys = {"training_risk_reference", "validation_risk_reference"}
    if set(four) != set(BENCHMARKS) or any(set(v) != expected_keys for v in four.values()):
        raise QuantError("Every nonselected pair also needs all four original score components.")
    gap = abs(score - recorded)
    if recorded != min(number(v) for row in four.values() for v in row.values()):
        raise QuantError("The recorded score must be its actual four-component minimum.")
    independent_gap = 0.0
    with localcontext() as precision:
        precision.prec = 50
        scale = Decimal(252).sqrt()
        independently_used = {b: math.fsum([number(means[b]), -(cost / 21)]) for b in BENCHMARKS}
        minimum = min(Decimal.from_float(v) for v in independently_used.values())
        independent_sign = "positive" if minimum > 0 else "negative" if minimum < 0 else "zero"
        if independent_sign != sign:
            raise QuantError("The actual binary64 numerator sign disagrees with the scalar oracle.")
        expected = []
        for b in BENCHMARKS:
            for window, risk in risks.items():
                value = scale * Decimal.from_float(independently_used[b]) / Decimal.from_float(risk)
                expected.append(value)
                independent_gap = max(
                    independent_gap, abs(float(value) - number(four[b][window + "_risk_reference"]))
                )
        independent_gap = max(independent_gap, abs(float(min(expected)) - recorded))
    if gap > TOLERANCE or independent_gap > TOLERANCE:
        raise QuantError("The fixed independent signed-normalizer score identity failed.")
    return {
        "pair": names,
        "was_recorded_selected": selected,
        "status": "recorded_score_identity_verified",
        "numerator_sign": sign,
        "used_binary64_post_cost_numerator_hex": float(worst).hex(),
        "worst_benchmark_set": [b for b, v in numerators.items() if v == worst],
        "binding_risk_window_set": windows,
        "continuous_score_risk_slope": "increasing"
        if worst < 0
        else "decreasing"
        if worst > 0
        else "flat",
        "larger_risk_is_continuously_rewarded_at_fixed_numerator": worst < 0,
        "recorded_score": recorded,
        "recorded_score_key": pair["selection_score_key"],
        "risk_window_max_to_min_ratio": max(risks.values()) / min(risks.values()),
        "score_identity_gap": gap,
        "independent_decimal_score_gap": independent_gap,
        "derivative_is_actual_discrete_ranking_or_causal_drawdown_effect": False,
    }


def diagnose(audits, retain=None, before_case=None):
    decisions = []
    expected = {
        tuple(sorted((pool.EXPERT_IDS[i], pool.EXPERT_IDS[j])))
        for i, j in itertools.combinations(range(8), 2)
    }
    for index, audit in enumerate(audits):
        if before_case is not None:
            before_case()
        if retain is not None:
            retain({"record": "normalizer_input", "index": index, "source_audit": audit})
        if (
            audit["decision_index"] != index
            or set(audit["candidates"]) != set(IDS)
            or audit["decision_information_clock"] != pool.CLOCK
            or audit["source_forecaster_refitted"] is not False
            or audit["source_risk_weights_retrained"] is not False
            or audit["source_shadow_wealth_or_floor_transferred"] is not False
        ):
            raise QuantError("Keep both same-clock saved policies without new fits or capital.")
        models = {}
        for identifier, model in zip(IDS, MODELS, strict=True):
            source = audit["candidates"][identifier]
            if source["model"] != model or source["safety_or_success_claimed"] is not False:
                raise QuantError(
                    "A saved forecast diagnosis cannot invent a safe or successful source."
                )
            if not source["scores_available"]:
                if source["selected_pair"] is not None:
                    raise QuantError("An unscored source cannot invent a selected pair.")
                rows = [case(p, False) for p in source["pairs"]]
                if any(r["status"] != "source_pair_unscorable_not_evaluated" for r in rows):
                    raise QuantError("A mature scored menu cannot be relabeled as unscored.")
                models[model] = {
                    "status": "original_unscored_origin_retained",
                    "original_reason": source["reason"],
                    "pairs": rows,
                }
                continue
            pairs = source["pairs"]
            if len(pairs) != 28 or {tuple(p["pair"]) for p in pairs} != expected:
                raise QuantError("Retain every one of28pairs, not selected-only or partial menus.")
            rows = [case(p, p["pair"] == source["selected_pair"]) for p in pairs]
            eligible = [r for r in rows if r["status"] == "recorded_score_identity_verified"]
            chosen = [r for r in eligible if r["was_recorded_selected"]]
            if len(chosen) != 1:
                raise QuantError("Exactly one original selected pair must remain identifiable.")
            actual = chosen[0]
            if actual["recorded_score_key"] != max(r["recorded_score_key"] for r in eligible):
                raise QuantError("The saved choice cannot disagree with its recorded maximum key.")
            models[model] = {
                "status": "recorded_scored_origin_verified",
                "pairs": rows,
                "selected_numerator_sign": actual["numerator_sign"],
                "all_eligible_numerators_negative": all(
                    r["numerator_sign"] == "negative" for r in eligible
                ),
                "selected_negative_with_any_nonnegative_menu": (
                    actual["numerator_sign"] == "negative"
                    and any(r["numerator_sign"] != "negative" for r in eligible)
                ),
            }
        result = {"decision_session": audit["decision_session"], "index": index, "models": models}
        decisions.append(result)
        if retain is not None:
            retain({"record": "normalizer_resolved", "index": index, "result": result})
    if not decisions:
        raise QuantError("No retained decisions is not a successful diagnosis.")
    return aggregate(decisions), decisions


def aggregate(decisions):
    result = {}
    for model in MODELS:
        origins = [d["models"][model] for d in decisions]
        scored = [o for o in origins if o["status"] == "recorded_scored_origin_verified"]
        rows = [
            r
            for o in scored
            for r in o["pairs"]
            if r["status"] == "recorded_score_identity_verified"
        ]
        selected = [r for r in rows if r["was_recorded_selected"]]
        result[model] = {
            "decision_origins": len(origins),
            "scored_origins": len(scored),
            "unscored_origins": len(origins) - len(scored),
            "all_recorded_pair_rows": sum(len(o["pairs"]) for o in origins),
            "evaluable_pair_rows": len(rows),
            "original_unscorable_pair_rows": sum(
                r["status"] == "source_pair_unscorable_not_evaluated"
                for o in origins
                for r in o["pairs"]
            ),
            "all_pairs_numerator_sign_counts": {
                s: sum(r["numerator_sign"] == s for r in rows)
                for s in ("positive", "negative", "zero")
            },
            "selected_numerator_sign_counts": {
                s: sum(r["numerator_sign"] == s for r in selected)
                for s in ("positive", "negative", "zero")
            },
            "all_pairs_continuous_risk_slope_counts": {
                s: sum(r["continuous_score_risk_slope"] == s for r in rows)
                for s in ("increasing", "decreasing", "flat")
            },
            "selected_continuous_risk_slope_counts": {
                s: sum(r["continuous_score_risk_slope"] == s for r in selected)
                for s in ("increasing", "decreasing", "flat")
            },
            "selected_binding_risk_window_counts": {
                "training_only": sum(
                    r["binding_risk_window_set"] == ["training"] for r in selected
                ),
                "validation_only": sum(
                    r["binding_risk_window_set"] == ["validation"] for r in selected
                ),
                "both": sum(len(r["binding_risk_window_set"]) == 2 for r in selected),
            },
            "all_eligible_numerators_negative_origins": sum(
                o["all_eligible_numerators_negative"] for o in scored
            ),
            "selected_negative_with_any_nonnegative_menu_origins": sum(
                o["selected_negative_with_any_nonnegative_menu"] for o in scored
            ),
            "maximum_score_identity_gap": max(
                (r["score_identity_gap"] for r in rows), default=None
            ),
            "maximum_independent_decimal_score_gap": max(
                (r["independent_decimal_score_gap"] for r in rows), default=None
            ),
        }
    return {
        "decision_count": len(decisions),
        "models": result,
        "derivative_scope": "continuous_unrounded_score_with_numerator_fixed",
        "main_drawdown_or_causal_loss_attribution": False,
        "alternative_pair_ranking_or_selection_generated": False,
        "new_targets_accounts_or_financial_trials": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
