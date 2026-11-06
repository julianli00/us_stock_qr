from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from us_quant import strategy_expert_pool_eight_v1 as original
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.saved_inclusion_execution_v1 import _target
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-active-forecast-pair-v1.json"
PRIOR_HEAD = "ddbdf3b81a693cd6c3b459ec85ea49bfd65ff47617edee7825465c4277274c51"
MODELS = ("unconditional_unit_ridge", "contextual_unit_ridge")
BENCHMARKS = ("SPY", "QQQ")
EXPERT_IDS, MECHANISMS, CLOCK = original.EXPERT_IDS, original.MECHANISMS, original.CLOCK
CANDIDATES = (
    {
        "id": "expert_pool_eight_unconditional_forecast_pair_v1",
        "weighting": "saved_unconditional_active_amount_original_pair_risk",
    },
    {
        "id": "expert_pool_eight_contextual_forecast_pair_v1",
        "weighting": "saved_contextual_active_amount_original_pair_risk",
    },
)
IDS = tuple(c["id"] for c in CANDIDATES)


def validate(policy):
    source = read_json(original.POLICY)
    inherited = (
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
        **{k: source[k] for k in inherited},
        "schema_version": 1,
        "study_id": "ssd_expert_active_forecast_pair_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 175938,
        "prior_journal_sequence": 1246,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 283,
        "all_pairs_per_mature_decision": 28,
        "forecast_models": list(MODELS),
        "forecast_benchmarks": list(BENCHMARKS),
        "ranking": (
            "minimum_saved_forecast_active_rate_after_proxy_over_each_original_RF_risk_window"
        ),
        "ranking_risk_windows_are_independent_predictor_validation": False,
        "pair_training_ERC_weights_copied_unchanged": True,
        "forecast_model_or_original_source_selector_replayed": False,
        "forecaster_is_an_additional_strategy_expert": False,
        "new_Ridge_fits": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "model_selected_from_complete_forecast_errors": False,
        "benchmark_translation_is_pair_specific_alpha": False,
        "nominal_proxy_is_actual_main_cost_forecast": False,
        "independent_target_tolerance": 1e-12,
        "independent_score_tolerance": 1e-10,
        "candidate_count_in_this_study": 2,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in expected.items()
    ):
        raise QuantError("Freeze both forecast selectors and every original risk/execution law.")
    if set(policy) != set(expected) | {
        "source_risk_tape",
        "source_forecast_tape",
        "source_forecast_freeze",
        "source_forecast_result",
        "source_forecast_completion",
        "cached_original_eight_results",
    }:
        raise QuantError("Only declared saved-source references may augment the fixed law.")


def checked_join(row):
    before, resolved = row["source_before"], row["source_resolved"]
    fb, fr = row["forecast_before"], row["forecast_resolved"]
    old, forecast = resolved["audit"], fr["audit"]
    day, index = before["as_of"], before["decision_index"]
    observed = before["observations"]
    if (
        before["record"] != "decision_input"
        or resolved["record"] != "decision_resolved"
        or fb["record"] != "forecast_input"
        or fr["record"] != "forecast_resolved"
        or resolved["decision_index"] != index
        or fb["index"] != index
        or fr["index"] != index
        or old["decision_session"] != day
        or fb["as_of"] != day
        or forecast["as_of"] != day
        or before["information_clock"] != CLOCK
        or fb["information_clock"] != CLOCK
        or old["decision_information_clock"] != CLOCK
        or before["binary64_values_truncated_to_fixed_decimals"] is not False
        or fb["binary64_values_truncated_to_fixed_decimals"] is not False
        or tuple(o["id"] for o in observed) != EXPERT_IDS
        or tuple(o["mechanism_group"] for o in observed) != MECHANISMS
        or tuple(resolved["asset_order"]) != ASSETS
        or len(fb["observations"]) != 8
        or old["shadow_wealth_added_to_main_capital"] is not False
        or old["source_feedback_floor_transferred_to_main_account"] is not False
        or forecast["forecast_is_main_account_return_or_qualification"] is not False
    ):
        raise QuantError("Join exact same-clock source records, not a future or selected forecast.")
    dates, rf = observed[0]["return_dates"], observed[0]["RF"]
    for o, other in zip(observed, fb["observations"], strict=True):
        if (
            any(
                o[k] != other[k]
                for k in (
                    "id",
                    "mechanism_group",
                    "introduced_at",
                    "as_of",
                    "target_known_at",
                    "return_dates",
                    "NET_returns",
                    "RF",
                    "score_mature",
                )
            )
            or o["as_of"] != day
            or o["target_known_at"] is None
            or o["target_known_at"] > day
            or other["target_available"] is not True
            or tuple(o["asset_order"]) != ASSETS
            or o["return_dates"] != dates
            or o["RF"] != rf
            or len(o["NET_returns"]) != len(dates)
            or len(rf) != len(dates)
            or len(dates) > 126
            or type(o["score_mature"]) is not bool
            or o["score_mature"] != (len(dates) == 126)
            or np.asarray([o["NET_returns"], o["RF"]]).dtype.kind not in "fiu"
            or not np.isfinite(np.asarray([o["NET_returns"], o["RF"]], dtype=float)).all()
        ):
            raise QuantError(
                "No source date, RF, opinion clock or binary64 observation may change."
            )
    if dates and (
        dates[-1] != day or dates != sessions(dates[0], dates[-1]).strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Only complete matured original source sessions enter ranking risk.")
    mature = len(dates) == 126
    if forecast["forecast_available"] is not mature:
        raise QuantError("An absent mature forecast is not an unscored equal-weight fallback.")
    predictions = forecast["predictions"]
    if mature:
        if (
            not isinstance(predictions, dict)
            or set(predictions) != set(MODELS)
            or any(
                np.asarray(predictions[m]).shape != (16,)
                or np.asarray(predictions[m]).dtype.kind not in "fiu"
                or not np.isfinite(np.asarray(predictions[m], dtype=float)).all()
                for m in MODELS
            )
        ):
            raise QuantError("Both saved predictors need all sixteen finite assessment rates.")
        expected = {
            tuple(sorted((EXPERT_IDS[i], EXPERT_IDS[j])))
            for i, j in itertools.combinations(range(8), 2)
        }
        pairs = old["selection"]["pairs"]
        if len(pairs) != 28 or {tuple(p["pair"]) for p in pairs} != expected:
            raise QuantError("Retain all28source risk assessments, including unscorable pairs.")
    elif predictions is not None or old["selection"]["scores_available"]:
        raise QuantError("The original unmatured state must have no forecast or successful score.")
    opinions = np.stack([_target(o["target"], len(ASSETS)) for o in observed])
    return before, old, forecast, opinions


def _weights(value):
    original_value = np.asarray(value)
    if original_value.dtype.kind not in "fiu":
        raise QuantError("Saved numeric expert weights cannot be coerced from strings or booleans.")
    weights = original_value.astype(float, copy=False)
    if (
        weights.shape != (8,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
    ):
        raise QuantError("Expert statistical weights must be finite, nonnegative and unit-sum.")
    return weights


def allocate(row, previous, policy):
    validate(policy)
    before, old, forecast, opinions = checked_join(row)
    if set(previous) != set(IDS):
        raise QuantError("Both new policies own separate previous statistical allocations.")
    observed = before["observations"]
    targets, candidates = {}, {}
    excess = np.column_stack([np.asarray(o["NET_returns"]) - np.asarray(o["RF"]) for o in observed])
    for candidate, model in zip(CANDIDATES, MODELS, strict=True):
        name = candidate["id"]
        prior = _weights(previous[name])
        pairs = []
        if forecast["forecast_available"]:
            predictions = np.asarray(forecast["predictions"][model]).reshape(8, 2)
            reference = prior @ opinions
            for source in old["selection"]["pairs"]:
                record = {"pair": source["pair"]}
                if "unscorable_reason" in source:
                    record["unscorable_reason"] = source["unscorable_reason"]
                    pairs.append(record)
                    continue
                weights = _weights(source["expert_weights"])
                if [EXPERT_IDS[i] for i in np.flatnonzero(weights)] != [
                    n for n in EXPERT_IDS if n in source["pair"]
                ] or (weights > 0).sum() != 2:
                    raise QuantError("Keep the exact two-source support of each archived ERC pair.")
                target = weights @ opinions
                change = abs(target - reference)
                cost = float(change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000)
                training_risk = math.sqrt(
                    math.fsum(source["training_marginal_variance_contributions"])
                )
                validation_risk = float(np.std(excess[84:] @ weights, ddof=1))
                if min(training_risk, validation_risk) <= 0 or not math.isfinite(validation_risk):
                    raise QuantError("Do not repair a retained pair's undefined risk with a floor.")
                active = weights @ predictions
                scores = {
                    b: {
                        "training_risk_reference": math.sqrt(252)
                        * (active[j] - cost / 21)
                        / training_risk,
                        "validation_risk_reference": math.sqrt(252)
                        * (active[j] - cost / 21)
                        / validation_risk,
                    }
                    for j, b in enumerate(BENCHMARKS)
                }
                score = min(v for values in scores.values() for v in values.values())
                if not math.isfinite(score):
                    raise QuantError("A nonfinite forecast risk score cannot become success.")
                record.update(
                    expert_weights=weights.tolist(),
                    forecast_mean_daily_active=dict(zip(BENCHMARKS, active.tolist(), strict=True)),
                    forecast_risk_scores=scores,
                    selection_score=score,
                    selection_score_key=round(score, 12),
                    source_training_correlation=source["training_correlation"],
                    source_training_marginal_variance_contributions=(
                        source["training_marginal_variance_contributions"]
                    ),
                    training_RF_excess_pair_risk=training_risk,
                    validation_RF_excess_pair_risk=validation_risk,
                    estimated_incremental_switch_cost_fraction=cost,
                    estimated_incremental_changed_assets=int((change > 1e-12).sum()),
                    cost_proxy_is_actual_main_account_cost=False,
                )
                pairs.append(record)
        usable = [p for p in pairs if "selection_score_key" in p]
        if usable:
            best = sorted(usable, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
            weights = np.asarray(best["expert_weights"])
            target = weights @ opinions
            selected = best["pair"]
            reason = "saved_forecast_amount_ranked_over_original_pair_risk"
        else:
            weights = np.full(8, 0.125)
            target = _target(resolved_equal(row), len(ASSETS)).copy()
            selected = None
            reason = (
                "original_unmatured_equal8_no_successful_forecast"
                if not forecast["forecast_available"]
                else "original_unscorable_risk_menu_equal8_no_success_claim"
            )
        targets[name] = target
        candidates[name] = {
            "model": model,
            "previous_weights": prior.tolist(),
            "expert_weights": weights.tolist(),
            "selected_pair": selected,
            "scores_available": bool(usable),
            "reason": reason,
            "pairs": pairs,
            "safety_or_success_claimed": False,
        }
    return targets, {
        "decision_index": before["decision_index"],
        "decision_session": before["as_of"],
        "decision_information_clock": CLOCK,
        "candidates": candidates,
        "source_NET_RF_binary64_values_checked": sum(2 * len(o["NET_returns"]) for o in observed),
        "forecaster_is_an_additional_strategy_expert": False,
        "source_forecaster_refitted": False,
        "source_risk_weights_retrained": False,
        "source_shadow_wealth_or_floor_transferred": False,
        "independent_forward_validation": False,
        "order_authority": False,
    }


def resolved_equal(row):
    return row["source_resolved"]["combined_targets"][original.CANDIDATES[0]["id"]]


def verify(row, previous, targets, audit, policy):
    validate(policy)
    before, old, forecast, opinions = checked_join(row)
    if audit["decision_session"] != before["as_of"] or set(targets) != set(IDS):
        raise QuantError("Verify both new allocations on the same original decision lattice.")
    gaps = {"target": 0.0, "weight": 0.0, "score": 0.0, "risk": 0.0}
    values = [
        [float(n) - float(r) for n, r in zip(o["NET_returns"], o["RF"], strict=True)]
        for o in before["observations"]
    ]
    for name, model in zip(IDS, MODELS, strict=True):
        actual = audit["candidates"][name]
        if (
            actual["model"] != model
            or actual["previous_weights"] != list(previous[name])
            or actual["safety_or_success_claimed"] is not False
        ):
            raise QuantError("The two predictors must retain separate causal allocation states.")
        if not forecast["forecast_available"] and (
            actual["pairs"] != []
            or actual["reason"] != "original_unmatured_equal8_no_successful_forecast"
        ):
            raise QuantError(
                "A warm-up row cannot invent scored pairs or replace its null forecast."
            )
        if forecast["forecast_available"]:
            source_pairs = {tuple(p["pair"]): p for p in old["selection"]["pairs"]}
            if len(actual["pairs"]) != 28 or {tuple(p["pair"]) for p in actual["pairs"]} != set(
                source_pairs
            ):
                raise QuantError("Every nonselected pair also needs a verified forecast score.")
            means = [math.fsum(v[:84]) / 84 for v in values]
            deviations = [
                math.sqrt(math.fsum((x - m) ** 2 for x in v[:84]) / 83)
                for v, m in zip(values, means, strict=True)
            ]
            for p in actual["pairs"]:
                source = source_pairs[tuple(p["pair"])]
                if "unscorable_reason" in source:
                    if p != {
                        "pair": source["pair"],
                        "unscorable_reason": source["unscorable_reason"],
                    }:
                        raise QuantError(
                            "A source-unscorable pair cannot be rescued by forecasting."
                        )
                    continue
                weights = _weights(source["expert_weights"])
                i, j = [EXPERT_IDS.index(n) for n in p["pair"]]
                expected = np.zeros(8)
                expected[i], expected[j] = (
                    deviations[j] / (deviations[i] + deviations[j]),
                    deviations[i] / (deviations[i] + deviations[j]),
                )
                gaps["weight"] = max(gaps["weight"], float(max(abs(expected - weights))))
                if p["expert_weights"] != source["expert_weights"]:
                    raise QuantError("The new rule must copy the actual original pair weights.")
                portfolio = [
                    math.fsum(weights[k] * values[k][t] for k in range(8)) for t in range(126)
                ]
                risks = []
                for sample in (portfolio[:84], portfolio[84:]):
                    mean = math.fsum(sample) / len(sample)
                    risks.append(
                        math.sqrt(math.fsum((x - mean) ** 2 for x in sample) / (len(sample) - 1))
                    )
                covariance = (
                    math.fsum(
                        (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                contribution = np.zeros(8)
                contribution[i] = weights[i] * (
                    weights[i] * deviations[i] ** 2 + weights[j] * covariance
                )
                contribution[j] = weights[j] * (
                    weights[j] * deviations[j] ** 2 + weights[i] * covariance
                )
                if (
                    max(
                        abs(
                            contribution
                            - np.asarray(p["source_training_marginal_variance_contributions"])
                        )
                    )
                    > 1e-12
                    or abs(
                        p["source_training_correlation"]
                        - covariance / (deviations[i] * deviations[j])
                    )
                    > 1e-10
                ):
                    raise QuantError("The archived covariance dependence or marginal risk changed.")
                gaps["risk"] = max(
                    gaps["risk"],
                    abs(risks[0] - p["training_RF_excess_pair_risk"]),
                    abs(risks[1] - p["validation_RF_excess_pair_risk"]),
                )
                target = [
                    math.fsum(weights[k] * opinions[k, s] for k in range(8))
                    for s in range(len(ASSETS))
                ]
                prior = [
                    math.fsum(previous[name][k] * opinions[k, s] for k in range(8))
                    for s in range(len(ASSETS))
                ]
                change = [abs(a - b) for a, b in zip(target, prior, strict=True)]
                cost = math.fsum(change) * 5 / 10000 + sum(x > 1e-12 for x in change) / 10000
                active = [
                    math.fsum(
                        weights[k] * forecast["predictions"][model][2 * k + b] for k in range(8)
                    )
                    for b in range(2)
                ]
                scores = [math.sqrt(252) * (v - cost / 21) / risk for v in active for risk in risks]
                gaps["score"] = max(gaps["score"], abs(min(scores) - p["selection_score"]))
                if (
                    abs(cost - p["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or p["estimated_incremental_changed_assets"] != sum(x > 1e-12 for x in change)
                    or p["selection_score_key"] != round(p["selection_score"], 12)
                    or p["cost_proxy_is_actual_main_account_cost"] is not False
                    or max(
                        abs(active[b] - p["forecast_mean_daily_active"][benchmark])
                        for b, benchmark in enumerate(BENCHMARKS)
                    )
                    > 1e-12
                ):
                    raise QuantError(
                        "The independent saved-forecast mean or own-state cost differs."
                    )
                expected_scores = [
                    p["forecast_risk_scores"][b][k]
                    for b in BENCHMARKS
                    for k in ("training_risk_reference", "validation_risk_reference")
                ]
                gaps["score"] = max(
                    gaps["score"],
                    max(abs(a - b) for a, b in zip(scores, expected_scores, strict=True)),
                )
                if p["selection_score"] != min(expected_scores):
                    raise QuantError(
                        "Rank by the complete worst-benchmark/worst-risk-window score."
                    )
        usable = [p for p in actual["pairs"] if "selection_score_key" in p]
        if usable:
            best = sorted(usable, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
            if actual["selected_pair"] != best["pair"] or actual["scores_available"] is not True:
                raise QuantError("Preserve the frozen12decimal score key and lexical pair tie.")
            weights = best["expert_weights"]
        else:
            weights = [0.125] * 8
            if actual["scores_available"] or actual["selected_pair"] is not None:
                raise QuantError("Missing mature information is not a successful neutral forecast.")
        if list(weights) != actual["expert_weights"]:
            raise QuantError("Issued expert weights differ from their complete selection record.")
        scalar = [
            math.fsum(weights[k] * opinions[k, s] for k in range(8)) for s in range(len(ASSETS))
        ]
        gaps["target"] = max(gaps["target"], float(max(abs(np.asarray(scalar) - targets[name]))))
        _target(targets[name], len(ASSETS))
    if (
        gaps["target"] > policy["independent_target_tolerance"]
        or gaps["weight"] > 1e-12
        or gaps["risk"] > 1e-12
        or gaps["score"] > policy["independent_score_tolerance"]
    ):
        raise QuantError("Independent target, ERC weight, risk or forecast-score agreement failed.")
    return {f"maximum_independent_{k}_difference": v for k, v in gaps.items()}


def decision_sessions(calendar):
    return original.decision_sessions(calendar, read_json(original.POLICY))
