from __future__ import annotations

import math
from datetime import date
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-active-forecast-diagnosis-v1.json"
MODELS = ("unconditional_unit_ridge", "contextual_unit_ridge")
BENCHMARKS = ("SPY", "QQQ")
TOLERANCE = 1e-12


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_active_forecast_diagnosis_20261106_v1",
        "actual_native_event_seq": 175233,
        "prior_journal_sequence": 1246,
        "prior_journal_head_sha256": (
            "ddbdf3b81a693cd6c3b459ec85ea49bfd65ff47617edee7825465c4277274c51"
        ),
        "prior_combined_completed": 283,
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "context_symbols": ["SPY", "IEF", "BIL"],
        "context": "intercept_and_126_session_SPY_BIL_and_IEF_BIL_relative_price_returns",
        "learning_clock": "completed_full_exchange_month_intervals",
        "learning_memory": "expanding_completed_months",
        "initial_partial_source_month_assimilated": False,
        "maximum_complete_learning_intervals": 119,
        "label": "mean_daily_source_NET_minus_original_full_capital_NET_benchmark",
        "models": list(MODELS),
        "ridge_penalty": 1,
        "intercept_penalized": True,
        "source_count": 8,
        "benchmark_order": list(BENCHMARKS),
        "forecast_available_after_common_source_observations": 126,
        "unmatured_forecast": None,
        "grade_forecast_before_assimilating_its_outcome": True,
        "month_errors_equally_weighted": True,
        "independent_numeric_tolerance": TOLERANCE,
        "source_native_RF_identity_claimed_without_admitted_views": False,
        "new_source_requests": 0,
        "new_RF_views": 0,
        "new_shadow_runs": 0,
        "new_allocations": 0,
        "new_funded_paths": 0,
        "new_financial_candidates": 0,
        "select_best_expert_or_model_from_complete_results": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in fixed.items()
    ):
        raise QuantError("Freeze prediction rates, chronology and zero-allocation scope.")


def exact(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise QuantError("Use finite recorded numbers, not filled or successful predictions.")
    return Fraction.from_float(float(value))


def context(record):
    before, after = record["earlier_prices"], record["current_prices"]
    if (
        set(before) != {"SPY", "IEF", "BIL"}
        or set(after) != set(before)
        or any(exact(v) <= 0 for v in [*before.values(), *after.values()])
    ):
        raise QuantError("Both continuous price contexts need positive admitted observations.")
    return [
        1.0,
        *[
            float(
                exact(after[s]) * exact(before["BIL"]) / (exact(before[s]) * exact(after["BIL"]))
                - 1
            )
            for s in ("SPY", "IEF")
        ],
    ]


def ridge(rows, current, dimension):
    if dimension not in (1, 3) or len(current) != 3:
        raise QuantError("Only the frozen intercept-only and two-context predictors exist.")
    matrix = [[Fraction(int(i == j)) for j in range(dimension)] for i in range(dimension)]
    cross = [[Fraction(0) for _ in range(16)] for _ in range(dimension)]
    for features, outcome in rows:
        if len(features) != 3 or len(outcome) != 16 or features[0] != 1.0:
            raise QuantError("A learning row needs the same sixteen source-benchmark labels.")
        x, y = [exact(v) for v in features[:dimension]], [exact(v) for v in outcome]
        for i in range(dimension):
            for j in range(dimension):
                matrix[i][j] += x[i] * x[j]
            for j in range(16):
                cross[i][j] += x[i] * y[j]
    augmented = [a + b for a, b in zip(matrix, cross, strict=True)]
    for i in range(dimension):
        pivot = augmented[i][i]
        if pivot <= 0:
            raise QuantError("The fixed positive unit prior must have a positive exact pivot.")
        augmented[i] = [v / pivot for v in augmented[i]]
        for j in range(dimension):
            if j != i:
                ratio = augmented[j][i]
                augmented[j] = [
                    a - ratio * b for a, b in zip(augmented[j], augmented[i], strict=True)
                ]
    x = [exact(v) for v in current[:dimension]]
    result = [
        float(sum((x[i] * augmented[i][dimension + j] for i in range(dimension)), Fraction(0)))
        for j in range(16)
    ]
    if not all(math.isfinite(v) for v in result):
        raise QuantError("An undefined forecast cannot become zero or a successful score.")
    return result


def checked_input(row, position, previous):
    day = pd.Timestamp(row["as_of"])
    if (
        row["record"] != "forecast_input"
        or row["index"] != position
        or row["as_of"] != date.fromisoformat(row["as_of"]).isoformat()
        or position == 0
        and row["as_of"] != "2016-10-05"
        or position > 0
        and (day <= previous or not is_month_end(day))
        or row["information_clock"] != pool.CLOCK
        or row["context"]["as_of"] != row["as_of"]
        or len(sessions(row["context"]["earlier_session"], day)) != 127
        or row["binary64_values_truncated_to_fixed_decimals"] is not False
    ):
        raise QuantError("Preserve every original anchor, monthly clock and full-precision input.")
    observed = row["observations"]
    if (
        tuple(o["id"] for o in observed) != pool.EXPERT_IDS
        or tuple(o["mechanism_group"] for o in observed) != pool.MECHANISMS
        or any(
            o["as_of"] != row["as_of"]
            or o["target_available"] is not True
            or o["target_known_at"] is None
            or o["target_known_at"] > row["as_of"]
            for o in observed
        )
    ):
        raise QuantError("Retain all eight complete admitted identities and opinion clocks.")
    dates = observed[0]["return_dates"]
    if (
        len(dates) > 126
        or position == 0
        and dates
        or position > 0
        and not dates
        or dates
        and dates[-1] != row["as_of"]
        or dates
        and dates != sessions(dates[0], dates[-1]).strftime("%Y-%m-%d").tolist()
        or row["benchmark_dates"] != dates
        or set(row["benchmarks"]) != set(BENCHMARKS)
        or any(len(row["benchmarks"][b]) != len(dates) for b in BENCHMARKS)
        or any(
            o["return_dates"] != dates
            or len(o["NET_returns"]) != len(dates)
            or len(o["RF"]) != len(dates)
            or o["RF"] != observed[0]["RF"]
            or o["score_mature"] != (len(dates) == 126)
            for o in observed
        )
    ):
        raise QuantError("Missing or future source/benchmark dates are errors, never padding.")
    for values in (
        *[o[k] for o in observed for k in ("NET_returns", "RF")],
        *row["benchmarks"].values(),
    ):
        if any(exact(v) <= -1 for v in values):
            raise QuantError("Cash-funded NET or reference observations must exceed minus one.")
    context(row["context"])
    return day


def complete_period(previous, day):
    if previous is None or not is_month_end(previous):
        return []
    if day.year * 12 + day.month != previous.year * 12 + previous.month + 1:
        raise QuantError("A missing exchange month cannot be silently skipped.")
    return sessions(next_session(previous), day).strftime("%Y-%m-%d").tolist()


def label(row, period):
    positions = [row["benchmark_dates"].index(d) for d in period]
    return [
        float(
            sum(
                (exact(o["NET_returns"][i]) - exact(row["benchmarks"][b][i]) for i in positions),
                Fraction(0),
            )
            / len(positions)
        )
        for o in row["observations"]
        for b in BENCHMARKS
    ]


def metrics(predicted, actual):
    if len(predicted) != len(actual):
        raise QuantError("Compare the same genuinely graded forecast intervals.")
    n = len(actual)
    if not n:
        return {
            "count": 0,
            "rmse": None,
            "mae": None,
            "mean_error": None,
            "direction_accuracy": None,
            "zero_active_reference_mse": None,
            "skill_vs_zero_active": None,
            "skill_defined": False,
        }
    errors = [p - a for p, a in zip(predicted, actual, strict=True)]
    mse = math.fsum(e * e for e in errors) / n
    zero = math.fsum(a * a for a in actual) / n
    return {
        "count": n,
        "rmse": math.sqrt(mse),
        "mae": math.fsum(abs(e) for e in errors) / n,
        "mean_error": math.fsum(errors) / n,
        "direction_accuracy": sum(
            (p > 0) - (p < 0) == (a > 0) - (a < 0) for p, a in zip(predicted, actual, strict=True)
        )
        / n,
        "zero_active_reference_mse": zero,
        "skill_vs_zero_active": 1 - mse / zero if zero > 0 else None,
        "skill_defined": zero > 0,
    }


def summarize(audits):
    graded = [a["graded_prior_forecast"] for a in audits if a["graded_prior_forecast"] is not None]
    result = {}
    for model in MODELS:
        result[model] = {}
        for b, benchmark in enumerate(BENCHMARKS):
            actual, predicted, sources = [], [], {}
            for i, identifier in enumerate(pool.EXPERT_IDS):
                a = [g["actual"][2 * i + b] for g in graded]
                p = [g["predictions"][model][2 * i + b] for g in graded]
                sources[identifier] = metrics(p, a)
                actual.extend(a)
                predicted.extend(p)
            result[model][benchmark] = {"pooled": metrics(predicted, actual), "experts": sources}
    return {
        "decision_count": len(audits),
        "complete_learning_months": audits[-1]["complete_learning_months"] if audits else 0,
        "graded_months": len(graded),
        "unscored_decisions": sum(not a["forecast_available"] for a in audits),
        "models": result,
        "pooled_expert_errors_are_independent_observations": False,
        "zero_active_reference_is_a_funded_strategy": False,
        "new_allocations": 0,
        "new_financial_trials": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def analyze(inputs, retain=None, before_decision=None):
    learning, audits = [], []
    previous = previous_x = pending = None
    for position, row in enumerate(inputs):
        if before_decision is not None:
            before_decision()
        if retain is not None:
            retain(row)
        day = checked_input(row, position, previous)
        current_x = context(row["context"])
        period = complete_period(previous, day)
        actual = label(row, period) if period else None
        graded = (
            {
                "issued_at": str(previous.date()),
                "period": period,
                "actual": actual,
                "predictions": pending,
            }
            if actual is not None and pending is not None
            else None
        )
        if actual is not None:
            learning.append((previous_x, actual))
        if len(learning) > 119:
            raise QuantError("Keep the full original source horizon without appended new months.")
        available = all(o["score_mature"] for o in row["observations"])
        if available and not learning:
            raise QuantError("A mature forecast needs genuinely assimilated complete months.")
        forecast = (
            {name: ridge(learning, current_x, d) for name, d in zip(MODELS, (1, 3), strict=True)}
            if available
            else None
        )
        audit = {
            "as_of": row["as_of"],
            "context_features": current_x,
            "assimilated_period": period,
            "assimilated_label": actual,
            "graded_prior_forecast": graded,
            "complete_learning_months": len(learning),
            "forecast_available": available,
            "unscored_reason": None if available else "unmatured_source_history_no_forecast",
            "predictions": forecast,
            "forecast_is_main_account_return_or_qualification": False,
        }
        audits.append(audit)
        if retain is not None:
            retain({"record": "forecast_resolved", "index": position, "audit": audit})
        previous, previous_x, pending = day, current_x, forecast
    if not audits:
        raise QuantError("An empty source tape is not a successful forecast diagnosis.")
    return summarize(audits), audits


def verify(inputs, audits, summary):
    learning_x, learning_y, independent = [], [], []
    previous = previous_x = pending = None
    largest = 0.0
    inputs = list(inputs)
    if len(inputs) != len(audits):
        raise QuantError("Every actual retained input needs exactly one resolved forecast.")
    for position, (row, saved) in enumerate(zip(inputs, audits, strict=True)):
        day = checked_input(row, position, previous)
        before, after = row["context"]["earlier_prices"], row["context"]["current_prices"]
        x = [
            1.0,
            *[(after[s] / before[s]) / (after["BIL"] / before["BIL"]) - 1 for s in ("SPY", "IEF")],
        ]
        period = complete_period(previous, day)
        actual = None
        if period:
            locations = [row["benchmark_dates"].index(d) for d in period]
            actual = [
                (
                    math.fsum(o["NET_returns"][i] for i in locations)
                    - math.fsum(row["benchmarks"][b][i] for i in locations)
                )
                / len(locations)
                for o in row["observations"]
                for b in BENCHMARKS
            ]
        graded = (
            {
                "issued_at": str(previous.date()),
                "period": period,
                "actual": actual,
                "predictions": pending,
            }
            if actual is not None and pending is not None
            else None
        )
        if actual is not None:
            learning_x.append(previous_x)
            learning_y.append(actual)
        available = all(o["score_mature"] for o in row["observations"])
        forecasts = None
        if available:
            forecasts = {}
            for model, dimension in zip(MODELS, (1, 3), strict=True):
                features = np.asarray(learning_x)[:, :dimension]
                coefficients = np.linalg.solve(
                    np.eye(dimension) + features.T @ features,
                    features.T @ np.asarray(learning_y),
                )
                forecasts[model] = (np.asarray(x[:dimension]) @ coefficients).tolist()
        expected = {
            "as_of": row["as_of"],
            "context_features": x,
            "assimilated_period": period,
            "assimilated_label": actual,
            "graded_prior_forecast": graded,
            "complete_learning_months": len(learning_x),
            "forecast_available": available,
            "unscored_reason": None if available else "unmatured_source_history_no_forecast",
            "predictions": forecasts,
            "forecast_is_main_account_return_or_qualification": False,
        }
        largest = max(largest, compare(saved, expected))
        independent.append(expected)
        previous, previous_x, pending = day, x, forecasts
    # Three-way signs describe the recorded forecasts, not the oracle's rounding noise.
    largest = max(largest, compare(summary, summarize(audits)))
    return {
        "maximum_independent_numeric_difference": largest,
        "independent_batch_numpy_vs_exact_rational_ridge": True,
        "every_nonselected_source_and_both_predictors_checked": True,
    }


def compare(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise QuantError("Forecast evidence keys or claimed scope changed.")
        return max((compare(actual[k], v) for k, v in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise QuantError("Forecast evidence shape changed.")
        return max((compare(a, b) for a, b in zip(actual, expected, strict=True)), default=0.0)
    if type(expected) is float:
        if type(actual) not in (float, int) or not math.isfinite(actual):
            raise QuantError("Numerical forecast evidence must remain finite.")
        gap = abs(actual - expected)
        if gap > TOLERANCE:
            raise QuantError("The frozen independent prediction agreement failed.")
        return gap
    if type(actual) is not type(expected) or actual != expected:
        raise QuantError("Forecast chronology, missingness or non-numerical evidence changed.")
    return 0.0
