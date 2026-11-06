from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.five_hierarchical_risk_v1 import EXPERT_IDS
from us_quant.pair_risk_forecast_diagnosis_v1 import future_calendar

PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
WINDOWS = {
    key: ("2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05") for key in PATHS
}
CHANNELS = ("fixed_source_fraction_NET_RF", "actual_main_account_NET_RF")


def require(condition, message):
    if not condition:
        raise QuantError(message)


def variance(values):
    values = [float(v) for v in values]
    require(
        len(values) >= 2 and all(math.isfinite(v) for v in values),
        "Variance requires complete finite observations, not a filled label.",
    )
    if all(v == values[0] for v in values):
        return 0.0
    mean = math.fsum(values) / len(values)
    result = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    require(math.isfinite(result) and result >= 0, "A negative or nonfinite risk label is invalid.")
    return result


def forecast_variance(audit):
    weights = np.asarray(audit["expert_weights"], dtype=float)
    allocation = audit["allocation"]
    require(
        weights.shape == (5,)
        and np.isfinite(weights).all()
        and (weights >= 0).all()
        and abs(math.fsum(weights) - 1) <= 1e-12
        and allocation["risk_estimate_available"] is True
        and allocation["risk_sample_end"] == audit["decision_session"],
        "Use the exact original dated risk forecast and statistical expert allocation.",
    )
    vol = float(allocation["forecast_annualized_RF_excess_volatility"])
    covariance = np.asarray(allocation["covariance"], dtype=float)
    require(
        math.isfinite(vol)
        and vol >= 0
        and covariance.shape == (5, 5)
        and np.isfinite(covariance).all(),
        "The recorded forecast needs finite risk and unchanged covariance coordinates.",
    )
    value = vol * vol / 252
    projected = math.fsum(
        float(weights[i]) * float(weights[j]) * float(covariance[i, j])
        for i in range(5)
        for j in range(5)
    )
    require(
        math.isfinite(value) and value >= 0 and projected >= 0 and abs(value - projected) <= 1e-12,
        "Stored covariance and recorded volatility do not reconcile.",
    )
    return value, weights, abs(value - projected)


def assess(predicted, weights, source_net, risk_free, main_net):
    source_net, risk_free, main_net = (
        np.asarray(source_net, dtype=float),
        np.asarray(risk_free, dtype=float),
        np.asarray(main_net, dtype=float),
    )
    require(
        source_net.ndim == 2
        and source_net.shape[1] == 5
        and len(source_net) >= 2
        and risk_free.shape == main_net.shape == (len(source_net),)
        and weights.shape == (5,)
        and np.isfinite(weights).all()
        and (weights >= 0).all()
        and abs(math.fsum(weights) - 1) <= 1e-12
        and np.isfinite(source_net).all()
        and (source_net > -1).all()
        and np.isfinite(risk_free).all()
        and np.isfinite(main_net).all()
        and (main_net > -1).all()
        and math.isfinite(predicted)
        and predicted >= 0,
        "Use complete exact source/main NET-RF label inputs and original forecast weights.",
    )
    excess = source_net - risk_free[:, None]
    scalar_source = [
        math.fsum(float(w) * float(v) for w, v in zip(weights, row, strict=True)) for row in excess
    ]
    source_value = variance(scalar_source)
    main_values = main_net - risk_free
    main_value = variance(main_values)
    projected_source = excess @ weights
    independent_source = float(np.var(projected_source - projected_source[0], ddof=1))
    independent_main = float(np.var(main_values - main_values[0], ddof=1))
    for first, second in ((source_value, independent_source), (main_value, independent_main)):
        require(
            math.isfinite(second) and math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-18),
            "Independent scalar and shifted-vector label variances disagree.",
        )
    return {
        "predicted_daily_variance": predicted,
        "source_statistical_NET_RF_daily_variance": source_value,
        "actual_main_NET_RF_daily_variance": main_value,
        "source_signed_forecast_error": predicted - source_value,
        "main_signed_forecast_error": predicted - main_value,
        "source_absolute_forecast_error": abs(predicted - source_value),
        "main_absolute_forecast_error": abs(predicted - main_value),
        "source_underforecast": predicted < source_value,
        "main_underforecast": predicted < main_value,
        "source_to_forecast_variance_ratio": source_value / predicted if predicted > 0 else None,
        "main_to_forecast_variance_ratio": main_value / predicted if predicted > 0 else None,
        "main_minus_source_realized_variance": main_value - source_value,
        "main_minus_source_absolute_forecast_error": abs(predicted - main_value)
        - abs(predicted - source_value),
        "independent_source_variance_difference": abs(source_value - independent_source),
        "independent_main_variance_difference": abs(main_value - independent_main),
        "exact_constant_source_mixture": all(v == scalar_source[0] for v in scalar_source),
        "exact_constant_main_excess_returns": bool((main_values == main_values[0]).all()),
        "zero_forecast_risk_ratio_undefined": predicted == 0,
        "zero_variance_is_safe_success": False,
        "source_mixture_is_main_account_or_tradable_portfolio": False,
        "label_gap_is_causal_fee_or_source_profit_attribution": False,
    }


def _inputs(source_net, source_rf, mains):
    require(
        isinstance(source_net, pd.DataFrame)
        and not source_net.empty
        and tuple(source_net.columns) == EXPERT_IDS
        and source_net.index.is_unique
        and source_net.index.is_monotonic_increasing
        and source_net.index.equals(sessions("2016-10-06", source_net.index[-1]))
        and source_rf.index.equals(source_net.index)
        and np.isfinite(source_net.to_numpy()).all()
        and (source_net.to_numpy() > -1).all()
        and np.isfinite(source_rf).all()
        and tuple(mains) == PATHS,
        "Keep the original formal source NET-RF calendar and all four main contexts.",
    )
    end = source_net.index[-1]
    for key, main in mains.items():
        wanted = sessions(*WINDOWS[key])
        available = wanted[wanted <= end]
        require(
            isinstance(main, pd.DataFrame)
            and {"return", "risk_free"}.issubset(main)
            and main.index.equals(available)
            and np.isfinite(main[["return", "risk_free"]].to_numpy()).all()
            and (main["return"] > -1).all()
            and main["risk_free"].equals(source_rf.loc[main.index]),
            "Each main label context needs its exact original observed window and identical RF.",
        )


def diagnose(audits, source_net, source_rf, mains, before_case=None, retain=None):
    _inputs(source_net, source_rf, mains)
    results = []
    for position, audit in enumerate(audits):
        day = pd.Timestamp(audit["decision_session"])
        require(day <= source_net.index[-1], "A future origin cannot be a stored forecast.")
        allocation = audit["allocation"]
        for key in PATHS:
            if before_case is not None:
                before_case()
            row = {
                "decision_index": position,
                "decision_session": str(day.date()),
                "path_key": key,
                "forecast_available": allocation["risk_estimate_available"],
                "original_forecast_reason": allocation["reason"],
                "predicted_daily_variance": None,
                "label": None,
            }
            before = {"record": "risk_transfer_input", "case": row.copy(), "original_audit": audit}
            if not allocation["risk_estimate_available"]:
                require(
                    allocation["safety_or_success_claimed"] is False,
                    "Unscored source history cannot claim safe risk.",
                )
                row["status"] = "original_unscored_no_forecast"
            else:
                predicted, weights, gap = forecast_variance(audit)
                wanted = future_calendar(day)
                source_dates = wanted.intersection(source_net.index)
                main_dates = wanted.intersection(mains[key].index)
                row.update(
                    {
                        "predicted_daily_variance": predicted,
                        "recorded_forecast_projection_gap": gap,
                        "required_label_start": str(wanted[0].date()),
                        "required_label_end": str(wanted[-1].date()),
                        "required_label_sessions": len(wanted),
                        "available_source_sessions": len(source_dates),
                        "available_main_sessions": len(main_dates),
                        "initial_window_boundary_censored": wanted[0]
                        < pd.Timestamp(WINDOWS[key][0])
                        and len(main_dates) > 0,
                        "terminal_window_boundary_censored": wanted[-1]
                        > pd.Timestamp(WINDOWS[key][1]),
                    }
                )
                before.update(
                    {
                        "required_dates": wanted.strftime("%Y-%m-%d").tolist(),
                        "source_dates": source_dates.strftime("%Y-%m-%d").tolist(),
                        "main_dates": main_dates.strftime("%Y-%m-%d").tolist(),
                        "source_NET_binary64_hex": [
                            [float(v).hex() for v in values]
                            for values in source_net.loc[source_dates].to_numpy()
                        ],
                        "source_RF_binary64_hex": [
                            float(v).hex() for v in source_rf.loc[source_dates]
                        ],
                        "main_NET_binary64_hex": [
                            float(v).hex() for v in mains[key].loc[main_dates, "return"]
                        ],
                        "main_RF_binary64_hex": [
                            float(v).hex() for v in mains[key].loc[main_dates, "risk_free"]
                        ],
                    }
                )
                if wanted[-1] < pd.Timestamp(WINDOWS[key][0]):
                    row["status"] = "outside_main_path_window_not_assessed"
                elif not source_dates.equals(wanted) or not main_dates.equals(wanted):
                    row["status"] = "future_month_incomplete_not_assessed"
                else:
                    row["status"] = "complete_paired_risk_label"
            if retain is not None:
                retain(before)
            if row["status"] == "complete_paired_risk_label":
                row["label"] = assess(
                    predicted,
                    weights,
                    source_net.loc[wanted].to_numpy(),
                    source_rf.loc[wanted].to_numpy(),
                    mains[key].loc[wanted, "return"].to_numpy(),
                )
            results.append(row)
            if retain is not None:
                retain({"record": "risk_transfer_resolved", "result": row})
    return results


def summarize(rows):
    output = {}
    for key in PATHS:
        group = [r for r in rows if r["path_key"] == key]
        complete = [r["label"] for r in group if r["status"] == "complete_paired_risk_label"]
        summary = {
            "all_original_origins_retained": len(group),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "paired_complete_origins": len(complete),
            "source_mean_absolute_variance_error": None,
            "main_mean_absolute_variance_error": None,
            "source_underforecast_count": 0,
            "main_underforecast_count": 0,
            "zero_forecast_ratio_undefined_count": 0,
        }
        if complete:
            summary.update(
                {
                    "source_mean_absolute_variance_error": math.fsum(
                        r["source_absolute_forecast_error"] for r in complete
                    )
                    / len(complete),
                    "main_mean_absolute_variance_error": math.fsum(
                        r["main_absolute_forecast_error"] for r in complete
                    )
                    / len(complete),
                    "mean_main_minus_source_realized_variance": math.fsum(
                        r["main_minus_source_realized_variance"] for r in complete
                    )
                    / len(complete),
                    "mean_main_minus_source_absolute_forecast_error": math.fsum(
                        r["main_minus_source_absolute_forecast_error"] for r in complete
                    )
                    / len(complete),
                    "source_underforecast_count": sum(r["source_underforecast"] for r in complete),
                    "main_underforecast_count": sum(r["main_underforecast"] for r in complete),
                    "zero_forecast_ratio_undefined_count": sum(
                        r["zero_forecast_risk_ratio_undefined"] for r in complete
                    ),
                    "maximum_independent_source_variance_difference": max(
                        r["independent_source_variance_difference"] for r in complete
                    ),
                    "maximum_independent_main_variance_difference": max(
                        r["independent_main_variance_difference"] for r in complete
                    ),
                }
            )
        output[key] = summary
    return {
        "path_summaries": output,
        "path_contexts_are_independent_samples": False,
        "new_model_choices_targets_accounts_or_financial_trials": 0,
        "label_gap_is_causal_fee_or_source_profit_attribution": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
