from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import conditional_variance_v1 as model
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import CORE, FACTOR_IDS, UNIVERSE
from us_quant.data import MarketData
from us_quant.storage import digest_bytes

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/conditional-variance-growth-v1.json"
PRIOR_HEAD = "0c5fddb20f2e2f3f78f5993310c25b6648c31ac6c1c95b5e691020820371cb71"
CANDIDATES = (
    {"id": "symmetric_garch_growth_bil_v1", "asymmetric": False},
    {"id": "downside_gjr_growth_bil_v1", "asymmetric": True},
)
STOCKS = ("SOXX", *CORE)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_conditional_variance_growth_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 205,
        "prior_journal_sequence": 321,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "lookback_return_sessions": 252,
        "forecast_sessions": 21,
        "conditional_mean": 0.0,
        "initial_normalized_variance": 1.0,
        "normalized_intercept_bounds": [1e-8, 10.0],
        "dynamic_logit_bounds": [-20.0, 20.0],
        "garch_initial_parameters": [0.05, 0.05, 0.90, 0.0],
        "gjr_initial_parameters": [0.05, 0.05, 0.85, 0.10],
        "optimizer": "L-BFGS-B",
        "optimizer_initializations_per_fit": 1,
        "optimizer_max_iterations": 1000,
        "optimizer_ftol": 1e-12,
        "optimizer_gtol": 1e-7,
        "optimizer_max_line_search_steps": 40,
        "independent_target_tolerance": 1e-10,
        "independent_variance_relative_tolerance": 1e-9,
        "independent_mean_likelihood_tolerance": 1e-8,
        "original_source_months": 131,
        "eligible_source_months": 122,
        "undefined_prehistory_months": 9,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "growth_share_of_equity": 0.5,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed paired conditional-variance and financial contract.")
    if (
        (model.LOOKBACK, model.HORIZON) != (252, 21)
        or model.OMEGA_BOUNDS != (1e-8, 10.0)
        or model.LOGIT_LIMIT != 20.0
        or model.MAX_ITERATIONS != 1000
        or model.FTOL != 1e-12
        or model.GTOL != 1e-7
        or model.MAX_LINE_SEARCH != 40
    ):
        raise QuantError("The numerical kernel differs from its fixed policy.")


def allocation(pooled: float, forecast: float) -> pd.Series:
    if any(not math.isfinite(value) or value <= 0 for value in (pooled, forecast)):
        raise QuantError("Missing or nonpositive risk is not a full-stock/cash default.")
    stock = 0.98 if forecast <= pooled else 0.98 * (pooled / forecast)
    row = pd.Series(0.0, index=UNIVERSE)
    row["SOXX"] = stock / 2
    row.loc[list(CORE)] = stock / 8
    row["BIL"] = 0.98 - stock
    if (row < 0).any() or abs(row.sum() - 0.98) > 1e-12 or not (row.loc[list(STOCKS)] > 0).all():
        raise QuantError(
            "Risk budgeting must retain positive stock participation and cash funding."
        )
    return row


def eligible_dates(data, calendar) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [day for day in calendar.dropna(how="all").index if data.close.index.get_loc(day) >= 252]
    )


def observed_returns(data: MarketData, day) -> tuple[np.ndarray, dict]:
    position = data.close.index.get_loc(day)
    if position < 252:
        raise QuantError("Conditional variance needs252actualpast returns,not synthetic warmup.")
    prices = data.close.loc[:, list(STOCKS)].iloc[position - 252 : position + 1].to_numpy()
    if prices.shape != (253, 5) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise QuantError("Require253finitepositiveknowncloses for each stock proxy constituent.")
    gross = prices[1:] / prices[:-1]
    values = np.log(0.5 * gross[:, 0] + 0.125 * gross[:, 1:].sum(axis=1))
    if not np.isfinite(values).all():
        raise QuantError("The stock risk proxy must be a finite observed log-return series.")
    return values, {
        "decision_session": str(day.date()),
        "first_prior_close": str(data.close.index[position - 252].date()),
        "last_observation": str(day.date()),
        "observed_returns": 252,
        "stock_proxy_symbols": list(STOCKS),
        "stock_proxy_input_sha256": digest_bytes(np.asarray(values, dtype="<f8").tobytes()),
    }


def target_for_day(data, day, candidate: dict, saved: dict | None = None):
    if candidate not in CANDIDATES:
        raise QuantError("Unknown conditional-variance candidate.")
    values, information = observed_returns(data, day)
    if saved is not None:
        if (
            saved.get("status") != "completed"
            or saved.get("candidate_id") != candidate["id"]
            or saved.get("information") != information
            or saved.get("model", {}).get("asymmetric") is not candidate["asymmetric"]
        ):
            raise QuantError("Preserve a conflicting/failed dated fit without replay.")
        fitted = saved["model"]
    else:
        try:
            fitted = model.fit(values, candidate["asymmetric"])
        except QuantError as error:
            raise model.FitError(
                f"Conditional variance stopped for {candidate['id']} at {day.date()}: {error}",
                {
                    "status": "failed",
                    "candidate_id": candidate["id"],
                    "information": information,
                    "error": str(error),
                    "optimizer_receipt": error.receipt
                    if isinstance(error, model.FitError)
                    else None,
                },
            ) from error
    row = allocation(fitted["pooled_daily_squared_return"], fitted["mean_forecast_variance_daily"])
    audit = {
        "status": "completed",
        "candidate_id": candidate["id"],
        "information": information,
        "model": fitted,
        "stock_target": float(row.loc[list(STOCKS)].sum()),
        "BIL_target": float(row["BIL"]),
        "past_targets_revised": False,
        "risk_proxy_is_a_funded_account": False,
        "future_risk_guarantee_claimed": False,
    }
    if saved is not None and saved != audit:
        raise QuantError("A saved fit cannot acquire changed targets or claims.")
    return row, audit


def build_targets(
    data: MarketData,
    calendar: pd.DataFrame,
    policy: dict,
    retain_fit=None,
    before_fit=None,
    load_fit=None,
):
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not calendar.index.equals(data.close.index)
        or not calendar.columns.equals(data.close.columns)
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or (calendar.isna().any(axis=1) & ~calendar.isna().all(axis=1)).any()
        or not all(is_month_end(day) for day in calendar.dropna(how="all").index)
    ):
        raise QuantError("Keep the original complete market and monthly calendar.")
    dates = eligible_dates(data, calendar)
    if dates.empty:
        raise QuantError("No original opportunity has252observed returns.")
    targets = {
        candidate["id"]: pd.DataFrame(np.nan, index=calendar.index, columns=calendar.columns)
        for candidate in CANDIDATES
    }
    audits = {candidate["id"]: [] for candidate in CANDIDATES}
    for day in dates:
        for candidate in CANDIDATES:
            identifier = candidate["id"]
            saved = load_fit(identifier, day) if load_fit is not None else None
            if before_fit is not None:
                before_fit(identifier, day, saved is not None)
            try:
                row, audit = target_for_day(data, day, candidate, saved)
            except QuantError as error:
                if retain_fit is not None and saved is None:
                    retain_fit(
                        error.receipt
                        if isinstance(error, model.FitError)
                        else {
                            "status": "failed",
                            "candidate_id": identifier,
                            "information": {"decision_session": str(day.date())},
                            "error": str(error),
                        }
                    )
                raise
            if retain_fit is not None:
                retain_fit(audit)
            targets[identifier].loc[day] = row
            audits[identifier].append(audit)
    return targets, audits


def independent_row(data, day, audit):
    position = data.close.index.get_loc(day)
    if position < 252:
        raise QuantError("The independent reader cannot fill missing training history.")
    values = [
        math.log(
            math.fsum(
                (0.5 if name == "SOXX" else 0.125)
                * float(data.close[name].iloc[i])
                / float(data.close[name].iloc[i - 1])
                for name in STOCKS
            )
        )
        for i in range(position - 251, position + 1)
    ]
    fitted = audit["model"]
    oracle = model.independent(
        values, fitted["parameters_daily"], fitted["pooled_daily_squared_return"]
    )
    ratio = oracle["pooled_squared_return"] / oracle["mean_forecast_variance"]
    stock = 0.98 * min(1.0, ratio)
    row = pd.Series(0.0, index=UNIVERSE)
    row["SOXX"] = 0.5 * stock
    for name in CORE:
        row[name] = 0.125 * stock
    row["BIL"] = 0.98 - stock
    return row, oracle
