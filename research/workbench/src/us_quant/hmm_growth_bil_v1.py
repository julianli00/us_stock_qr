from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import volatility_hmm_v1 as hmm
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import CORE, FACTOR_IDS, UNIVERSE
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/hmm-growth-bil-v1.json"
PRIOR_HEAD = "c62b145cdf93aa8dedd1a5db9d47fa5d6afba2e71d5dcb1eca6b09b9b98cf8a3"
CANDIDATES = (
    {
        "id": "causal_two_state_growth_bil_v1",
        "construction": "filtered_volatility_regime_growth_BIL_package",
    },
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_hmm_growth_bil_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 200,
        "prior_journal_sequence": 263,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "lookback_return_sessions": 252,
        "states": 2,
        "emission_means_fixed_zero": True,
        "initial_probabilities": [0.5, 0.5],
        "initial_transition": [[0.95, 0.05], [0.05, 0.95]],
        "transition_dirichlet_alpha": 2.0,
        "variance_inverse_gamma_shape": 2.0,
        "variance_prior_scale_in_pooled_units": 3.0,
        "variance_pseudocount": 6.0,
        "max_em_iterations": 500,
        "relative_objective_tolerance": 1e-7,
        "independent_probability_tolerance": 1e-10,
        "independent_likelihood_tolerance": 1e-8,
        "original_source_months": 131,
        "eligible_source_months": 122,
        "undefined_prehistory_months": 9,
        "stock_target_minimum": 0.147,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "growth_share_of_equity": 0.50,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in expected.items()
    ) or any(
        policy.get("methodology", {}).get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed HMM/package/prior/financial contract.")
    if (
        hmm.MAX_ITERATIONS != 500
        or hmm.RELATIVE_OBJECTIVE_TOLERANCE != 1e-7
        or hmm.TRANSITION_PSEUDOCOUNT != 1.0
        or hmm.VARIANCE_PSEUDOCOUNT != 6.0
        or hmm.INITIAL.tolist() != [0.5, 0.5]
    ):
        raise QuantError("The statistical kernel differs from the frozen model.")


def allocation(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise QuantError("A model probability outside0..1 is not an allocation default.")
    stock = 0.147 + (0.98 - 0.147) * probability
    row = pd.Series(0.0, index=UNIVERSE)
    row["SOXX"] = stock / 2
    row.loc[list(CORE)] = stock / 8
    row["BIL"] = 0.98 - stock
    if (row < 0).any() or abs(row.sum() - 0.98) > 1e-12:
        raise QuantError("Probability blending must retain long-only cash funding.")
    return row


def eligible_dates(data, calendar):
    return pd.DatetimeIndex(
        [day for day in calendar.dropna(how="all").index if data.close.index.get_loc(day) >= 252]
    )


def target_for_day(data, day):
    position = data.close.index.get_loc(day)
    if position < 252:
        raise QuantError("The HMM needs252actualpast returns,not synthetic warmup.")
    close = data.close["SPY"].iloc[position - 252 : position + 1].to_numpy()
    if close.shape != (253,) or not np.isfinite(close).all() or (close <= 0).any():
        raise QuantError("Require253finitepositiveactualSPYcloses.")
    returns = np.log(close[1:] / close[:-1])
    try:
        model = hmm.fit(returns)
    except QuantError as error:
        raise QuantError(f"Regime fit stopped at {day.date()}: {error}") from error
    row = allocation(model["low_variance_state_probability"])
    return row, {
        "decision_session": str(day.date()),
        "first_prior_close": str(data.close.index[position - 252].date()),
        "last_observation": str(day.date()),
        "observed_returns": 252,
        "model": model,
        "stock_target": float(row.loc[[*CORE, "SOXX"]].sum()),
        "BIL_target": float(row["BIL"]),
        "past_targets_revised": False,
        "future_probability_calibration_or_risk_guarantee_claimed": False,
    }


def build_targets(
    data: MarketData, calendar: pd.DataFrame, policy: dict, retain_fit=None, before_fit=None
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
        raise QuantError("Keep the actual original calendar and immutable market.")
    dates = eligible_dates(data, calendar)
    if dates.empty:
        raise QuantError("No original opportunity has a full observed training window.")
    target = pd.DataFrame(np.nan, index=calendar.index, columns=calendar.columns)
    audits = []
    for day in dates:
        if before_fit is not None:
            before_fit()
        row, audit = target_for_day(data, day)
        target.loc[day] = row
        audits.append(audit)
        if retain_fit is not None:
            retain_fit(audit)
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(data, day, audit):
    position = data.close.index.get_loc(day)
    if position < 252:
        raise QuantError("Independent filtering cannot fill missing history.")
    returns = [
        math.log(float(data.close["SPY"].iloc[i]) / float(data.close["SPY"].iloc[i - 1]))
        for i in range(position - 251, position + 1)
    ]
    model = audit["model"]
    posterior, likelihood = hmm.independent_filter(
        returns, model["transition"], model["state_variances_daily"]
    )
    stock = 0.147 * (1 - posterior[0]) + 0.98 * posterior[0]
    row = pd.Series(0.0, index=UNIVERSE)
    row["SOXX"] = stock * 0.5
    for name in CORE:
        row[name] = stock * 0.125
    row["BIL"] = 0.98 - stock
    return row, posterior, likelihood, math.fsum(value * value for value in returns) / 252
