from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import FACTOR_IDS, UNIVERSE
from us_quant.data import MarketData
from us_quant.dual_horizon import seed_window

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/nearest-band-execution-v1.json"
PRIOR_HEAD = "ce504e95884a2db90484b1b6f3cc32b656db96d13cfc1d4933731d4453a2e267"
BAND = 0.025
BUDGET = 0.98
TOL = 1e-10
CANDIDATES = (
    {
        "id": "daily_proxy_nearest_band_execution_v1",
        "execution": "project_actual_weights_to_source_band",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_nearest_band_execution_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 190,
        "prior_journal_sequence": 142,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_daily_variance_disagreement_v1",
        "capital_usd": 10000,
        "commission_per_order": 1.0,
        "cash_reserve": 0.02,
        "weight_band": BAND,
        "band_trigger_tolerance": 1e-12,
        "projection_tolerance": TOL,
        "projection_iterations": 100,
        "monthly_decisions": 131,
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
        "source_registration": {
            "path": "evidence/daily_variance_disagreement_20261028_registration_v1.json",
            "sha256": "fe9d39e5bab50f33a71c36a303893e7ff6e9e2450bb39750fdd5855a4ee3d176",
        },
        "source_targets": {
            "path": (
                "data/ssd-daily-variance-disagreement-20261028-v1/"
                "dollar_option_daily_variance_disagreement_v1/targets.csv"
            ),
            "sha256": "665f4a7c8e43fc6030eddf6b6c0bb4b342beeef9caeb683dd87271008a49ec8d",
        },
    }
    if any(
        policy.get(k) != x or type(policy.get(k)) is not type(x) for k, x in expected.items()
    ) or any(
        policy.get("methodology", {}).get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the single fixed nearest-band law, source and accounting limits.")


def bounds(current: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        current.ndim != 1
        or current.shape != desired.shape
        or not len(current)
        or not np.isfinite(current).all()
        or not np.isfinite(desired).all()
        or (current < 0).any()
        or (desired < 0).any()
        or current.sum() > 1 + TOL
        or not np.isclose(desired.sum(), BUDGET, rtol=0, atol=1e-12)
    ):
        raise QuantError(
            "Projection needs actual unleveraged weights and a funded original target."
        )
    active = (current > 0) | (desired > 0)
    lower = np.where(active, np.maximum(0, desired - BAND), 0)
    upper = np.where(active, np.minimum(1, desired + BAND), 0)
    if lower.sum() > BUDGET + TOL or upper.sum() < BUDGET - TOL:
        raise QuantError("The original target band has no feasible funded allocation.")
    return lower, upper


def certificate(current: np.ndarray, desired: np.ndarray, target: np.ndarray, dual: float) -> dict:
    lower, upper = bounds(current, desired)
    gradient = target - current + dual
    adjustable = upper > lower
    interior = adjustable & (target > lower + TOL) & (target < upper - TOL)
    at_lower = adjustable & (target <= lower + TOL)
    at_upper = adjustable & (target >= upper - TOL)
    errors = {
        "budget_error": abs(float(target.sum()) - BUDGET),
        "box_violation": float(max(0, np.max(lower - target), np.max(target - upper))),
        "interior_stationarity_error": float(np.max(np.abs(gradient[interior]), initial=0)),
        "lower_sign_error": float(np.max(np.maximum(-gradient[at_lower], 0), initial=0)),
        "upper_sign_error": float(np.max(np.maximum(gradient[at_upper], 0), initial=0)),
    }
    if (
        target.shape != current.shape
        or not np.isfinite(target).all()
        or not np.isfinite(dual)
        or any(value > TOL or not np.isfinite(value) for value in errors.values())
    ):
        raise QuantError("Independent convex projection conditions failed.")
    return {
        **errors,
        "dual_lambda": float(dual),
        "squared_weight_movement": float(np.square(target - current).sum()),
        "original_center_squared_movement": float(np.square(desired - current).sum()),
        "minimum_fee_or_future_cost_claim": False,
    }


def independent_projection(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    lower, upper = bounds(current, desired)
    points = np.unique(np.r_[current - upper, current - lower])
    for left, right in zip(points[:-1], points[1:], strict=True):
        middle = (left + right) / 2
        free = (current - middle > lower) & (current - middle < upper)
        if not free.any():
            trial = np.clip(current - middle, lower, upper)
            if abs(float(trial.sum()) - BUDGET) <= 1e-12:
                return trial
            continue
        fixed = np.clip(current - middle, lower, upper)
        dual = (float(current[free].sum() + fixed[~free].sum()) - BUDGET) / int(free.sum())
        if left - 1e-12 <= dual <= right + 1e-12:
            trial = np.clip(current - dual, lower, upper)
            if abs(float(trial.sum()) - BUDGET) <= 1e-12:
                return trial
    for point in points:
        trial = np.clip(current - point, lower, upper)
        if abs(float(trial.sum()) - BUDGET) <= 1e-12:
            return trial
    raise QuantError("Independent breakpoint projection could not satisfy the budget.")


def project(current: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, dict]:
    current, desired = np.asarray(current, dtype=float), np.asarray(desired, dtype=float)
    lower, upper = bounds(current, desired)
    left = float(np.min(current - upper)) - 1
    right = float(np.max(current - lower)) + 1
    for _ in range(100):
        middle = (left + right) / 2
        target = np.clip(current - middle, lower, upper)
        if target.sum() > BUDGET:
            left = middle
        else:
            right = middle
    dual = (left + right) / 2
    target = np.clip(current - dual, lower, upper)
    proof = certificate(current, desired, target, dual)
    oracle = independent_projection(current, desired)
    difference = float(np.max(np.abs(target - oracle)))
    if difference > 1e-12 or proof["squared_weight_movement"] > (
        proof["original_center_squared_movement"] + TOL
    ):
        raise QuantError("The bisection and independent breakpoint projections disagree.")
    return target, {**proof, "independent_breakpoint_maximum_difference": difference}


def run(
    data: MarketData,
    desired: pd.DataFrame,
    policy: dict,
    start: str,
    end: str,
    cost: float,
    delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != tuple(UNIVERSE)
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
        or type(delay) is not int
        or delay not in (1, 2)
        or cost not in (5.0, 20.0)
        or not np.isfinite(cost)
    ):
        raise QuantError(
            "Use the complete original monthly source and original execution scenarios."
        )
    active = desired.dropna()
    if (
        active.empty
        or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active).all().all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), BUDGET, rtol=0, atol=1e-12)
    ):
        raise QuantError("The frozen desired source must remain monthly, finite and fully funded.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2 or not dates.equals(sessions(start, end)):
        raise QuantError("Preserve every original formal return session.")
    known = seed_window(desired, str(dates[0].date()))
    anchor = previous_session(dates[0])
    issued = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    pending, decisions, rows, weight_rows = {}, [], [], []
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = float(policy["capital_usd"])

    def decide(day: pd.Timestamp, actual: pd.Series, initial: bool = False) -> None:
        center = known.loc[day]
        gap = float(np.max(np.abs(actual - center)))
        reason = "initial_capital" if initial else "hold"
        proof = None
        target = None
        if initial:
            target = center.to_numpy().copy()
        elif gap + policy["band_trigger_tolerance"] >= BAND:
            target, proof = project(actual.to_numpy(), center.to_numpy())
            reason = "nearest_band"
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if target is not None:
            if execution in pending:
                raise QuantError(
                    "A projected request cannot overwrite an existing delayed request."
                )
            pending[execution] = target.copy()
            issued.loc[day] = target
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()),
                "initial_capital": initial,
                "reason": reason,
                "maximum_observed_source_gap": gap,
                "actual_closing_weights": actual.tolist(),
                "source_desired_weights": center.tolist(),
                "issued_projected_target": target.tolist() if target is not None else None,
                "projection_certificate": proof,
            }
        )

    decide(anchor, pd.Series(0.0, index=desired.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        fee = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, fee, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, policy["commission_per_order"]
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The projected account became insolvent or borrowed cash.")
        held = pd.Series(values / nav, index=desired.columns)
        rows.append((nav, nav / previous_nav - 1, cash, float(held.sum()), turnover, fee, orders))
        weight_rows.append(held.to_numpy())
        if not known.loc[day].isna().all():
            if not is_month_end(day):
                raise QuantError("Projection cannot introduce an extra nonmonthly decision.")
            decide(day, held)
        previous_nav = nav
    account = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    account["risk_free"] = data.risk_free.loc[dates]
    return (
        BacktestResult(account, pd.DataFrame(weight_rows, index=dates, columns=desired.columns)),
        issued,
        decisions,
    )
