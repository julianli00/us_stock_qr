from __future__ import annotations

from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import next_session, previous_session
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import seed_window
from us_quant.multifactor_stability import FACTORS
from us_quant.portfolio_refinement import (
    RefinementRun,
    observed_covariance,
    rebalance_reason,
    validate_candidate,
)
from us_quant.semiconductor_trend_transfer import equity_signal
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/equity-first-regimes-v1.json"
CANDIDATES = (
    {"id": "equity_first_trend200_band025_v1", "variance_cap": False},
    {"id": "equity_first_vix_variance_band025_v1", "variance_cap": True},
)
COMPOSITION = {"SOXX": 0.5, **dict.fromkeys(FACTORS, 0.125)}
UNIVERSE = {"SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", "XLK", "SOXX", *FACTORS}


def validate(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("study_id") != "ssd_equity_first_regimes_20261027_v1"
        or policy.get("protocol_id") != "ssd_dual_benchmark_net_cagr_v1"
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("data_start") != "2015-08-10"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("prior_completed_configurations") != 178
        or policy.get("prior_recurring_reviews") != 94
        or policy.get("prior_recurring_registered_versions") != 96
        or policy.get("prior_terminal_technical_versions") != 2
        or policy.get("minimum_economic_factor_families") != 3
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("candidate_count_in_this_frozen_pair") != 2
        or policy.get("pair_size_is_a_total_research_quota") is not False
        or policy.get("equity_composition") != COMPOSITION
        or policy.get("cash_reserve") != 0.02
        or policy.get("capital_usd") != 10000
        or policy.get("commission_per_order") != 1
        or policy.get("weight_band") != 0.025
        or policy.get("trend_observations") != 200
        or policy.get("variance_reference_observations") != 252
        or policy.get("common_warmup_observations") != 253
        or policy.get("candidates") != list(CANDIDATES)
        or any(type(candidate["variance_cap"]) is not bool for candidate in policy["candidates"])
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed equity-first pair, existing families or risk boundary changed.")


def validate_inputs(data: MarketData, vix: pd.Series) -> None:
    data.validate()
    if (
        not data.close.columns.is_unique
        or set(data.close.columns) != UNIVERSE
        or not isinstance(vix, pd.Series)
        or not vix.index.equals(data.close.index)
        or not np.isfinite(vix).all()
        or (vix <= 0).any()
    ):
        raise QuantError("Require the original complete twelve-asset panel and actual aligned VIX.")
    with np.errstate(over="ignore"):
        finite_variance = np.isfinite(vix.to_numpy() ** 2).all()
    if not finite_variance:
        raise QuantError("Observed VIX cannot create a nonfinite variance or a safe scale default.")


def build_targets(data: MarketData, vix: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    validate_inputs(data, vix)
    equity = equity_signal(data.close)
    eligible = equity > equity.rolling(200, min_periods=200).mean()
    gold = data.close["GLD"] > data.close["GLD"].rolling(200, min_periods=200).mean()
    reference = vix.pow(2).shift(1).rolling(252, min_periods=252).median()
    variance_scale = (reference / vix.pow(2)).clip(upper=1.0)
    result = {}
    for candidate in CANDIDATES:
        scale = variance_scale if candidate["variance_cap"] else 1.0
        equity_weight = 0.98 * eligible.astype(float) * scale
        targets = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
        for symbol, share in COMPOSITION.items():
            targets[symbol] = equity_weight * share
        remaining = 0.98 - equity_weight
        targets["GLD"] = remaining.where(gold, 0.0)
        targets["BIL"] = remaining.where(~gold, 0.0)
        targets.iloc[:252] = np.nan
        defined = targets.dropna(how="all")
        if (
            (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
            or not np.isfinite(defined.to_numpy()).all()
            or (defined < 0).any().any()
            or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("A desired target is missing, negative or not cash funded.")
        result[candidate["id"]] = targets
    return result


def independent_targets(data: MarketData, vix: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    validate_inputs(data, vix)
    columns = list(data.close.columns)
    positions = {symbol: columns.index(symbol) for symbol in COMPOSITION}
    gold_column, bill_column = columns.index("GLD"), columns.index("BIL")
    prices = data.close.to_numpy()
    variance = vix.to_numpy() ** 2
    levels = [1.0]
    matrices = {
        candidate["id"]: np.full(prices.shape, np.nan, dtype=float) for candidate in CANDIDATES
    }
    for index in range(len(prices)):
        if index:
            change = sum(
                share
                * (prices[index, positions[symbol]] / prices[index - 1, positions[symbol]] - 1)
                for symbol, share in COMPOSITION.items()
            )
            levels.append(levels[-1] * (1 + change))
        if index < 252:
            continue
        equity_on = levels[-1] > sum(levels[-200:]) / 200
        gold_on = (
            prices[index, gold_column] > sum(prices[index - 199 : index + 1, gold_column]) / 200
        )
        risk_scale = min(1.0, median(variance[index - 252 : index].tolist()) / variance[index])
        for candidate in CANDIDATES:
            equity_weight = 0.98 * equity_on
            if candidate["variance_cap"]:
                equity_weight *= risk_scale
            row = np.zeros(len(columns))
            for symbol, share in COMPOSITION.items():
                row[positions[symbol]] = share * equity_weight
            row[gold_column if gold_on else bill_column] = 0.98 - equity_weight
            matrices[candidate["id"]][index] = row
    return {
        name: pd.DataFrame(values, index=data.close.index, columns=data.close.columns)
        for name, values in matrices.items()
    }


def pure_band() -> tuple[dict, int]:
    original = read_json(ROOT / "config/portfolio-refinement.json")
    selected = [
        row for row in original["candidates"] if row["id"] == "portfolio_observed_weight_band025"
    ]
    if (
        len(selected) != 1
        or selected[0]["weight_band"] != 0.025
        or selected[0]["qld_cap"] is not None
        or selected[0]["volatility_target"] is not None
    ):
        raise QuantError("Require the actual unchanged pure observed-holdings band.")
    validate_candidate(selected[0])
    return selected[0], original["covariance_sessions"]


def simulate_targets(
    data: MarketData,
    targets: pd.DataFrame,
    policy: dict,
    start: str,
    end: str,
    *,
    cost_bps: float,
    delay: int,
) -> RefinementRun:
    validate(policy)
    data.validate()
    if (
        type(delay) is not int
        or delay not in (1, 2)
        or (cost_bps, delay) not in ((5.0, 1), (20.0, 2))
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
    ):
        raise QuantError(
            "Use the original base/stress execution on a complete desired-target lattice."
        )
    defined = targets.dropna(how="all")
    if (
        defined.empty
        or not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Daily target intents must remain complete, unleveraged and funded.")
    dates = data.close.loc[start:end].index
    known = seed_window(targets, start)
    if len(dates) < 2 or known.loc[dates].isna().any().any():
        raise QuantError("Do not turn an unavailable daily intent into an implicit safe holding.")
    band, covariance_sessions = pure_band()
    issued = pd.DataFrame(np.nan, index=targets.index, columns=targets.columns)
    decisions: list[dict] = []
    pending: tuple[pd.Timestamp, np.ndarray] | None = None
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = float(policy["capital_usd"])
    rows, weights = [], []

    def decide(day: pd.Timestamp, current: pd.Series, *, initial: bool = False) -> None:
        nonlocal pending
        desired = known.loc[day]
        if pending is not None:
            reason = "pending_request_preserved"
        elif initial:
            reason = "initial_capital"
        else:
            covariance = observed_covariance(data, day, covariance_sessions)
            reason = rebalance_reason(current, desired, covariance, band)
        execution = None
        if reason not in {"hold", "pending_request_preserved"}:
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending = (execution, desired.to_numpy().copy())
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()) if execution is not None else None,
                "reason": reason,
                "maximum_observed_target_gap": float(abs(current - desired).max()),
            }
        )

    decide(previous_session(dates[0]), pd.Series(0.0, index=targets.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if pending is not None:
            execution, desired = pending
            if execution < day:
                raise QuantError(
                    "A dated pending request was skipped; preserve the failed account."
                )
            if execution == day:
                values, cash, costs, turnover, orders = rebalance(
                    units * opening, cash, desired, cost_bps, policy["commission_per_order"]
                )
                if turnover > 0 or orders > 0:
                    units = values / opening
                pending = None
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The new funded account became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=targets.columns)
        rows.append(
            (nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders)
        )
        weights.append(current.to_numpy())
        decide(day, current)
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    result = BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=targets.columns))
    return RefinementRun(result, issued, decisions)
