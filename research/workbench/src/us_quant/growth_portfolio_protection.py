from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.cash_funded_accounting_v2 import rebalance, simulate
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import seed_window
from us_quant.growth_factor_satellite import composed_monthly_target
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/growth-portfolio-protection.json"
PRIOR = ROOT / "evidence/growth_factor_satellite_20261011_registration.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("growth_share_of_equity") != 0.5
        or policy.get("drawdown_target") != 0.15
        or policy.get("actual_weight_trade_band") != 0.05
        or policy.get("cash_reserve") != 0.02
        or policy.get("capital_usd") != 10000
        or policy.get("commission_per_order") != 1
        or policy.get("candidates")
        != [
            {"id": "four_factor_growth_gold_monthly_control", "portfolio_insurance": False},
            {"id": "four_factor_growth_gold_ratchet_protection", "portfolio_insurance": True},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed growth core, protection target or funding assumptions changed.")


def monthly_targets(data: MarketData, policy: dict) -> pd.DataFrame:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS}:
        raise QuantError("Protection requires the actual unleveraged growth/factor/defense snapshot.")
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for i, day in enumerate(data.close.index):
        if i >= 63 and is_month_end(day):
            targets.loc[day] = composed_monthly_target(
                data.close.iloc[: i + 1],
                {"growth_share_of_equity": policy["growth_share_of_equity"]},
            )
    return targets


def protect_target(monthly: pd.Series, nav: float, peak: float, limit: float) -> pd.Series:
    if (
        not np.isfinite([nav, peak, limit]).all()
        or nav <= 0
        or peak < nav
        or not 0 < limit < 1
        or not np.isfinite(monthly).all()
        or not monthly.index.is_unique
        or (monthly < 0).any()
        or "BIL" not in monthly.index
        or monthly["BIL"] != 0
        or not np.isclose(monthly.sum(), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("The protection budget needs an observed funded NAV and known monthly core.")
    scale = float(np.clip((nav - (1 - limit) * peak) / (limit * nav), 0, 1))
    target = monthly * scale
    target["BIL"] = 0.98 * (1 - scale)
    if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
        raise QuantError("Protection cannot borrow or introduce negative cash-ETF exposure.")
    return target


def run(
    data: MarketData, baseline: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate_policy(policy)
    data.validate()
    if (
        candidate not in policy["candidates"]
        or cost not in (5, 20)
        or type(delay) is not int
        or delay not in (1, 2)
        or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()
    ):
        raise QuantError("Only the frozen protection candidates and two cost/delay paths are supported.")
    known = seed_window(baseline, start)
    if not candidate["portfolio_insurance"]:
        own = simulate(
            data, known, start, end, initial_capital=10000,
            cost_bps=cost, commission=1, delay=delay,
        )
        return own, known, []
    dates = data.close.loc[start:end].index
    anchor = previous_session(dates[0])
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    monthly = known.loc[anchor].copy()
    cash = prior_nav = peak = float(policy["capital_usd"])
    units = np.zeros(len(data.close.columns))
    pending, decisions = {}, []
    rows, weights = [], []
    requested = False

    def schedule(day, target, reason):
        if pending:
            raise QuantError("Do not overwrite or overlap pending protection requests.")
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        targets.loc[day] = target
        pending[execution] = target.to_numpy()
        decisions.append({
            "signal_session": str(day.date()),
            "execution_session": str(execution.date()),
            "reason": reason,
            "known_nav_usd": prior_nav,
            "known_peak_usd": peak,
            "bil_target_weight": float(target["BIL"]),
        })

    schedule(anchor, protect_target(monthly, prior_nav, peak, 0.15), "initial_capital")
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        cost_paid = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, cost_paid, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, policy["commission_per_order"]
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < 0:
            raise QuantError("The protected portfolio became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=data.close.columns)
        rows.append((nav, nav / prior_nav - 1, cash, current.sum(), turnover, cost_paid, orders))
        weights.append(current.to_numpy())
        peak = max(peak, nav)
        prior_nav = nav
        if not known.loc[day].isna().all():
            monthly = known.loc[day].copy()
            requested = True
        if not pending:
            desired = protect_target(monthly, nav, peak, policy["drawdown_target"])
            if requested or float(abs(desired - current).max()) >= policy["actual_weight_trade_band"]:
                schedule(day, desired, "monthly_target" if requested else "actual_weight_protection_band")
                requested = False
    frame = pd.DataFrame(
        rows, index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    own = BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns))
    replay = simulate(
        data, targets, start, end, initial_capital=10000,
        cost_bps=cost, commission=1, delay=delay,
    )
    if not np.allclose(replay.frame, own.frame, rtol=0, atol=1e-8):
        raise QuantError("Feedback protection targets do not replay from the actual funded-account engine.")
    return own, targets, decisions


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    previous = read_json(PRIOR)
    verified = verified_etf_market(previous["readiness"], ROOT)
    market = previous["candidates"][0]["spec"]["market"]
    data = review_market({"market": market}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        retained, actual = getattr(data, name), getattr(verified, name)
        if (
            not retained.index.equals(actual.index)
            or not retained.columns.equals(actual.columns)
            or not np.allclose(retained, actual, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("Protection must use the exact already audited actual fund panels.")
    if not np.allclose(data.risk_free, verified.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The protection study cannot change the frozen risk-free proxy.")
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Protection evidence must remain inside the isolated workbench.")
    new_output_directory(output)
    readiness = {**previous["readiness"], "checked_at": utc_now()}
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY, Path(__file__),
            ROOT / "config/growth-factor-satellite.json",
            Path(__file__).with_name("growth_factor_satellite.py"),
            Path(__file__).with_name("multifactor_stability.py"),
            Path(__file__).with_name("cash_funded_accounting_v2.py"),
        )
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "strategy_outcomes_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare growth/factor/gold portfolio protection.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/growth-protection-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(json.dumps({
            "prepared": [row["id"] for row in result["specs"]],
            "strategy_outcomes_computed": False,
        }))
    except QuantError as exc:
        parser.exit(2, f"Growth protection blocked: {exc}\n")


if __name__ == "__main__":
    main()
