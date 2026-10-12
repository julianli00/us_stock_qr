from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.calendar import next_session, previous_session
from us_quant.cash_funded_accounting_v2 import rebalance, simulate
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import build_targets as dollar_targets
from us_quant.dual_horizon import seed_window
from us_quant.growth_portfolio_protection import protect_target
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import (
    file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/semiconductor-portfolio-protection.json"
CANDIDATES = (
    {"id": "semiconductor_ratchet_protection", "dollar_guard": False},
    {"id": "semiconductor_dollar_ratchet_protection", "dollar_guard": True},
)
REFERENCES = ("source_registration", "protection_policy", "protection_implementation")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "semiconductor_portfolio_protection_20261011",
        "data_scope": "factor_etf_portfolio", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 124, "new_configurations": 2,
        "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "drawdown_target": 0.15, "actual_weight_trade_band": 0.05,
        "cash_reserve": 0.02, "capital_usd": 10000.0, "commission_per_order": 1.0,
        "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Keep the original ratchet law, two fixed baselines and original funding goals.")


def protected_target(monthly: pd.Series, nav: float, peak: float) -> pd.Series:
    if (
        not isinstance(monthly, pd.Series) or not monthly.index.is_unique or "BIL" not in monthly
        or not np.isfinite(monthly).all() or (monthly < 0).any()
        or not np.isclose(monthly.sum(), 0.98, rtol=0, atol=1e-12)
        or not np.isfinite([nav, peak]).all() or nav <= 0 or peak < nav
    ):
        raise QuantError("Protection needs a complete funded monthly target and observed account wealth.")
    risky = monthly.drop("BIL")
    risk_budget = float(risky.sum())
    if risk_budget == 0:
        return monthly.copy()
    normalized = monthly.copy()
    normalized["BIL"] = 0.0
    normalized *= 0.98 / risk_budget
    protected = protect_target(normalized, nav, peak, 0.15)
    result = protected * (risk_budget / 0.98)
    result["BIL"] += monthly["BIL"]
    if (
        (result < 0).any()
        or (result.drop("BIL") > risky + 1e-12).any()
        or result["BIL"] + 1e-12 < monthly["BIL"]
        or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("A ratchet cannot increase risky capital or discard existing dollar-guard bills.")
    return result


def monthly_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    plain = sector_targets(
        data, read_json(ROOT / "config/sector-growth-balance.json"),
    )["four_factor_semiconductor_sector50"]
    guarded = dollar_targets(
        data, read_json(ROOT / "config/dollar-risk-guard.json"),
    )["semiconductor_dollar_half_guard"]
    if (
        not plain.index.equals(guarded.index) or not plain.columns.equals(guarded.columns)
        or not plain.isna().equals(guarded.isna())
    ):
        raise QuantError("Both ratchet sources must retain the original identical monthly decision dates.")
    return {
        CANDIDATES[0]["id"]: plain,
        CANDIDATES[1]["id"]: guarded,
    }


def run(
    data: MarketData, baseline: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict], list[dict]]:
    validate(policy)
    data.validate()
    if (
        candidate not in CANDIDATES or (cost, delay) not in ((5.0, 1), (20.0, 2))
        or type(delay) is not int
        or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()
    ):
        raise QuantError("Use the exact protected source and original base/stress path contract.")
    known = seed_window(baseline, start)
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("A protected account requires at least two actual return sessions.")
    anchor = previous_session(dates[0])
    monthly = known.loc[anchor].copy()
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    units = np.zeros(len(data.close.columns))
    cash = prior_nav = peak = 10000.0
    pending, decisions = {}, []
    rows, weights, states = [], [], []
    requested = False

    def schedule(day: pd.Timestamp, target: pd.Series, reason: str) -> None:
        if pending:
            raise QuantError("Never overwrite or overlap a delayed protection request.")
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        pending[execution] = target.to_numpy()
        issued.loc[day] = target
        decisions.append({
            "signal_session": day.strftime("%Y-%m-%d"), "execution_session": execution.strftime("%Y-%m-%d"),
            "reason": reason, "known_nav_usd": prior_nav, "known_peak_usd": peak,
            "monthly_bil_weight": float(monthly["BIL"]), "protected_bil_weight": float(target["BIL"]),
        })

    schedule(anchor, protected_target(monthly, prior_nav, peak), "initial_capital")
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        cost_paid = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, cost_paid, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, 1,
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < 0:
            raise QuantError("A protected source became insolvent or borrowed cash.")
        actual = pd.Series(values / nav, index=data.close.columns)
        rows.append((nav, nav / prior_nav - 1, cash, actual.sum(), turnover, cost_paid, orders))
        weights.append(actual.to_numpy())
        peak = max(peak, nav)
        prior_nav = nav
        if not known.loc[day].isna().all():
            monthly = known.loc[day].copy()
            requested = True
        desired = protected_target(monthly, nav, peak)
        if not pending and (requested or float(abs(desired - actual).max()) >= 0.05):
            schedule(day, desired, "monthly_target" if requested else "actual_weight_protection_band")
            requested = False
        states.append({
            "session": day.strftime("%Y-%m-%d"), "nav_usd": nav, "peak_usd": peak,
            "floor_usd": 0.85 * peak,
            "protection_scale": float(np.clip((nav - 0.85 * peak) / (0.15 * nav), 0, 1)),
            "monthly_bil_weight": float(monthly["BIL"]), "desired_bil_weight": float(desired["BIL"]),
            "pending_execution_sessions": [execution.strftime("%Y-%m-%d") for execution in pending],
            "monthly_update_requested": requested,
        })
    frame = pd.DataFrame(
        rows, index=dates, columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    result = BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns))
    replay = simulate(data, issued, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    if not np.allclose(replay.frame, frame, rtol=0, atol=1e-8):
        raise QuantError("The issued cost-dependent ratchet targets do not replay in frozen funded accounting.")
    return result, issued, decisions, states


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    previous = read_json(safe_file(
        ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"],
    ))
    records = [entry for entry in previous["candidates"] if entry["spec"]["id"] == "semiconductor_dollar_half_guard"]
    if len(records) != 1:
        raise QuantError("Require the original single actual dollar-guard source registration.")
    old = records[0]["spec"]
    data = review_market({"market": old["market"]}, ROOT)
    verified = verified_etf_market(previous["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (
            not left.index.equals(right.index)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("Reuse only the exact original audited actual fund/risk-free panels.")
    baselines = monthly_baselines(data, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private source directory without replacing old evidence.")
    new_output_directory(output)
    readiness = deepcopy(previous["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(old["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/semiconductor_portfolio_protection.py"] = file_digest(Path(__file__))
    frozen["src/us_quant/cash_funded_accounting_v2.py"] = file_digest(
        Path(__file__).with_name("cash_funded_accounting_v2.py"),
    )
    specs = []
    for candidate in CANDIDATES:
        spec = {
            "id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
            "market": old["market"], "frozen_files": frozen, "asset_leverage": old["asset_leverage"],
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False, "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(
            output / f"{candidate['id']}-monthly-baseline.csv",
            baselines[candidate["id"]].to_csv(float_format="%.17g"),
        )
        specs.append(spec)
    return {"specs": specs, "baseline_month_ends": len(next(iter(baselines.values())).dropna(how="all")),
            "candidate_returns_or_dynamic_protection_accounts_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed source baselines for ratcheted protection; no orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/semiconductor-protection-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({
            "prepared_configurations": len(result["specs"]), "baseline_month_ends": result["baseline_month_ends"],
            "candidate_returns_or_dynamic_protection_accounts_computed": False,
            "investment_objective_verified": False,
        }, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Semiconductor protection preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
