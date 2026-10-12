from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, rebalance, simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import load_protocol, seed_window
from us_quant.multifactor_stability import FACTORS, cap_volatility, load_market
from us_quant.research_program import ResearchProgram, review_market, safe_file, timestamp
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/factor-gold-risk.json"


def validate(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("risk_lookback") != 63
        or policy.get("equity_sleeve_min") != 0.30
        or policy.get("equity_sleeve_max") != 0.70
        or policy.get("cash_reserve") != 0.02
        or policy.get("capital_usd") != 10000
        or policy.get("commission_per_order") != 1.0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("candidates")
        != [
            {
                "id": "four_factor_gold_risk_balance",
                "daily_volatility_target": None,
                "daily_weight_band": None,
            },
            {
                "id": "four_factor_gold_daily_vol12",
                "daily_volatility_target": 0.12,
                "daily_weight_band": 0.05,
            },
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Factor/gold study must keep both fixed unleveraged hypotheses.")


def monthly_targets(data: MarketData, policy: dict) -> pd.DataFrame:
    validate(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "GLD", "BIL", *FACTORS}:
        raise QuantError("The study requires the verified eight-instrument ETF snapshot.")
    daily = data.close.pct_change(fill_method=None)
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for i, day in enumerate(data.close.index):
        if i < policy["risk_lookback"] or not is_month_end(day):
            continue
        past = daily.iloc[i - policy["risk_lookback"] + 1 : i + 1]
        equity_vol = float(past.loc[:, list(FACTORS)].mean(axis=1).std(ddof=1))
        gold_vol = float(past["GLD"].std(ddof=1))
        if not np.isfinite([equity_vol, gold_vol]).all() or min(equity_vol, gold_vol) <= 1e-10:
            raise QuantError("Risk balance needs positive observed sleeve volatilities.")
        equity = np.clip(
            gold_vol / (equity_vol + gold_vol),
            policy["equity_sleeve_min"],
            policy["equity_sleeve_max"],
        )
        targets.loc[day] = 0.0
        targets.loc[day, list(FACTORS)] = (1 - policy["cash_reserve"]) * equity / 4
        targets.loc[day, "GLD"] = (1 - policy["cash_reserve"]) * (1 - equity)
    return targets


def run_candidate(
    data: MarketData,
    baseline: pd.DataFrame,
    candidate: dict,
    policy: dict,
    start: str,
    end: str,
    cost: float,
    delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    if candidate not in policy["candidates"] or delay not in (1, 2) or cost not in (5, 20):
        raise QuantError("Only registered candidates and base/stress scenarios are supported.")
    data.validate()
    known = seed_window(baseline, start)
    dates = data.close.loc[start:end].index
    anchor = previous_session(dates[0])
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    monthly = known.loc[anchor].copy()
    pending, decisions = {}, []
    cash = previous_nav = float(policy["capital_usd"])
    units = np.zeros(len(data.close.columns))
    rows, weights = [], []
    requested = True

    def desired(day: pd.Timestamp) -> pd.Series:
        if candidate["daily_volatility_target"] is None:
            return monthly.copy()
        past = (
            data.close.loc[:day]
            .tail(policy["risk_lookback"] + 1)
            .pct_change(fill_method=None)
            .iloc[1:]
        )
        if len(past) != policy["risk_lookback"]:
            raise QuantError("Daily risk forecast requires complete prior observations.")
        return cap_volatility(
            monthly,
            past.cov() * 252,
            candidate["daily_volatility_target"],
            1 - policy["cash_reserve"],
        )

    def schedule(day: pd.Timestamp, target: pd.Series, reason: str) -> None:
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if pending:
            raise QuantError("Modeled decisions cannot overlap or overwrite pending targets.")
        issued.loc[day] = target
        pending[execution] = target.to_numpy()
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()),
                "reason": reason,
                "target_equity": float(target.loc[list(FACTORS)].sum()),
                "target_gold": float(target["GLD"]),
                "target_bil": float(target["BIL"]),
            }
        )

    schedule(anchor, desired(anchor), "initial_capital")
    requested = False
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        cost_paid = turnover = 0.0
        orders = 0
        if day in pending:
            dollars, cash, cost_paid, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, policy["commission_per_order"]
            )
            units = dollars / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The risk-balanced portfolio became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=data.close.columns)
        rows.append((nav, nav / previous_nav - 1, cash, current.sum(), turnover, cost_paid, orders))
        weights.append(current.to_numpy())
        if not known.loc[day].isna().all():
            monthly = known.loc[day].copy()
            requested = True
        if not pending and (requested or candidate["daily_volatility_target"] is not None):
            target = desired(day)
            if requested or float(abs(target - current).max()) >= candidate["daily_weight_band"]:
                schedule(day, target, "monthly_target" if requested else "daily_risk_band")
                requested = False
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    own = BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns))
    replay = simulate(
        data,
        issued,
        start,
        end,
        initial_capital=policy["capital_usd"],
        cost_bps=cost,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    if not np.allclose(replay.frame, own.frame, rtol=0, atol=1e-8):
        raise QuantError("Dynamic risk decisions do not reproduce in the frozen accounting engine.")
    return own, issued, decisions


def prepare(output: Path) -> dict:
    policy = read_json(CONFIG)
    validate(policy)
    original_policy = ROOT / "config/multifactor-stability.json"
    original_registration = ROOT / "evidence/multifactor_stability_20261010_registration_v2.json"
    base = ROOT / "data/factor-round-20261007/base"
    funds = ROOT / "data/multifactor-stability-20261010"
    data = load_market(read_json(original_policy), read_json(original_registration), base, funds)
    new_output_directory(output)
    market = {}
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"market-{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        market[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    source = {
        "adapter": "frozen_multifactor_etf_20261010",
        "policy": original_policy.relative_to(ROOT).as_posix(),
        "policy_sha256": file_digest(original_policy),
        "registration": original_registration.relative_to(ROOT).as_posix(),
        "registration_sha256": file_digest(original_registration),
        "base_manifest": (base / "manifest.json").relative_to(ROOT).as_posix(),
        "base_manifest_sha256": file_digest(base / "manifest.json"),
        "factor_manifest": (funds / "manifest.json").relative_to(ROOT).as_posix(),
        "factor_manifest_sha256": file_digest(funds / "manifest.json"),
    }
    readiness = {
        "schema_version": 1,
        "checked_at": timestamp().isoformat(),
        "data_scope": "factor_etf_portfolio",
        "verified_etf_source": source,
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": source["policy"]
                if key == "factor_mandates"
                else source["factor_manifest"],
                "evidence_sha256": source["policy_sha256"]
                if key == "factor_mandates"
                else source["factor_manifest_sha256"],
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "stock_data_still_blocked": True,
        "price_as_of": policy["as_of"],
        "revalidation_not_new_market_observation": True,
    }
    write_json(output / "readiness.json", readiness)
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio",
            "configuration": candidate,
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": {
                CONFIG.relative_to(ROOT).as_posix(): file_digest(CONFIG),
                Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
            },
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    return {"readiness": readiness, "specs": specs, "strategy_outcomes_computed": False}


def register(prepared: Path, receipt: Path) -> dict:
    if receipt.exists():
        raise QuantError("Refusing to replace a candidate registration receipt.")
    policy = read_json(CONFIG)
    readiness = read_json(prepared / "readiness.json")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        records = []
        for candidate in policy["candidates"]:
            spec = read_json(prepared / f"{candidate['id']}-spec.json")
            previous = program.db.execute(
                "SELECT spec_sha FROM candidates WHERE id=?", (candidate["id"],)
            ).fetchone()
            if previous is None:
                registration = program.register_candidate(spec, readiness)
            else:
                program.verify()
                if previous["spec_sha"] != digest_json(spec):
                    raise QuantError(
                        "An interrupted registration has a different frozen specification."
                    )
                registrations = [
                    json.loads(row["body"])
                    for row in program.db.execute(
                        "SELECT body FROM events WHERE kind='candidate' ORDER BY seq"
                    )
                ]
                registration = next(
                    row for row in registrations if row["candidate_id"] == candidate["id"]
                )
            records.append({"spec": spec, "registration": registration})
        result = {
            "study": policy,
            "readiness": readiness,
            "candidates": records,
            "registered_before_outcomes": True,
        }
        write_json(receipt, result)
        return result
    finally:
        program.close()


def evaluate(receipt: Path, output: Path) -> dict:
    registered = read_json(receipt)
    policy = read_json(CONFIG)
    validate(policy)
    if registered["study"] != policy or {row["spec"]["id"] for row in registered["candidates"]} != {
        row["id"] for row in policy["candidates"]
    }:
        raise QuantError("The study registration changed or omitted a frozen candidate.")
    for record in registered["candidates"]:
        for relative, digest in record["spec"]["frozen_files"].items():
            safe_file(ROOT, relative, digest)
    new_output_directory(output)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    summary = {"candidates": {}, "order_authority": False, "independent_forward_validation": False}
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    try:
        for record in registered["candidates"]:
            spec = record["spec"]
            existing = program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (spec["id"],)
            ).fetchone()
            if existing is not None:
                summary["candidates"][spec["id"]] = json.loads(existing["body"])
                continue
            data = review_market({"market": spec["market"]}, ROOT)
            baseline = monthly_targets(data, policy)
            candidate = spec["configuration"]
            paths = []
            for window in comparison.windows():
                years, start, end = (
                    window["years"],
                    window["first_return_session"],
                    window["last_session"],
                )
                for scenario, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                    own, targets, decisions = run_candidate(
                        data, baseline, candidate, policy, start, end, cost, delay
                    )
                    independent = independent_equity(
                        data,
                        targets,
                        start,
                        end,
                        capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    spy_targets = buy_and_hold_signals(data.close, "SPY", start)
                    spy = simulate(
                        data,
                        spy_targets,
                        start,
                        end,
                        initial_capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    spy_bt = independent_equity(
                        data,
                        spy_targets,
                        start,
                        end,
                        capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    entry = {
                        "years": years,
                        "scenario": scenario,
                        "capital_usd": 10000,
                        "cost_bps": cost,
                        "delay_sessions": delay,
                        "commission_per_order": 1,
                    }
                    prefix = output / spec["id"] / f"{years}y-{scenario}"
                    for key, frame in (
                        ("strategy", own.frame),
                        ("strategy_bt", independent),
                        ("spy", spy.frame),
                        ("spy_bt", spy_bt),
                        ("weights", own.weights),
                        ("targets", targets),
                    ):
                        path = prefix.parent / f"{prefix.name}-{key}.csv"
                        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                        entry[key] = {
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": file_digest(path),
                        }
                    write_json(
                        prefix.parent / f"{prefix.name}-decisions.json", {"decisions": decisions}
                    )
                    paths.append(entry)
            bundle = {
                "candidate_spec_sha256": record["registration"]["spec_sha256"],
                "completed_at": timestamp().isoformat(),
                "as_of": policy["as_of"],
                "market": spec["market"],
                "data_scope": "factor_etf_portfolio",
                "leveraged_products_allowed": False,
                "order_authority": False,
                "history_status": spec["history_status"],
                "paths": paths,
            }
            write_json(output / spec["id"] / "bundle.json", bundle)
            result = program.review(spec["id"], bundle)
            summary["candidates"][spec["id"]] = result
            write_json(output / "progress.json", {"completed": list(summary["candidates"])})
        summary["program"] = program.status()
        write_json(output / "results.json", summary)
        return summary
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two frozen unleveraged factor/gold risk hypotheses."
    )
    parser.add_argument("stage", choices=("prepare", "register", "evaluate"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/factor-gold-risk-20261010")
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "evidence/factor_gold_risk_20261010_registration.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/factor-gold-risk-20261010")
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            result = prepare(args.prepared)
            print(json.dumps({"prepared": len(result["specs"]), "returns_computed": False}))
        elif args.stage == "register":
            result = register(args.prepared, args.registration)
            print(json.dumps({"registered": [x["spec"]["id"] for x in result["candidates"]]}))
        else:
            result = evaluate(args.registration, args.output)
            print(
                json.dumps(
                    {
                        name: {
                            "status": row["status"],
                            "paths": [
                                {
                                    "years": p["years"],
                                    "scenario": p["scenario"],
                                    **p["metrics"],
                                    "gates": p["gates"],
                                }
                                for p in row["paths"]
                            ],
                        }
                        for name, row in result["candidates"].items()
                    },
                    indent=2,
                )
            )
    except QuantError as exc:
        parser.exit(2, f"Factor/gold research blocked: {exc}\n")


if __name__ == "__main__":
    main()
