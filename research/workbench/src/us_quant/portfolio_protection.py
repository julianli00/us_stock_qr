from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, rebalance
from us_quant.bt_audit import independent_equity
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import (
    DualProtocol,
    gates,
    load_prices,
    load_protocol,
    metrics,
    run_window,
)
from us_quant.legacy_horizons import fetch_supplement
from us_quant.metrics import block_bootstrap
from us_quant.storage import (
    digest_json,
    file_digest,
    implementation_fingerprint,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals

EXPECTED_CANDIDATES = [
    {"id": "tipp_tqqq_m3", "risk_asset": "TQQQ", "cushion_multiplier": 3.0},
    {"id": "tipp_tqqq_m6", "risk_asset": "TQQQ", "cushion_multiplier": 6.0},
    {"id": "tipp_upro_m3", "risk_asset": "UPRO", "cushion_multiplier": 3.0},
    {"id": "tipp_upro_m6", "risk_asset": "UPRO", "cushion_multiplier": 6.0},
]


def validate_policy(policy: dict, comparison: DualProtocol) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("prior_disclosed_trials") != 48
        or policy.get("as_of") != comparison.as_of
        or policy.get("horizons_years") != list(comparison.horizons_years)
        or policy.get("capital_usd") != comparison.capital_usd
        or policy.get("data_start") != "2010-03-01"
        or policy.get("cash_reserve") != 0.02
        or policy.get("floor_fraction_of_high_water") != 0.90
        or policy.get("defensive_asset") != "BIL"
        or policy.get("risk_symbols") != ["TQQQ", "UPRO"]
        or policy.get("rebalance_band") != 0.01
        or policy.get("cost_bps_per_side") != 7.5
        or policy.get("stress_cost_bps_per_side") != 30
        or policy.get("commission_per_order") != 1
        or policy.get("base_delay_sessions") != 1
        or policy.get("stress_delay_sessions") != 2
        or policy.get("candidates") != EXPECTED_CANDIDATES
        or policy.get("parameter_search") is not False
        or policy.get("broker_order_authority") is not False
    ):
        raise QuantError("Portfolio protection must use its fixed predeclared research policy.")


def fingerprint() -> str:
    return digest_json(
        {
            "portfolio_protection": file_digest(Path(__file__)),
            "independent_accounting": file_digest(Path(__file__).with_name("bt_audit.py")),
            "original_frozen_core": implementation_fingerprint(),
        }
    )


def sources_hashes(data_root: Path, sources: Path) -> dict:
    return {
        "existing_market_manifest": file_digest(data_root / "manifest.json"),
        "methodology_sources": file_digest(sources),
    }


def register(policy: dict, comparison: DualProtocol, source_hashes: dict, output: Path) -> dict:
    validate_policy(policy, comparison)
    if output.exists():
        raise QuantError("Refusing to replace portfolio-protection preregistration.")
    result = {
        "registered_at": utc_now(),
        "policy": policy,
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "windows": comparison.windows(),
        "source_fingerprints": source_hashes,
        "required_supplemental_symbols": policy["risk_symbols"],
        "prior_disclosed_trials": 48,
        "new_trials": 4,
        "global_trials_after_round": 52,
        "registered_before_new_price_download_and_results": True,
        "risk_definition_unchanged": True,
        "order_authority": False,
        "floor_is_not_guaranteed": True,
    }
    write_json(output, result)
    return result


def verify_registration(
    policy: dict, comparison: DualProtocol, source_hashes: dict, path: Path
) -> dict:
    validate_policy(policy, comparison)
    record = read_json(path)
    if (
        record.get("policy_sha256") != digest_json(policy)
        or record.get("comparison_sha256") != digest_json(asdict(comparison))
        or record.get("source_fingerprints") != source_hashes
        or record.get("windows") != comparison.windows()
        or record.get("prior_disclosed_trials") != 48
        or record.get("new_trials") != 4
        or record.get("global_trials_after_round") != 52
    ):
        raise QuantError("Portfolio-protection rules, data evidence, or trial count changed.")
    return record


def load_data(
    policy: dict, comparison: DualProtocol, data_root: Path, supplement: Path, registration: dict
) -> MarketData:
    original = load_prices(comparison, data_root)
    manifest = read_json(supplement / "manifest.json")
    if (
        manifest.get("comparison_registration_sha256") != digest_json(registration)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or set(manifest.get("sources", {})) != set(policy["risk_symbols"])
    ):
        raise QuantError("The protection supplement does not match its preregistered coverage.")
    expected_files = {f"{symbol}.csv" for symbol in policy["risk_symbols"]}
    if not expected_files <= set(manifest["files"]):
        raise QuantError("The protection price manifest is incomplete.")
    for relative, expected in manifest["files"].items():
        path = (supplement / relative).resolve()
        if (
            not path.is_relative_to(supplement.resolve())
            or not path.is_file()
            or file_digest(path) != expected
        ):
            raise QuantError(f"Protection data artifact is missing, unsafe, or revised: {relative}")
    extra = {
        symbol: pd.read_csv(supplement / f"{symbol}.csv", index_col="date", parse_dates=["date"])
        for symbol in policy["risk_symbols"]
    }
    base_symbols = ["SPY", "QQQ", "BIL"]
    index = sessions(policy["data_start"], policy["as_of"])

    def assemble(source: pd.DataFrame, column: str) -> pd.DataFrame:
        base = source.loc[index, base_symbols]
        new = pd.DataFrame({symbol: frame[column] for symbol, frame in extra.items()})
        if not new.index.equals(index):
            raise QuantError("Supplemental leveraged ETF history has missing sessions.")
        return pd.concat([base, new], axis=1)

    result = MarketData(
        assemble(original.open, "adj_open"),
        assemble(original.close, "adj_close"),
        assemble(original.raw_close, "close"),
        assemble(original.volume, "volume"),
        original.risk_free.loc[index],
    )
    result.validate()
    return result


def risk_budget(
    equity: float, high_water: float, multiplier: float, floor_fraction: float, cash_reserve: float
) -> tuple[float, float]:
    if (
        not all(
            math.isfinite(value)
            for value in (equity, high_water, multiplier, floor_fraction, cash_reserve)
        )
        or equity <= 0
        or high_water < equity - 1e-8
        or multiplier <= 0
        or not 0 < floor_fraction < 1
        or not 0 <= cash_reserve < 1
    ):
        raise QuantError(
            "Invalid current equity, running high water, or portfolio-protection rule."
        )
    floor = floor_fraction * high_water
    risky = min(1 - cash_reserve, multiplier * max(equity - floor, 0) / equity)
    return float(risky), float(floor)


@dataclass(frozen=True)
class ProtectionRun:
    result: BacktestResult
    issued_signals: pd.DataFrame
    effective_signals: pd.DataFrame
    decisions: list[dict]
    skipped_numeric_dust_orders: int


def simulate_protection(
    data: MarketData,
    candidate: dict,
    policy: dict,
    start: str,
    end: str,
    *,
    delay: int,
    cost_bps: float,
    commission: float,
) -> ProtectionRun:
    data.validate()
    if (
        type(delay) is not int
        or delay < 1
        or not 0 <= cost_bps < 100
        or not math.isfinite(commission)
        or commission < 0
        or candidate["risk_asset"] not in data.close.columns
        or policy["defensive_asset"] not in data.close.columns
        or policy["capital_usd"] <= 0
    ):
        raise QuantError("Invalid protection simulation funding, assets, or execution delay.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("Portfolio protection needs at least two observed market sessions.")
    anchor = previous_session(dates[0])
    if anchor not in data.close.index:
        raise QuantError("Initial capital allocation requires the prior known session.")
    symbols = data.close.columns
    risk_index = symbols.get_loc(candidate["risk_asset"])
    defense_index = symbols.get_loc(policy["defensive_asset"])
    reserve = policy["cash_reserve"]
    multiplier = candidate["cushion_multiplier"]
    fraction = policy["floor_fraction_of_high_water"]
    initial = float(policy["capital_usd"])
    units = np.zeros(len(symbols), dtype=float)
    cash = last_nav = high_water = initial
    floor = fraction * initial
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=symbols)
    effective = issued.copy()
    pending = {}
    decisions, rows, weights_history = [], [], []
    skipped = 0

    def schedule(
        day: pd.Timestamp,
        allocation: float,
        equity: float,
        peak: float,
        current_risk: float,
        reason: str,
    ) -> None:
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        targets = np.zeros(len(symbols))
        targets[risk_index] = allocation
        targets[defense_index] = 1 - reserve - allocation
        issued.loc[day] = targets
        effective.loc[day] = targets
        if execution in pending:
            raise QuantError("A pending protection signal would be overwritten.")
        pending[execution] = (day, targets)
        decisions.append(
            {
                "signal_session": day.date().isoformat(),
                "execution_session": execution.date().isoformat(),
                "observed_equity": float(equity),
                "observed_high_water": float(peak),
                "floor": float(fraction * peak),
                "desired_risk_weight": float(allocation),
                "actual_risk_weight_at_decision": float(current_risk),
                "reason": reason,
            }
        )

    first_risk, _ = risk_budget(initial, initial, multiplier, fraction, reserve)
    schedule(anchor, first_risk, initial, initial, 0.0, "registered_initial_capital")
    for day in dates:
        opening = data.open.loc[day].to_numpy()
        closing = data.close.loc[day].to_numpy()
        open_values = units * opening
        open_nav = float(open_values.sum() + cash)
        open_floor_breach = open_nav < floor - 1e-7
        cost = turnover = 0.0
        tickets = 0
        source = None
        if day in pending:
            source, target = pending.pop(day)
            changes = target * open_nav - open_values
            if np.all(np.abs(changes) <= 1e-6):
                effective.loc[source] = np.nan
                skipped += 1
            else:
                exits = (target == 0) & (open_values > 1e-6)
                entries = (target * open_nav > 1e-6) & (open_values <= 1e-6)
                minimum_cost = (
                    int(exits.sum() + entries.sum()) * commission
                    + float(open_values[exits].sum()) * cost_bps / 10000
                )
                if minimum_cost >= open_nav:
                    raise QuantError(f"Protection cannot fund required costs on {day.date()}.")
                new_values, cash, cost, turnover, tickets = rebalance(
                    open_values, cash, target, cost_bps, commission
                )
                units = new_values / opening
        close_values = units * closing
        nav = float(close_values.sum() + cash)
        if not np.isfinite([nav, cash]).all() or nav <= 0 or cash < -1e-7:
            raise QuantError("Protection accounting became nonfinite or unfunded.")
        high_water = max(high_water, nav)
        desired_risk, floor = risk_budget(nav, high_water, multiplier, fraction, reserve)
        weights = close_values / nav
        actual_risk = float(weights[risk_index])
        close_floor_breach = nav < floor - 1e-7
        immediate_exit = desired_risk == 0 and actual_risk > 1e-10
        trigger = abs(desired_risk - actual_risk) >= policy["rebalance_band"] or immediate_exit
        rows.append(
            {
                "equity": nav,
                "return": nav / last_nav - 1,
                "cash": cash,
                "gross_exposure": float(weights.sum()),
                "turnover": turnover,
                "cost": cost,
                "orders": tickets,
                "risk_free": float(data.risk_free.loc[day]),
                "opening_equity_before_orders": open_nav,
                "high_water": high_water,
                "floor": floor,
                "risky_asset_weight": actual_risk,
                "desired_risk_weight": desired_risk,
                "opening_floor_breach": bool(open_floor_breach),
                "closing_floor_breach": bool(close_floor_breach),
                "decision_triggered": bool(trigger),
                "executed_signal_session": source.date().isoformat()
                if source is not None
                else None,
            }
        )
        weights_history.append(weights)
        if trigger:
            schedule(
                day,
                desired_risk,
                nav,
                high_water,
                actual_risk,
                "floor_risk_exit" if immediate_exit else "risk_weight_band",
            )
        last_nav = nav
    return ProtectionRun(
        BacktestResult(
            pd.DataFrame(rows, index=dates),
            pd.DataFrame(weights_history, index=dates, columns=symbols),
        ),
        issued,
        effective,
        decisions,
        skipped,
    )


def independent_check(
    data: MarketData,
    run: ProtectionRun,
    candidate: dict,
    policy: dict,
    start: str,
    end: str,
    delay: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict]:
    independent = independent_equity(
        data,
        run.effective_signals,
        start,
        end,
        capital=policy["capital_usd"],
        cost_bps=cost_bps,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    difference = float(abs(independent["equity"] - run.result.frame["equity"]).max())
    tolerance = policy["capital_usd"] * 1e-8
    if difference > tolerance:
        raise QuantError(f"Independent protection accounting disagrees by ${difference:.8f}.")
    peaks = np.maximum.accumulate(np.r_[policy["capital_usd"], independent["equity"]])[1:]
    expected = [
        risk_budget(
            float(nav),
            float(peak),
            candidate["cushion_multiplier"],
            policy["floor_fraction_of_high_water"],
            policy["cash_reserve"],
        )[0]
        for nav, peak in zip(independent["equity"], peaks, strict=True)
    ]
    if not np.allclose(expected, run.result.frame["desired_risk_weight"], atol=1e-10, rtol=1e-10):
        raise QuantError(
            "Risk targets do not follow the independent observed equity/high-water path."
        )
    return independent, {
        "max_equity_difference_usd": difference,
        "tolerance_usd": tolerance,
        "independent_high_water_feedback_passed": True,
        "skipped_numeric_dust_orders": run.skipped_numeric_dust_orders,
    }


def protection_metrics(frame: pd.DataFrame) -> dict:
    result = metrics(frame)
    underexposed = frame["risky_asset_weight"] < 0.01
    longest = current = 0
    for below in underexposed:
        current = current + 1 if below else 0
        longest = max(longest, current)
    result.update(
        {
            "opening_floor_breach_sessions": int(frame["opening_floor_breach"].sum()),
            "closing_floor_breach_sessions": int(frame["closing_floor_breach"].sum()),
            "average_risky_asset_weight": float(frame["risky_asset_weight"].mean()),
            "maximum_risky_asset_weight": float(frame["risky_asset_weight"].max()),
            "fraction_sessions_risk_below_one_percent": float(underexposed.mean()),
            "longest_risk_below_one_percent_sessions": longest,
            "floor_is_a_guarantee": False,
        }
    )
    return result


def evaluate(
    policy: dict,
    comparison: DualProtocol,
    registration: dict,
    data: MarketData,
    output: Path,
) -> dict:
    validate_policy(policy, comparison)
    if output.exists():
        raise QuantError("Refusing to overwrite prior portfolio-protection results.")
    if data.close.index[-1].date().isoformat() != comparison.as_of:
        raise QuantError("Protection data does not reach the common declared endpoint.")
    benchmark_paths = {}
    for window in comparison.windows():
        start, end = window["first_return_session"], window["last_session"]
        benchmark_paths[f"{window['years']}y"] = run_window(
            data, buy_and_hold_signals(data.close, "SPY", start), start, end, comparison
        )
    candidates, artifacts, checks, decisions = {}, {}, [], {}
    for candidate in policy["candidates"]:
        identifier = candidate["id"]
        record = {"candidate": candidate, "windows": {}}
        for window in comparison.windows():
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            record["windows"][key] = {}
            for stress in (False, True):
                label = "stress" if stress else "base"
                delay = policy["stress_delay_sessions"] if stress else policy["base_delay_sessions"]
                cost = policy["stress_cost_bps_per_side"] if stress else policy["cost_bps_per_side"]
                result = simulate_protection(
                    data,
                    candidate,
                    policy,
                    start,
                    end,
                    delay=delay,
                    cost_bps=cost,
                    commission=policy["commission_per_order"],
                )
                independently, audit = independent_check(
                    data, result, candidate, policy, start, end, delay, cost
                )
                values = protection_metrics(result.result.frame)
                conditions = gates(values, metrics(benchmark_paths[key].frame), comparison)
                record["windows"][key][label] = {
                    "metrics": values,
                    "gates": conditions,
                    "all_numeric_gates_passed": all(conditions.values()),
                }
                if not stress:
                    record["windows"][key]["conditional_bootstrap"] = block_bootstrap(
                        result.result.frame["return"],
                        benchmark_paths[key].frame["return"],
                        result.result.frame["risk_free"],
                        samples=1000,
                        block=21,
                        seed=20261006,
                    )
                checks.append({"candidate": identifier, "horizon": key, "scenario": label, **audit})
                prefix = f"{identifier}/{key}-{label}"
                artifacts[f"{prefix}.csv"] = result.result.frame.to_csv(float_format="%.12g")
                artifacts[f"{prefix}-bt.csv"] = independently.to_csv(float_format="%.12g")
                artifacts[f"{prefix}-weights.csv"] = result.result.weights.to_csv(
                    float_format="%.12g"
                )
                artifacts[f"{prefix}-issued.csv"] = result.issued_signals.dropna(how="all").to_csv(
                    float_format="%.12g"
                )
                decisions[f"{prefix}-decisions.json"] = result.decisions
        record["both_horizons_pass"] = all(
            value["base"]["all_numeric_gates_passed"] for value in record["windows"].values()
        )
        record["both_horizons_stress_pass"] = all(
            value["stress"]["all_numeric_gates_passed"] for value in record["windows"].values()
        )
        candidates[identifier] = record
    result = {
        "stage": "fixed_high_water_portfolio_protection_dual_horizon",
        "created_at": utc_now(),
        "registration": registration,
        "policy": policy,
        "implementation_sha256": fingerprint(),
        "windows": comparison.windows(),
        "prior_trials": 48,
        "new_trials": 4,
        "global_trials": 52,
        "candidates": candidates,
        "independent_bt_checks": checks,
        "benchmark_spy": {key: metrics(value.frame) for key, value in benchmark_paths.items()},
        "base_joint_passes": [key for key, row in candidates.items() if row["both_horizons_pass"]],
        "stress_joint_passes": [
            key
            for key, row in candidates.items()
            if row["both_horizons_pass"] and row["both_horizons_stress_pass"]
        ],
        "paper_order_authority": False,
        "investment_objective_verified": False,
        "risk_free_floor_guaranteed": False,
        "limitations": policy["disclosures"],
    }
    new_output_directory(output)
    for relative, content in artifacts.items():
        write_text_atomic(output / relative, content)
    for relative, content in decisions.items():
        write_json(output / relative, content)
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(output / "results.json", result)
    lines = [
        "# TIPP-inspired portfolio-level risk control",
        "",
        "**A ratcheting floor is a hypothesis, not a guarantee; next-open gap risk remains.**",
        "",
        "| Rule | 10y CAGR | Sharpe | Drawdown | 5y CAGR | Sharpe | Drawdown | Both pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in candidates.items():
        cells = []
        for horizon in ("10y", "5y"):
            value = row["windows"][horizon]["base"]["metrics"]
            sharpe = "undefined" if value["sharpe"] is None else f"{value['sharpe']:.2f}"
            cells.append(f"{value['cagr']:.2%} | {sharpe} | {value['max_drawdown']:.2%}")
        lines.append(f"| {name} | {' | '.join(cells)} | {row['both_horizons_pass']} |")
    lines.extend(["", *policy["disclosures"], ""])
    write_text_atomic(output / "report.md", "\n".join(lines))
    return result


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser("protection", help="Fixed TIPP-inspired research; no orders.")
    command.add_argument("stage", choices=["register", "fetch", "evaluate"])
    command.add_argument("--policy", type=Path, default=Path("config/portfolio-protection.json"))
    command.add_argument("--comparison", type=Path, default=Path("config/dual-horizon.json"))
    command.add_argument(
        "--sources", type=Path, default=Path("data/portfolio-protection/sources.json")
    )
    command.add_argument(
        "--registration", type=Path, default=Path("reports/portfolio-protection/registration.json")
    )
    command.add_argument("--data", type=Path, default=Path("data/dual-horizon/market"))
    command.add_argument(
        "--supplement", type=Path, default=Path("data/portfolio-protection/market")
    )
    command.add_argument(
        "--output", type=Path, default=Path("reports/portfolio-protection/evaluation")
    )


def dispatch_protection(args: argparse.Namespace) -> dict:
    policy, comparison = read_json(args.policy), load_protocol(args.comparison)
    sources = sources_hashes(args.data, args.sources)
    if args.stage == "register":
        record = register(policy, comparison, sources, args.registration)
        return {
            key: record[key]
            for key in (
                "registered_at",
                "prior_disclosed_trials",
                "new_trials",
                "global_trials_after_round",
                "order_authority",
                "floor_is_not_guaranteed",
            )
        }
    registration = verify_registration(policy, comparison, sources, args.registration)
    if args.stage == "fetch":
        manifest = fetch_supplement(
            replace(comparison, data_start=policy["data_start"]), registration, args.supplement
        )
        return {
            "series": len(manifest["sources"]),
            "data_start": manifest["data_start"],
            "data_end": manifest["data_end"],
        }
    data = load_data(policy, comparison, args.data, args.supplement, registration)
    result = evaluate(policy, comparison, registration, data, args.output)
    return {
        key: result[key]
        for key in (
            "stage",
            "global_trials",
            "base_joint_passes",
            "stress_joint_passes",
            "risk_free_floor_guaranteed",
            "paper_order_authority",
            "investment_objective_verified",
        )
    }
