"""Python ports of public OLMAR/PAMR update rules, with separate execution accounting.

Based on OLPS by Bin Li, Doyen Sahoo, and Steven C.H. Hoi, Copyright 2009-2015.
Licensed under Apache-2.0; see notices/OLPS-LICENSE.txt and notices/OLPS-NOTICE.txt.
Modified 2026-10-06: Python/NumPy port, explicit input guards, warmup, 2% cash,
and next-open execution with separately modeled fees. Not an upstream run engine.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, rebalance
from us_quant.bt_audit import independent_equity
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import (
    DualProtocol,
    gates,
    load_prices,
    load_protocol,
    metrics,
    rolling_diagnostics,
    run_window,
    seed_window,
)
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals


def validate(policy: dict, comparison: DualProtocol) -> None:
    expected_candidates = [
        {"id": "olmar_w5_eps10", "method": "olmar", "window": 5, "epsilon": 10.0},
        {"id": "pamr_eps05", "method": "pamr", "window": 1, "epsilon": 0.5},
    ]
    if (
        policy.get("schema_version") != 1
        or policy.get("prior_disclosed_trials") != 46
        or policy.get("as_of") != comparison.as_of
        or policy.get("horizons_years") != list(comparison.horizons_years)
        or policy.get("capital_usd") != comparison.capital_usd
        or policy.get("cash_reserve") != comparison.cash_reserve
        or policy.get("warmup_sessions") != 6
        or policy.get("parameter_search") is not False
        or policy.get("broker_order_authority") is not False
        or policy.get("research_history_previously_viewed") is not True
        or policy.get("candidates") != expected_candidates
        or policy.get("upstream", {}).get("license") != "Apache-2.0"
    ):
        raise QuantError("Online reversion must follow its fixed two-method research policy.")


def simplex_projection(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise QuantError("A finite nonempty vector is required for a portfolio projection.")
    centered = values - values.max()
    ordered = np.sort(centered)[::-1]
    cumulative = np.cumsum(ordered) - 1
    coordinates = np.arange(1, len(values) + 1)
    active = np.flatnonzero(ordered > cumulative / coordinates)
    if not len(active):
        raise QuantError("No feasible unit-simplex projection exists.")
    last = active[-1]
    threshold = cumulative[last] / (last + 1)
    result = np.maximum(centered - threshold, 0)
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1, atol=1e-9):
        raise QuantError("The projected portfolio does not have a finite unit budget.")
    return result / result.sum()


def update_weights(
    weights: np.ndarray, relative: np.ndarray, method: str, epsilon: float
) -> np.ndarray:
    old, ratio = np.asarray(weights, dtype=float), np.asarray(relative, dtype=float)
    if (
        old.ndim != 1
        or old.shape != ratio.shape
        or len(old) == 0
        or not np.isfinite(old).all()
        or not np.isfinite(ratio).all()
        or (old < 0).any()
        or (ratio <= 0).any()
        or not np.isclose(old.sum(), 1, atol=1e-10)
        or not np.isfinite(epsilon)
    ):
        raise QuantError("Online updates require finite positive relatives and simplex weights.")
    centered = ratio - ratio.mean()
    variance = float(centered @ centered)
    if method == "olmar" and epsilon >= 1:
        loss = max(0.0, epsilon - float(old @ ratio))
        sign = 1
    elif method == "pamr" and epsilon >= 0:
        loss = max(0.0, float(old @ ratio) - epsilon)
        sign = -1
    else:
        raise QuantError("Unknown online method or invalid threshold.")
    if variance == 0 or loss == 0:
        return old.copy()
    multiplier = loss / variance
    proposal = old + sign * multiplier * centered
    if not np.isfinite(proposal).all():
        raise QuantError("Online update overflowed; do not replace it with an invented allocation.")
    return simplex_projection(proposal)


def build_online_signals(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    data.validate()
    close = data.close.to_numpy()
    dimension = len(data.close.columns)
    if dimension < 2:
        raise QuantError("Online relative-asset rules require at least two assets.")
    results = {}
    for candidate in policy["candidates"]:
        weights = np.full(dimension, 1 / dimension, dtype=float)
        signal = np.full_like(close, np.nan)
        for index in range(policy["warmup_sessions"] - 1, len(close)):
            if candidate["method"] == "olmar":
                prediction = (
                    close[index - candidate["window"] + 1 : index + 1].mean(axis=0) / close[index]
                )
            else:
                prediction = close[index] / close[index - 1]
            weights = update_weights(weights, prediction, candidate["method"], candidate["epsilon"])
            signal[index] = weights * (1 - policy["cash_reserve"])
        frame = pd.DataFrame(signal, index=data.close.index, columns=data.close.columns)
        known = frame.dropna(how="all")
        if (known < 0).any().any() or not np.allclose(
            known.sum(axis=1), 1 - policy["cash_reserve"], atol=1e-10
        ):
            raise QuantError("Online portfolio violated the long-only cash budget.")
        results[candidate["id"]] = frame
    return results


class SimulationFundingError(QuantError):
    def __init__(self, day: pd.Timestamp, equity: float):
        self.day = day
        self.equity = equity
        super().__init__(
            f"Required modeled transaction charges exceed available portfolio capital "
            f"on {day.date()} (${equity:.6f}); the rule cannot continue."
        )


def actionable_schedule(
    data: MarketData,
    signals: pd.DataFrame,
    start: str,
    end: str,
    comparison: DualProtocol,
    *,
    stress: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Do not create a paid ticket when every intended dollar change is numerical dust."""
    delay = 1 + (comparison.stress_additional_delay_sessions if stress else 0)
    cost = comparison.stress_cost_bps_per_side if stress else comparison.cost_bps_per_side
    if not signals.index.equals(data.open.index) or not signals.columns.equals(data.open.columns):
        raise QuantError("Actionable schedule must align with all opening prices.")
    if (signals.isna().any(axis=1) & ~signals.isna().all(axis=1)).any():
        raise QuantError("A target may not have partially unspecified asset weights.")
    output = signals.copy()
    scheduled = signals.shift(delay)
    units = np.zeros(len(data.open.columns))
    cash = comparison.capital_usd
    skipped = 0
    for day in data.open.loc[start:end].index:
        target = scheduled.loc[day].to_numpy()
        if np.isnan(target).all():
            continue
        if not np.isfinite(target).all() or (target < 0).any() or target.sum() > 1 + 1e-10:
            raise QuantError("An actionable target must be finite, long-only, and cash funded.")
        price = data.open.loc[day].to_numpy()
        holdings = units * price
        nav = float(holdings.sum() + cash)
        source = signals.index.get_loc(day) - delay
        if source < 0:
            raise QuantError("Actionable orders must follow an earlier observed target.")
        # This is the same $1e-6 ticket threshold used by the existing fee model.
        if np.all(np.abs(target * nav - holdings) <= 1e-6):
            output.iloc[source] = np.nan
            skipped += 1
            continue
        exits = (target == 0) & (holdings > 1e-6)
        entries = (target * nav > 1e-6) & (holdings <= 1e-6)
        unavoidable_cost = comparison.commission_per_order * int(
            exits.sum() + entries.sum()
        ) + cost / 10000 * float(holdings[exits].sum())
        if unavoidable_cost >= nav:
            raise SimulationFundingError(day, nav)
        try:
            values, cash, _, _, _ = rebalance(
                holdings, cash, target, cost, comparison.commission_per_order
            )
        except QuantError as exc:
            if str(exc) != "Transaction charges would exhaust the portfolio.":
                raise
            raise SimulationFundingError(day, nav) from exc
        units = values / price
    return output, skipped


def audited_scenario(
    data: MarketData,
    signals: pd.DataFrame,
    start: str,
    end: str,
    comparison: DualProtocol,
    *,
    stress: bool = False,
) -> tuple[BacktestResult, pd.DataFrame, dict]:
    actual_end = end
    stopped = None
    try:
        actionable, skipped = actionable_schedule(
            data, signals, start, end, comparison, stress=stress
        )
    except SimulationFundingError as exc:
        actual_end = previous_session(exc.day).date().isoformat()
        stopped = {
            "reason": "fixed_strategy_cannot_fund_required_modeled_transaction_costs",
            "session": exc.day.date().isoformat(),
            "equity_at_rejected_rebalance": exc.equity,
            "no_capital_injection_or_zero_fee_fallback": True,
        }
        if len(data.close.loc[start:actual_end]) < 2:
            raise QuantError("Insufficient funded history to audit the failed strategy.") from exc
        actionable, skipped = actionable_schedule(
            data, signals, start, actual_end, comparison, stress=stress
        )
    result = run_window(data, actionable, start, actual_end, comparison, stress=stress)
    independent = independent_equity(
        data,
        actionable,
        start,
        actual_end,
        capital=comparison.capital_usd,
        cost_bps=comparison.stress_cost_bps_per_side if stress else comparison.cost_bps_per_side,
        commission=comparison.commission_per_order,
        delay=1 + (comparison.stress_additional_delay_sessions if stress else 0),
    )
    discrepancy = float(abs(result.frame["equity"] - independent["equity"]).max())
    if discrepancy > comparison.capital_usd * 1e-8:
        raise QuantError(f"Online independent accounting disagrees by ${discrepancy:.8f}.")
    return (
        result,
        independent,
        {
            "requested_start": start,
            "requested_end": end,
            "audited_end": actual_end,
            "completed_requested_window": stopped is None,
            "stopped": stopped,
            "skipped_zero_dollar_rebalances": skipped,
            "max_equity_difference_usd": discrepancy,
        },
    )


def implementation_hash() -> str:
    root = Path(__file__).parent
    return digest_json(
        {
            path: file_digest(root / path)
            for path in ("online_reversion.py", "bt_audit.py", "dual_horizon.py")
        }
    )


def register(policy: dict, comparison: DualProtocol, sources: dict, output: Path) -> dict:
    validate(policy, comparison)
    if output.exists():
        raise QuantError("Refusing to overwrite online-method preregistration.")
    result = {
        "registered_at": utc_now(),
        "policy": policy,
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "source_fingerprints": sources,
        "prior_trials": 46,
        "new_trials": 2,
        "global_trials_after_round": 48,
        "windows": comparison.windows(),
        "raw_future_prices_not_used_for_signals": True,
        "zero_cost_runs_are_diagnostics_only": True,
        "order_authority": False,
    }
    write_json(output, result)
    return result


def verify_registration(policy: dict, comparison: DualProtocol, sources: dict, path: Path) -> dict:
    validate(policy, comparison)
    result = read_json(path)
    if (
        result.get("policy_sha256") != digest_json(policy)
        or result.get("comparison_sha256") != digest_json(asdict(comparison))
        or result.get("source_fingerprints") != sources
        or result.get("candidate_ids") != [item["id"] for item in policy["candidates"]]
    ):
        raise QuantError("Online-method source, parameters, or comparison windows changed.")
    return result


def evaluate(
    policy: dict,
    comparison: DualProtocol,
    data: MarketData,
    registration: dict,
    output: Path,
) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite online mean-reversion results.")
    signals = build_online_signals(data, policy)
    reports, files, audits = {}, {}, []
    benchmarks = {}
    for window in comparison.windows():
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        benchmark = run_window(
            data, buy_and_hold_signals(data.close, "SPY", start), start, end, comparison
        )
        benchmarks[key] = metrics(benchmark.frame)
    for candidate in policy["candidates"]:
        rule = candidate["id"]
        result = {"method": candidate, "windows": {}}
        for window in comparison.windows():
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            known = seed_window(signals[rule], start)
            result["windows"][key] = {}
            for scenario in ("base", "stress", "zero_cost_diagnostic"):
                costs = (
                    replace(comparison, cost_bps_per_side=0.0, commission_per_order=0.0)
                    if scenario == "zero_cost_diagnostic"
                    else comparison
                )
                stress = scenario == "stress"
                simulated, independent, accounting = audited_scenario(
                    data, known, start, end, costs, stress=stress
                )
                audits.append(
                    {
                        "candidate": rule,
                        "window": key,
                        "scenario": scenario,
                        **accounting,
                    }
                )
                values = metrics(simulated.frame)
                evidence = {
                    "metrics": values if accounting["completed_requested_window"] else None,
                    "incomplete_prefix_metrics": values
                    if not accounting["completed_requested_window"]
                    else None,
                    "accounting": accounting,
                    "acceptance_eligible": scenario != "zero_cost_diagnostic",
                    "maximum_end_of_day_asset_weight": float(simulated.weights.max().max()),
                    "maximum_end_of_day_gross_exposure": float(simulated.weights.sum(axis=1).max()),
                }
                if scenario != "zero_cost_diagnostic":
                    evidence["gates"] = (
                        gates(values, benchmarks[key], comparison)
                        if accounting["completed_requested_window"]
                        else {
                            "complete_requested_horizon": False,
                            "can_fund_declared_transaction_costs": False,
                        }
                    )
                    evidence["all_gates_passed"] = bool(
                        accounting["completed_requested_window"] and all(evidence["gates"].values())
                    )
                result["windows"][key][scenario] = evidence
                files[f"{rule}/{key}-{scenario}.csv"] = simulated.frame.to_csv(float_format="%.12g")
                files[f"{rule}/{key}-{scenario}-bt.csv"] = independent.to_csv(float_format="%.12g")
        first = data.close.index[policy["warmup_sessions"]]
        continuous, _, continuous_audit = audited_scenario(
            data,
            signals[rule],
            first.date().isoformat(),
            comparison.as_of,
            comparison,
        )
        passive = run_window(
            data,
            buy_and_hold_signals(data.close, "SPY", first.date().isoformat()),
            first.date().isoformat(),
            comparison.as_of,
            comparison,
        )
        result["rolling_windows"] = {
            f"{year}y": rolling_diagnostics(continuous.frame, passive.frame, year, comparison)
            for year in comparison.horizons_years
        }
        result["both_horizons_pass"] = all(
            item["base"]["all_gates_passed"] for item in result["windows"].values()
        )
        result["both_horizons_stress_pass"] = all(
            item["stress"]["all_gates_passed"] for item in result["windows"].values()
        )
        result["signals_sha256"] = digest_json(signals[rule].dropna().to_numpy().tolist())
        result["continuous_accounting"] = continuous_audit
        files[f"{rule}/signals.csv"] = signals[rule].dropna(how="all").to_csv(float_format="%.12g")
        reports[rule] = result
    result = {
        "created_at": utc_now(),
        "stage": "public_online_reversion_dual_horizon",
        "implementation_sha256": implementation_hash(),
        "registration": registration,
        "windows": comparison.windows(),
        "prior_trials": 46,
        "new_trials": 2,
        "global_trials": 48,
        "candidates": reports,
        "benchmark_spy": benchmarks,
        "base_joint_passes": [key for key, row in reports.items() if row["both_horizons_pass"]],
        "stress_joint_passes": [
            key
            for key, row in reports.items()
            if row["both_horizons_pass"] and row["both_horizons_stress_pass"]
        ],
        "independent_bt_checks": audits,
        "order_authority": False,
        "investment_objective_verified": False,
        "limitations": [
            "Public algorithms are not evidence of profitable deployment in these instruments.",
            "Unit-simplex weights can concentrate almost the entire research account in one asset.",
            "Daily rebalancing is sensitive to minimum commissions, execution lag, and spread.",
            "Actual following-open fills differ from idealized close-to-close academic backtests.",
            "There is no new independent research holdout; both requested windows overlap.",
            "All-target changes <=$1e-6 are no orders; rounding must not trigger fixed fees.",
            "Fee exhaustion stops the scenario; prefix returns are not full-horizon metrics.",
        ],
    }
    new_output_directory(output)
    for filename, content in files.items():
        write_text_atomic(output / filename, content)
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "results.json", result)
    lines = [
        "# Public online reversion: same-rule ten/five-year results",
        "",
        "**Only net-of-cost results count toward acceptance; zero-cost paths are diagnostic.**",
        "",
        "| Rule/scenario | 10y CAGR | Sharpe | Drawdown | 5y CAGR | Sharpe | Drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in reports.items():
        for scenario in ("base", "stress", "zero_cost_diagnostic"):
            values = []
            for window in ("10y", "5y"):
                m = row["windows"][window][scenario]["metrics"]
                if m is None:
                    values.append("fee funding exhausted | incomplete | incomplete")
                else:
                    sharpe = "undefined" if m["sharpe"] is None else f"{m['sharpe']:.2f}"
                    values.append(f"{m['cagr']:.2%} | {sharpe} | {m['max_drawdown']:.2%}")
            lines.append(f"| {name}/{scenario} | {' | '.join(values)} |")
    lines.extend(["", *result["limitations"], ""])
    write_text_atomic(output / "report.md", "\n".join(lines))
    return result


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "online-reversion", help="Fixed OLMAR/PAMR research; no order authority."
    )
    command.add_argument("stage", choices=["register", "evaluate"])
    command.add_argument("--policy", type=Path, default=Path("config/online-reversion.json"))
    command.add_argument("--comparison", type=Path, default=Path("config/dual-horizon.json"))
    command.add_argument(
        "--registration", type=Path, default=Path("reports/online-reversion/registration.json")
    )
    command.add_argument("--data", type=Path, default=Path("data/dual-horizon/market"))
    command.add_argument("--output", type=Path, default=Path("reports/online-reversion/evaluation"))


def dispatch_online(args: argparse.Namespace) -> dict:
    policy, comparison = read_json(args.policy), load_protocol(args.comparison)
    validate(policy, comparison)
    sources = {
        "data_manifest_sha256": file_digest(args.data / "manifest.json"),
        "open_source_manifest_sha256": file_digest(Path("data/online-reversion/sources.json")),
        "LICENSE": file_digest(Path(__file__).parent / "notices/OLPS-LICENSE.txt"),
        "NOTICE": file_digest(Path(__file__).parent / "notices/OLPS-NOTICE.txt"),
    }
    if args.stage == "register":
        record = register(policy, comparison, sources, args.registration)
        return {
            key: record[key]
            for key in (
                "registered_at",
                "prior_trials",
                "new_trials",
                "global_trials_after_round",
                "order_authority",
            )
        }
    registered = verify_registration(policy, comparison, sources, args.registration)
    data = load_prices(comparison, args.data)
    result = evaluate(policy, comparison, data, registered, args.output)
    return {
        key: result[key]
        for key in (
            "stage",
            "global_trials",
            "base_joint_passes",
            "stress_joint_passes",
            "order_authority",
            "investment_objective_verified",
        )
    }
