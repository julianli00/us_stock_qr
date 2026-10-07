from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import ffn
import numpy as np
import pandas as pd

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.dual_horizon import DualProtocol, load_protocol, metrics, seed_window
from us_quant.factor_research import (
    build_signals,
    goal_gates,
    load_data,
    validate_policy,
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


def independent_metrics(
    frame: pd.DataFrame, independent: pd.DataFrame, risk_free: pd.Series, capital: float
) -> tuple[dict, float]:
    if (
        len(frame) < 2
        or not frame.index.equals(independent.index)
        or not frame.index.equals(risk_free.index)
        or not frame.index.equals(sessions(frame.index[0], frame.index[-1]))
        or not np.isfinite(frame[["equity", "return", "risk_free"]].to_numpy()).all()
        or not np.isfinite(independent[["equity", "return"]].to_numpy()).all()
        or not np.isfinite(risk_free).all()
        or capital <= 0
        or (independent["equity"] <= 0).any()
        or not np.allclose(frame["risk_free"], risk_free, rtol=0, atol=1e-12)
    ):
        raise QuantError("Independent metric audit found incomplete or inconsistent inputs.")
    difference = float(abs(frame["equity"] - independent["equity"]).max())
    if difference > capital * 1e-8:
        raise QuantError("Independent and original equity paths disagree.")
    equity = independent["equity"].to_numpy()
    returns = equity / np.r_[capital, equity[:-1]] - 1
    if not np.allclose(returns, frame["return"], rtol=0, atol=1e-10) or not np.allclose(
        returns, independent["return"], rtol=0, atol=1e-10
    ):
        raise QuantError("Recorded returns do not follow the independently accounted equity.")
    nav = pd.concat(
        [
            pd.Series([capital], index=[previous_session(frame.index[0])]),
            independent["equity"],
        ]
    )
    excess = returns - risk_free.to_numpy()
    deviation = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() / deviation * np.sqrt(252))
        if (deviation > 1e-12 and returns.std(ddof=1) > 1e-12)
        else None
    )
    return {
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
        "sessions": len(frame),
        "cagr": float(ffn.calc_cagr(nav)),
        "sharpe": sharpe,
        "max_drawdown": float(-ffn.calc_max_drawdown(nav)),
    }, difference


def check_metrics(actual: dict, recorded: dict) -> None:
    for key in ("start", "end", "sessions"):
        if actual[key] != recorded[key]:
            raise QuantError("Published metric window differs from the actual observed ledger.")
    for key in ("cagr", "sharpe", "max_drawdown"):
        left, right = actual[key], recorded[key]
        if (left is None) != (right is None) or (
            left is not None
            and (not math.isfinite(right) or not math.isclose(left, right, rel_tol=0, abs_tol=1e-8))
        ):
            raise QuantError(f"Published {key} disagrees with independent metric calculation.")


def audit_round(specification: dict, risk_free: pd.Series, comparison: DualProtocol) -> dict:
    policy = read_json(Path(specification["policy"]))
    registration = read_json(Path(specification["registration"]))
    result = read_json(Path(specification["results"]))
    root = Path(specification["artifacts"])
    validate_policy(policy)
    configured = {item["id"] for item in policy["candidates"]}
    if (
        registration["policy_sha256"] != digest_json(policy)
        or registration["comparison_sha256"] != digest_json(asdict(comparison))
        or registration["windows"] != comparison.windows()
        or result["as_of"] != comparison.as_of
        or result["registration_sha256"] != digest_json(registration)
        or result["implementation_sha256"] != registration["implementation_sha256"]
        or configured != set(registration["candidate_ids"])
        or configured != set(result["candidates"])
        or result["new_configurations"] != len(configured)
        or result["total_disclosed_configurations"] != registration["total_disclosed_after_round"]
    ):
        raise QuantError("Round provenance or complete candidate disclosure is inconsistent.")
    for name, expected in result["artifact_sha256"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or file_digest(path) != expected
        ):
            raise QuantError("A retained research ledger is missing, unsafe or revised.")
    audited = {}
    ranges = {
        f"{window['years']}y": (window["first_return_session"], window["last_session"])
        for window in comparison.windows()
    }
    ranges["nonoverlap_earlyy"] = (
        ranges["10y"][0],
        str(previous_session(ranges["5y"][0]).date()),
    )
    paths, largest_difference = 0, 0.0
    for identifier in ("spy", *sorted(configured)):
        row = (
            result["benchmarks"]
            if identifier == "spy"
            else result["candidates"][identifier]["windows"]
        )
        audited[identifier] = {}
        if set(row) != {"10y", "5y", "nonoverlap_earlyy"}:
            raise QuantError("A primary or non-overlapping chronological result was omitted.")
        for window in row:
            audited[identifier][window] = {}
            for scenario in ("base", "stress"):
                prefix = root / identifier / f"{window}-{scenario}"
                frame = pd.read_csv(f"{prefix}.csv", index_col=0, parse_dates=True)
                independent = pd.read_csv(f"{prefix}-bt.csv", index_col=0, parse_dates=True)
                if not frame.index.equals(sessions(*ranges[window])):
                    raise QuantError("An evaluated return interval differs from its registration.")
                actual, difference = independent_metrics(
                    frame, independent, risk_free.loc[frame.index], policy["capital_usd"]
                )
                check_metrics(actual, row[window][scenario]["metrics"])
                audited[identifier][window][scenario] = actual
                paths += 1
                largest_difference = max(largest_difference, difference)
    base_passes, stress_passes, old_passes = [], [], []
    for identifier in sorted(configured):
        passes, old = {}, {}
        for window, scenarios in audited[identifier].items():
            for scenario, actual in scenarios.items():
                gates = goal_gates(actual, audited["spy"][window][scenario], policy)
                if scenario == "stress":
                    gates["also_beats_base_cost_spy"] = bool(
                        actual["cagr"] > audited["spy"][window]["base"]["cagr"]
                    )
                secondary = {
                    "net_cagr_above_20pct": actual["cagr"] > 0.20,
                    "max_drawdown_at_most_15pct": actual["max_drawdown"] <= 0.15,
                }
                reported = result["candidates"][identifier]["windows"][window][scenario]
                if (
                    gates != reported["primary_gates"]
                    or secondary != reported["secondary_gates"]
                    or reported["primary_pass"] != all(gates.values())
                ):
                    raise QuantError("Published target flags disagree with independent metrics.")
                passes[window, scenario] = all(gates.values())
                old[window, scenario] = all(secondary.values())
        if all(passes[window, "base"] for window in ("10y", "5y")):
            base_passes.append(identifier)
        keys = [(window, scenario) for window in ("10y", "5y") for scenario in ("base", "stress")]
        if all(passes[key] for key in keys):
            stress_passes.append(identifier)
            if all(old[key] for key in keys):
                old_passes.append(identifier)
    if (
        set(base_passes) != set(result["base_joint_passes"])
        or set(stress_passes) != set(result["base_and_stress_joint_passes"])
        or set(old_passes) != set(result["old_all_four_gates_joint_passes"])
        or paths != result["independent_accounting_paths"]
    ):
        raise QuantError("A published round-level success or accounting count is incorrect.")
    return {
        "round_id": result["round_id"],
        "implementation_revision": specification["implementation_revision"],
        "results_sha256": file_digest(Path(specification["results"])),
        "registration_sha256": file_digest(Path(specification["registration"])),
        "configurations_audited": len(configured),
        "accounting_paths_audited": paths,
        "max_equity_difference_usd": largest_difference,
        "base_joint_passes": base_passes,
        "base_and_stress_joint_passes": stress_passes,
        "old_all_four_gates_joint_passes": old_passes,
        "independent_metrics_passed": True,
    }


def checked_scenario(
    data, signal: pd.DataFrame, start: str, end: str, policy: dict, cost: float, delay: int
) -> tuple[pd.DataFrame, dict]:
    result = simulate(
        data,
        signal,
        start,
        end,
        initial_capital=policy["capital_usd"],
        cost_bps=cost,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    independent = independent_equity(
        data,
        signal,
        start,
        end,
        capital=policy["capital_usd"],
        cost_bps=cost,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    actual, difference = independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], policy["capital_usd"]
    )
    qld = result.weights["QLD"]
    return result.frame, {
        "metrics": actual,
        "max_equity_difference_usd": difference,
        "mean_qld_weight": float(qld.mean()),
        "max_qld_weight": float(qld.max()),
        "max_approximate_underlying_gross": float((result.frame["gross_exposure"] + qld).max()),
    }


def rolling_comparison(
    frame: pd.DataFrame, benchmark: pd.DataFrame, years: int, policy: dict
) -> dict:
    records = []
    for end in frame.index:
        if not is_month_end(end):
            continue
        anchor = end - pd.DateOffset(years=years)
        if anchor < previous_session(frame.index[0]):
            continue
        selected = frame.loc[(frame.index > anchor) & (frame.index <= end)]
        values = metrics(selected)
        bench = metrics(benchmark.loc[selected.index])
        records.append(
            {
                "start": values["start"],
                "end": values["end"],
                "cagr": values["cagr"],
                "sharpe": values["sharpe"],
                "max_drawdown": values["max_drawdown"],
                "spy_cagr": bench["cagr"],
                "primary_pass": all(goal_gates(values, bench, policy).values()),
            }
        )
    if not records:
        raise QuantError("No complete rolling windows are available.")
    return {
        "windows": len(records),
        "primary_pass_count": sum(item["primary_pass"] for item in records),
        "primary_pass_fraction": sum(item["primary_pass"] for item in records) / len(records),
        "worst_sharpe": min(item["sharpe"] for item in records if item["sharpe"] is not None),
        "worst_excess_cagr": min(item["cagr"] - item["spy_cagr"] for item in records),
        "records": records,
    }


def validate(plan: dict, comparison_path: Path, data_path: Path, output: Path) -> dict:
    if (
        plan.get("schema_version") != 1
        or plan.get("order_authority") is not False
        or plan.get("independent_forward_validation") is not False
    ):
        raise QuantError("Invalid or trading-enabled factor validation plan.")
    current = next(item for item in plan["rounds"] if item["id"] == plan["qualification_source"])
    policy = read_json(Path(current["policy"]))
    registration = read_json(Path(current["registration"]))
    comparison = load_protocol(comparison_path)
    data = load_data(policy, comparison, registration, data_path)
    audits = [audit_round(item, data.risk_free, comparison) for item in plan["rounds"]]
    candidates = next(item for item in audits if item["round_id"] == current["id"])[
        "base_and_stress_joint_passes"
    ]
    if not candidates:
        raise QuantError("No preregistered primary-qualified candidates exist to validate.")
    signals = build_signals(data, policy)
    for identifier in candidates:
        retained = pd.read_csv(
            Path(current["artifacts"]) / identifier / "issued-signals.csv",
            index_col=0,
            parse_dates=True,
        ).astype(float)
        pd.testing.assert_frame_equal(
            signals[identifier].dropna(how="all"),
            retained,
            check_freq=False,
            check_exact=False,
            rtol=0,
            atol=5e-12,
        )
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "validation_id": plan["validation_id"],
        "created_at": utc_now(),
        "validation_plan_sha256": digest_json(plan),
        "validator_sha256": file_digest(Path(__file__)),
        "round_audits": audits,
        "total_new_configurations_audited": sum(item["configurations_audited"] for item in audits),
        "original_accounting_paths_audited": sum(
            item["accounting_paths_audited"] for item in audits
        ),
        "total_disclosed_configurations": registration["total_disclosed_after_round"],
        "primary_qualified_candidates": candidates,
        "additional_scenarios": {},
        "rolling_windows": {},
        "additional_accounting_paths": 0,
        "latest_requested_historical_thresholds_verified": True,
        "selection_adjusted_statistical_alpha_verified": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    for scenario in plan["additional_scenarios"]:
        result["additional_scenarios"][scenario["id"]] = {}
        for window in comparison.windows():
            key, start, end = (
                f"{window['years']}y",
                window["first_return_session"],
                window["last_session"],
            )
            benchmark, bench = checked_scenario(
                data,
                buy_and_hold_signals(data.close, "SPY", start),
                start,
                end,
                policy,
                scenario["cost_bps_per_side"],
                scenario["delay_sessions"],
            )
            result["additional_accounting_paths"] += 1
            result["additional_scenarios"][scenario["id"]][key] = {"spy": bench["metrics"]}
            for identifier in candidates:
                frame, values = checked_scenario(
                    data,
                    seed_window(signals[identifier], start),
                    start,
                    end,
                    policy,
                    scenario["cost_bps_per_side"],
                    scenario["delay_sessions"],
                )
                values["primary_gates"] = goal_gates(values["metrics"], bench["metrics"], policy)
                values["primary_pass"] = all(values["primary_gates"].values())
                result["additional_scenarios"][scenario["id"]][key][identifier] = values
                result["additional_accounting_paths"] += 1
                write_text_atomic(
                    output / f"{scenario['id']}/{identifier}-{key}.csv",
                    frame.to_csv(float_format="%.12g"),
                )
    first_decision = max(signals[name].dropna(how="all").index[0] for name in candidates)
    start = str(data.close.index[data.close.index.get_loc(first_decision) + 1].date())
    for scenario, cost, delay in (
        ("base", policy["cost_bps_per_side"], policy["base_delay_sessions"]),
        ("stress", policy["stress_cost_bps_per_side"], policy["stress_delay_sessions"]),
    ):
        benchmark, _ = checked_scenario(
            data,
            buy_and_hold_signals(data.close, "SPY", start),
            start,
            comparison.as_of,
            policy,
            cost,
            delay,
        )
        result["additional_accounting_paths"] += 1
        for identifier in candidates:
            frame, values = checked_scenario(
                data, signals[identifier], start, comparison.as_of, policy, cost, delay
            )
            result["rolling_windows"].setdefault(identifier, {})[scenario] = {
                "continuous_account": values,
                **{
                    f"{years}y": rolling_comparison(frame, benchmark, years, policy)
                    for years in plan["rolling_windows"]["years"]
                },
            }
            result["additional_accounting_paths"] += 1
            write_text_atomic(
                output / f"continuous/{identifier}-{scenario}.csv",
                frame.to_csv(float_format="%.12g"),
            )
    result["limitations"] = [
        "The latest primary historical windows pass, not all possible historical windows.",
        "Stronger execution scenarios are sensitivity diagnostics; all failures remain reported.",
        "Seventy-two registered configurations include adaptive selection and correlated variants.",
        "Conditional intervals do not prove future Sharpe above one or future positive alpha.",
        "The old 15% drawdown limit remains unsatisfied; QLD has embedded daily leverage.",
        "Old forward ledgers remain paused; no independent forward observations were fabricated.",
    ]
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent metric and robustness audit; no trading."
    )
    parser.add_argument("--plan", type=Path, default=Path("config/factor-validation.json"))
    parser.add_argument("--comparison", type=Path, default=Path("config/dual-horizon.json"))
    parser.add_argument("--data", type=Path, default=Path("data/factor-round-20261007"))
    parser.add_argument("--output", type=Path, default=Path("reports/factor-validation-20261007"))
    args = parser.parse_args()
    try:
        result = validate(read_json(args.plan), args.comparison, args.data, args.output)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "total_new_configurations_audited",
                        "original_accounting_paths_audited",
                        "additional_accounting_paths",
                        "primary_qualified_candidates",
                        "latest_requested_historical_thresholds_verified",
                        "selection_adjusted_statistical_alpha_verified",
                        "independent_forward_validation",
                    )
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Factor validation blocked: {exc}\n")


if __name__ == "__main__":
    main()
