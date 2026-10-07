from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from us_quant.backtest import BacktestResult, rebalance, simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import load_protocol, metrics, seed_window
from us_quant.factor_research import (
    build_signals,
    goal_gates,
    load_data,
)
from us_quant.factor_research import (
    fingerprint as baseline_fingerprint,
)
from us_quant.factor_validation import check_metrics, independent_metrics, rolling_comparison
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

ROOT = Path(__file__).resolve().parents[2]
CONTROL = {
    "id": "unchanged_baseline",
    "qld_cap": None,
    "volatility_target": None,
    "weight_band": None,
}


def validate_policy(policy: dict) -> None:
    cases = [
        (0.30, None, None),
        (None, 0.12, None),
        (0.30, 0.12, None),
        (None, None, 0.025),
        (None, 0.12, 0.025),
        (0.30, 0.12, 0.025),
    ]
    if (
        policy.get("schema_version") != 1
        or policy.get("prior_disclosed_configurations") != 72
        or policy.get("new_configurations") != 6
        or policy.get("capital_usd") != 10000
        or policy.get("cash_reserve") != 0.02
        or policy.get("commission_per_order") != 1.0
        or policy.get("covariance_sessions") != 63
        or policy.get("defensive_asset") != "BIL"
        or policy.get("embedded_leverage_asset") != "QLD"
        or policy.get("scenarios")
        != [
            {"id": "base", "cost_bps_per_side": 5.0, "delay_sessions": 1},
            {"id": "stress", "cost_bps_per_side": 20.0, "delay_sessions": 2},
            {"id": "higher_cost", "cost_bps_per_side": 50.0, "delay_sessions": 2},
        ]
        or [
            tuple(candidate.get(key) for key in ("qld_cap", "volatility_target", "weight_band"))
            for candidate in policy.get("candidates", [])
        ]
        != cases
    ):
        raise QuantError("Refinement policy may not change the frozen funding or six hypotheses.")
    identifiers = [item["id"] for item in policy["candidates"]]
    if (
        len(set(identifiers)) != 6
        or CONTROL["id"] in identifiers
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name)
            for name in identifiers
        )
    ):
        raise QuantError("Refinement candidate IDs must remain distinct from the baseline.")
    authority = policy.get("methodology", {})
    if any(
        authority.get(key) is not False
        for key in (
            "order_authority",
            "automatic_retuning",
            "old_forward_ledger_modified",
            "persistent_automation_started",
        )
    ):
        raise QuantError("Portfolio research has no trading or persistent automation authority.")
    promotion = policy["promotion_contract"]
    if (
        promotion.get("max_drawdown_worsening_tolerance") != 0.01
        or promotion.get("minimum_turnover_reduction_fraction") != 0.10
        or promotion.get("old_cagr_goal_strictly_above") != 0.20
        or promotion.get("old_drawdown_goal_at_most") != 0.15
        or promotion.get("automatic_baseline_replacement") is not False
        or promotion.get("independent_forward_evidence_required_for_live_promotion") is not True
    ):
        raise QuantError("Refinement promotion criteria may not be relaxed.")


def validate_candidate(candidate: dict) -> None:
    for key, upper in (("qld_cap", 0.98), ("volatility_target", 1.0), ("weight_band", 1.0)):
        value = candidate.get(key)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper
        ):
            raise QuantError("Risk controls and trade bands must be finite and positive.")


def fingerprint() -> str:
    return digest_json(
        {
            "refinement": file_digest(Path(__file__)),
            "unchanged_factor_engine": baseline_fingerprint(),
            "independent_metrics": file_digest(Path(__file__).with_name("factor_validation.py")),
            "scipy": version("scipy"),
            "ffn": version("ffn"),
        }
    )


def baseline_context(policy: dict, data_path: Path) -> tuple[dict, dict, MarketData]:
    validate_policy(policy)
    baseline = policy["baseline"]
    inputs = {}
    for name in ("policy", "registration", "results"):
        path = ROOT / baseline[name]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(ROOT)
            or file_digest(path) != baseline[f"{name}_sha256"]
        ):
            raise QuantError("The preserved baseline definition or evidence has changed.")
        inputs[name] = read_json(path)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    data = load_data(inputs["policy"], comparison, inputs["registration"], data_path)
    return inputs["policy"], inputs["results"], data


def register(policy: dict, data_path: Path, output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite a portfolio refinement registration.")
    base, _, _ = baseline_context(policy, data_path)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    sources = read_json(ROOT / policy["baseline"]["registration"])["sources"]
    record = {
        "schema_version": 1,
        "round_id": policy["round_id"],
        "registered_at": utc_now(),
        "policy_sha256": digest_json(policy),
        "implementation_sha256": fingerprint(),
        "baseline_policy_sha256": digest_json(base),
        "baseline_sources_sha256": digest_json(sources),
        "windows": comparison.windows(),
        "prior_disclosed_configurations": 72,
        "new_configurations": 6,
        "total_disclosed_configurations": 78,
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "history_previously_exposed": True,
        "order_authority": False,
    }
    write_json(output, record)
    return record


def verify_registration(policy: dict, record: dict) -> None:
    validate_policy(policy)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    sources = read_json(ROOT / policy["baseline"]["registration"])["sources"]
    if (
        record.get("round_id") != policy["round_id"]
        or record.get("policy_sha256") != digest_json(policy)
        or record.get("implementation_sha256") != fingerprint()
        or record.get("baseline_sources_sha256") != digest_json(sources)
        or record.get("baseline_policy_sha256")
        != digest_json(read_json(ROOT / policy["baseline"]["policy"]))
        or record.get("windows") != comparison.windows()
        or record.get("candidate_ids") != [item["id"] for item in policy["candidates"]]
        or record.get("total_disclosed_configurations") != 78
        or record.get("history_previously_exposed") is not True
        or record.get("order_authority") is not False
    ):
        raise QuantError("Refinement rules, code or provenance changed after registration.")
    timestamp = pd.Timestamp(record.get("registered_at"))
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise QuantError("Refinement registration needs a timezone-aware timestamp.")


def observed_covariance(data: MarketData, day: pd.Timestamp, lookback: int) -> pd.DataFrame:
    observed = data.close.loc[:day].tail(lookback + 1)
    if len(observed) != lookback + 1 or observed.index[-1] != day:
        raise QuantError("Risk controls require a complete trailing, already observed window.")
    result = observed.pct_change(fill_method=None).iloc[1:].cov() * 252
    values = result.to_numpy()
    if (
        not np.isfinite(values).all()
        or not np.allclose(values, values.T)
        or np.linalg.eigvalsh(values).min() < -1e-10
    ):
        raise QuantError("Observed covariance is invalid; no synthetic risk fallback.")
    return result


def predicted_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    if (
        not weights.index.equals(covariance.index)
        or not covariance.index.equals(covariance.columns)
        or not np.isfinite(weights).all()
    ):
        raise QuantError("Risk weights and covariance must be finite and exactly aligned.")
    variance = float(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy())
    if not math.isfinite(variance) or variance < -1e-10:
        raise QuantError("Invalid portfolio variance.")
    return math.sqrt(max(0, variance))


def constrained_target(
    baseline: pd.Series, covariance: pd.DataFrame, candidate: dict, budget: float
) -> pd.Series:
    validate_candidate(candidate)
    if (
        not np.isfinite(baseline).all()
        or (baseline < 0).any()
        or not np.isclose(baseline.sum(), budget, rtol=0, atol=1e-10)
        or not {"QLD", "BIL"} <= set(baseline.index)
    ):
        raise QuantError("Portfolio controls require a complete cash-funded baseline target.")
    target = baseline.copy()
    cap = candidate["qld_cap"]
    if cap is not None and target["QLD"] > cap:
        reduction = target["QLD"] - cap
        target["QLD"] = cap
        target["BIL"] += reduction
    limit = candidate["volatility_target"]
    if limit is not None and predicted_volatility(target, covariance) > limit:
        risky = target.copy()
        risky["BIL"] = 0

        def scaled(scale: float) -> pd.Series:
            result = risky * scale
            result["BIL"] = budget - result.sum()
            return result

        if predicted_volatility(scaled(0), covariance) > limit:
            raise QuantError("The defensive ETF cannot meet the declared volatility target.")
        scale = brentq(
            lambda value: predicted_volatility(scaled(value), covariance) - limit,
            0,
            1,
            xtol=1e-14,
        )
        target = scaled(scale)
    if (
        (target < -1e-12).any()
        or not np.isclose(target.sum(), budget, rtol=0, atol=1e-10)
        or (cap is not None and target["QLD"] > cap + 1e-10)
        or (limit is not None and predicted_volatility(target, covariance) > limit + 1e-10)
    ):
        raise QuantError("Projected portfolio target violates its declared constraints.")
    return target.clip(lower=0)


def rebalance_reason(
    current: pd.Series, desired: pd.Series, covariance: pd.DataFrame, candidate: dict
) -> str:
    validate_candidate(candidate)
    if (
        not current.index.equals(desired.index)
        or not np.isfinite(current).all()
        or (current < -1e-12).any()
        or current.sum() > 1 + 1e-10
    ):
        raise QuantError("No-trade decisions require valid observed portfolio weights.")
    band = candidate["weight_band"]
    if band is None:
        return "scheduled_rebalance"
    if (candidate["qld_cap"] is not None and current["QLD"] > candidate["qld_cap"] + 1e-10) or (
        candidate["volatility_target"] is not None
        and predicted_volatility(current, covariance) > candidate["volatility_target"] + 1e-10
    ):
        return "risk_limit_override"
    return "observed_weight_gap" if float(abs(current - desired).max()) + 1e-12 >= band else "hold"


@dataclass(frozen=True)
class RefinementRun:
    result: BacktestResult
    issued: pd.DataFrame
    decisions: list[dict]


def simulate_refinement(
    data: MarketData,
    baseline_signals: pd.DataFrame,
    candidate: dict,
    policy: dict,
    start: str,
    end: str,
    *,
    cost_bps: float,
    delay: int,
) -> RefinementRun:
    data.validate()
    validate_candidate(candidate)
    if (
        type(delay) is not int
        or delay < 1
        or not math.isfinite(cost_bps)
        or not 0 <= cost_bps < 100
        or not baseline_signals.index.equals(data.close.index)
        or not baseline_signals.columns.equals(data.close.columns)
        or (baseline_signals.isna().any(axis=1) & ~baseline_signals.isna().all(axis=1)).any()
    ):
        raise QuantError("Invalid causal refinement prices, signals, delay or costs.")
    defined = baseline_signals.dropna(how="all")
    if (
        defined.empty
        or not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or not np.allclose(defined.sum(axis=1), 1 - policy["cash_reserve"], rtol=0, atol=1e-10)
    ):
        raise QuantError("Baseline targets must remain complete and cash funded.")
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("Refinement requires a complete return window.")
    anchor = previous_session(dates[0])
    known = seed_window(baseline_signals, str(dates[0].date()))
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    pending, decisions = {}, []
    cash = last_nav = policy["capital_usd"]
    units = np.zeros(len(data.close.columns))
    rows, weights = [], []
    budget = 1 - policy["cash_reserve"]

    def decide(day: pd.Timestamp, current: pd.Series, initial: bool = False) -> None:
        covariance = observed_covariance(data, day, policy["covariance_sessions"])
        desired = constrained_target(known.loc[day], covariance, candidate, budget)
        reason = (
            "initial_capital"
            if initial
            else rebalance_reason(current, desired, covariance, candidate)
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if reason != "hold":
            if execution in pending:
                raise QuantError("Refinement must not overwrite a pending causal decision.")
            pending[execution] = desired.to_numpy()
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()),
                "initial_capital": initial,
                "reason": reason,
                "observed_qld_weight": float(current["QLD"]),
                "observed_target_gap": float(abs(current - desired).max()),
                "observed_portfolio_volatility": predicted_volatility(current, covariance),
                "target_qld_weight": float(desired["QLD"]),
                "target_portfolio_volatility": predicted_volatility(desired, covariance),
            }
        )

    decide(anchor, pd.Series(0.0, index=data.close.columns), initial=True)
    for day in dates:
        opening = data.open.loc[day].to_numpy()
        closing = data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if day in pending:
            dollars, cash, costs, turnover, orders = rebalance(
                units * opening,
                cash,
                pending.pop(day),
                cost_bps,
                policy["commission_per_order"],
            )
            units = dollars / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not math.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("Refinement accounting became insolvent or borrowed cash.")
        observed = pd.Series(values / nav, index=data.close.columns)
        rows.append((nav, nav / last_nav - 1, cash, float(observed.sum()), turnover, costs, orders))
        weights.append(observed.to_numpy())
        if not known.loc[day].isna().all():
            if not is_month_end(day):
                raise QuantError("New refinement decisions must use completed month ends.")
            decide(day, observed)
        last_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    return RefinementRun(
        BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns)),
        issued,
        decisions,
    )


def audit_run(
    data: MarketData,
    baseline_signals: pd.DataFrame,
    run: RefinementRun,
    candidate: dict,
    policy: dict,
    start: str,
    end: str,
    cost_bps: float,
    delay: int,
) -> tuple[pd.DataFrame, dict]:
    replay = simulate(
        data,
        run.issued,
        start,
        end,
        initial_capital=policy["capital_usd"],
        cost_bps=cost_bps,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    if not np.allclose(replay.frame, run.result.frame, rtol=0, atol=1e-8) or not np.allclose(
        replay.weights, run.result.weights, rtol=0, atol=1e-10
    ):
        raise QuantError("Causal refinement decisions do not replay through the original engine.")
    independent = independent_equity(
        data,
        run.issued,
        start,
        end,
        capital=policy["capital_usd"],
        cost_bps=cost_bps,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    actual, difference = independent_metrics(
        run.result.frame,
        independent,
        data.risk_free.loc[run.result.frame.index],
        policy["capital_usd"],
    )
    check_metrics(actual, metrics(run.result.frame))
    known = seed_window(baseline_signals, start)
    for decision in run.decisions:
        day = pd.Timestamp(decision["signal_session"])
        covariance = observed_covariance(data, day, policy["covariance_sessions"])
        observed = (
            pd.Series(0.0, index=data.close.columns)
            if decision["initial_capital"]
            else replay.weights.loc[day]
        )
        desired = constrained_target(
            known.loc[day], covariance, candidate, 1 - policy["cash_reserve"]
        )
        expected = (
            "initial_capital"
            if decision["initial_capital"]
            else rebalance_reason(observed, desired, covariance, candidate)
        )
        if expected != decision["reason"] or not math.isclose(
            decision["observed_qld_weight"], observed["QLD"], rel_tol=0, abs_tol=1e-10
        ):
            raise QuantError("The decision did not follow actual causally replayed positions.")
        if (expected == "hold") != run.issued.loc[day].isna().all():
            raise QuantError("A held or executed signal contradicts the observed-weight gate.")
        if expected != "hold" and not np.allclose(run.issued.loc[day], desired, rtol=0, atol=1e-12):
            raise QuantError("An issued target differs from the registered risk projection.")
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if str(execution.date()) != decision["execution_session"] or decision[
            "initial_capital"
        ] != (day == previous_session(start)):
            raise QuantError("A refinement decision has incorrect initialization or timing.")
    return independent, {
        "max_equity_difference_usd": difference,
        "original_engine_replay_passed": True,
        "position_feedback_replay_passed": True,
        "independent_metrics_passed": True,
    }


def promotion_assessment(candidate: dict, baseline: dict, policy: dict) -> dict:
    keys = [(window, scenario) for window in ("10y", "5y") for scenario in ("base", "stress")]
    primary = all(candidate[window][scenario]["primary_pass"] for window, scenario in keys)
    tolerable_drawdown = all(
        candidate[window][scenario]["metrics"]["max_drawdown"]
        <= baseline[window][scenario]["metrics"]["max_drawdown"]
        + policy["promotion_contract"]["max_drawdown_worsening_tolerance"]
        for window, scenario in keys
    )
    risk_improved = all(
        candidate[window][scenario]["metrics"]["max_drawdown"] <= 0.15 for window, scenario in keys
    )
    cost_improved = all(
        candidate[window]["higher_cost"]["primary_pass"]
        and candidate[window]["higher_cost"]["metrics"]["annualized_one_way_turnover"]
        <= baseline[window]["higher_cost"]["metrics"]["annualized_one_way_turnover"]
        * (1 - policy["promotion_contract"]["minimum_turnover_reduction_fraction"])
        for window in ("10y", "5y")
    )
    return {
        "primary_both_windows_base_and_stress": primary,
        "drawdown_not_materially_worse": tolerable_drawdown,
        "all_primary_drawdowns_at_most_15pct": risk_improved,
        "higher_cost_goal_and_turnover_improved": cost_improved,
        "research_improvement_qualified": bool(
            primary and tolerable_drawdown and (risk_improved or cost_improved)
        ),
        "automatic_baseline_replacement": False,
    }


def evaluate(policy: dict, registration: dict, data_path: Path, output: Path) -> dict:
    verify_registration(policy, registration)
    baseline_policy, prior_result, data = baseline_context(policy, data_path)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    baseline_signals = build_signals(data, baseline_policy)[policy["baseline"]["candidate"]]
    windows = comparison.windows()
    windows.append(
        {
            "years": "nonoverlap_early",
            "first_return_session": windows[0]["first_return_session"],
            "last_session": str(previous_session(windows[1]["first_return_session"]).date()),
        }
    )
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "round_id": policy["round_id"],
        "created_at": utc_now(),
        "registration_sha256": digest_json(registration),
        "implementation_sha256": fingerprint(),
        "prior_disclosed_configurations": 72,
        "new_configurations": 6,
        "total_disclosed_configurations": 78,
        "as_of": comparison.as_of,
        "baseline": policy["baseline"],
        "candidates": {},
        "benchmarks": {},
        "primary_qualified_candidates": [],
        "improvement_qualified_candidates": [],
        "independent_accounting_paths": 0,
        "history_previously_exposed": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
        "automatic_baseline_replacement": False,
        "limitations": [
            "New rules were designed after exposed prior results; not independent holdouts.",
            "Risk caps constrain targets at monthly decisions, not gaps or intervening weights.",
            "A volatility target cannot guarantee a maximum drawdown.",
            "Fractional units, model transaction costs and pre-tax accounts remain assumptions.",
            "The old forward ledger stays paused; no orders, service changes or auto-retuning.",
        ],
    }
    for window in windows:
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        result["benchmarks"][key] = {}
        for scenario in policy["scenarios"]:
            label, cost, delay = (
                scenario["id"],
                scenario["cost_bps_per_side"],
                scenario["delay_sessions"],
            )
            signal = buy_and_hold_signals(data.close, "SPY", start)
            own = simulate(
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
                own.frame, independent, data.risk_free.loc[own.frame.index], policy["capital_usd"]
            )
            check_metrics(actual, metrics(own.frame))
            result["benchmarks"][key][label] = {
                "metrics": metrics(own.frame),
                "max_equity_difference_usd": difference,
            }
            result["independent_accounting_paths"] += 1
            write_text_atomic(
                output / f"spy/{key}-{label}.csv", own.frame.to_csv(float_format="%.12g")
            )
            write_text_atomic(
                output / f"spy/{key}-{label}-bt.csv", independent.to_csv(float_format="%.12g")
            )
    for candidate in (CONTROL, *policy["candidates"]):
        identifier = candidate["id"]
        row = {"definition": candidate, "windows": {}}
        for window in windows:
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            row["windows"][key] = {}
            for scenario in policy["scenarios"]:
                label, cost, delay = (
                    scenario["id"],
                    scenario["cost_bps_per_side"],
                    scenario["delay_sessions"],
                )
                run = simulate_refinement(
                    data,
                    baseline_signals,
                    candidate,
                    policy,
                    start,
                    end,
                    cost_bps=cost,
                    delay=delay,
                )
                independent, audit = audit_run(
                    data, baseline_signals, run, candidate, policy, start, end, cost, delay
                )
                values = metrics(run.result.frame)
                if identifier == CONTROL["id"] and label in {"base", "stress"}:
                    saved = prior_result["candidates"][policy["baseline"]["candidate"]]["windows"][
                        key
                    ][label]["metrics"]
                    check_metrics(values, saved)
                    relative = f"{policy['baseline']['candidate']}/{key}-{label}.csv"
                    path = ROOT / policy["baseline"]["artifacts"] / relative
                    if (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(ROOT)
                        or file_digest(path) != prior_result["artifact_sha256"][relative]
                    ):
                        raise QuantError("The baseline's retained accounting path changed.")
                    saved_frame = pd.read_csv(path, index_col=0, parse_dates=True)
                    if (
                        not saved_frame.index.equals(run.result.frame.index)
                        or not saved_frame.columns.equals(run.result.frame.columns)
                        or not np.allclose(saved_frame, run.result.frame, rtol=1e-10, atol=1e-6)
                    ):
                        raise QuantError(
                            "The unchanged control differs from the full prior ledger."
                        )
                    audit["unchanged_baseline_full_ledger_replay_passed"] = True
                gates = goal_gates(
                    values, result["benchmarks"][key][label]["metrics"], baseline_policy
                )
                if label != "base":
                    gates["also_beats_base_cost_spy"] = bool(
                        values["cagr"] > result["benchmarks"][key]["base"]["metrics"]["cagr"]
                    )
                row["windows"][key][label] = {
                    "metrics": values,
                    "primary_gates": gates,
                    "primary_pass": all(gates.values()),
                    "old_goals": {
                        "cagr_above_20pct": values["cagr"] > 0.20,
                        "drawdown_at_most_15pct": values["max_drawdown"] <= 0.15,
                    },
                    "decisions": len(run.decisions),
                    "skipped_monthly_rebalances": sum(x["reason"] == "hold" for x in run.decisions),
                    "risk_override_decisions": sum(
                        x["reason"] == "risk_limit_override" for x in run.decisions
                    ),
                    "max_realized_qld_weight": float(run.result.weights["QLD"].max()),
                    "mean_realized_qld_weight": float(run.result.weights["QLD"].mean()),
                    "max_approximate_underlying_gross": float(
                        (run.result.frame["gross_exposure"] + run.result.weights["QLD"]).max()
                    ),
                    "audit": audit,
                }
                prefix = f"{identifier}/{key}-{label}"
                for suffix, frame in (
                    ("", run.result.frame),
                    ("-bt", independent),
                    ("-weights", run.result.weights),
                    ("-issued", run.issued.dropna(how="all")),
                ):
                    write_text_atomic(
                        output / f"{prefix}{suffix}.csv", frame.to_csv(float_format="%.12g")
                    )
                write_json(output / f"{prefix}-decisions.json", {"decisions": run.decisions})
                result["independent_accounting_paths"] += 1
        result["candidates"][identifier] = row
        if identifier != CONTROL["id"]:
            assessment = promotion_assessment(
                row["windows"], result["candidates"][CONTROL["id"]]["windows"], policy
            )
            row["promotion_assessment"] = assessment
            if assessment["primary_both_windows_base_and_stress"]:
                result["primary_qualified_candidates"].append(identifier)
            if assessment["research_improvement_qualified"]:
                result["improvement_qualified_candidates"].append(identifier)
        write_json(
            output / "progress.json",
            {
                "complete": False,
                "completed": list(result["candidates"]),
                "expected": [CONTROL["id"], *registration["candidate_ids"]],
            },
        )
    result["rolling_validation"] = {}
    if result["improvement_qualified_candidates"]:
        first = baseline_signals.dropna(how="all").index[0]
        start = str(next_session(first).date())
        chosen = {CONTROL["id"], *result["improvement_qualified_candidates"]}
        for scenario in policy["scenarios"][:2]:
            label, cost, delay = (
                scenario["id"],
                scenario["cost_bps_per_side"],
                scenario["delay_sessions"],
            )
            signal = buy_and_hold_signals(data.close, "SPY", start)
            benchmark = simulate(
                data,
                signal,
                start,
                comparison.as_of,
                initial_capital=policy["capital_usd"],
                cost_bps=cost,
                commission=policy["commission_per_order"],
                delay=delay,
            )
            independent_benchmark = independent_equity(
                data,
                signal,
                start,
                comparison.as_of,
                capital=policy["capital_usd"],
                cost_bps=cost,
                commission=policy["commission_per_order"],
                delay=delay,
            )
            audited_benchmark, _ = independent_metrics(
                benchmark.frame,
                independent_benchmark,
                data.risk_free.loc[benchmark.frame.index],
                policy["capital_usd"],
            )
            check_metrics(audited_benchmark, metrics(benchmark.frame))
            result["independent_accounting_paths"] += 1
            write_text_atomic(
                output / f"rolling/spy-{label}.csv", benchmark.frame.to_csv(float_format="%.12g")
            )
            write_text_atomic(
                output / f"rolling/spy-{label}-bt.csv",
                independent_benchmark.to_csv(float_format="%.12g"),
            )
            for candidate in (CONTROL, *policy["candidates"]):
                if candidate["id"] not in chosen:
                    continue
                run = simulate_refinement(
                    data,
                    baseline_signals,
                    candidate,
                    policy,
                    start,
                    comparison.as_of,
                    cost_bps=cost,
                    delay=delay,
                )
                independent, audit = audit_run(
                    data,
                    baseline_signals,
                    run,
                    candidate,
                    policy,
                    start,
                    comparison.as_of,
                    cost,
                    delay,
                )
                result["independent_accounting_paths"] += 1
                result["rolling_validation"].setdefault(candidate["id"], {})[label] = {
                    "audit": audit,
                    **{
                        f"{years}y": rolling_comparison(
                            run.result.frame, benchmark.frame, years, baseline_policy
                        )
                        for years in (10, 5)
                    },
                }
                prefix = f"rolling/{candidate['id']}-{label}"
                write_text_atomic(
                    output / f"{prefix}.csv", run.result.frame.to_csv(float_format="%.12g")
                )
                write_text_atomic(
                    output / f"{prefix}-bt.csv", independent.to_csv(float_format="%.12g")
                )
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "progress.json"
    }
    write_json(output / "results.json", result)
    write_json(
        output / "progress.json", {"complete": True, "completed": list(result["candidates"])}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen portfolio risk/turnover research; no orders."
    )
    parser.add_argument("stage", choices=("register", "evaluate"))
    parser.add_argument("--policy", type=Path, default=Path("config/portfolio-refinement.json"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path("evidence/portfolio_refinement_20261007_registration.json"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/factor-round-20261007"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/portfolio-refinement-20261007")
    )
    args = parser.parse_args()
    try:
        policy = read_json(args.policy)
        if args.stage == "register":
            result = register(policy, args.data, args.registration)
            keys = ("registered_at", "candidate_ids", "total_disclosed_configurations")
        else:
            result = evaluate(policy, read_json(args.registration), args.data, args.output)
            keys = (
                "total_disclosed_configurations",
                "primary_qualified_candidates",
                "improvement_qualified_candidates",
                "independent_accounting_paths",
                "automatic_baseline_replacement",
                "independent_forward_validation",
            )
        print(json.dumps({key: result[key] for key in keys}, indent=2))
    except QuantError as exc:
        parser.exit(2, f"Portfolio refinement blocked: {exc}\n")


if __name__ == "__main__":
    main()
