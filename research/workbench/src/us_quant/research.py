from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, simulate
from us_quant.calendar import completed_session, is_month_end, next_session, previous_session
from us_quant.config import QuantError, ResearchConfig
from us_quant.data import MarketData, load_market, verify_dataset
from us_quant.metrics import acceptance, block_bootstrap, performance
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
from us_quant.strategy import buy_and_hold_signals, monthly_signals, target_weights


def _simulate(
    config: ResearchConfig,
    data: MarketData,
    signals: pd.DataFrame,
    start: str,
    end: str,
    *,
    stress: bool = False,
) -> BacktestResult:
    return simulate(
        data,
        signals,
        start,
        end,
        initial_capital=config.initial_capital,
        cost_bps=config.stress.cost_bps_per_side if stress else config.cost_bps_per_side,
        commission=config.commission_per_order,
        delay=config.execution_delay_sessions
        + (config.stress.extra_execution_delay_sessions if stress else 0),
    )


def _metrics(result: BacktestResult, start: str | None = None, end: str | None = None) -> dict:
    frame = result.frame.loc[start:end]
    values = performance(frame["return"], frame["risk_free"])
    years = len(frame) / 252.0
    values["annual_turnover_one_way_sum"] = float(frame["turnover"].sum() / years)
    values["total_cost_dollars"] = float(frame["cost"].sum())
    values["order_count"] = int(frame["orders"].sum())
    values["average_gross_exposure"] = float(frame["gross_exposure"].mean())
    return values


def benchmark_results(
    config: ResearchConfig, data: MarketData, start: str, end: str
) -> dict[str, BacktestResult]:
    results = {
        symbol: _simulate(config, data, buy_and_hold_signals(data.close, symbol, start), start, end)
        for symbol in (config.primary_benchmark, config.secondary_benchmark)
    }
    balanced = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for day in balanced.index:
        if is_month_end(day):
            balanced.loc[day] = 0.0
            balanced.at[day, "SPY"] = 0.6
            balanced.at[day, "IEF"] = 0.4
    results["60_40_monthly"] = _simulate(config, data, balanced, start, end)
    return results


def select_candidate(
    config: ResearchConfig, results: dict[str, BacktestResult], start: str, end: str
) -> tuple[str | None, dict]:
    records = {key: _metrics(value, start, end) for key, value in results.items()}
    eligible = [
        key
        for key, value in records.items()
        if value["max_drawdown"] <= config.selection.max_training_drawdown
        and value["cagr"] > config.selection.minimum_training_cagr
        and value["sharpe"] is not None
    ]
    eligible.sort(key=lambda key: (-records[key]["sharpe"], -records[key]["cagr"], key))
    chosen = eligible[0] if eligible else None
    return chosen, {
        "training_start": start,
        "training_end": end,
        "selected_candidate_id": chosen,
        "eligible_candidates": eligible,
        "selection_reason": "highest net training Sharpe among risk-eligible candidates"
        if chosen
        else "no eligible candidate; explicit cash allocation, not a fitted fallback",
        "training_metrics": records,
    }


def verify_freeze(config: ResearchConfig, development: Path, path: Path) -> dict:
    verify_dataset(development, config, "development")
    frozen = read_json(path)
    if frozen.get("protocol_sha256") != digest_json(config.to_dict()):
        raise QuantError(
            "Frozen strategy protocol changed; do not silently reuse a tested holdout."
        )
    if frozen.get("implementation_sha256") != implementation_fingerprint():
        raise QuantError("Research implementation changed after freezing. Audit before proceeding.")
    if frozen.get("development_manifest_sha256") != file_digest(development / "manifest.json"):
        raise QuantError("The development snapshot differs from the frozen evidence.")
    result_path = path.parent / "results.json"
    if frozen.get("development_results_sha256") != file_digest(result_path):
        raise QuantError("Frozen development results were changed or moved without their evidence.")
    development_result = read_json(result_path)
    if (
        frozen["walk_forward_gates"] != development_result["gates"]
        or frozen["walk_forward_stress_gates"] != development_result["stress_gates"]
        or frozen["selected_candidate_id"]
        != development_result["final_selection"]["selected_candidate_id"]
    ):
        raise QuantError("Frozen selection or gates disagree with development evidence.")
    candidate = config.candidate(frozen["selected_candidate_id"])
    if digest_json(frozen.get("selected_candidate")) != digest_json(
        asdict(candidate) if candidate else None
    ):
        raise QuantError("Frozen candidate parameters disagree with the registered protocol.")
    return frozen


def run_development(config: ResearchConfig, development: Path, output: Path) -> dict:
    data = load_market(config, development)
    if data.close.index[-1] > pd.Timestamp(config.development_end):
        raise QuantError("Holdout contamination: development data extends beyond its boundary.")
    signals = {item.id: monthly_signals(data.close, item, config) for item in config.candidates}
    simulations = {
        key: _simulate(config, data, value, config.simulation_start, config.development_end)
        for key, value in signals.items()
    }
    combined = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    folds = []
    final_year = pd.Timestamp(config.development_end).year
    for year in range(
        config.selection.first_test_year, final_year + 1, config.selection.test_years
    ):
        train_start = f"{year - config.selection.training_years}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = min(f"{year + config.selection.test_years - 1}-12-31", config.development_end)
        chosen, record = select_candidate(config, simulations, train_start, train_end)
        first = data.close.index[data.close.index.searchsorted(pd.Timestamp(test_start))]
        boundary = previous_session(first)
        selected_signals = (
            signals[chosen] if chosen is not None else monthly_signals(data.close, None, config)
        )
        combined.loc[boundary:test_end] = selected_signals.loc[boundary:test_end]
        record.update({"test_start": first.date().isoformat(), "test_end": test_end})
        folds.append(record)
    walk_start = folds[0]["test_start"]
    walk = _simulate(config, data, combined, walk_start, config.development_end)
    stressed = _simulate(config, data, combined, walk_start, config.development_end, stress=True)
    benchmarks = benchmark_results(config, data, walk_start, config.development_end)
    for fold in folds:
        fold["out_of_sample"] = _metrics(walk, fold["test_start"], fold["test_end"])
        fold["spy"] = _metrics(
            benchmarks[config.primary_benchmark], fold["test_start"], fold["test_end"]
        )
    chosen, final_selection = select_candidate(
        config, simulations, config.selection.final_training_start, config.development_end
    )
    walk_metrics = _metrics(walk)
    stress_metrics = _metrics(stressed)
    benchmark_metrics = {key: _metrics(value) for key, value in benchmarks.items()}
    primary = benchmark_metrics[config.primary_benchmark]
    result = {
        "stage": "development",
        "created_at": utc_now(),
        "protocol_sha256": digest_json(config.to_dict()),
        "implementation_sha256": implementation_fingerprint(),
        "candidates_tested": len(config.candidates),
        "targets": asdict(config.targets),
        "candidate_full_development_diagnostics": {
            key: _metrics(value) for key, value in simulations.items()
        },
        "walk_forward": walk_metrics,
        "walk_forward_stress": stress_metrics,
        "benchmarks": benchmark_metrics,
        "gates": acceptance(walk_metrics, primary, config),
        "stress_gates": acceptance(stress_metrics, primary, config),
        "folds": folds,
        "final_selection": final_selection,
        "warning": (
            "Candidate diagnostics and final training metrics are not untouched holdout results."
        ),
    }
    new_output_directory(output)
    write_json(output / "results.json", result)
    write_text_atomic(output / "walk_forward.csv", walk.frame.to_csv(float_format="%.12g"))
    write_text_atomic(
        output / "walk_forward_weights.csv", walk.weights.to_csv(float_format="%.12g")
    )
    write_text_atomic(
        output / "walk_forward_stress.csv", stressed.frame.to_csv(float_format="%.12g")
    )
    for key, value in benchmarks.items():
        write_text_atomic(output / f"benchmark_{key}.csv", value.frame.to_csv(float_format="%.12g"))
    frozen = {
        "schema_version": 1,
        "frozen_at": utc_now(),
        "protocol_sha256": digest_json(config.to_dict()),
        "implementation_sha256": implementation_fingerprint(),
        "development_manifest_sha256": file_digest(development / "manifest.json"),
        "development_results_sha256": file_digest(output / "results.json"),
        "selected_candidate_id": chosen,
        "selected_candidate": asdict(config.candidate(chosen)) if chosen else None,
        "selection": final_selection,
        "holdout_start": config.holdout_start,
        "holdout_end": config.as_of,
        "walk_forward_gates": result["gates"],
        "walk_forward_stress_gates": result["stress_gates"],
        "protocol": config.to_dict(),
    }
    write_json(output / "frozen.json", frozen)
    write_text_atomic(output / "report.md", render_report(result))
    return result


def make_signal(
    config: ResearchConfig,
    data: MarketData,
    frozen: dict,
    freeze_path: Path,
    qualification: dict,
) -> dict:
    day = data.close.index[-1]
    candidate = config.candidate(frozen["selected_candidate_id"])
    weights = target_weights(data.close, candidate, config)
    qualified = qualification.get("research_gates_passed") is True
    month_end = is_month_end(day)
    fresh = day == completed_session()
    reasons = []
    if not qualified:
        reasons.append("research_acceptance_failed")
    if not month_end:
        reasons.append("not_month_end")
    if not fresh:
        reasons.append("stale_completed_session")
    if candidate is None:
        reasons.append("no_eligible_candidate")
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "signal_date": day.date().isoformat(),
        "execution_session": next_session(day).date().isoformat(),
        "strategy_id": frozen["selected_candidate_id"],
        "freeze_sha256": file_digest(freeze_path),
        "protocol_sha256": digest_json(config.to_dict()),
        "qualification_sha256": digest_json(qualification),
        "research_qualified": qualified,
        "executable": not reasons,
        "block_reasons": reasons,
        "mode": "paper_only",
        "weights": {key: float(value) for key, value in weights.items()},
        "cash_weight": float(1.0 - weights.sum()),
        "warning": (
            "Non-month-end weights are diagnostic only; never execute stale or unqualified signals."
        ),
    }


def run_holdout(
    config: ResearchConfig, development: Path, holdout: Path, freeze_path: Path, output: Path
) -> dict:
    frozen = verify_freeze(config, development, freeze_path)
    manifest = verify_dataset(holdout, config, "holdout")
    if pd.Timestamp(manifest["retrieved_at"]) <= pd.Timestamp(frozen["frozen_at"]):
        raise QuantError("Holdout snapshot predates the freeze; it is not a fresh held-back fetch.")
    data = load_market(config, development, holdout)
    candidate = config.candidate(frozen["selected_candidate_id"])
    signals = monthly_signals(data.close, candidate, config)
    base = _simulate(config, data, signals, config.holdout_start, config.as_of)
    stressed = _simulate(config, data, signals, config.holdout_start, config.as_of, stress=True)
    benchmarks = benchmark_results(config, data, config.holdout_start, config.as_of)
    base_metrics, stress_metrics = _metrics(base), _metrics(stressed)
    benchmark_metrics = {key: _metrics(value) for key, value in benchmarks.items()}
    primary = benchmark_metrics[config.primary_benchmark]
    gates = acceptance(base_metrics, primary, config)
    stress_gates = acceptance(stress_metrics, primary, config)
    passed = (
        all(
            all(group.values())
            for group in (
                gates,
                stress_gates,
                frozen["walk_forward_gates"],
                frozen["walk_forward_stress_gates"],
            )
        )
        and candidate is not None
    )
    result = {
        "stage": "holdout",
        "created_at": utc_now(),
        "selected_candidate_id": frozen["selected_candidate_id"],
        "targets": asdict(config.targets),
        "freeze_sha256": file_digest(freeze_path),
        "protocol_sha256": digest_json(config.to_dict()),
        "implementation_sha256": implementation_fingerprint(),
        "holdout_manifest_sha256": file_digest(holdout / "manifest.json"),
        "holdout": base_metrics,
        "holdout_stress": stress_metrics,
        "benchmarks": benchmark_metrics,
        "gates": gates,
        "stress_gates": stress_gates,
        "walk_forward_gates": frozen["walk_forward_gates"],
        "walk_forward_stress_gates": frozen["walk_forward_stress_gates"],
        "research_gates_passed": bool(passed),
        "objective_verified": False,
        "forward_paper_sessions": 0,
        "required_forward_paper_sessions": config.targets.minimum_forward_paper_sessions,
        "paper_submission_eligible": bool(passed),
        "bootstrap": block_bootstrap(
            base.frame["return"],
            benchmarks[config.primary_benchmark].frame["return"],
            base.frame["risk_free"],
            samples=config.stress.bootstrap_samples,
            block=config.stress.bootstrap_block_sessions,
            seed=config.stress.bootstrap_seed,
        ),
        "limitations": [
            "Historical acceptance, even if passed, cannot guarantee future returns or risk.",
            "The ETF universe is retrospective, not a point-in-time constituent database.",
            "Adjusted OHLC models gross dividend reinvestment, not exact cash-dividend accounting.",
            "Returns are before personal taxes, dividend withholding, and currency conversion.",
            "Cash earns zero; Sharpe uses a lagged ^IRX 91-day bank-discount-yield approximation.",
            "Fractional synthetic holdings and ideal next-open liquidity are research assumptions.",
            "Paper sizing, IOC fills, settled cash, and safety halts differ from the backtest.",
            "Six candidates were considered; this consumed holdout must not be tuned against.",
            "The historical holdout is not prospective live-market evidence.",
            "Forward paper performance is required; zero forward sessions have been observed.",
        ],
    }
    new_output_directory(output)
    write_json(output / "results.json", result)
    write_text_atomic(output / "equity.csv", base.frame.to_csv(float_format="%.12g"))
    write_text_atomic(output / "weights.csv", base.weights.to_csv(float_format="%.12g"))
    write_text_atomic(output / "stress.csv", stressed.frame.to_csv(float_format="%.12g"))
    for key, value in benchmarks.items():
        write_text_atomic(output / f"benchmark_{key}.csv", value.frame.to_csv(float_format="%.12g"))
    write_json(
        output / "latest_signal.json", make_signal(config, data, frozen, freeze_path, result)
    )
    write_text_atomic(output / "report.md", render_report(result))
    return result


def verify_qualification(config: ResearchConfig, freeze_path: Path, result_path: Path) -> dict:
    result = read_json(result_path)
    if result.get("stage") != "holdout" or result.get("freeze_sha256") != file_digest(freeze_path):
        raise QuantError("Qualification evidence does not match the frozen holdout.")
    if (
        result.get("protocol_sha256") != digest_json(config.to_dict())
        or result.get("implementation_sha256") != implementation_fingerprint()
    ):
        raise QuantError("Qualification protocol or research implementation changed.")
    groups = ("gates", "stress_gates", "walk_forward_gates", "walk_forward_stress_gates")
    expected = acceptance(result["holdout"], result["benchmarks"][config.primary_benchmark], config)
    stressed = acceptance(
        result["holdout_stress"], result["benchmarks"][config.primary_benchmark], config
    )
    frozen = read_json(freeze_path)
    if (
        result["gates"] != expected
        or result["stress_gates"] != stressed
        or result["walk_forward_gates"] != frozen["walk_forward_gates"]
        or result["walk_forward_stress_gates"] != frozen["walk_forward_stress_gates"]
        or result["selected_candidate_id"] != frozen["selected_candidate_id"]
    ):
        raise QuantError(
            "Qualification gates disagree with metrics or frozen development evidence."
        )
    calculated = all(result.get(group) and all(result[group].values()) for group in groups)
    calculated = bool(calculated and result.get("selected_candidate_id") is not None)
    if result.get("research_gates_passed") is not calculated:
        raise QuantError("Qualification flag disagrees with its acceptance gates.")
    return result


def render_report(result: dict) -> str:
    stage = result["stage"]
    label = "holdout" if stage == "holdout" else "walk_forward"
    strategy = result[label]
    stressed = result[f"{label}_stress"]

    def row(name: str, item: dict) -> str:
        sharpe = "undefined" if item["sharpe"] is None else f"{item['sharpe']:.2f}"
        return (
            f"| {name} | {item['cagr']:.2%} | {sharpe} | "
            f"{item['max_drawdown']:.2%} | {item['annualized_volatility']:.2%} |"
        )

    lines = [
        f"# {stage.title()} research evidence",
        "",
        f"Window: {strategy['start']} through {strategy['end']}.",
        "",
        "**These are historical, net-of-modeled-cost, pre-personal-tax results, not a forecast.**",
        "",
        "| Portfolio | CAGR | Excess-return Sharpe | Max drawdown | Annual volatility |",
        "|---|---:|---:|---:|---:|",
        row("Strategy", strategy),
        row("Strategy: 20 bps per side + one extra session", stressed),
        *(row(key, value) for key, value in result["benchmarks"].items()),
        "",
        "## Acceptance",
        "",
        f"Required: CAGR > {result['targets']['cagr_strictly_above']:.0%}, "
        f"Sharpe > {result['targets']['sharpe_strictly_above']:g}, "
        f"drawdown <= {result['targets']['max_drawdown_at_most']:.0%}, and beat SPY.",
        "",
        *(f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in result["gates"].items()),
        "",
        "Costs include turnover on purchases and sales plus a fixed commission per nonzero ticket.",
        "Signals use month-end closes and execute no earlier than the following session's open.",
        "Cash has no assumed interest. Dividends are included through adjusted prices.",
        "",
    ]
    if stage == "holdout":
        lines.extend(
            [
                f"Frozen candidate: `{result['selected_candidate_id']}`.",
                f"All historical research gates passed: **{result['research_gates_passed']}**.",
                "**Overall investment objective is NOT verified. Forward paper sessions: 0.**",
                "",
                "## Conditional uncertainty",
                "",
                f"Paired 21-session block-bootstrap intervals: `{result['bootstrap']}`",
                "",
                "## Limitations",
                "",
                *(f"- {item}" for item in result["limitations"]),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Walk-forward decisions",
                "",
                "| Training end | Forward window | Candidate | Forward CAGR | Forward drawdown |",
                "|---|---|---|---:|---:|",
                *(
                    f"| {fold['training_end']} | {fold['test_start']} - {fold['test_end']} | "
                    f"{fold['selected_candidate_id'] or 'CASH'} | "
                    f"{fold['out_of_sample']['cagr']:.2%} | "
                    f"{fold['out_of_sample']['max_drawdown']:.2%} |"
                    for fold in result["folds"]
                ),
                "",
                "Final candidate uses only 2016-2021 selection data; do not retune on the holdout.",
                "",
            ]
        )
    return "\n".join(lines)
