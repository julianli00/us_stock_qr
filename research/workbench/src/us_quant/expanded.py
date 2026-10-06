from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, simulate
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.config import Candidate, QuantError, ResearchConfig
from us_quant.data import MarketData, fetch_dataset, load_market
from us_quant.metrics import acceptance, block_bootstrap, performance
from us_quant.research import select_candidate
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

ORIGINAL_SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


@dataclass(frozen=True)
class ExpandedCandidate:
    id: str
    family: str
    universe: str
    rebalance: str
    top_k: int
    max_weight: float
    target_volatility: float
    components: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpandedProtocol:
    round: int
    registered_on: str
    base_protocol_sha256: str
    data_start: str
    simulation_start: str
    development_end: str
    reused_diagnostic_start: str
    as_of: str
    first_test_year: int
    training_years: int
    test_years: int
    final_training_start: str
    sectors: tuple[str, ...]
    indices: tuple[str, ...]
    momentum_lookback: int
    momentum_skip: int
    trend_lookback: int
    volatility_lookback: int
    rsi_period: int
    rsi_entry: float
    rsi_exit: float
    pullback_exit_average: int
    maximum_holding_sessions: int
    weight_rebalance_band: float
    cash_reserve: float
    prior_candidate_trials: int
    reuse_disclosure: str
    universe_rationale: str
    candidates: tuple[ExpandedCandidate, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.indices + self.sectors

    def validate(self, base: ResearchConfig) -> None:
        if self.base_protocol_sha256 != digest_json(base.to_dict()):
            raise QuantError("Expanded round no longer matches its unchanged base protocol.")
        if self.round != 2 or self.prior_candidate_trials < len(base.candidates):
            raise QuantError("All preceding candidate trials must be disclosed.")
        if self.sectors != ORIGINAL_SECTORS or self.indices != ("SPY", "QQQ"):
            raise QuantError(
                "This round requires the complete original sector universe plus SPY/QQQ."
            )
        if not self.reuse_disclosure or not self.universe_rationale:
            raise QuantError("Historical reuse and universe limitations must be disclosed.")
        integer_values = (
            self.momentum_lookback,
            self.momentum_skip,
            self.trend_lookback,
            self.volatility_lookback,
            self.rsi_period,
            self.pullback_exit_average,
            self.maximum_holding_sessions,
        )
        if any(type(value) is not int or value < 1 for value in integer_values):
            raise QuantError("Indicator windows and holding periods must be positive integers.")
        if not 0 < self.momentum_skip < self.momentum_lookback:
            raise QuantError("Momentum must skip a positive interval shorter than its lookback.")
        if not 0 <= self.rsi_entry < self.rsi_exit <= 100:
            raise QuantError("RSI entry and exit thresholds are inconsistent.")
        if (
            not math.isfinite(self.weight_rebalance_band)
            or not 0 <= self.weight_rebalance_band <= 0.05
            or not 0.02 <= self.cash_reserve < 1
        ):
            raise QuantError("Invalid turnover band or cash reserve.")
        known = {}
        for candidate in self.candidates:
            if (
                not re.fullmatch("[a-z0-9_]+", candidate.id)
                or candidate.id in known
                or candidate.family not in {"momentum", "trend", "pullback", "blend"}
                or candidate.universe not in {"sectors", "indices"}
                or candidate.rebalance not in {"daily", "weekly", "monthly"}
            ):
                raise QuantError("Invalid or duplicate expanded candidate.")
            size = len(self.sectors if candidate.universe == "sectors" else self.indices)
            if type(candidate.top_k) is not int or not 1 <= candidate.top_k <= size:
                raise QuantError("Invalid candidate position count.")
            if not 0 < candidate.max_weight <= 0.5 or not 0 < candidate.target_volatility <= 0.2:
                raise QuantError("Expanded candidates remain unleveraged and risk-limited.")
            if candidate.family == "blend":
                if len(candidate.components) != 2 or any(
                    key not in known or known[key].family == "blend" for key in candidate.components
                ):
                    raise QuantError("A blend needs exactly two preceding non-blend candidates.")
            elif candidate.components:
                raise QuantError("Only blend candidates may specify components.")
            known[candidate.id] = candidate
        if not 1 <= len(known) <= 8:
            raise QuantError("This registered round is bounded to at most eight candidates.")
        self.data_config(base)

    def data_config(self, base: ResearchConfig) -> ResearchConfig:
        # This descriptor adapts the existing data API; it is never a tested strategy.
        descriptor = Candidate(
            "expanded_data_container_not_a_strategy", self.symbols, "trend", 1, 0.5, 0.15
        )
        return replace(
            base,
            data_start=self.data_start,
            simulation_start=self.simulation_start,
            development_end=self.development_end,
            holdout_start=self.reused_diagnostic_start,
            as_of=self.as_of,
            symbols=self.symbols,
            candidates=(descriptor,),
            selection=replace(
                base.selection,
                first_test_year=self.first_test_year,
                training_years=self.training_years,
                test_years=self.test_years,
                final_training_start=self.final_training_start,
            ),
        )


def load_expanded(path: Path, base: ResearchConfig) -> ExpandedProtocol:
    try:
        raw = read_json(path)
        raw["sectors"], raw["indices"] = tuple(raw["sectors"]), tuple(raw["indices"])
        raw["candidates"] = tuple(
            ExpandedCandidate(**{**item, "components": tuple(item.get("components", []))})
            for item in raw["candidates"]
        )
        protocol = ExpandedProtocol(**raw)
        protocol.validate(base)
        return protocol
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantError(f"Invalid expanded protocol: {exc}") from exc


def expanded_fingerprint() -> str:
    return digest_json(
        {
            "reused_core": implementation_fingerprint(),
            "expanded": file_digest(Path(__file__)),
        }
    )


def register(protocol: ExpandedProtocol, path: Path) -> dict:
    if path.exists():
        raise QuantError("The expanded research registration is immutable; refusing overwrite.")
    receipt = {
        "registered_at": utc_now(),
        "protocol": asdict(protocol),
        "protocol_sha256": digest_json(asdict(protocol)),
        "prior_candidate_trials": protocol.prior_candidate_trials,
        "registered_new_candidates": len(protocol.candidates),
        "cumulative_candidate_trials": protocol.prior_candidate_trials + len(protocol.candidates),
        "historical_reuse_disclosed": True,
        "paper_submission_eligible": False,
    }
    write_json(path, receipt)
    return receipt


def verify_registration(protocol: ExpandedProtocol, path: Path) -> dict:
    receipt = read_json(path)
    if receipt.get("protocol_sha256") != digest_json(asdict(protocol)) or digest_json(
        receipt.get("protocol")
    ) != digest_json(asdict(protocol)):
        raise QuantError("Expanded protocol was changed after registration.")
    return receipt


def fetch_expanded_data(
    protocol: ExpandedProtocol, base: ResearchConfig, phase: str, output: Path
) -> dict:
    manifest = fetch_dataset(protocol.data_config(base), phase, output)
    manifest["research_round"] = protocol.round
    manifest["evidence_role"] = (
        "development" if phase == "development" else "reused_diagnostic_not_untouched_holdout"
    )
    manifest["independent_prospective_evidence"] = False
    write_json(output / "manifest.json", manifest)
    return manifest


def rsi_ewm(close: pd.DataFrame, period: int) -> pd.DataFrame:
    if type(period) is not int or period < 1:
        raise QuantError("RSI period must be a positive integer.")
    difference = close.diff()
    gains = difference.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    losses = (
        (-difference.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    )
    total = gains + losses
    return (100 * gains / total).mask(total == 0, 50.0)


@dataclass(frozen=True)
class Features:
    trend: np.ndarray
    momentum: np.ndarray
    rsi: np.ndarray
    short_average: np.ndarray
    volatility: np.ndarray
    covariance: np.ndarray
    warmup: int


def features(close: pd.DataFrame, protocol: ExpandedProtocol) -> Features:
    returns = close.pct_change(fill_method=None)
    rolling = returns.rolling(protocol.volatility_lookback)
    covariance = (
        rolling.cov().to_numpy().reshape(len(close), len(close.columns), len(close.columns))
    )
    return Features(
        trend=close.rolling(protocol.trend_lookback).mean().to_numpy(),
        momentum=(
            close.shift(protocol.momentum_skip) / close.shift(protocol.momentum_lookback) - 1
        ).to_numpy(),
        rsi=rsi_ewm(close, protocol.rsi_period).to_numpy(),
        short_average=close.rolling(protocol.pullback_exit_average).mean().to_numpy(),
        volatility=rolling.std(ddof=1).to_numpy() * np.sqrt(252),
        covariance=covariance * 252,
        warmup=max(
            protocol.momentum_lookback,
            protocol.trend_lookback,
            protocol.volatility_lookback,
            protocol.rsi_period,
        ),
    )


def scheduled(day: pd.Timestamp, cadence: str) -> bool:
    if cadence == "daily":
        return True
    if cadence == "monthly":
        return is_month_end(day)
    if cadence == "weekly":
        return day.isocalendar()[:2] != next_session(day).isocalendar()[:2]
    raise QuantError(f"Unknown rebalance cadence: {cadence}")


def scale_risk(weights: np.ndarray, covariance: np.ndarray, target: float) -> np.ndarray:
    active = np.flatnonzero(weights)
    if len(active) == 0:
        return weights
    matrix = covariance[np.ix_(active, active)]
    variance = float(weights[active] @ matrix @ weights[active])
    if not np.isfinite(variance) or variance < -1e-12:
        raise QuantError("Nonfinite or invalid covariance in expanded risk control.")
    volatility = math.sqrt(max(variance, 0.0))
    return weights * min(1.0, target / volatility) if volatility > 0 else weights


@dataclass(frozen=True)
class Intent:
    signals: pd.DataFrame
    targets: pd.DataFrame


def build_intents(data: MarketData, protocol: ExpandedProtocol) -> dict[str, Intent]:
    data.validate()
    if tuple(data.close.columns) != protocol.symbols:
        raise QuantError("Expanded market columns must exactly match the registered universe.")
    close = data.close.to_numpy()
    indicator = features(data.close, protocol)
    output = {}
    for candidate in protocol.candidates:
        universe = protocol.sectors if candidate.universe == "sectors" else protocol.indices
        locations = [data.close.columns.get_loc(symbol) for symbol in universe]
        signals = np.full(close.shape, np.nan)
        targets = np.zeros(close.shape)
        previous = np.zeros(len(data.close.columns))
        selected: dict[int, int] = {}
        for i, day in enumerate(data.close.index):
            if i < indicator.warmup:
                continue
            rebalance_day = scheduled(day, candidate.rebalance)
            eligible = [
                position
                for position in locations
                if close[i, position] > indicator.trend[i, position]
                and np.isfinite(indicator.volatility[i, position])
                and indicator.volatility[i, position] > 1e-8
            ]
            if candidate.family in {"momentum", "trend"}:
                eligible = [j for j in eligible if indicator.momentum[i, j] > 0]
                if rebalance_day:
                    ranked = sorted(
                        eligible,
                        key=lambda j: (-indicator.momentum[i, j], str(data.close.columns[j])),
                    )
                    selected = {j: i for j in ranked[: candidate.top_k]}
                else:
                    selected = {j: entered for j, entered in selected.items() if j in eligible}
            elif candidate.family == "pullback":
                exiting = {
                    j
                    for j, entered in selected.items()
                    if j not in eligible
                    or indicator.rsi[i, j] >= protocol.rsi_exit
                    or close[i, j] > indicator.short_average[i, j]
                    or i - entered >= protocol.maximum_holding_sessions
                }
                selected = {j: entered for j, entered in selected.items() if j not in exiting}
                entries = sorted(
                    (
                        j
                        for j in eligible
                        if j not in selected
                        and j not in exiting
                        and indicator.rsi[i, j] <= protocol.rsi_entry
                    ),
                    key=lambda j: (indicator.rsi[i, j], str(data.close.columns[j])),
                )
                for j in entries[: candidate.top_k - len(selected)]:
                    selected[j] = i
            weights = np.zeros(len(data.close.columns))
            if candidate.family == "blend":
                weights = sum(
                    output[key].targets.iloc[i].to_numpy() for key in candidate.components
                ) / len(candidate.components)
                weights = np.minimum(weights, candidate.max_weight)
            elif selected:
                chosen = sorted(selected)
                inverse = 1.0 / indicator.volatility[i, chosen]
                budget = (1 - protocol.cash_reserve) * len(chosen) / candidate.top_k
                weights[chosen] = np.minimum(budget * inverse / inverse.sum(), candidate.max_weight)
            weights = scale_risk(weights, indicator.covariance[i], candidate.target_volatility)
            membership_changed = not np.array_equal(weights > 0, previous > 0)
            allocation_changed = (
                np.max(np.abs(weights - previous)) >= protocol.weight_rebalance_band
            )
            if rebalance_day or membership_changed or allocation_changed:
                signals[i] = weights
                previous = weights
            targets[i] = previous
        if (
            (targets < 0).any()
            or (targets > candidate.max_weight + 1e-10).any()
            or (targets.sum(axis=1) > 1 - protocol.cash_reserve + 1e-10).any()
        ):
            raise QuantError("Expanded targets breached long-only portfolio limits.")
        output[candidate.id] = Intent(
            pd.DataFrame(signals, index=data.close.index, columns=data.close.columns),
            pd.DataFrame(targets, index=data.close.index, columns=data.close.columns),
        )
    return output


def run_path(
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


def describe(result: BacktestResult, start: str | None = None, end: str | None = None) -> dict:
    frame = result.frame.loc[start:end]
    report = performance(frame["return"], frame["risk_free"])
    report.update(
        {
            "annual_turnover": float(frame["turnover"].sum() / (len(frame) / 252)),
            "total_cost_dollars": float(frame["cost"].sum()),
            "order_count": int(frame["orders"].sum()),
            "average_gross_exposure": float(frame["gross_exposure"].mean()),
        }
    )
    return report


def benchmark(
    config: ResearchConfig, data: MarketData, symbol: str, start: str, end: str
) -> BacktestResult:
    return run_path(config, data, buy_and_hold_signals(data.close, symbol, start), start, end)


def export_result(output: Path, name: str, result: BacktestResult) -> None:
    write_text_atomic(output / f"{name}.csv", result.frame.to_csv(float_format="%.12g"))
    write_text_atomic(output / f"{name}_weights.csv", result.weights.to_csv(float_format="%.12g"))


def develop(
    protocol: ExpandedProtocol,
    base: ResearchConfig,
    development: Path,
    registration: Path,
    output: Path,
) -> dict:
    receipt = verify_registration(protocol, registration)
    config = protocol.data_config(base)
    data = load_market(config, development)
    manifest = read_json(development / "manifest.json")
    if manifest.get("evidence_role") != "development" or manifest.get("research_round") != 2:
        raise QuantError("Expanded development provenance is missing or inconsistent.")
    if pd.Timestamp(manifest["retrieved_at"]) <= pd.Timestamp(receipt["registered_at"]):
        raise QuantError("Expanded data must be fetched after the round was registered.")
    intents = build_intents(data, protocol)
    results = {
        key: run_path(config, data, intent.signals, config.simulation_start, config.development_end)
        for key, intent in intents.items()
    }
    combined = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    folds = []
    for year in range(
        protocol.first_test_year,
        pd.Timestamp(protocol.development_end).year + 1,
        protocol.test_years,
    ):
        train_start, train_end = f"{year - protocol.training_years}-01-01", f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = min(f"{year + protocol.test_years - 1}-12-31", protocol.development_end)
        chosen, selection = select_candidate(config, results, train_start, train_end)
        first = data.close.index[data.close.index.searchsorted(pd.Timestamp(test_start))]
        boundary = previous_session(first)
        if chosen is None:
            combined.loc[boundary:test_end] = 0.0
        else:
            combined.loc[boundary:test_end] = intents[chosen].signals.loc[boundary:test_end]
            combined.loc[boundary] = intents[chosen].targets.loc[boundary]
        selection.update({"test_start": first.date().isoformat(), "test_end": test_end})
        folds.append(selection)
    start = folds[0]["test_start"]
    walk = run_path(config, data, combined, start, protocol.development_end)
    stressed = run_path(config, data, combined, start, protocol.development_end, stress=True)
    spy = benchmark(config, data, "SPY", start, protocol.development_end)
    qqq = benchmark(config, data, "QQQ", start, protocol.development_end)
    for fold in folds:
        fold["forward_metrics"] = describe(walk, fold["test_start"], fold["test_end"])
        fold["spy_metrics"] = describe(spy, fold["test_start"], fold["test_end"])
    selected, selection = select_candidate(
        config, results, protocol.final_training_start, protocol.development_end
    )
    report = {
        "round": protocol.round,
        "stage": "expanded_development",
        "created_at": utc_now(),
        "protocol_sha256": digest_json(asdict(protocol)),
        "implementation_sha256": expanded_fingerprint(),
        "registration_sha256": file_digest(registration),
        "development_manifest_sha256": file_digest(development / "manifest.json"),
        "cumulative_candidate_trials": receipt["cumulative_candidate_trials"],
        "candidate_diagnostics": {key: describe(value) for key, value in results.items()},
        "walk_forward": describe(walk),
        "walk_forward_stress": describe(stressed),
        "benchmarks": {"SPY": describe(spy), "QQQ": describe(qqq)},
        "gates": acceptance(describe(walk), describe(spy), config),
        "stress_gates": acceptance(describe(stressed), describe(spy), config),
        "folds": folds,
        "final_selection": selection,
        "selected_candidate_id": selected,
        "objective_verified": False,
        "paper_submission_eligible": False,
        "reuse_disclosure": protocol.reuse_disclosure,
    }
    new_output_directory(output)
    for key, value in results.items():
        export_result(output / "candidates", key, value)
    for key, value in (
        ("walk_forward", walk),
        ("walk_forward_stress", stressed),
        ("benchmark_SPY", spy),
        ("benchmark_QQQ", qqq),
    ):
        export_result(output, key, value)
    write_json(output / "results.json", report)
    write_json(
        output / "frozen.json",
        {
            "frozen_at": utc_now(),
            "protocol_sha256": digest_json(asdict(protocol)),
            "implementation_sha256": expanded_fingerprint(),
            "registration_sha256": file_digest(registration),
            "development_manifest_sha256": file_digest(development / "manifest.json"),
            "development_results_sha256": file_digest(output / "results.json"),
            "selected_candidate_id": selected,
            "walk_forward_gates": report["gates"],
            "walk_forward_stress_gates": report["stress_gates"],
            "paper_submission_eligible": False,
        },
    )
    write_text_atomic(output / "report.md", render_expanded(report))
    return report


def verify_expanded_freeze(
    protocol: ExpandedProtocol, development: Path, registration: Path, freeze: Path
) -> dict:
    verify_registration(protocol, registration)
    frozen = read_json(freeze)
    expected = {
        "protocol_sha256": digest_json(asdict(protocol)),
        "implementation_sha256": expanded_fingerprint(),
        "registration_sha256": file_digest(registration),
        "development_manifest_sha256": file_digest(development / "manifest.json"),
        "development_results_sha256": file_digest(freeze.parent / "results.json"),
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise QuantError("Expanded freeze no longer matches the registered code/data/results.")
    report = read_json(freeze.parent / "results.json")
    if (
        frozen["selected_candidate_id"] != report["selected_candidate_id"]
        or frozen["walk_forward_gates"] != report["gates"]
        or frozen["walk_forward_stress_gates"] != report["stress_gates"]
    ):
        raise QuantError("Expanded selection was changed after freezing.")
    return frozen


def audit_reused(
    protocol: ExpandedProtocol,
    base: ResearchConfig,
    development: Path,
    reused: Path,
    registration: Path,
    freeze: Path,
    output: Path,
) -> dict:
    frozen = verify_expanded_freeze(protocol, development, registration, freeze)
    config = protocol.data_config(base)
    data = load_market(config, development, reused)
    manifest = read_json(reused / "manifest.json")
    if (
        manifest.get("evidence_role") != "reused_diagnostic_not_untouched_holdout"
        or manifest.get("independent_prospective_evidence") is not False
    ):
        raise QuantError("Reused history must not be mislabeled as independent evidence.")
    selected = frozen["selected_candidate_id"]
    if selected is None:
        signal = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
    else:
        intent = build_intents(data, protocol)[selected]
        signal = intent.signals.copy()
        # A new diagnostic account starts at this known boundary, not at an arbitrary later signal.
        boundary = previous_session(pd.Timestamp(protocol.reused_diagnostic_start))
        signal.loc[boundary] = intent.targets.loc[boundary]
    start, end = protocol.reused_diagnostic_start, protocol.as_of
    result = run_path(config, data, signal, start, end)
    stressed = run_path(config, data, signal, start, end, stress=True)
    spy = benchmark(config, data, "SPY", start, end)
    qqq = benchmark(config, data, "QQQ", start, end)
    historical_gates = acceptance(describe(result), describe(spy), config)
    stress_gates = acceptance(describe(stressed), describe(spy), config)
    all_historical = all(
        all(group.values())
        for group in (
            historical_gates,
            stress_gates,
            frozen["walk_forward_gates"],
            frozen["walk_forward_stress_gates"],
        )
    )
    report = {
        "round": protocol.round,
        "stage": "reused_history_diagnostic",
        "created_at": utc_now(),
        "selected_candidate_id": selected,
        "freeze_sha256": file_digest(freeze),
        "protocol_sha256": digest_json(asdict(protocol)),
        "implementation_sha256": expanded_fingerprint(),
        "reused_manifest_sha256": file_digest(reused / "manifest.json"),
        "cumulative_candidate_trials": protocol.prior_candidate_trials + len(protocol.candidates),
        "diagnostic": describe(result),
        "diagnostic_stress": describe(stressed),
        "benchmarks": {"SPY": describe(spy), "QQQ": describe(qqq)},
        "historical_numeric_gates": historical_gates,
        "historical_stress_gates": stress_gates,
        "all_historical_numeric_gates_passed": all_historical,
        "independent_prospective_evidence": False,
        "objective_verified": False,
        "paper_submission_eligible": False,
        "forward_paper_sessions": 0,
        "reuse_disclosure": protocol.reuse_disclosure,
        "bootstrap": block_bootstrap(
            result.frame["return"],
            spy.frame["return"],
            result.frame["risk_free"],
            samples=config.stress.bootstrap_samples,
            block=config.stress.bootstrap_block_sessions,
            seed=config.stress.bootstrap_seed,
        ),
        "block_reasons": [
            "historical_period_already_examined",
            "no_independent_forward_validation",
            "daily_sector_execution_not_qualified_for_monthly_v1_paper_adapter",
        ]
        + ([] if all_historical else ["historical_numeric_targets_failed"]),
    }
    new_output_directory(output)
    for key, value in (
        ("diagnostic", result),
        ("diagnostic_stress", stressed),
        ("benchmark_SPY", spy),
        ("benchmark_QQQ", qqq),
    ):
        export_result(output, key, value)
    write_json(output / "results.json", report)
    write_text_atomic(output / "report.md", render_expanded(report))
    return report


def render_expanded(report: dict) -> str:
    development = report["stage"] == "expanded_development"
    label = "walk_forward" if development else "diagnostic"
    lines = [
        "# Expanded strategy research: round 2",
        "",
        f"Stage: `{report['stage']}`. "
        f"Candidate trials so far: {report['cumulative_candidate_trials']}.",
        "",
        "**The investment objective is NOT verified. This round cannot authorize orders.**",
        "",
        report["reuse_disclosure"],
        "",
        "| Portfolio | CAGR | Sharpe | Max drawdown | Annual turnover |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in [
        ("Strategy", report[label]),
        ("Higher cost + delayed execution", report[f"{label}_stress"]),
        *report["benchmarks"].items(),
    ]:
        sharpe = "undefined" if value["sharpe"] is None else f"{value['sharpe']:.2f}"
        lines.append(
            f"| {name} | {value['cagr']:.2%} | {sharpe} | "
            f"{value['max_drawdown']:.2%} | {value['annual_turnover']:.2f} |"
        )
    lines.extend(["", f"Selected candidate: `{report['selected_candidate_id'] or 'CASH'}`.", ""])
    if development:
        lines.extend(
            [
                "## Every registered candidate: development diagnostics, not untouched tests",
                "",
                "| Candidate | CAGR | Sharpe | Max drawdown | Orders |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, value in report["candidate_diagnostics"].items():
            sharpe = "undefined" if value["sharpe"] is None else f"{value['sharpe']:.2f}"
            lines.append(
                f"| {name} | {value['cagr']:.2%} | {sharpe} | "
                f"{value['max_drawdown']:.2%} | {value['order_count']} |"
            )
    gates = report["gates"] if development else report["historical_numeric_gates"]
    lines.extend(
        [
            "",
            "## Numeric checks",
            "",
            *(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in gates.items()),
            "",
            "After modeled costs, before personal taxes. No leverage or short sales.",
            "Daily controls react after the close and cannot avoid an already-occurring price gap.",
            "Sector definitions changed; ETF survival and data revisions remain limitations.",
            "Momentum and reversal studies motivate hypotheses, not this implementation's profits.",
            "",
        ]
    )
    return "\n".join(lines)


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "expanded", help="Bounded second-round strategy research; no orders."
    )
    command.add_argument(
        "stage", choices=["register", "fetch-development", "develop", "fetch-reused", "audit"]
    )
    command.add_argument("--protocol", type=Path, default=Path("config/expanded.json"))
    command.add_argument(
        "--registration", type=Path, default=Path("reports/expanded/registration.json")
    )
    command.add_argument("--development", type=Path, default=Path("data/expanded/development"))
    command.add_argument("--reused", type=Path, default=Path("data/expanded/reused"))
    command.add_argument(
        "--freeze", type=Path, default=Path("reports/expanded/development/frozen.json")
    )
    command.add_argument("--output", type=Path)


def dispatch_expanded(args: argparse.Namespace, base: ResearchConfig) -> dict:
    protocol = load_expanded(args.protocol, base)
    if args.stage == "register":
        return register(protocol, args.registration)
    verify_registration(protocol, args.registration)
    if args.stage == "fetch-development":
        return fetch_expanded_data(protocol, base, "development", args.output or args.development)
    if args.stage == "develop":
        return develop(
            protocol,
            base,
            args.development,
            args.registration,
            args.output or Path("reports/expanded/development"),
        )
    verify_expanded_freeze(protocol, args.development, args.registration, args.freeze)
    if args.stage == "fetch-reused":
        return fetch_expanded_data(protocol, base, "holdout", args.output or args.reused)
    if args.stage == "audit":
        return audit_reused(
            protocol,
            base,
            args.development,
            args.reused,
            args.registration,
            args.freeze,
            args.output or Path("reports/expanded/reused"),
        )
    raise QuantError("Unknown expanded research stage.")
