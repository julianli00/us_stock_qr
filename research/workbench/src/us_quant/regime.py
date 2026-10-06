from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult, simulate
from us_quant.calendar import previous_session
from us_quant.config import QuantError, ResearchConfig, load_config
from us_quant.data import MarketData, load_market
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


@dataclass(frozen=True)
class RegimeCandidate:
    id: str
    family: str
    universe: tuple[str, ...]
    fast_average: int
    slow_average: int
    momentum_lookback: int
    top_k: int
    target_volatility: float
    max_weight: float


@dataclass(frozen=True)
class RegimeProtocol:
    schema_version: int
    round: int
    registered_on: str
    base_protocol_sha256: str
    prior_disclosed_trials: int
    rebalance_band: float
    candidates: tuple[RegimeCandidate, ...]

    def validate(self, base: ResearchConfig) -> None:
        if (
            self.schema_version != 1
            or self.round != 5
            or self.base_protocol_sha256 != digest_json(base.to_dict())
            or self.prior_disclosed_trials != 28
            or not math.isfinite(self.rebalance_band)
            or not 0.01 <= self.rebalance_band <= 0.10
            or len(self.candidates) != 6
        ):
            raise QuantError("Invalid daily-regime research registration.")
        identifiers = set()
        for item in self.candidates:
            if (
                not re.fullmatch(r"[a-z0-9_]+", item.id)
                or item.id in identifiers
                or item.family not in {"single_trend", "rotation"}
                or not item.universe
                or not set(item.universe) <= set(base.symbols)
                or type(item.fast_average) is not int
                or type(item.slow_average) is not int
                or not 0 <= item.fast_average < item.slow_average <= 252
                or type(item.momentum_lookback) is not int
                or not 0 <= item.momentum_lookback <= 252
                or type(item.top_k) is not int
                or not 1 <= item.top_k <= len(item.universe)
                or not 0.10 <= item.target_volatility <= 0.18
                or not 0 < item.max_weight <= 0.98
            ):
                raise QuantError("Invalid, duplicate, or over-risk daily-regime candidate.")
            if (
                item.family == "single_trend"
                and (len(item.universe) != 1 or item.momentum_lookback != 0)
            ) or (
                item.family == "rotation"
                and (len(item.universe) < 2 or item.momentum_lookback < 20)
            ):
                raise QuantError("Daily-regime family fields are inconsistent.")
            identifiers.add(item.id)


def load_protocol(path: Path, base: ResearchConfig) -> RegimeProtocol:
    try:
        raw = read_json(path)
        raw["candidates"] = tuple(
            RegimeCandidate(**{**item, "universe": tuple(item["universe"])})
            for item in raw["candidates"]
        )
        protocol = RegimeProtocol(**raw)
        protocol.validate(base)
        return protocol
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantError(f"Invalid daily-regime protocol: {exc}") from exc


def fingerprint() -> str:
    return digest_json(
        {
            "regime": file_digest(Path(__file__)),
            "causal_backtest_core": implementation_fingerprint(),
        }
    )


@dataclass(frozen=True)
class Intent:
    signals: pd.DataFrame
    targets: pd.DataFrame


def build_intent(
    data: MarketData,
    candidate: RegimeCandidate,
    protocol: RegimeProtocol,
    base: ResearchConfig,
) -> Intent:
    data.validate()
    columns = data.close.columns
    locations = [columns.get_loc(symbol) for symbol in candidate.universe]
    close = data.close.to_numpy()
    returns = data.close.pct_change(fill_method=None)
    volatility = returns.rolling(base.volatility_lookback).std(ddof=1).to_numpy() * np.sqrt(252)
    slow = data.close.rolling(candidate.slow_average).mean().to_numpy()
    fast = (
        data.close.rolling(candidate.fast_average).mean().to_numpy()
        if candidate.fast_average
        else None
    )
    momentum = (
        (data.close / data.close.shift(candidate.momentum_lookback) - 1).to_numpy()
        if candidate.momentum_lookback
        else None
    )
    warmup = max(
        candidate.slow_average,
        candidate.fast_average,
        candidate.momentum_lookback,
        base.volatility_lookback,
    )
    signals = np.full(close.shape, np.nan)
    targets = np.zeros(close.shape)
    prior = np.zeros(len(columns))
    for index in range(warmup, len(data.close)):
        eligible = [
            location
            for location in locations
            if close[index, location] > slow[index, location]
            and (fast is None or fast[index, location] > slow[index, location])
            and np.isfinite(volatility[index, location])
            and volatility[index, location] > 1e-8
            and (
                momentum is None
                or np.isfinite(momentum[index, location])
                and momentum[index, location] > 0
            )
        ]
        if momentum is not None:
            eligible.sort(
                key=lambda location: (
                    -momentum[index, location],
                    str(columns[location]),
                )
            )
        else:
            eligible.sort(key=lambda location: str(columns[location]))
        chosen = eligible[: candidate.top_k]
        weights = np.zeros(len(columns))
        if chosen:
            inverse = 1 / volatility[index, chosen]
            allocation = inverse / inverse.sum() * (1 - base.cash_reserve)
            allocation = np.minimum(allocation, candidate.max_weight)
            covariance = (
                returns.iloc[
                    index - base.volatility_lookback + 1 : index + 1,
                    chosen,
                ]
                .cov()
                .to_numpy()
                * 252
            )
            variance = float(allocation @ covariance @ allocation)
            if not math.isfinite(variance) or variance < -1e-12:
                raise QuantError("Invalid trailing variance in daily-regime signal.")
            predicted = math.sqrt(max(variance, 0))
            if predicted > candidate.target_volatility:
                allocation *= candidate.target_volatility / predicted
            weights[chosen] = allocation
        membership_changed = not np.array_equal(weights > 0, prior > 0)
        allocation_changed = np.max(np.abs(weights - prior)) >= protocol.rebalance_band
        if membership_changed or allocation_changed:
            signals[index] = weights
            prior = weights
        targets[index] = prior
    if (
        (targets < 0).any()
        or (targets.sum(axis=1) > 1 - base.cash_reserve + 1e-10).any()
        or (targets > candidate.max_weight + 1e-10).any()
    ):
        raise QuantError("Daily-regime target breached long-only portfolio limits.")
    return Intent(
        pd.DataFrame(signals, index=data.close.index, columns=columns),
        pd.DataFrame(targets, index=data.close.index, columns=columns),
    )


def run(
    base: ResearchConfig,
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
        initial_capital=base.initial_capital,
        cost_bps=base.stress.cost_bps_per_side if stress else base.cost_bps_per_side,
        commission=base.commission_per_order,
        delay=base.execution_delay_sessions
        + (base.stress.extra_execution_delay_sessions if stress else 0),
    )


def describe(result: BacktestResult, start: str | None = None, end: str | None = None) -> dict:
    frame = result.frame.loc[start:end]
    values = performance(frame["return"], frame["risk_free"])
    values.update(
        {
            "annual_turnover": float(frame["turnover"].sum() / (len(frame) / 252)),
            "total_cost_dollars": float(frame["cost"].sum()),
            "order_count": int(frame["orders"].sum()),
            "average_gross_exposure": float(frame["gross_exposure"].mean()),
        }
    )
    return values


def benchmark(
    base: ResearchConfig, data: MarketData, symbol: str, start: str, end: str
) -> BacktestResult:
    return run(
        base,
        data,
        buy_and_hold_signals(data.close, symbol, start),
        start,
        end,
    )


def verify_registration(protocol: RegimeProtocol, base: ResearchConfig, path: Path) -> dict:
    registration = read_json(path)
    if (
        registration.get("protocol_sha256") != digest_json(asdict(protocol))
        or registration.get("base_protocol_sha256") != digest_json(base.to_dict())
        or registration.get("candidate_ids") != [item.id for item in protocol.candidates]
    ):
        raise QuantError("Daily-regime registration changed or omits candidates.")
    return registration


def develop(
    protocol: RegimeProtocol,
    base: ResearchConfig,
    data_path: Path,
    registration_path: Path,
    output: Path,
) -> dict:
    registration = verify_registration(protocol, base, registration_path)
    data = load_market(base, data_path)
    if registration.get("history_already_viewed") is not True:
        raise QuantError("Daily-regime inputs must disclose that their price history was viewed.")
    intents = {item.id: build_intent(data, item, protocol, base) for item in protocol.candidates}
    results = {
        key: run(base, data, intent.signals, base.simulation_start, base.development_end)
        for key, intent in intents.items()
    }
    combined = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    folds = []
    for year in range(
        base.selection.first_test_year,
        pd.Timestamp(base.development_end).year + 1,
        base.selection.test_years,
    ):
        train_start = f"{year - base.selection.training_years}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = min(
            f"{year + base.selection.test_years - 1}-12-31",
            base.development_end,
        )
        chosen, record = select_candidate(base, results, train_start, train_end)
        first = data.close.index[data.close.index.searchsorted(pd.Timestamp(test_start))]
        boundary = previous_session(first)
        if chosen is None:
            combined.loc[boundary:test_end] = 0.0
        else:
            combined.loc[boundary:test_end] = intents[chosen].signals.loc[boundary:test_end]
            combined.loc[boundary] = intents[chosen].targets.loc[boundary]
        record.update({"test_start": first.date().isoformat(), "test_end": test_end})
        folds.append(record)
    start = folds[0]["test_start"]
    walk = run(base, data, combined, start, base.development_end)
    stressed = run(base, data, combined, start, base.development_end, stress=True)
    spy = benchmark(base, data, base.primary_benchmark, start, base.development_end)
    qqq = benchmark(base, data, base.secondary_benchmark, start, base.development_end)
    for fold in folds:
        fold["forward"] = describe(walk, fold["test_start"], fold["test_end"])
    chosen, selection = select_candidate(
        base,
        results,
        base.selection.final_training_start,
        base.development_end,
    )
    walk_metrics, stress_metrics, spy_metrics = (
        describe(walk),
        describe(stressed),
        describe(spy),
    )
    report = {
        "stage": "daily_regime_development",
        "created_at": utc_now(),
        "protocol_sha256": digest_json(asdict(protocol)),
        "base_protocol_sha256": digest_json(base.to_dict()),
        "implementation_sha256": fingerprint(),
        "registration_sha256": file_digest(registration_path),
        "data_manifest_sha256": file_digest(data_path / "manifest.json"),
        "prior_disclosed_trials": protocol.prior_disclosed_trials,
        "new_candidate_trials": len(protocol.candidates),
        "candidate_diagnostics": {key: describe(value) for key, value in results.items()},
        "walk_forward": walk_metrics,
        "walk_forward_stress": stress_metrics,
        "benchmarks": {"SPY": spy_metrics, "QQQ": describe(qqq)},
        "gates": acceptance(walk_metrics, spy_metrics, base),
        "stress_gates": acceptance(stress_metrics, spy_metrics, base),
        "folds": folds,
        "final_selection": selection,
        "selected_candidate_id": chosen,
        "objective_verified": False,
        "order_authority": False,
        "warning": "All input history was previously viewed; walk-forward is diagnostic.",
    }
    new_output_directory(output)
    for key, value in results.items():
        write_text_atomic(
            output / "candidates" / f"{key}.csv",
            value.frame.to_csv(float_format="%.12g"),
        )
    for name, value in (
        ("walk_forward", walk),
        ("walk_forward_stress", stressed),
        ("benchmark_SPY", spy),
        ("benchmark_QQQ", qqq),
    ):
        write_text_atomic(output / f"{name}.csv", value.frame.to_csv(float_format="%.12g"))
    write_json(output / "results.json", report)
    freeze = {
        "frozen_at": utc_now(),
        "protocol_sha256": report["protocol_sha256"],
        "base_protocol_sha256": report["base_protocol_sha256"],
        "implementation_sha256": report["implementation_sha256"],
        "registration_sha256": report["registration_sha256"],
        "data_manifest_sha256": report["data_manifest_sha256"],
        "results_sha256": file_digest(output / "results.json"),
        "selected_candidate_id": chosen,
        "walk_forward_gates": report["gates"],
        "walk_forward_stress_gates": report["stress_gates"],
        "order_authority": False,
    }
    write_json(output / "frozen.json", freeze)
    return report


def verify_freeze(
    protocol: RegimeProtocol,
    base: ResearchConfig,
    data_path: Path,
    registration_path: Path,
    freeze_path: Path,
) -> dict:
    verify_registration(protocol, base, registration_path)
    freeze = read_json(freeze_path)
    expected = {
        "protocol_sha256": digest_json(asdict(protocol)),
        "base_protocol_sha256": digest_json(base.to_dict()),
        "implementation_sha256": fingerprint(),
        "registration_sha256": file_digest(registration_path),
        "data_manifest_sha256": file_digest(data_path / "manifest.json"),
        "results_sha256": file_digest(freeze_path.parent / "results.json"),
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise QuantError("Daily-regime freeze no longer matches code, data, or registration.")
    return freeze


def diagnostic(
    protocol: RegimeProtocol,
    base: ResearchConfig,
    development: Path,
    recent: Path,
    registration: Path,
    freeze_path: Path,
    output: Path,
) -> dict:
    freeze = verify_freeze(protocol, base, development, registration, freeze_path)
    data = load_market(base, development, recent)
    boundary = previous_session(pd.Timestamp(base.holdout_start))
    candidate = next(
        (item for item in protocol.candidates if item.id == freeze["selected_candidate_id"]),
        None,
    )
    if freeze["selected_candidate_id"] is not None and candidate is None:
        raise QuantError("Frozen daily-regime candidate is absent from the protocol.")
    if candidate is None:
        signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        signals.loc[boundary] = 0.0
    else:
        intent = build_intent(data, candidate, protocol, base)
        signals = intent.signals.copy()
        signals.loc[boundary] = intent.targets.loc[boundary]
    start, end = base.holdout_start, base.as_of
    result = run(base, data, signals, start, end)
    stressed = run(base, data, signals, start, end, stress=True)
    spy = benchmark(base, data, base.primary_benchmark, start, end)
    qqq = benchmark(base, data, base.secondary_benchmark, start, end)
    metrics, stress_metrics, spy_metrics = (
        describe(result),
        describe(stressed),
        describe(spy),
    )
    report = {
        "stage": "daily_regime_reused_history_diagnostic",
        "created_at": utc_now(),
        "selected_candidate_id": candidate.id if candidate else None,
        "freeze_sha256": file_digest(freeze_path),
        "protocol_sha256": digest_json(asdict(protocol)),
        "implementation_sha256": fingerprint(),
        "recent_manifest_sha256": file_digest(recent / "manifest.json"),
        "diagnostic": metrics,
        "diagnostic_stress": stress_metrics,
        "benchmarks": {"SPY": spy_metrics, "QQQ": describe(qqq)},
        "gates": acceptance(metrics, spy_metrics, base),
        "stress_gates": acceptance(stress_metrics, spy_metrics, base),
        "walk_forward_gates": freeze["walk_forward_gates"],
        "walk_forward_stress_gates": freeze["walk_forward_stress_gates"],
        "bootstrap": block_bootstrap(
            result.frame["return"],
            spy.frame["return"],
            result.frame["risk_free"],
            samples=base.stress.bootstrap_samples,
            block=base.stress.bootstrap_block_sessions,
            seed=base.stress.bootstrap_seed,
        ),
        "independent_holdout": False,
        "objective_verified": False,
        "order_authority": False,
    }
    new_output_directory(output)
    for name, value in (
        ("diagnostic", result),
        ("diagnostic_stress", stressed),
        ("benchmark_SPY", spy),
        ("benchmark_QQQ", qqq),
    ):
        write_text_atomic(output / f"{name}.csv", value.frame.to_csv(float_format="%.12g"))
    write_json(output / "results.json", report)
    return report


def add_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "regime", help="Daily leveraged-ETF regime diagnostics; no order authority."
    )
    parser.add_argument("stage", choices=["develop", "diagnostic"])
    parser.add_argument("--protocol", type=Path, default=Path("config/daily-regime.json"))
    parser.add_argument("--base-config", type=Path, default=Path("config/geared-etf.json"))
    parser.add_argument(
        "--registration", type=Path, default=Path("reports/daily-regime/registration.json")
    )
    parser.add_argument("--development", type=Path, default=Path("data/geared-etf/development"))
    parser.add_argument("--recent", type=Path, default=Path("data/geared-etf/reused"))
    parser.add_argument(
        "--freeze", type=Path, default=Path("reports/daily-regime/development/frozen.json")
    )
    parser.add_argument("--output", type=Path)


def dispatch_regime(args: argparse.Namespace) -> dict:
    base = load_config(args.base_config)
    protocol = load_protocol(args.protocol, base)
    if args.stage == "develop":
        return develop(
            protocol,
            base,
            args.development,
            args.registration,
            args.output or Path("reports/daily-regime/development"),
        )
    return diagnostic(
        protocol,
        base,
        args.development,
        args.recent,
        args.registration,
        args.freeze,
        args.output or Path("reports/daily-regime/reused"),
    )
