from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from us_quant.backtest import BacktestResult, simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import completed_session, is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart, risk_free_returns
from us_quant.metrics import block_bootstrap, performance
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
class PublicRule:
    id: str
    family: str
    offensive: tuple[str, ...]
    defensive: tuple[str, ...]
    canary: tuple[str, ...]
    offensive_count: int
    defensive_count: int
    components: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class DualProtocol:
    schema_version: int
    registered_on: str
    request: str
    data_start: str
    as_of: str
    horizons_years: tuple[int, ...]
    prior_disclosed_trials: int
    capital_usd: float
    cash_reserve: float
    cost_bps_per_side: float
    commission_per_order: float
    stress_cost_bps_per_side: float
    stress_additional_delay_sessions: int
    cagr_strictly_above: float
    sharpe_strictly_above: float
    drawdown_at_most: float
    primary_benchmark: str
    secondary_benchmark: str
    risk_free_symbol: str
    symbols: tuple[str, ...]
    candidates: tuple[PublicRule, ...]
    methodology: dict

    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.horizons_years != (10, 5)
            or self.prior_disclosed_trials < 34
            or self.capital_usd != 10000
            or self.cash_reserve != 0.02
            or self.cagr_strictly_above != 0.20
            or self.sharpe_strictly_above != 1.0
            or self.drawdown_at_most != 0.15
            or self.primary_benchmark != "SPY"
            or self.secondary_benchmark != "QQQ"
            or self.risk_free_symbol != "^IRX"
            or self.methodology.get("order_authority") is not False
        ):
            raise QuantError(
                "Dual-horizon research may not relax the declared acceptance or authority."
            )
        numeric = (self.cost_bps_per_side, self.commission_per_order, self.stress_cost_bps_per_side)
        if (
            not all(math.isfinite(value) for value in numeric)
            or not 0 < self.cost_bps_per_side < self.stress_cost_bps_per_side < 100
            or self.commission_per_order <= 0
            or type(self.stress_additional_delay_sessions) is not int
            or self.stress_additional_delay_sessions < 1
        ):
            raise QuantError("Invalid dual-horizon costs or execution stress.")
        for day in (self.registered_on, self.data_start, self.as_of):
            value = pd.Timestamp(day)
            if value.tzinfo is not None or pd.isna(value) or value.date().isoformat() != day:
                raise QuantError("Dual-horizon dates must be ISO session dates.")
        if (
            pd.Timestamp(self.data_start) >= pd.Timestamp(self.as_of) - pd.DateOffset(years=11)
            or not self.symbols
            or len(set(self.symbols)) != len(self.symbols)
            or any(not re.fullmatch(r"[A-Z]{1,6}", symbol) for symbol in self.symbols)
            or not {"SPY", "QQQ", "BIL", "TIP"} <= set(self.symbols)
            or len(self.candidates) != 8
        ):
            raise QuantError("The registered long-history universe or candidate count is invalid.")
        known = {}
        for rule in self.candidates:
            if (
                not re.fullmatch(r"[a-z0-9_]+", rule.id)
                or rule.id in known
                or rule.family not in {"vaa", "daa", "baa", "haa", "gem", "adm", "ensemble"}
                or not rule.source
                or any(
                    len(values) != len(set(values))
                    for values in (rule.offensive, rule.defensive, rule.canary, rule.components)
                )
                or not set(rule.offensive + rule.defensive + rule.canary) <= set(self.symbols)
            ):
                raise QuantError("Invalid public-rule identity, universe, or attribution.")
            if rule.family == "ensemble":
                if (
                    tuple(known) != rule.components
                    or len(rule.components) != 7
                    or rule.offensive
                    or rule.defensive
                    or rule.canary
                ):
                    raise QuantError(
                        "The ensemble must equally combine all seven earlier fixed rules."
                    )
            elif (
                rule.components
                or type(rule.offensive_count) is not int
                or type(rule.defensive_count) is not int
                or not 1 <= rule.offensive_count <= len(rule.offensive)
                or not 1 <= rule.defensive_count <= len(rule.defensive)
            ):
                raise QuantError("Invalid fixed selection counts.")
            if rule.family in {"vaa", "daa", "baa", "haa", "gem"} and not rule.canary:
                raise QuantError("This rule requires its declared canary universe.")
            known[rule.id] = rule

    def windows(self) -> list[dict]:
        result = []
        end = pd.Timestamp(self.as_of)
        if self.as_of not in sessions(self.as_of, self.as_of).strftime("%Y-%m-%d"):
            raise QuantError("The common end date must be a completed exchange session.")
        for years in self.horizons_years:
            anchor = end - pd.DateOffset(years=years)
            index = sessions(anchor + pd.Timedelta(days=1), end)
            result.append(
                {
                    "years": years,
                    "calendar_anchor": anchor.date().isoformat(),
                    "first_return_session": index[0].date().isoformat(),
                    "last_session": index[-1].date().isoformat(),
                    "sessions": len(index),
                }
            )
        return result


def load_protocol(path: Path) -> DualProtocol:
    try:
        raw = read_json(path)
        raw["symbols"] = tuple(raw["symbols"])
        raw["horizons_years"] = tuple(raw["horizons_years"])
        raw["candidates"] = tuple(
            PublicRule(
                **{
                    **item,
                    **{
                        key: tuple(item[key])
                        for key in ("offensive", "defensive", "canary", "components")
                    },
                }
            )
            for item in raw["candidates"]
        )
        protocol = DualProtocol(**raw)
        protocol.validate()
        return protocol
    except (TypeError, ValueError, KeyError) as exc:
        raise QuantError(f"Invalid dual-horizon protocol: {exc}") from exc


def fingerprint() -> str:
    return digest_json(
        {
            "dual_horizon": file_digest(Path(__file__)),
            "bt_audit": file_digest(Path(__file__).with_name("bt_audit.py")),
            "original_frozen_core": implementation_fingerprint(),
            "bt_version": version("bt"),
        }
    )


def verify_registration(protocol: DualProtocol, path: Path) -> dict:
    record = read_json(path)
    if (
        record.get("protocol_sha256") != digest_json(asdict(protocol))
        or record.get("candidate_ids") != [rule.id for rule in protocol.candidates]
        or record.get("windows") != protocol.windows()
        or record.get("same_rule_required_in_both_windows") is not True
        or record.get("prior_disclosed_trials") != protocol.prior_disclosed_trials
        or record.get("new_trials") != len(protocol.candidates)
        or record.get("cumulative_trials_after_round")
        != protocol.prior_disclosed_trials + len(protocol.candidates)
    ):
        raise QuantError(
            "The dual-horizon rules, windows, or trial count changed after registration."
        )
    return record


def verify_sources(root: Path, registration: dict) -> None:
    manifest_path = root / "manifest.json"
    if registration.get("source_manifest_sha256") != file_digest(manifest_path):
        raise QuantError("The reviewed open-source/license manifest changed.")
    for project in read_json(manifest_path)["items"]:
        if project.get("license") != "MIT":
            raise QuantError("An unreviewed upstream license was substituted.")
        for item in project["files"]:
            path = (root / item["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or not path.is_file()
                or file_digest(path) != item["sha256"]
            ):
                raise QuantError("A retained upstream source or license notice changed.")


def fetch_prices(protocol: DualProtocol, registration: Path, output: Path) -> dict:
    verify_registration(protocol, registration)
    if pd.Timestamp(protocol.as_of) > completed_session():
        raise QuantError("The declared end session has not fully completed.")
    if output.exists():
        raise QuantError("Refusing to overwrite a dual-horizon market snapshot.")
    collected = {}
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-quant-research/0.1 (personal research)"
        for symbol in (*protocol.symbols, protocol.risk_free_symbol):
            start = (
                (pd.Timestamp(protocol.data_start) - pd.Timedelta(days=14)).date().isoformat()
                if symbol == protocol.risk_free_symbol
                else protocol.data_start
            )
            params = {
                "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
                "period2": int(
                    (pd.Timestamp(protocol.as_of, tz="UTC") + pd.Timedelta(days=1)).timestamp()
                ),
                "interval": "1d",
                "events": "div,splits",
            }
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
            try:
                response = client.get(url, params=params, timeout=(10, 60))
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise QuantError(f"Dual-horizon data request for {symbol} failed: {exc}") from exc
            frame = parse_chart(payload, symbol, start, protocol.as_of)
            record = payload["chart"]["result"][0]
            action_dates = set()
            for kind in ("dividends", "splits", "capitalGains"):
                for event in record.get("events", {}).get(kind, {}).values():
                    action_dates.add(
                        pd.Timestamp(event["date"], unit="s", tz="UTC")
                        .tz_convert("America/New_York")
                        .normalize()
                        .tz_localize(None)
                    )
            if symbol != protocol.risk_free_symbol:
                factor_changes = (frame["adj_close"] / frame["close"]).pct_change().abs()
                unexplained = [
                    day.date().isoformat()
                    for day in factor_changes[factor_changes > 0.0001].index
                    if day not in action_dates
                ]
                if unexplained:
                    raise QuantError(
                        f"{symbol} has unexplained corporate-action adjustments: {unexplained}"
                    )
            collected[symbol] = (
                response.text,
                frame,
                {
                    "url": response.url,
                    "rows": len(frame),
                    "first_session": frame.index[0].date().isoformat(),
                    "last_session": frame.index[-1].date().isoformat(),
                    "dividends": len(record.get("events", {}).get("dividends", {})),
                    "splits": len(record.get("events", {}).get("splits", {})),
                    "adjustment_jumps_explained": symbol != protocol.risk_free_symbol,
                },
            )
    files, sources = {}, {}
    for symbol, (raw, frame, metadata) in collected.items():
        filename = "IRX" if symbol == protocol.risk_free_symbol else symbol
        raw_path, csv_path = output / "raw" / f"{filename}.json", output / f"{filename}.csv"
        write_text_atomic(raw_path, raw)
        write_text_atomic(csv_path, frame.to_csv(float_format="%.12g"))
        files[raw_path.relative_to(output).as_posix()] = file_digest(raw_path)
        files[csv_path.name] = file_digest(csv_path)
        sources[symbol] = metadata
    manifest = {
        "retrieved_at": utc_now(),
        "protocol_sha256": digest_json(asdict(protocol)),
        "registration_sha256": file_digest(registration),
        "data_start": protocol.data_start,
        "data_end": protocol.as_of,
        "provider": "Yahoo public chart; research only, not an execution feed",
        "files": files,
        "sources": sources,
        "synthetic_preinception_prices": False,
        "history_status": "retrospective_diagnostics_not_independent_forward_evidence",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prices(protocol: DualProtocol, root: Path) -> MarketData:
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("protocol_sha256") != digest_json(asdict(protocol))
        or manifest.get("data_start") != protocol.data_start
        or manifest.get("data_end") != protocol.as_of
    ):
        raise QuantError("The dual-horizon snapshot does not match the registered date/universe.")
    required = {f"{symbol}.csv" for symbol in protocol.symbols} | {"IRX.csv"}
    if not required <= set(manifest["files"]):
        raise QuantError("Dual-horizon manifest omits a required series.")
    for name, expected in manifest["files"].items():
        file = (root / name).resolve()
        if (
            not file.is_relative_to(root.resolve())
            or not file.is_file()
            or file_digest(file) != expected
        ):
            raise QuantError(f"Market evidence path is unsafe, missing, or changed: {name}")
    frames = {
        symbol: pd.read_csv(root / f"{symbol}.csv", index_col="date", parse_dates=["date"])
        for symbol in protocol.symbols
    }
    close = pd.DataFrame({symbol: frame["adj_close"] for symbol, frame in frames.items()})
    rates = pd.read_csv(root / "IRX.csv", index_col="date", parse_dates=["date"])["close"]
    data = MarketData(
        open=pd.DataFrame({symbol: frame["adj_open"] for symbol, frame in frames.items()}),
        close=close,
        raw_close=pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()}),
        volume=pd.DataFrame({symbol: frame["volume"] for symbol, frame in frames.items()}),
        risk_free=risk_free_returns(close.index, rates),
    )
    data.validate()
    if not data.close.index.equals(sessions(protocol.data_start, protocol.as_of)):
        raise QuantError("The dual-horizon panel does not cover its entire declared interval.")
    return data


def month_end_weights(
    month_prices: pd.DataFrame, rule: PublicRule, protocol: DualProtocol
) -> pd.Series:
    if len(month_prices) < 13:
        raise QuantError("Public monthly rules need thirteen completed month-end observations.")
    values = month_prices.tail(13)
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise QuantError("Invalid observed prices for monthly momentum.")
    latest = values.iloc[-1]
    returns = {n: latest / values.iloc[-n - 1] - 1 for n in (1, 3, 6, 12)}
    weighted = sum(factor * returns[n] for n, factor in ((1, 12), (3, 4), (6, 2), (12, 1)))
    simple = sum(returns.values()) / 4
    relative = latest / values.mean()
    weights = pd.Series(0.0, index=month_prices.columns)

    def ranked(symbols: tuple[str, ...], score: pd.Series, number: int) -> list[str]:
        return sorted(symbols, key=lambda symbol: (-float(score[symbol]), symbol))[:number]

    def allocate(symbols: list[str], budget: float) -> None:
        if not symbols:
            raise QuantError("A public rule produced an empty allocation.")
        for symbol in symbols:
            weights[symbol] += budget / len(symbols)

    if rule.family == "vaa":
        risk_on = bool((weighted.loc[list(rule.canary)] > 0).all())
        universe = rule.offensive if risk_on else rule.defensive
        allocate(ranked(universe, weighted, 1), 1)
    elif rule.family == "daa":
        defensive = float((weighted.loc[list(rule.canary)] <= 0).mean())
        allocate(ranked(rule.offensive, weighted, rule.offensive_count), 1 - defensive)
        allocate(ranked(rule.defensive, weighted, 1), defensive)
    elif rule.family == "baa":
        risk_on = bool((weighted.loc[list(rule.canary)] > 0).all())
        if risk_on:
            allocate(ranked(rule.offensive, relative, rule.offensive_count), 1)
        else:
            chosen = ranked(rule.defensive, relative, rule.defensive_count)
            chosen = [symbol if relative[symbol] >= relative["BIL"] else "BIL" for symbol in chosen]
            allocate(chosen, 1)
    elif rule.family == "haa":
        safe = ranked(rule.defensive, simple, 1)[0]
        if simple[rule.canary[0]] > 0:
            chosen = ranked(rule.offensive, simple, rule.offensive_count)
            allocate([symbol if simple[symbol] > 0 else safe for symbol in chosen], 1)
        else:
            allocate([safe], 1)
    elif rule.family == "gem":
        universe = (
            rule.offensive if returns[12]["SPY"] > returns[12][rule.canary[0]] else rule.defensive
        )
        allocate(ranked(universe, returns[12], 1), 1)
    elif rule.family == "adm":
        score = (returns[1] + returns[3] + returns[6]) / 3
        chosen = ranked(rule.offensive, score, 1)[0]
        allocate([chosen] if score[chosen] > 0 else [rule.defensive[0]], 1)
    else:
        raise QuantError("Ensembles require the independently computed component signals.")
    return weights * (1 - protocol.cash_reserve)


def build_signals(close: pd.DataFrame, protocol: DualProtocol) -> dict[str, pd.DataFrame]:
    if tuple(close.columns) != protocol.symbols or not close.index.is_unique:
        raise QuantError("Public-rule price columns must exactly match their declared universe.")
    monthly = close.loc[[is_month_end(day) for day in close.index]]
    signals = {}
    for rule in protocol.candidates:
        frame = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for index in range(12, len(monthly)):
            day = monthly.index[index]
            if rule.family == "ensemble":
                frame.loc[day] = sum(signals[key].loc[day] for key in rule.components) / len(
                    rule.components
                )
            else:
                frame.loc[day] = month_end_weights(monthly.iloc[: index + 1], rule, protocol)
        allocated = frame.dropna(how="all")
        if (
            not np.isfinite(allocated.to_numpy()).all()
            or (allocated < 0).any().any()
            or not np.allclose(allocated.sum(axis=1), 1 - protocol.cash_reserve, atol=1e-12)
        ):
            raise QuantError("Public-rule weights violated the cash reserve or long-only limits.")
        signals[rule.id] = frame
    return signals


def seed_window(signals: pd.DataFrame, start: str) -> pd.DataFrame:
    first = pd.Timestamp(start)
    anchor = previous_session(first)
    earlier = signals.loc[:anchor].dropna(how="all")
    if earlier.empty or anchor not in signals.index:
        raise QuantError("The evaluation window lacks a previously known allocation.")
    result = signals.copy()
    result.loc[anchor] = earlier.iloc[-1]
    return result


def run_window(
    data: MarketData,
    signal: pd.DataFrame,
    start: str,
    end: str,
    protocol: DualProtocol,
    *,
    stress: bool = False,
) -> BacktestResult:
    return simulate(
        data,
        signal,
        start,
        end,
        initial_capital=protocol.capital_usd,
        cost_bps=protocol.stress_cost_bps_per_side if stress else protocol.cost_bps_per_side,
        commission=protocol.commission_per_order,
        delay=1 + (protocol.stress_additional_delay_sessions if stress else 0),
    )


def metrics(frame: pd.DataFrame) -> dict:
    values = performance(frame["return"], frame["risk_free"])
    if "cost" in frame:
        values.update(
            {
                "total_cost_dollars": float(frame["cost"].sum()),
                "annualized_one_way_turnover": float(frame["turnover"].sum() / (len(frame) / 252)),
                "order_tickets": int(frame["orders"].sum()),
            }
        )
    return values


def gates(strategy: dict, benchmark: dict, protocol: DualProtocol) -> dict:
    if (
        strategy["start"] != benchmark["start"]
        or strategy["end"] != benchmark["end"]
        or strategy["sessions"] != benchmark["sessions"]
    ):
        raise QuantError("A target comparison may not use different market intervals.")
    return {
        "cagr_above_20pct": bool(strategy["cagr"] > protocol.cagr_strictly_above),
        "sharpe_above_1": bool(
            strategy["sharpe"] is not None and strategy["sharpe"] > protocol.sharpe_strictly_above
        ),
        "max_drawdown_at_most_15pct": bool(strategy["max_drawdown"] <= protocol.drawdown_at_most),
        "beats_spy": bool(strategy["cagr"] > benchmark["cagr"]),
    }


def rolling_diagnostics(
    frame: pd.DataFrame, benchmark: pd.DataFrame, years: int, protocol: DualProtocol
) -> dict:
    records = []
    for end in frame.index:
        if not is_month_end(end):
            continue
        anchor = end - pd.DateOffset(years=years)
        if anchor < previous_session(frame.index[0]):
            continue
        chosen = frame.loc[(frame.index > anchor) & (frame.index <= end)]
        bench = benchmark.loc[chosen.index]
        strategy_metrics, benchmark_metrics = metrics(chosen), metrics(bench)
        records.append(
            {
                "start": strategy_metrics["start"],
                "end": strategy_metrics["end"],
                "cagr": strategy_metrics["cagr"],
                "sharpe": strategy_metrics["sharpe"],
                "max_drawdown": strategy_metrics["max_drawdown"],
                "spy_cagr": benchmark_metrics["cagr"],
                "joint_pass": all(gates(strategy_metrics, benchmark_metrics, protocol).values()),
            }
        )
    return {
        "years": years,
        "observation_frequency": "completed month ends",
        "accounting": "slices of one continuously managed account; not separately refitted windows",
        "windows": len(records),
        "joint_pass_count": sum(row["joint_pass"] for row in records),
        "minimum_cagr": min((row["cagr"] for row in records), default=None),
        "median_cagr": float(np.median([row["cagr"] for row in records])) if records else None,
        "maximum_drawdown": max((row["max_drawdown"] for row in records), default=None),
        "records": records,
    }


def evaluate(
    protocol: DualProtocol, registration_path: Path, prices_path: Path, output: Path
) -> dict:
    import ffn

    registration = verify_registration(protocol, registration_path)
    data = load_prices(protocol, prices_path)
    manifest = read_json(prices_path / "manifest.json")
    if manifest.get("registration_sha256") != file_digest(registration_path) or pd.Timestamp(
        manifest["retrieved_at"]
    ) < pd.Timestamp(registration["registered_at"]):
        raise QuantError("The market snapshot was not obtained under this preregistration.")
    windows = protocol.windows()
    if output.exists():
        raise QuantError("Refusing to overwrite an existing simultaneous-horizon result.")
    signals = build_signals(data.close, protocol)
    candidates, independent_checks = {}, []
    benchmarks = {}
    for window in windows:
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        benchmarks[key] = {}
        for symbol in (protocol.primary_benchmark, protocol.secondary_benchmark):
            benchmark_signal = buy_and_hold_signals(data.close, symbol, start)
            result = run_window(data, benchmark_signal, start, end, protocol)
            benchmarks[key][symbol] = result
    continuous_start = next(iter(signals.values())).dropna(how="all").index[0]
    following = data.close.index[data.close.index.get_loc(continuous_start) + 1]
    continuous_start = following.date().isoformat()
    continuous_spy = run_window(
        data,
        buy_and_hold_signals(data.close, "SPY", continuous_start),
        continuous_start,
        protocol.as_of,
        protocol,
    )
    artifacts = {}
    for rule in protocol.candidates:
        candidate = {"family": rule.family, "source": rule.source, "windows": {}}
        for window in windows:
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            known_signals = seed_window(signals[rule.id], start)
            bench_metrics = metrics(benchmarks[key]["SPY"].frame)
            evaluation = {"window": window, "benchmark_spy": bench_metrics}
            for stress in (False, True):
                name = "stress" if stress else "base"
                result = run_window(data, known_signals, start, end, protocol, stress=stress)
                record = metrics(result.frame)
                condition = gates(record, bench_metrics, protocol)
                independent = independent_equity(
                    data,
                    known_signals,
                    start,
                    end,
                    capital=protocol.capital_usd,
                    cost_bps=protocol.stress_cost_bps_per_side
                    if stress
                    else protocol.cost_bps_per_side,
                    commission=protocol.commission_per_order,
                    delay=1 + (protocol.stress_additional_delay_sessions if stress else 0),
                )
                difference = float(abs(independent["equity"] - result.frame["equity"]).max())
                tolerance = protocol.capital_usd * 1e-8
                if difference > tolerance:
                    raise QuantError(
                        f"Independent bt mismatch for {rule.id}/{key}/{name}: "
                        f"${difference:.8f} exceeds ${tolerance:.8f}."
                    )
                independent_prices = pd.concat(
                    [
                        pd.Series(
                            [protocol.capital_usd],
                            index=[previous_session(independent.index[0])],
                        ),
                        independent["equity"],
                    ]
                )
                ffn_cagr = float(ffn.calc_cagr(independent_prices))
                ffn_drawdown = float(-ffn.calc_max_drawdown(independent_prices))
                excess = independent["return"] - result.frame["risk_free"]
                deviation = float(excess.std(ddof=1))
                audit_sharpe = (
                    float(excess.mean() / deviation * np.sqrt(252))
                    if deviation > 1e-12 and independent["return"].std(ddof=1) > 1e-12
                    else None
                )
                if (
                    abs(ffn_cagr - record["cagr"]) > 1e-8
                    or abs(ffn_drawdown - record["max_drawdown"]) > 1e-8
                    or (audit_sharpe is None) != (record["sharpe"] is None)
                    or (audit_sharpe is not None and abs(audit_sharpe - record["sharpe"]) > 1e-8)
                ):
                    raise QuantError(
                        "Independent equity/statistics disagree with reported performance."
                    )
                independent_checks.append(
                    {
                        "candidate": rule.id,
                        "window": key,
                        "scenario": name,
                        "max_equity_difference_usd": difference,
                        "tolerance_usd": tolerance,
                        "ffn_cagr": ffn_cagr,
                        "ffn_max_drawdown": ffn_drawdown,
                        "independent_excess_sharpe": audit_sharpe,
                        "passed": True,
                    }
                )
                evaluation[name] = {
                    "metrics": record,
                    "gates": condition,
                    "all_numeric_gates_passed": all(condition.values()),
                }
                if not stress:
                    evaluation["conditional_bootstrap"] = block_bootstrap(
                        result.frame["return"],
                        benchmarks[key]["SPY"].frame["return"],
                        result.frame["risk_free"],
                        samples=1000,
                        block=21,
                        seed=20261006,
                    )
                artifacts[f"{rule.id}/{key}-{name}.csv"] = result.frame.to_csv(float_format="%.12g")
                artifacts[f"{rule.id}/{key}-{name}-bt.csv"] = independent.to_csv(
                    float_format="%.12g"
                )
            candidate["windows"][key] = evaluation
        ongoing = run_window(data, signals[rule.id], continuous_start, protocol.as_of, protocol)
        candidate["rolling_windows"] = {
            f"{years}y": rolling_diagnostics(ongoing.frame, continuous_spy.frame, years, protocol)
            for years in protocol.horizons_years
        }
        chronology = {}
        for first, last in (
            (continuous_start, "2015-12-31"),
            ("2016-01-01", "2020-12-31"),
            ("2021-01-01", protocol.as_of),
        ):
            frame = ongoing.frame.loc[first:last]
            if len(frame) >= 2:
                chronology[f"{first}/{last}"] = {
                    "strategy": metrics(frame),
                    "spy": metrics(continuous_spy.frame.loc[frame.index]),
                    "interpretation": "fixed-rule chronological diagnostics; not a new holdout",
                }
        candidate["chronological_diagnostics"] = chronology
        candidate["both_horizons_pass"] = all(
            value["base"]["all_numeric_gates_passed"] for value in candidate["windows"].values()
        )
        candidate["both_horizons_stress_pass"] = all(
            value["stress"]["all_numeric_gates_passed"] for value in candidate["windows"].values()
        )
        candidate["robust_numeric_pass"] = (
            candidate["both_horizons_pass"] and candidate["both_horizons_stress_pass"]
        )
        candidate["order_authority"] = False
        candidates[rule.id] = candidate
        artifacts[f"{rule.id}/continuous.csv"] = ongoing.frame.to_csv(float_format="%.12g")
        artifacts[f"{rule.id}/monthly_signals.csv"] = (
            signals[rule.id].dropna(how="all").to_csv(float_format="%.12g")
        )
    result = {
        "stage": "simultaneous_trailing_10y_5y_research",
        "created_at": utc_now(),
        "protocol_sha256": digest_json(asdict(protocol)),
        "implementation_sha256": fingerprint(),
        "registration_sha256": file_digest(registration_path),
        "market_manifest_sha256": file_digest(prices_path / "manifest.json"),
        "prior_trials": protocol.prior_disclosed_trials,
        "new_trials": len(protocol.candidates),
        "cumulative_trials": protocol.prior_disclosed_trials + len(protocol.candidates),
        "windows": windows,
        "same_rule_and_shared_endpoint": True,
        "numeric_joint_passes": [
            key for key, value in candidates.items() if value["both_horizons_pass"]
        ],
        "stress_robust_joint_passes": [
            key for key, value in candidates.items() if value["robust_numeric_pass"]
        ],
        "independent_bt": {
            "version": version("bt"),
            "ffn_version": version("ffn"),
            "checks": independent_checks,
            "all_passed": all(row["passed"] for row in independent_checks),
            "scope": "same causal weights; independent bt/fee accounting and cash-budget solver",
        },
        "candidates": candidates,
        "benchmarks": {
            key: {symbol: metrics(value.frame) for symbol, value in items.items()}
            for key, items in benchmarks.items()
        },
        "registration": registration,
        "investment_objective_verified": False,
        "paper_order_authority": False,
        "limitations": [
            "Eight methods/proxies follow earlier failed research; all forty-two trials count.",
            "The 5-year window is inside the 10-year window; they are not independent samples.",
            "Historical data was partly examined previously; no untouched holdout claim.",
            "Adjusted OHLC includes gross distributions, not exact pay-date cash or personal tax.",
            "Fractional units, opening liquidity, and flat ticket fees are modeling assumptions.",
            "The risk-free proxy is lagged ^IRX, not cash earned in the IBKR account.",
            "Disclosed proxies, 2% cash, and next-open fills differ from published examples.",
            "Historical numeric passes do not prove future returns or broker execution quality.",
        ],
    }
    new_output_directory(output)
    for relative_path, content in artifacts.items():
        write_text_atomic(output / "candidates" / relative_path, content)
    for key, items in benchmarks.items():
        for symbol, value in items.items():
            write_text_atomic(
                output / f"benchmark-{symbol}-{key}.csv", value.frame.to_csv(float_format="%.12g")
            )
    result["artifact_sha256"] = {
        str(path.relative_to(output)): file_digest(path) for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "results.json", result)
    write_text_atomic(output / "report.md", render_report(result))
    return result


def render_report(result: dict) -> str:
    lines = [
        "# Same-rule ten-year and five-year evaluation",
        "",
        "**Retrospective research only: no order authority or future-return promise.**",
        "",
        f"Disclosed trials: {result['cumulative_trials']}. "
        f"Common endpoint: {result['windows'][0]['last_session']}.",
        "",
        "| Method | 10y CAGR | Sharpe | Drawdown | 5y CAGR | Sharpe | Drawdown | Both pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fields(value: dict) -> str:
        sharpe = "undefined" if value["sharpe"] is None else f"{value['sharpe']:.2f}"
        return f"{value['cagr']:.2%} | {sharpe} | {value['max_drawdown']:.2%}"

    for key, candidate in result["candidates"].items():
        ten, five = (candidate["windows"][name]["base"]["metrics"] for name in ("10y", "5y"))
        lines.append(
            f"| {key} | {fields(ten)} | {fields(five)} | "
            f"{'PASS' if candidate['both_horizons_pass'] else 'FAIL'} |"
        )
    for symbol in ("SPY", "QQQ"):
        lines.append(
            f"| {symbol} benchmark | {fields(result['benchmarks']['10y'][symbol])} | "
            f"{fields(result['benchmarks']['5y'][symbol])} | comparison only |"
        )
    lines.extend(
        [
            "",
            "Required in both windows: net CAGR >20%, excess-return Sharpe >1, "
            "max drawdown <=15%, and higher CAGR than same-period SPY.",
            "Costs: 5 bps/side + $1/ticket. Stress: 20 bps/side + one extra session.",
            "Each window starts with $10,000; same rules/parameters, no separate optimization.",
            "",
            f"Independent bt 1.3.0 checks passed: {result['independent_bt']['all_passed']}.",
            f"Base joint passes: {result['numeric_joint_passes']}.",
            f"Base plus stress joint passes: {result['stress_robust_joint_passes']}.",
            "",
            "## Limitations",
            "",
            *(f"- {text}" for text in result["limitations"]),
            "",
        ]
    )
    return "\n".join(lines)


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "dual-horizon", help="Fixed-method simultaneous 10y/5y evaluation."
    )
    command.add_argument("stage", choices=["fetch", "evaluate"])
    command.add_argument("--protocol", type=Path, default=Path("config/dual-horizon.json"))
    command.add_argument(
        "--registration", type=Path, default=Path("reports/dual-horizon/registration.json")
    )
    command.add_argument("--data", type=Path, default=Path("data/dual-horizon/market"))
    command.add_argument("--sources", type=Path, default=Path("data/dual-horizon/open-source"))
    command.add_argument("--output", type=Path, default=Path("reports/dual-horizon/evaluation"))


def dispatch_dual(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol)
    registered = verify_registration(protocol, args.registration)
    verify_sources(args.sources, registered)
    if args.stage == "fetch":
        report = fetch_prices(protocol, args.registration, args.data)
        return {
            "data_start": report["data_start"],
            "data_end": report["data_end"],
            "series": len(report["sources"]),
            "saved_to": str(args.data),
        }
    report = evaluate(protocol, args.registration, args.data, args.output)
    return {
        key: report[key]
        for key in (
            "stage",
            "windows",
            "cumulative_trials",
            "numeric_joint_passes",
            "stress_robust_joint_passes",
            "investment_objective_verified",
            "paper_order_authority",
        )
    }
