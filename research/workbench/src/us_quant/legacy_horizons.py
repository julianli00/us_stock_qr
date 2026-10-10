from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import requests

from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError, ResearchConfig, load_config
from us_quant.data import MarketData, parse_chart
from us_quant.dual_horizon import (
    DualProtocol,
    gates,
    load_prices,
    load_protocol,
    metrics,
    run_window,
    seed_window,
)
from us_quant.evolution import candidate_config, seed_genome
from us_quant.expanded import build_intents, load_expanded
from us_quant.regime import build_intent
from us_quant.regime import load_protocol as load_regime
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals, monthly_signals

INPUTS = (
    "config/research.json",
    "config/expanded.json",
    "config/evolution.json",
    "config/geared-etf.json",
    "config/daily-regime.json",
    "reports/evolution/batch-screens/2026-09-29-registration.json",
)


@dataclass(frozen=True)
class LegacyCase:
    id: str
    family: str
    config: ResearchConfig
    description: dict


def cases(root: Path) -> list[LegacyCase]:
    base = load_config(root / "config/research.json")
    geared = load_config(root / "config/geared-etf.json")
    expanded = load_expanded(root / "config/expanded.json", base)
    regime = load_regime(root / "config/daily-regime.json", geared)
    evolution_policy = read_json(root / "config/evolution.json")
    result = []
    for config, family in ((base, "original_monthly"), (geared, "geared_monthly")):
        for candidate in config.candidates:
            result.append(LegacyCase(candidate.id, family, config, asdict(candidate)))
    for candidate in expanded.candidates:
        result.append(
            LegacyCase(
                candidate.id, "expanded_family", expanded.data_config(base), asdict(candidate)
            )
        )
    for candidate in regime.candidates:
        result.append(LegacyCase(candidate.id, "daily_regime", geared, asdict(candidate)))
    genomes = [
        {
            "id": "evo_" + digest_json(seed_genome(base, evolution_policy))[:12],
            "genome": seed_genome(base, evolution_policy),
        },
        *read_json(root / "reports/evolution/batch-screens/2026-09-29-registration.json")[
            "candidates"
        ],
    ]
    for item in genomes:
        config = candidate_config(base, evolution_policy, item["id"], item["genome"])
        result.append(LegacyCase(item["id"], "evolution_monthly", config, item["genome"]))
    if len(result) != 34 or len({item.id for item in result}) != 34:
        raise QuantError("The legacy comparison must disclose exactly 34 original configurations.")
    return result


def source_fingerprints(root: Path) -> dict:
    return {name: file_digest(root / name) for name in INPUTS}


def register_comparison(root: Path, protocol: DualProtocol, output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite the frozen legacy-horizon comparison plan.")
    prior_cases = cases(root)
    all_symbols = sorted({symbol for item in prior_cases for symbol in item.config.symbols})
    result = {
        "registered_at": utc_now(),
        "mode": "same_rule_recomparison_not_new_hypotheses",
        "prior_configurations": [asdict(item) for item in prior_cases],
        "case_ids": [item.id for item in prior_cases],
        "windows": protocol.windows(),
        "comparison_protocol_sha256": digest_json(asdict(protocol)),
        "source_fingerprints": source_fingerprints(root),
        "required_supplemental_symbols": sorted(set(all_symbols) - set(protocol.symbols)),
        "capital_usd": protocol.capital_usd,
        "cost_policy": "per-case max(original cost, new comparison cost), including stress",
        "new_hypotheses": 0,
        "warning": "Re-windowing known rules is retrospective exposure, not new holdout data.",
        "order_authority": False,
    }
    write_json(output, result)
    return result


def verify_comparison(root: Path, protocol: DualProtocol, registration: Path) -> dict:
    record = read_json(registration)
    if (
        record.get("comparison_protocol_sha256") != digest_json(asdict(protocol))
        or record.get("source_fingerprints") != source_fingerprints(root)
        or record.get("case_ids") != [item.id for item in cases(root)]
        or digest_json(record.get("prior_configurations"))
        != digest_json([asdict(item) for item in cases(root)])
        or record.get("windows") != protocol.windows()
        or record.get("new_hypotheses") != 0
    ):
        raise QuantError("The legacy rules, windows, or comparison plan changed.")
    return record


def fetch_supplement(protocol: DualProtocol, record: dict, output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite supplemental legacy price history.")
    fetched = {}
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-quant-research/0.1 (personal research)"
        for symbol in record["required_supplemental_symbols"]:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            try:
                response = client.get(
                    url,
                    params={
                        "period1": int(pd.Timestamp(protocol.data_start, tz="UTC").timestamp()),
                        "period2": int(
                            (
                                pd.Timestamp(protocol.as_of, tz="UTC") + pd.Timedelta(days=1)
                            ).timestamp()
                        ),
                        "interval": "1d",
                        "events": "div,splits",
                    },
                    timeout=(10, 60),
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise QuantError(f"Supplemental data failed for {symbol}: {exc}") from exc
            frame = parse_chart(payload, symbol, protocol.data_start, protocol.as_of)
            fetched[symbol] = (response.text, frame, response.url)
    files, sources = {}, {}
    for symbol, (raw, frame, url) in fetched.items():
        raw_path, csv_path = output / "raw" / f"{symbol}.json", output / f"{symbol}.csv"
        write_text_atomic(raw_path, raw)
        write_text_atomic(csv_path, frame.to_csv(float_format="%.12g"))
        for path in (raw_path, csv_path):
            files[path.relative_to(output).as_posix()] = file_digest(path)
        sources[symbol] = {"url": url, "rows": len(frame)}
    manifest = {
        "retrieved_at": utc_now(),
        "comparison_registration_sha256": digest_json(record),
        "data_start": protocol.data_start,
        "data_end": protocol.as_of,
        "files": files,
        "sources": sources,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def combined_prices(
    protocol: DualProtocol, base_path: Path, supplement_path: Path, record: dict
) -> MarketData:
    original = load_prices(protocol, base_path)
    manifest = read_json(supplement_path / "manifest.json")
    if manifest.get("comparison_registration_sha256") != digest_json(record) or set(
        manifest["sources"]
    ) != set(record["required_supplemental_symbols"]):
        raise QuantError("Supplemental data does not match the comparison plan.")
    for name, expected in manifest["files"].items():
        path = (supplement_path / name).resolve()
        if (
            not path.is_relative_to(supplement_path.resolve())
            or not path.is_file()
            or file_digest(path) != expected
        ):
            raise QuantError("A supplemental price artifact is unsafe, missing, or revised.")
    extra = {
        symbol: pd.read_csv(
            supplement_path / f"{symbol}.csv", index_col="date", parse_dates=["date"]
        )
        for symbol in record["required_supplemental_symbols"]
    }

    def merge(existing: pd.DataFrame, field: str) -> pd.DataFrame:
        return pd.concat(
            [existing, pd.DataFrame({symbol: frame[field] for symbol, frame in extra.items()})],
            axis=1,
        )

    data = MarketData(
        merge(original.open, "adj_open"),
        merge(original.close, "adj_close"),
        merge(original.raw_close, "close"),
        merge(original.volume, "volume"),
        original.risk_free,
    )
    data.validate()
    return data


def subset(data: MarketData, symbols: tuple[str, ...]) -> MarketData:
    return MarketData(
        data.open.loc[:, list(symbols)],
        data.close.loc[:, list(symbols)],
        data.raw_close.loc[:, list(symbols)],
        data.volume.loc[:, list(symbols)],
        data.risk_free,
    )


def legacy_signals(root: Path, data: MarketData) -> dict[str, tuple[MarketData, pd.DataFrame]]:
    result = {}
    base = load_config(root / "config/research.json")
    expanded = load_expanded(root / "config/expanded.json", base)
    expanded_data = subset(data, expanded.symbols)
    expanded_intents = build_intents(expanded_data, expanded)
    geared = load_config(root / "config/geared-etf.json")
    regime = load_regime(root / "config/daily-regime.json", geared)
    for item in cases(root):
        market = subset(data, item.config.symbols)
        if item.family == "expanded_family":
            signal = expanded_intents[item.id].signals
        elif item.family == "daily_regime":
            candidate = next(rule for rule in regime.candidates if rule.id == item.id)
            signal = build_intent(market, candidate, regime, geared).signals
        else:
            signal = monthly_signals(market.close, item.config.candidate(item.id), item.config)
        result[item.id] = market, signal
    return result


def evaluate_legacy(
    root: Path,
    protocol: DualProtocol,
    registration: Path,
    market_path: Path,
    supplement_path: Path,
    output: Path,
) -> dict:
    record = verify_comparison(root, protocol, registration)
    if output.exists():
        raise QuantError("Refusing to overwrite prior dual-window comparisons.")
    data = combined_prices(protocol, market_path, supplement_path, record)
    signals = legacy_signals(root, data)
    benchmark_values, artifacts = {}, {}
    for window in protocol.windows():
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        benchmark = run_window(
            data, buy_and_hold_signals(data.close, "SPY", start), start, end, protocol
        )
        benchmark_values[key] = metrics(benchmark.frame)
    candidates, differences = {}, []
    for item in cases(root):
        market, signal = signals[item.id]
        comparison = replace(
            protocol,
            cost_bps_per_side=max(protocol.cost_bps_per_side, item.config.cost_bps_per_side),
            stress_cost_bps_per_side=max(
                protocol.stress_cost_bps_per_side, item.config.stress.cost_bps_per_side
            ),
        )
        row = {
            "family": item.family,
            "source_configuration": asdict(item.config),
            "unchanged_rule_description": item.description,
            "capital_usd": comparison.capital_usd,
            "cost_bps_per_side": comparison.cost_bps_per_side,
            "stress_cost_bps_per_side": comparison.stress_cost_bps_per_side,
            "windows": {},
        }
        for window in protocol.windows():
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            initialized = seed_window(signal, start)
            scenarios = {}
            for stress in (False, True):
                label = "stress" if stress else "base"
                result = run_window(market, initialized, start, end, comparison, stress=stress)
                independent = independent_equity(
                    market,
                    initialized,
                    start,
                    end,
                    capital=comparison.capital_usd,
                    cost_bps=comparison.stress_cost_bps_per_side
                    if stress
                    else comparison.cost_bps_per_side,
                    commission=comparison.commission_per_order,
                    delay=1 + (comparison.stress_additional_delay_sessions if stress else 0),
                )
                deviation = float(abs(result.frame["equity"] - independent["equity"]).max())
                if deviation > comparison.capital_usd * 1e-8:
                    raise QuantError(f"Legacy independent bt mismatch: {item.id}/{key}/{label}.")
                differences.append(
                    {
                        "candidate": item.id,
                        "window": key,
                        "scenario": label,
                        "maximum_equity_discrepancy_usd": deviation,
                    }
                )
                values = metrics(result.frame)
                checks = gates(values, benchmark_values[key], comparison)
                scenarios[label] = {
                    "metrics": values,
                    "gates": checks,
                    "passed": all(checks.values()),
                }
                artifacts[f"{item.id}/{key}-{label}.csv"] = result.frame.to_csv(
                    float_format="%.12g"
                )
            row["windows"][key] = scenarios
        row["both_horizons_pass"] = all(
            period["base"]["passed"] for period in row["windows"].values()
        )
        row["both_horizons_stress_pass"] = all(
            period["stress"]["passed"] for period in row["windows"].values()
        )
        candidates[item.id] = row
    report = {
        "created_at": utc_now(),
        "stage": "prior_34_rules_aligned_to_exact_10y_5y_windows",
        "registration_sha256": file_digest(registration),
        "implementation_sha256": file_digest(Path(__file__)),
        "comparison_protocol_sha256": digest_json(asdict(protocol)),
        "source_fingerprints": source_fingerprints(root),
        "new_hypotheses": 0,
        "recompared_original_configurations": len(candidates),
        "windows": protocol.windows(),
        "candidates": candidates,
        "benchmark_spy": benchmark_values,
        "base_joint_passes": [key for key, row in candidates.items() if row["both_horizons_pass"]],
        "stress_joint_passes": [
            key
            for key, row in candidates.items()
            if row["both_horizons_pass"] and row["both_horizons_stress_pass"]
        ],
        "independent_bt_checks": differences,
        "order_authority": False,
        "investment_objective_verified": False,
        "limitations": [
            "Original candidates are re-windowed at a uniform $10,000, not retuned.",
            "The original capital-matched control duplicates its parent at the same capital.",
            "The same historical data was previously examined; no independent holdout claim.",
            "Geared ETFs remain leveraged products; new public-rule methods are separate.",
            "Counts represent registered configurations, not independent statistical experiments.",
        ],
    }
    new_output_directory(output)
    for name, csv in artifacts.items():
        write_text_atomic(output / "candidates" / name, csv)
    report["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "results.json", report)
    lines = [
        "# Original 34 configurations: aligned dual-window comparison",
        "",
        "**No retuning, no new strategy hypotheses, no broker order authority.**",
        "",
        "| Rule | 10y CAGR | Sharpe | Drawdown | 5y CAGR | Sharpe | Drawdown | Joint pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, row in candidates.items():
        text = []
        for period in ("10y", "5y"):
            values = row["windows"][period]["base"]["metrics"]
            sharpe = "undefined" if values["sharpe"] is None else f"{values['sharpe']:.2f}"
            text.append(f"{values['cagr']:.2%} | {sharpe} | {values['max_drawdown']:.2%}")
        lines.append(f"| {key} | {' | '.join(text)} | {row['both_horizons_pass']} |")
    lines.extend(["", *report["limitations"], ""])
    write_text_atomic(output / "report.md", "\n".join(lines))
    return report


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "legacy-horizons", help="Recompare existing rules; no new hypotheses."
    )
    command.add_argument("stage", choices=["register", "fetch", "evaluate"])
    command.add_argument("--protocol", type=Path, default=Path("config/dual-horizon.json"))
    command.add_argument(
        "--registration",
        type=Path,
        default=Path("reports/dual-horizon/legacy-comparison-registration.json"),
    )
    command.add_argument("--data", type=Path, default=Path("data/dual-horizon/market"))
    command.add_argument(
        "--supplement", type=Path, default=Path("data/dual-horizon/legacy-supplement")
    )
    command.add_argument(
        "--output", type=Path, default=Path("reports/dual-horizon/legacy-comparison")
    )


def dispatch_legacy(args: argparse.Namespace) -> dict:
    protocol, root = load_protocol(args.protocol), Path.cwd()
    if args.stage == "register":
        result = register_comparison(root, protocol, args.registration)
        return {
            "registered_cases": len(result["case_ids"]),
            "new_hypotheses": 0,
            "supplemental_symbols": result["required_supplemental_symbols"],
        }
    record = verify_comparison(root, protocol, args.registration)
    if args.stage == "fetch":
        result = fetch_supplement(protocol, record, args.supplement)
        return {"series": len(result["sources"]), "end": result["data_end"]}
    report = evaluate_legacy(
        root, protocol, args.registration, args.data, args.supplement, args.output
    )
    return {
        key: report[key]
        for key in (
            "stage",
            "recompared_original_configurations",
            "new_hypotheses",
            "base_joint_passes",
            "stress_joint_passes",
            "investment_objective_verified",
        )
    }
