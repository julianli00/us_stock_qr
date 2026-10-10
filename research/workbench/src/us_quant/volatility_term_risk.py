from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.adaptive_factor_allocation import prepare as prepare_original_market
from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol, seed_window
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS
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
POLICY = ROOT / "config/volatility-term-risk.json"
SOURCES = ROOT / "data/volatility-term-source-20261010"


def validate(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("risk_indicator") != "VIX/VIX3M"
        or policy.get("risk_off_threshold") != 1.0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("capital_usd") != 10000
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "factor_gold_term_half_risk", "inverted_equity_scale": 0.5},
            {"id": "factor_gold_term_equity_exit", "inverted_equity_scale": 0.0},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("independent_forward_validation") is not False
    ):
        raise QuantError("The fixed term-structure risk hypothesis has changed.")


def load_terms(directory: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    manifest = read_json(directory / "manifest.json")
    records = {row["symbol"]: row for row in manifest["sources"]}
    if set(records) != {"VIX", "VIX3M"}:
        raise QuantError("Both distinct official volatility horizons are required.")
    values = {}
    for symbol in ("VIX", "VIX3M"):
        record = records[symbol]
        expected_url = (
            f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
        )
        path = directory / f"{symbol}.csv"
        if (
            record["url"] != expected_url
            or path.is_symlink()
            or file_digest(path) != record["sha256"]
        ):
            raise QuantError("The official volatility-index source has changed.")
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip()
        dates = pd.to_datetime(raw["DATE"], format="%m/%d/%Y")
        series = pd.Series(pd.to_numeric(raw["CLOSE"]).to_numpy(), index=dates)
        if not series.index.is_unique or not series.index.is_monotonic_increasing:
            raise QuantError("Volatility-index observations must be unique and ordered.")
        if len(index.difference(series.index)):
            raise QuantError("A volatility input lacks a required session; no forward filling.")
        chosen = series.loc[index]
        if not np.isfinite(chosen).all() or (chosen <= 0).any():
            raise QuantError("Volatility-index levels must be finite and positive.")
        values[symbol] = chosen
    return pd.DataFrame(values, index=index)


def build_targets(data, terms: pd.DataFrame, candidate: dict, policy: dict) -> pd.DataFrame:
    validate(policy)
    if (
        candidate not in policy["candidates"]
        or not terms.index.equals(data.close.index)
        or list(terms.columns) != ["VIX", "VIX3M"]
        or not np.isfinite(terms.to_numpy()).all()
        or (terms <= 0).any().any()
    ):
        raise QuantError("Term-structure inputs or candidate identity are invalid.")
    base_policy = read_json(ROOT / "config/factor-gold-risk.json")
    monthly = monthly_targets(data, base_policy)
    result = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    baseline, previous_state = None, None
    for day in data.close.index:
        updated = not monthly.loc[day].isna().all()
        if updated:
            baseline = monthly.loc[day].copy()
        if baseline is None:
            continue
        inverted = bool(terms.loc[day, "VIX"] >= terms.loc[day, "VIX3M"])
        if updated or previous_state is None or inverted != previous_state:
            weights = baseline.copy()
            if inverted:
                equity = weights.loc[list(FACTORS)].sum()
                weights.loc[list(FACTORS)] *= candidate["inverted_equity_scale"]
                weights["BIL"] += equity * (1 - candidate["inverted_equity_scale"])
            if (weights < -1e-12).any() or not np.isclose(weights.sum(), 0.98, atol=1e-10):
                raise QuantError("Risk overlay changed the cash-funded total budget.")
            result.loc[day] = weights
        previous_state = inverted
    return result


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    original = prepare_original_market(output)
    data = review_market({"market": original["specs"][0]["market"]}, ROOT)
    load_terms(SOURCES, data.close.index)
    frozen = {
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        "src/us_quant/factor_gold_risk.py": file_digest(ROOT / "src/us_quant/factor_gold_risk.py"),
        "config/factor-gold-risk.json": file_digest(ROOT / "config/factor-gold-risk.json"),
        (SOURCES / "manifest.json").relative_to(ROOT).as_posix(): file_digest(
            SOURCES / "manifest.json"
        ),
        **{
            (SOURCES / f"{symbol}.csv").relative_to(ROOT).as_posix(): file_digest(
                SOURCES / f"{symbol}.csv"
            )
            for symbol in ("VIX", "VIX3M")
        },
    }
    specs = [
        {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio",
            "evaluation_as_of": policy["as_of"],
            "market": original["specs"][0]["market"],
            "frozen_files": frozen,
            "asset_leverage": original["specs"][0]["asset_leverage"],
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        for candidate in policy["candidates"]
    ]
    result = {
        "study": policy,
        "readiness": original["readiness"],
        "specs": specs,
        "returns_computed": False,
    }
    write_json(output / "term-prepared.json", result)
    return result


def register(prepared: Path, receipt: Path):
    if receipt.exists():
        raise QuantError("An existing term-risk registration may not be overwritten.")
    inputs = read_json(prepared / "term-prepared.json")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        records = []
        for spec in inputs["specs"]:
            prior = program.db.execute(
                "SELECT spec_sha FROM candidates WHERE id=?", (spec["id"],)
            ).fetchone()
            if prior:
                if prior["spec_sha"] != digest_json(spec):
                    raise QuantError("Interrupted term-risk registration changed its definition.")
                old = [
                    json.loads(row["body"])
                    for row in program.db.execute("SELECT body FROM events WHERE kind='candidate'")
                ]
                registration = next(row for row in old if row["candidate_id"] == spec["id"])
            else:
                registration = program.register_candidate(spec, inputs["readiness"])
            records.append({"spec": spec, "registration": registration})
        result = {"study": inputs["study"], "readiness": inputs["readiness"], "candidates": records}
        write_json(receipt, result)
        return result
    finally:
        program.close()


def evaluate(receipt: Path, output: Path):
    inputs = read_json(receipt)
    policy = read_json(POLICY)
    validate(policy)
    if inputs["study"] != policy:
        raise QuantError("The term-risk study changed after its registration.")
    for item in inputs["candidates"]:
        for path, digest in item["spec"]["frozen_files"].items():
            safe_file(ROOT, path, digest)
    new_output_directory(output)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    summary = {"candidates": {}, "order_authority": False, "independent_forward_validation": False}
    try:
        for item in inputs["candidates"]:
            spec = item["spec"]
            old = program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (spec["id"],)
            ).fetchone()
            if old:
                summary["candidates"][spec["id"]] = json.loads(old["body"])
                continue
            data = review_market({"market": spec["market"]}, ROOT)
            terms = load_terms(SOURCES, data.close.index)
            daily_targets = build_targets(data, terms, spec["configuration"], policy)
            paths = []
            for window in load_protocol(ROOT / "config/dual-horizon.json").windows():
                start, end = window["first_return_session"], window["last_session"]
                for scenario, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                    targets = seed_window(daily_targets, start)
                    own = simulate(
                        data,
                        targets,
                        start,
                        end,
                        initial_capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
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
                    record = {
                        "years": window["years"],
                        "scenario": scenario,
                        "capital_usd": 10000,
                        "cost_bps": cost,
                        "commission_per_order": 1,
                        "delay_sessions": delay,
                    }
                    for key, frame in (
                        ("strategy", own.frame),
                        ("strategy_bt", independent),
                        ("spy", spy.frame),
                        ("spy_bt", spy_bt),
                        ("weights", own.weights),
                        ("targets", targets),
                    ):
                        path = output / spec["id"] / f"{window['years']}y-{scenario}-{key}.csv"
                        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                        record[key] = {
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": file_digest(path),
                        }
                    paths.append(record)
            bundle = {
                "candidate_spec_sha256": item["registration"]["spec_sha256"],
                "as_of": policy["as_of"],
                "completed_at": timestamp().isoformat(),
                "market": spec["market"],
                "data_scope": "factor_etf_portfolio",
                "leveraged_products_allowed": False,
                "order_authority": False,
                "history_status": spec["history_status"],
                "paths": paths,
            }
            write_json(output / spec["id"] / "bundle.json", bundle)
            summary["candidates"][spec["id"]] = program.review(spec["id"], bundle)
            write_json(output / "progress.json", {"completed": list(summary["candidates"])})
        summary["program"] = program.status()
        write_json(output / "results.json", summary)
        return summary
    finally:
        program.close()


def main():
    parser = argparse.ArgumentParser(
        description="Frozen option-implied term-risk overlay; no option trading."
    )
    parser.add_argument("stage", choices=("prepare", "register", "evaluate"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/term-risk-prepared-20261010")
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "evidence/term_risk_20261010_registration.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/term-risk-20261010")
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            result = prepare(args.prepared)
            print(json.dumps({"prepared": len(result["specs"]), "returns_computed": False}))
        elif args.stage == "register":
            result = register(args.prepared, args.receipt)
            print(json.dumps({"registered": [row["spec"]["id"] for row in result["candidates"]]}))
        else:
            result = evaluate(args.receipt, args.output)
            print(
                json.dumps(
                    {
                        name: {"status": row["status"], "paths": row["paths"]}
                        for name, row in result["candidates"].items()
                    },
                    indent=2,
                )
            )
    except QuantError as exc:
        parser.exit(2, f"Volatility term-risk research blocked: {exc}\n")


if __name__ == "__main__":
    main()
