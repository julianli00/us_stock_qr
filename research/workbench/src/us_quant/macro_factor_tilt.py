from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.adaptive_factor_allocation import prepare as prepare_market
from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
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
POLICY = ROOT / "config/macro-factor-tilt.json"
SOURCE = ROOT / "data/macro-rate-access-20261010"


def validate(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("publication_delay_sessions") != 2
        or policy.get("maximum_observation_age_days") != 7
        or policy.get("real_yield_change_sessions") != 63
        or policy.get("rising_real_yield_factor_shares") != [0.15, 0.35, 0.15, 0.35]
        or policy.get("nonrising_real_yield_factor_shares") != [0.35, 0.15, 0.35, 0.15]
        or policy.get("inverted_curve_equity_scale") != 0.50
        or policy.get("candidates")
        != [
            {"id": "macro_real_yield_factor_tilt", "curve_defense": False},
            {"id": "macro_real_yield_tilt_curve_defense", "curve_defense": True},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
    ):
        raise QuantError("The fixed macro factor tilt or timing assumptions were changed.")


def align_observations(
    observed: pd.Series, index: pd.DatetimeIndex, delay: int = 2, maximum_age: int = 7
) -> tuple[pd.Series, pd.DataFrame]:
    if (
        not observed.index.is_unique
        or not observed.index.is_monotonic_increasing
        or delay != 2
        or maximum_age != 7
    ):
        raise QuantError("Macro observations must be ordered and retain the declared release lag.")
    known = observed.dropna()
    if known.empty or not np.isfinite(known).all():
        raise QuantError("No finite macro observations are available.")
    calendar = sessions(known.index[0], max(known.index[-1], index[-1]) + pd.Timedelta(days=10))
    positions = calendar.searchsorted(known.index, side="right") + delay - 1
    available = pd.DatetimeIndex(calendar[positions])
    selected = available.searchsorted(index, side="right") - 1
    if (selected < 0).any():
        raise QuantError("A macro value was not released before the requested decision.")
    observation_dates = known.index[selected]
    availability_dates = available[selected]
    age = (index - observation_dates).days
    if (age > maximum_age).any():
        raise QuantError("Macro data are stale; do not fill an unbounded release gap.")
    values = pd.Series(known.to_numpy()[selected], index=index, name=observed.name)
    provenance = pd.DataFrame(
        {
            "observation_date": observation_dates,
            "available_session": availability_dates,
            "age_calendar_days": age,
            "carried_after_availability": availability_dates < index,
        },
        index=index,
    )
    return values, provenance


def load_macro(directory: Path, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    source = read_json(directory / "verified-manifest.json")
    rows = {row["series"]: row for row in source["sources"]}
    if set(rows) != {"T10Y3M", "DFII10"}:
        raise QuantError("Both official macro rate series are required.")
    values, audits = {}, {}
    for name, row in rows.items():
        path = directory / row["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or file_digest(path) != row["sha256"]
            or not row["url"].startswith(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}&"
            )
        ):
            raise QuantError("The macro source fingerprint or official series identity changed.")
        frame = pd.read_csv(path)
        if list(frame.columns) != ["observation_date", name]:
            raise QuantError("Unexpected macro source columns.")
        observed = pd.Series(
            pd.to_numeric(frame[name], errors="raise").to_numpy(),
            index=pd.to_datetime(frame["observation_date"]),
            name=name,
        )
        values[name], audits[name] = align_observations(observed, index)
    return pd.DataFrame(values, index=index), audits


def build_targets(data, macro: pd.DataFrame, candidate: dict, policy: dict) -> pd.DataFrame:
    validate(policy)
    if (
        candidate not in policy["candidates"]
        or not macro.index.equals(data.close.index)
        or set(macro.columns) != {"T10Y3M", "DFII10"}
        or not np.isfinite(macro.to_numpy()).all()
    ):
        raise QuantError("Macro targets require complete lagged indicators and fixed candidates.")
    base = monthly_targets(data, read_json(ROOT / "config/factor-gold-risk.json"))
    result = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for i, day in enumerate(data.close.index):
        if i < 63 or base.loc[day].isna().all():
            continue
        rising = macro.loc[day, "DFII10"] > macro["DFII10"].iloc[i - 63]
        shares = policy[
            "rising_real_yield_factor_shares" if rising else "nonrising_real_yield_factor_shares"
        ]
        weights = base.loc[day].copy()
        equity = float(weights.loc[list(FACTORS)].sum())
        scale = 0.5 if candidate["curve_defense"] and macro.loc[day, "T10Y3M"] <= 0 else 1.0
        weights.loc[list(FACTORS)] = equity * scale * np.array(shares)
        weights["BIL"] += equity * (1 - scale)
        if (weights < -1e-12).any() or not np.isclose(weights.sum(), 0.98):
            raise QuantError("Macro weighting violated its cash-funded budget.")
        result.loc[day] = weights
    return result


def prepare(output: Path):
    policy = read_json(POLICY)
    validate(policy)
    original = prepare_market(output)
    data = review_market({"market": original["specs"][0]["market"]}, ROOT)
    _, audits = load_macro(SOURCE, data.close.index)
    frozen = {
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        "src/us_quant/factor_gold_risk.py": file_digest(ROOT / "src/us_quant/factor_gold_risk.py"),
        "config/factor-gold-risk.json": file_digest(ROOT / "config/factor-gold-risk.json"),
        (SOURCE / "verified-manifest.json").relative_to(ROOT).as_posix(): file_digest(
            SOURCE / "verified-manifest.json"
        ),
    }
    for row in read_json(SOURCE / "verified-manifest.json")["sources"]:
        path = SOURCE / row["path"]
        frozen[path.relative_to(ROOT).as_posix()] = file_digest(path)
    specs = [
        {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio",
            "evaluation_as_of": policy["as_of"],
            "market": original["specs"][0]["market"],
            "asset_leverage": original["specs"][0]["asset_leverage"],
            "frozen_files": frozen,
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        for candidate in policy["candidates"]
    ]
    audit = {}
    for name, frame in audits.items():
        path = output / f"{name}-availability.csv"
        write_text_atomic(path, frame.to_csv(index_label="decision_session"))
        audit[name] = {
            "maximum_age_days": int(frame["age_calendar_days"].max()),
            "carried_after_availability_sessions": int(frame["carried_after_availability"].sum()),
            "decisions": len(frame),
            "sha256": file_digest(path),
        }
    result = {
        "study": policy,
        "specs": specs,
        "readiness": original["readiness"],
        "availability_audit": audit,
        "returns_computed": False,
    }
    write_json(output / "macro-prepared.json", result)
    return result


def register(prepared: Path, receipt: Path):
    if receipt.exists():
        raise QuantError("An existing macro registration must remain unchanged.")
    inputs = read_json(prepared / "macro-prepared.json")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        records = []
        for spec in inputs["specs"]:
            old = program.db.execute(
                "SELECT spec_sha FROM candidates WHERE id=?", (spec["id"],)
            ).fetchone()
            if old:
                if old["spec_sha"] != digest_json(spec):
                    raise QuantError("An interrupted macro registration has different inputs.")
                events = [
                    json.loads(row["body"])
                    for row in program.db.execute("SELECT body FROM events WHERE kind='candidate'")
                ]
                registration = next(row for row in events if row["candidate_id"] == spec["id"])
            else:
                registration = program.register_candidate(spec, inputs["readiness"])
            records.append({"spec": spec, "registration": registration})
        result = {
            "study": inputs["study"],
            "candidates": records,
            "readiness": inputs["readiness"],
            "availability_audit": inputs["availability_audit"],
        }
        write_json(receipt, result)
        return result
    finally:
        program.close()


def evaluate(receipt: Path, output: Path):
    inputs, policy = read_json(receipt), read_json(POLICY)
    validate(policy)
    if inputs["study"] != policy:
        raise QuantError("The macro study changed after preregistration.")
    for entry in inputs["candidates"]:
        for name, digest in entry["spec"]["frozen_files"].items():
            safe_file(ROOT, name, digest)
    new_output_directory(output)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    result = {"candidates": {}, "order_authority": False, "independent_forward_validation": False}
    try:
        for entry in inputs["candidates"]:
            spec = entry["spec"]
            previous = program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (spec["id"],)
            ).fetchone()
            if previous:
                result["candidates"][spec["id"]] = json.loads(previous["body"])
                continue
            data = review_market({"market": spec["market"]}, ROOT)
            macro, _ = load_macro(SOURCE, data.close.index)
            signals = build_targets(data, macro, spec["configuration"], policy)
            paths = []
            for window in load_protocol(ROOT / "config/dual-horizon.json").windows():
                start, end = window["first_return_session"], window["last_session"]
                for scenario, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                    targets = seed_window(signals, start)
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
                    benchmark_targets = buy_and_hold_signals(data.close, "SPY", start)
                    spy = simulate(
                        data,
                        benchmark_targets,
                        start,
                        end,
                        initial_capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    spy_bt = independent_equity(
                        data,
                        benchmark_targets,
                        start,
                        end,
                        capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    row = {
                        "years": window["years"],
                        "scenario": scenario,
                        "capital_usd": 10000,
                        "cost_bps": cost,
                        "commission_per_order": 1,
                        "delay_sessions": delay,
                    }
                    for name, frame in (
                        ("strategy", own.frame),
                        ("strategy_bt", independent),
                        ("spy", spy.frame),
                        ("spy_bt", spy_bt),
                        ("weights", own.weights),
                        ("targets", targets),
                    ):
                        path = output / spec["id"] / f"{window['years']}y-{scenario}-{name}.csv"
                        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                        row[name] = {
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": file_digest(path),
                        }
                    paths.append(row)
            bundle = {
                "candidate_spec_sha256": entry["registration"]["spec_sha256"],
                "market": spec["market"],
                "as_of": policy["as_of"],
                "completed_at": timestamp().isoformat(),
                "data_scope": "factor_etf_portfolio",
                "leveraged_products_allowed": False,
                "order_authority": False,
                "history_status": spec["history_status"],
                "paths": paths,
            }
            write_json(output / spec["id"] / "bundle.json", bundle)
            result["candidates"][spec["id"]] = program.review(spec["id"], bundle)
            write_json(output / "progress.json", {"completed": list(result["candidates"])})
        result["program"] = program.status()
        write_json(output / "results.json", result)
        return result
    finally:
        program.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fixed public macro factor tilt with explicit release lags."
    )
    parser.add_argument("stage", choices=("prepare", "register", "evaluate"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/macro-tilt-prepared-20261010")
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "evidence/macro_tilt_20261010_registration.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/macro-tilt-20261010")
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            result = prepare(args.prepared)
            print(
                json.dumps(
                    {
                        "prepared": len(result["specs"]),
                        "audit": result["availability_audit"],
                        "returns_computed": False,
                    }
                )
            )
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
        parser.exit(2, f"Macro factor research blocked: {exc}\n")


if __name__ == "__main__":
    main()
