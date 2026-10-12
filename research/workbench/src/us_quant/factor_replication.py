from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from us_quant.backtest import BacktestResult, rebalance, simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.dual_horizon import load_protocol, seed_window
from us_quant.multifactor_stability import cap_volatility, corporate_actions, load_market
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
POLICY = ROOT / "config/factor-implementation-replication.json"
FACTORS = ("MTUM", "VLUE", "SPHQ", "USMV")
COLUMNS = ("SPY", "IEF", "GLD", "BIL", *FACTORS)


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_start") != "2016-07-01"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("risk_lookback") != 63
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("equity_sleeve_min") != 0.30
        or policy.get("equity_sleeve_max") != 0.70
        or policy.get("capital_usd") != 10000
        or policy.get("cash_reserve") != 0.02
        or policy.get("commission_per_order") != 1.0
        or policy.get("candidates")
        != [
            {
                "id": "quality_implementation_gold_monthly",
                "daily_volatility_target": None,
                "daily_weight_band": None,
            },
            {
                "id": "quality_implementation_gold_daily_vol12",
                "daily_volatility_target": 0.12,
                "daily_weight_band": 0.05,
            },
        ]
        or [row.get("symbol") for row in policy.get("replacement_funds", [])] != ["SPHQ"]
        or any(row.get("daily_leverage") != 1 for row in policy["replacement_funds"])
        or policy.get("methodology", {}).get("order_authority") is not False
    ):
        raise QuantError("The fixed factor-implementation replication was altered.")
    revision = policy.get("data_quality_revision", {})
    audit_path = ROOT / revision["quality_audit"]
    if file_digest(audit_path) != revision["quality_audit_sha256"] or any(
        revision.get(key) is not False
        for key in (
            "strategy_outcomes_seen_before_revision",
            "formal_windows_changed",
            "allocation_rules_changed",
        )
    ):
        raise QuantError("The data-only revision cannot conceal performance selection.")
    records = {row["symbol"]: row for row in read_json(audit_path)["records"]}
    if records["SPMO"]["bad_rows"] != 238 or records["SPHQ"]["bad_rows"] != 0:
        raise QuantError("The quality-only replacement must follow the preserved source audit.")


def original_market() -> MarketData:
    return load_market(
        read_json(ROOT / "config/multifactor-stability.json"),
        read_json(ROOT / "evidence/multifactor_stability_20261010_registration_v2.json"),
        ROOT / "data/factor-round-20261007/base",
        ROOT / "data/multifactor-stability-20261010",
    )


def fetch_prices(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    if output.exists():
        raise QuantError("Refusing to replace factor replication source evidence.")
    collected = {}
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-quant-research/0.1 (personal research)"
        for fund in policy["replacement_funds"]:
            symbol = fund["symbol"]
            try:
                issuer = client.get(fund["issuer_url"], timeout=(10, 45))
                issuer.raise_for_status()
                if fund["mandate_keyword"] not in issuer.text or symbol not in issuer.text:
                    raise QuantError("The issuer page does not identify the expected factor fund.")
                response = client.get(
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={
                        "period1": int(pd.Timestamp(policy["data_start"], tz="UTC").timestamp()),
                        "period2": int(
                            (
                                pd.Timestamp(policy["as_of"], tz="UTC") + pd.Timedelta(days=1)
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
                raise QuantError(
                    f"Fund identity or history is unavailable for {symbol}: {exc}"
                ) from exc
            frame = parse_chart(payload, symbol, policy["data_start"], policy["as_of"])
            actions = corporate_actions(payload, frame, symbol)
            collected[symbol] = issuer, response, frame, actions
    new_output_directory(output)
    files, sources = {}, {}
    for symbol, (issuer, response, frame, actions) in collected.items():
        for relative, content in (
            (f"issuer/{symbol}.html", issuer.text),
            (f"raw/{symbol}.json", response.text),
            (f"{symbol}.csv", frame.to_csv(float_format="%.17g")),
        ):
            write_text_atomic(output / relative, content)
            files[relative] = file_digest(output / relative)
        sources[symbol] = {
            "issuer_url": issuer.url,
            "price_url": response.url,
            "rows": len(frame),
            "first_session": str(frame.index[0].date()),
            "last_session": str(frame.index[-1].date()),
            **actions,
        }
    manifest = {
        "schema_version": 1,
        "retrieved_at": utc_now(),
        "policy_sha256": file_digest(POLICY),
        "data_start": policy["data_start"],
        "data_end": policy["as_of"],
        "sources": sources,
        "files": files,
        "synthetic_history": False,
        "full_historical_methodology_verified": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_replication_market(policy: dict, directory: Path) -> MarketData:
    validate_policy(policy)
    manifest = read_json(directory / "manifest.json")
    required = {
        f"{p}{symbol}{s}"
        for symbol in ("SPHQ",)
        for p, s in (("", ".csv"), ("raw/", ".json"), ("issuer/", ".html"))
    }
    if (
        manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or set(manifest.get("files", {})) != required
        or manifest.get("synthetic_history") is not False
    ):
        raise QuantError("Factor replication source manifest does not match its protocol.")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or file_digest(path) != expected
        ):
            raise QuantError("Factor replication source was changed, missing or external.")
    old = original_market()
    index = sessions(policy["data_start"], policy["as_of"])
    new = {
        symbol: pd.read_csv(directory / f"{symbol}.csv", index_col="date", parse_dates=True)
        for symbol in ("SPHQ",)
    }

    def panel(old_frame: pd.DataFrame, field: str) -> pd.DataFrame:
        values = {}
        for symbol in COLUMNS:
            series = new[symbol][field] if symbol in new else old_frame.loc[index, symbol]
            if not series.index.equals(index):
                raise QuantError("Every replication instrument must cover all registered sessions.")
            values[symbol] = series
        return pd.DataFrame(values, index=index)

    result = MarketData(
        panel(old.open, "adj_open"),
        panel(old.close, "adj_close"),
        panel(old.raw_close, "close"),
        panel(old.volume, "volume"),
        old.risk_free.loc[index],
    )
    result.validate()
    return result


def monthly_targets(data: MarketData, policy: dict) -> pd.DataFrame:
    validate_policy(policy)
    data.validate()
    if tuple(data.close.columns) != COLUMNS:
        raise QuantError("Factor-slot replacements must retain their actual instrument symbols.")
    daily = data.close.pct_change(fill_method=None)
    result = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for i, day in enumerate(data.close.index):
        if i < 63 or not is_month_end(day):
            continue
        past = daily.iloc[i - 62 : i + 1]
        equity_vol = past.loc[:, list(FACTORS)].mean(axis=1).std(ddof=1)
        gold_vol = past["GLD"].std(ddof=1)
        if not np.isfinite([equity_vol, gold_vol]).all() or min(equity_vol, gold_vol) <= 1e-10:
            raise QuantError("Replication risk estimates require finite observed returns.")
        equity = np.clip(gold_vol / (equity_vol + gold_vol), 0.30, 0.70)
        result.loc[day] = 0.0
        result.loc[day, list(FACTORS)] = 0.98 * equity / 4
        result.loc[day, "GLD"] = 0.98 * (1 - equity)
    return result


def run(
    data: MarketData, candidate: dict, policy: dict, start: str, end: str, cost: float, delay: int
) -> tuple[BacktestResult, pd.DataFrame]:
    if candidate not in policy["candidates"] or cost not in (5, 20) or delay not in (1, 2):
        raise QuantError("Replication candidate or cost scenario was not registered.")
    baseline = seed_window(monthly_targets(data, policy), start)
    dates = data.close.loc[start:end].index
    anchor = previous_session(dates[0])
    monthly = baseline.loc[anchor].copy()
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    pending = {}
    cash = previous_nav = 10000.0
    units = np.zeros(len(data.close.columns))
    frames, weights = [], []
    requested = False

    def target(day):
        if candidate["daily_volatility_target"] is None:
            return monthly.copy()
        history = data.close.loc[:day].tail(64).pct_change(fill_method=None).iloc[1:]
        if len(history) != 63:
            raise QuantError("Incomplete trailing risk window.")
        return cap_volatility(monthly, history.cov() * 252, 0.12, 0.98)

    def schedule(day, allocation):
        if pending:
            raise QuantError("A pending target cannot be overwritten.")
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        issued.loc[day] = allocation
        pending[execution] = allocation.to_numpy()

    schedule(anchor, target(anchor))
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        charge = turnover = 0.0
        tickets = 0
        if day in pending:
            dollars, cash, charge, turnover, tickets = rebalance(
                units * opening, cash, pending.pop(day), cost, 1.0
            )
            units = dollars / opening
        values = units * closing
        equity = float(values.sum() + cash)
        if not np.isfinite(equity) or equity <= 0 or cash < -1e-7:
            raise QuantError("Replication accounting violates cash funding.")
        current = pd.Series(values / equity, index=data.close.columns)
        frames.append(
            (equity, equity / previous_nav - 1, cash, current.sum(), turnover, charge, tickets)
        )
        weights.append(current.to_numpy())
        if not baseline.loc[day].isna().all():
            monthly = baseline.loc[day].copy()
            requested = True
        if not pending and (requested or candidate["daily_volatility_target"] is not None):
            desired = target(day)
            if requested or float(abs(current - desired).max()) >= 0.05:
                schedule(day, desired)
                requested = False
        previous_nav = equity
    frame = pd.DataFrame(
        frames,
        index=dates,
        columns=[
            "equity",
            "return",
            "cash",
            "gross_exposure",
            "turnover",
            "cost",
            "orders",
        ],
    )
    frame["risk_free"] = data.risk_free.loc[dates]
    result = BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=data.close.columns))
    replay = simulate(
        data, issued, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    if not np.allclose(result.frame, replay.frame, atol=1e-8, rtol=0):
        raise QuantError("Replication targets do not reproduce the actual accounting path.")
    return result, issued


def prepare(source: Path, output: Path) -> dict:
    policy = read_json(POLICY)
    data = load_replication_market(policy, source)
    new_output_directory(output)
    market = {}
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        market[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    readiness = {
        "schema_version": 1,
        "data_scope": "factor_etf_portfolio",
        "checked_at": utc_now(),
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": POLICY.relative_to(ROOT).as_posix(),
                "evidence_sha256": file_digest(POLICY),
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "verified_etf_source": {
            "adapter": "factor_implementation_replication_20261010",
            "policy": POLICY.relative_to(ROOT).as_posix(),
            "policy_sha256": file_digest(POLICY),
            "factor_manifest": (source / "manifest.json").relative_to(ROOT).as_posix(),
            "factor_manifest_sha256": file_digest(source / "manifest.json"),
        },
        "stock_fundamental_data_still_blocked": True,
        "detailed_factsheet_bytes_verified": False,
    }
    specs = []
    for candidate in policy["candidates"]:
        specs.append(
            {
                "id": candidate["id"],
                "configuration": candidate,
                "factor_ids": policy["factor_ids"],
                "data_scope": "factor_etf_portfolio",
                "frozen_files": {
                    POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
                    Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
                },
                "market": market,
                "evaluation_as_of": policy["as_of"],
                "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
                "history_status": "exposed_history_not_independent_holdout",
                "leveraged_products_allowed": False,
                "order_authority": False,
            }
        )
    result = {"study": policy, "readiness": readiness, "specs": specs, "returns_computed": False}
    write_json(output / "prepared.json", result)
    return result


def register(prepared: Path, receipt: Path) -> dict:
    from us_quant.research_program import ResearchProgram

    if receipt.exists():
        raise QuantError("A replication registration receipt already exists.")
    inputs = read_json(prepared / "prepared.json")
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
                    raise QuantError("An interrupted replication has a different specification.")
                prior = [
                    json.loads(row["body"])
                    for row in program.db.execute("SELECT body FROM events WHERE kind='candidate'")
                ]
                registration = next(row for row in prior if row["candidate_id"] == spec["id"])
            else:
                registration = program.register_candidate(spec, inputs["readiness"])
            records.append({"spec": spec, "registration": registration})
        result = {"study": inputs["study"], "readiness": inputs["readiness"], "candidates": records}
        write_json(receipt, result)
        return result
    finally:
        program.close()


def evaluate(receipt: Path, output: Path) -> dict:
    from us_quant.research_program import ResearchProgram, review_market, safe_file

    inputs = read_json(receipt)
    policy = read_json(POLICY)
    if inputs["study"] != policy or len(inputs["candidates"]) != 2:
        raise QuantError("The replication study changed after registration.")
    for entry in inputs["candidates"]:
        for path, sha in entry["spec"]["frozen_files"].items():
            safe_file(ROOT, path, sha)
    new_output_directory(output)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    result = {"candidates": {}, "order_authority": False, "independent_forward_validation": False}
    try:
        for entry in inputs["candidates"]:
            spec = entry["spec"]
            old = program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (spec["id"],)
            ).fetchone()
            if old:
                result["candidates"][spec["id"]] = json.loads(old["body"])
                continue
            data = review_market({"market": spec["market"]}, ROOT)
            paths = []
            for window in load_protocol(ROOT / "config/dual-horizon.json").windows():
                start, end = window["first_return_session"], window["last_session"]
                for scenario, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                    own, targets = run(data, spec["configuration"], policy, start, end, cost, delay)
                    bt = independent_equity(
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
                    for key, frame in (
                        ("strategy", own.frame),
                        ("strategy_bt", bt),
                        ("spy", spy.frame),
                        ("spy_bt", spy_bt),
                        ("weights", own.weights),
                        ("targets", targets),
                    ):
                        path = output / spec["id"] / f"{window['years']}y-{scenario}-{key}.csv"
                        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                        row[key] = {
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": file_digest(path),
                        }
                    paths.append(row)
            bundle = {
                "candidate_spec_sha256": entry["registration"]["spec_sha256"],
                "as_of": policy["as_of"],
                "completed_at": utc_now(),
                "market": spec["market"],
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
        description="Fixed live-fund factor implementation replication."
    )
    parser.add_argument("stage", choices=("fetch", "prepare", "register", "evaluate"))
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/factor-replication-source-20261010"
    )
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "data/factor-replication-prepared-20261010"
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/factor_replication_20261010_registration.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/factor-replication-20261010")
    args = parser.parse_args()
    try:
        if args.stage == "fetch":
            result = fetch_prices(args.source)
            print(json.dumps(result["sources"], indent=2))
        elif args.stage == "prepare":
            result = prepare(args.source, args.prepared)
            print(json.dumps({"prepared": len(result["specs"]), "returns_computed": False}))
        elif args.stage == "register":
            result = register(args.prepared, args.receipt)
            print(json.dumps({"registered": [x["spec"]["id"] for x in result["candidates"]]}))
        else:
            result = evaluate(args.receipt, args.output)
            print(
                json.dumps(
                    {
                        key: {"status": value["status"], "paths": value["paths"]}
                        for key, value in result["candidates"].items()
                    },
                    indent=2,
                )
            )
    except QuantError as exc:
        parser.exit(2, f"Factor implementation replication blocked: {exc}\n")


if __name__ == "__main__":
    main()
