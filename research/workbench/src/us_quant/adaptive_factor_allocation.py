from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol, seed_window
from us_quant.multifactor_stability import FACTORS, load_market
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
POLICY = ROOT / "config/adaptive-factor-allocation.json"
ASSETS = (*FACTORS, "GLD", "BIL")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "data_scope": "factor_etf_portfolio",
        "as_of": "2026-10-05",
        "capital_usd": 10000.0,
        "cash_reserve": 0.02,
        "lookback_sessions": 252,
        "mean_half_life_sessions": 63,
        "mean_shrinkage": 0.50,
        "covariance_diagonal_shrinkage": 0.10,
        "risk_aversion": 3.0,
        "factor_min_total_weight": 0.025,
        "factor_max_total_weight": 0.25,
        "gold_max_total_weight": 0.50,
        "objective_turnover_penalty": 0.006,
    }
    if (
        any(policy.get(key) != value for key, value in constants.items())
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("universe") != list(ASSETS)
        or policy.get("candidates")
        != [
            {"id": "four_factor_static70_gold30", "method": "static"},
            {"id": "four_factor_causal_mean_variance", "method": "adaptive"},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Adaptive allocation must preserve the fixed preregistered protocol.")


def target(
    history: pd.DataFrame,
    rates: pd.Series,
    previous: pd.Series,
    candidate: dict,
    policy: dict,
) -> pd.Series:
    validate_policy(policy)
    if (
        len(history) < 253
        or not history.index.equals(rates.index)
        or not np.isfinite(history.to_numpy()).all()
        or (history <= 0).any().any()
        or not previous.index.equals(history.columns)
        or not np.isfinite(previous).all()
        or not set(ASSETS) <= set(history.columns)
        or not history.index.is_monotonic_increasing
        or not history.index.is_unique
    ):
        raise QuantError(
            "Adaptive allocation requires complete prior observations and aligned state."
        )
    result = pd.Series(0.0, index=history.columns)
    if candidate["method"] == "static":
        result.loc[list(FACTORS)] = 0.98 * 0.70 / 4
        result["GLD"] = 0.98 * 0.30
        return result
    if candidate["method"] != "adaptive":
        raise QuantError("Unregistered adaptive allocation method.")
    returns = history.loc[:, list(ASSETS)].pct_change(fill_method=None).iloc[1:].tail(252)
    risk_free = rates.loc[returns.index]
    excess = returns.sub(risk_free, axis=0)
    decay = 0.5 ** (np.arange(len(excess) - 1, -1, -1) / policy["mean_half_life_sessions"])
    decay /= decay.sum()
    expected = excess.to_numpy().T @ decay * 252
    common = expected[:-1].mean()
    expected[:-1] = 0.5 * expected[:-1] + 0.5 * common
    covariance = returns.cov().to_numpy() * 252
    covariance = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance))
    if not np.isfinite(expected).all() or not np.isfinite(covariance).all():
        raise QuantError("Adaptive moments are not finite.")
    before = previous.loc[list(ASSETS)].to_numpy()
    bounds = [(0.025, 0.25)] * 4 + [(0, 0.5), (0, 0.88)]
    initial = np.array([0.98 * 0.70 / 4] * 4 + [0.98 * 0.30, 0.0])

    def objective(values):
        weights, changes = values[:6], values[6:]
        return (
            0.5 * policy["risk_aversion"] * float(weights @ covariance @ weights)
            - float(expected @ weights)
            + policy["objective_turnover_penalty"] * changes.sum()
        )

    def gradient(values):
        return np.r_[
            policy["risk_aversion"] * covariance @ values[:6] - expected,
            np.full(6, policy["objective_turnover_penalty"]),
        ]

    change_jacobian = np.vstack(
        (
            np.column_stack((-np.eye(6), np.eye(6))),
            np.column_stack((np.eye(6), np.eye(6))),
        )
    )
    fit = minimize(
        objective,
        np.r_[initial, abs(initial - before)],
        jac=gradient,
        method="SLSQP",
        bounds=[*bounds, *[(0.0, 1.0)] * 6],
        constraints=[
            {
                "type": "eq",
                "fun": lambda values: values[:6].sum() - 0.98,
                "jac": lambda values: np.r_[np.ones(6), np.zeros(6)],
            },
            {
                "type": "ineq",
                "fun": lambda values: np.r_[
                    values[6:] - (values[:6] - before),
                    values[6:] + (values[:6] - before),
                ],
                "jac": lambda values: change_jacobian,
            },
        ],
        options={"ftol": 1e-10, "maxiter": 500},
    )
    weights = fit.x[:6]
    if (
        not fit.success
        or not np.isfinite(fit.x).all()
        or abs(weights.sum() - 0.98) > 1e-8
        or any(
            value < low - 1e-10 or value > high + 1e-10
            for value, (low, high) in zip(weights, bounds, strict=True)
        )
    ):
        raise QuantError("Constrained adaptive optimization failed; no silent fallback weights.")
    result.loc[list(ASSETS)] = weights
    return result


def build_targets(data, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    output = {}
    for candidate in policy["candidates"]:
        previous = pd.Series(0.0, index=data.close.columns)
        previous["BIL"] = 0.98
        signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for index, day in enumerate(data.close.index):
            if index < 252 or not is_month_end(day):
                continue
            prior = data.close.iloc[: index + 1]
            weights = target(prior, data.risk_free.loc[prior.index], previous, candidate, policy)
            signals.loc[day] = weights
            previous = weights
        if signals.dropna(how="all").empty:
            raise QuantError("No complete post-warmup monthly allocations exist.")
        output[candidate["id"]] = signals
    return output


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    p = ROOT / "config/multifactor-stability.json"
    r = ROOT / "evidence/multifactor_stability_20261010_registration_v2.json"
    base = ROOT / "data/factor-round-20261007/base"
    funds = ROOT / "data/multifactor-stability-20261010"
    data = load_market(read_json(p), read_json(r), base, funds)
    new_output_directory(output)
    market = {}
    for key, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"{key}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        market[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    readiness = {
        "schema_version": 1,
        "checked_at": timestamp().isoformat(),
        "data_scope": "factor_etf_portfolio",
        "verified_etf_source": {
            "adapter": "frozen_multifactor_etf_20261010",
            "policy": p.relative_to(ROOT).as_posix(),
            "policy_sha256": file_digest(p),
            "registration": r.relative_to(ROOT).as_posix(),
            "registration_sha256": file_digest(r),
            "base_manifest": (base / "manifest.json").relative_to(ROOT).as_posix(),
            "base_manifest_sha256": file_digest(base / "manifest.json"),
            "factor_manifest": (funds / "manifest.json").relative_to(ROOT).as_posix(),
            "factor_manifest_sha256": file_digest(funds / "manifest.json"),
        },
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": p.relative_to(ROOT).as_posix(),
                "evidence_sha256": file_digest(p),
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "stock_data_still_blocked": True,
        "new_market_observations": False,
    }
    specs = [
        {
            "id": candidate["id"],
            "configuration": candidate,
            "data_scope": "factor_etf_portfolio",
            "factor_ids": policy["factor_ids"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": {
                POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
                Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
            },
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        for candidate in policy["candidates"]
    ]
    result = {"policy": policy, "readiness": readiness, "specs": specs, "returns_computed": False}
    write_json(output / "prepared.json", result)
    return result


def register(prepared: Path, receipt: Path):
    if receipt.exists():
        raise QuantError("Do not replace an existing adaptive registration.")
    inputs = read_json(prepared / "prepared.json")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        entries = []
        for spec in inputs["specs"]:
            existing = program.db.execute(
                "SELECT spec_sha FROM candidates WHERE id=?", (spec["id"],)
            ).fetchone()
            if existing:
                if existing["spec_sha"] != digest_json(spec):
                    raise QuantError("Interrupted adaptive registration has changed inputs.")
                records = [
                    json.loads(row["body"])
                    for row in program.db.execute("SELECT body FROM events WHERE kind='candidate'")
                ]
                record = next(row for row in records if row["candidate_id"] == spec["id"])
            else:
                record = program.register_candidate(spec, inputs["readiness"])
            entries.append({"spec": spec, "registration": record})
        result = {
            "study": inputs["policy"],
            "readiness": inputs["readiness"],
            "candidates": entries,
        }
        write_json(receipt, result)
        return result
    finally:
        program.close()


def evaluate(receipt: Path, output: Path):
    registered = read_json(receipt)
    policy = read_json(POLICY)
    validate_policy(policy)
    if registered["study"] != policy:
        raise QuantError("Adaptive rules changed after registration.")
    for item in registered["candidates"]:
        for relative, digest in item["spec"]["frozen_files"].items():
            safe_file(ROOT, relative, digest)
    new_output_directory(output)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    result = {"candidates": {}, "order_authority": False, "independent_forward_validation": False}
    try:
        data = review_market({"market": registered["candidates"][0]["spec"]["market"]}, ROOT)
        all_targets = build_targets(data, policy)
        for item in registered["candidates"]:
            spec = item["spec"]
            previous = program.db.execute(
                "SELECT body FROM reviews WHERE candidate_id=?", (spec["id"],)
            ).fetchone()
            if previous:
                result["candidates"][spec["id"]] = json.loads(previous["body"])
                continue
            entries = []
            for window in load_protocol(ROOT / "config/dual-horizon.json").windows():
                start, end = window["first_return_session"], window["last_session"]
                for label, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                    targets = seed_window(all_targets[spec["id"]], start)
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
                    spy_target = buy_and_hold_signals(data.close, "SPY", start)
                    spy = simulate(
                        data,
                        spy_target,
                        start,
                        end,
                        initial_capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    spy_bt = independent_equity(
                        data,
                        spy_target,
                        start,
                        end,
                        capital=10000,
                        cost_bps=cost,
                        commission=1,
                        delay=delay,
                    )
                    record = {
                        "years": window["years"],
                        "scenario": label,
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
                        path = output / spec["id"] / f"{window['years']}y-{label}-{key}.csv"
                        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                        record[key] = {
                            "path": path.relative_to(ROOT).as_posix(),
                            "sha256": file_digest(path),
                        }
                    entries.append(record)
            bundle = {
                "candidate_spec_sha256": item["registration"]["spec_sha256"],
                "market": spec["market"],
                "as_of": policy["as_of"],
                "completed_at": timestamp().isoformat(),
                "data_scope": "factor_etf_portfolio",
                "leveraged_products_allowed": False,
                "order_authority": False,
                "history_status": spec["history_status"],
                "paths": entries,
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
        description="Frozen causal adaptive factor allocation comparison."
    )
    parser.add_argument("stage", choices=("prepare", "register", "evaluate"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/adaptive-factors-20261010")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/adaptive_factors_20261010_registration.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/adaptive-factors-20261010")
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
        parser.exit(2, f"Adaptive factor research blocked: {exc}\n")


if __name__ == "__main__":
    main()
