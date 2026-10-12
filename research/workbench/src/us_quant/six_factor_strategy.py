from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_family_sources import (
    POLICY as SOURCE_POLICY,
    SOURCE,
    prepare as prepare_source,
)
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS, bounded_factor_shares
from us_quant.research_program import ResearchProgram, review_market, safe_file, timestamp
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/six-factor-strategy.json"
BASELINE = ROOT / "config/factor-gold-risk.json"
FAMILIES = (*FACTORS, "IJR", "PKW")
FACTOR_IDS = (
    "price_momentum",
    "value_exposure",
    "quality_exposure",
    "low_volatility_exposure",
    "size_exposure_ijr_v1",
    "net_buyback_exposure_pkw_v1",
)


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("eligible_not_before") != "2026-10-17T09:00:00+08:00"
        or policy.get("planned_configurations") != 2
        or policy.get("factor_symbols") != list(FAMILIES)
        or policy.get("factor_ids") != list(FACTOR_IDS)
        or policy.get("baseline_policy") != BASELINE.relative_to(ROOT).as_posix()
        or policy.get("source_policy") != SOURCE_POLICY.relative_to(ROOT).as_posix()
        or policy.get("volatility_sessions") != 63
        or policy.get("minimum_share_of_equity") != 1 / 12
        or policy.get("maximum_share_of_equity") != 1 / 4
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "six_factor_equal_family", "factor_weighting": "equal"},
            {"id": "six_factor_bounded_inverse_volatility", "factor_weighting": "inverse_volatility"},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The two fixed six-family portfolios or shared risk budgets changed.")


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "GLD", "BIL", *FAMILIES}:
        raise QuantError("The six-family portfolio needs the complete audited ETF universe.")
    original = MarketData(
        data.open.drop(columns=["IJR", "PKW"]),
        data.close.drop(columns=["IJR", "PKW"]),
        data.raw_close.drop(columns=["IJR", "PKW"]),
        data.volume.drop(columns=["IJR", "PKW"]),
        data.risk_free,
    )
    baseline = monthly_targets(original, read_json(BASELINE))
    daily = data.close.pct_change(fill_method=None)
    output = {
        row["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for row in policy["candidates"]
    }
    for i, day in enumerate(data.close.index):
        if baseline.loc[day].isna().all():
            continue
        recent = daily.loc[:, list(FAMILIES)].iloc[i - 62 : i + 1]
        if len(recent) != 63 or not np.isfinite(recent.to_numpy()).all():
            raise QuantError("A full trailing volatility window is required for all six families.")
        inverse = bounded_factor_shares(
            recent.std(ddof=1),
            policy["minimum_share_of_equity"],
            policy["maximum_share_of_equity"],
        )
        equity = float(baseline.loc[day, list(FACTORS)].sum())
        for candidate in policy["candidates"]:
            shares = (
                pd.Series(1 / 6, index=FAMILIES)
                if candidate["factor_weighting"] == "equal"
                else inverse
            )
            target = baseline.loc[day].reindex(data.close.columns, fill_value=0.0)
            target.loc[list(FAMILIES)] = equity * shares
            if (
                not np.isfinite(target).all()
                or (target < 0).any()
                or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)
            ):
                raise QuantError("Six-family targets must remain long-only and cash funded.")
            output[candidate["id"]].loc[day] = target
    return output


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.exists() or output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Use a new, in-workbench directory for planned candidate specifications.")
    prepared = prepare_source(output / "sources")
    market = prepared["market"]
    data = review_market({"market": market}, ROOT)
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            BASELINE,
            SOURCE_POLICY,
            Path(__file__).with_name("factor_gold_risk.py"),
            Path(__file__).with_name("factor_family_sources.py"),
            Path(__file__).with_name("factor_replication.py"),
            Path(__file__).with_name("multifactor_stability.py"),
            SOURCE / "manifest.json",
        )
    }
    manifest = read_json(SOURCE / "manifest.json")
    for record in manifest["sources"].values():
        for name, digest in record["files"].items():
            frozen[(SOURCE / name).relative_to(ROOT).as_posix()] = digest
    for name, digest in manifest["additional_source_files"].items():
        frozen[(SOURCE / name).relative_to(ROOT).as_posix()] = digest
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": list(FACTOR_IDS),
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", prepared["readiness"])
    return {
        "specs": specs,
        "readiness": prepared["readiness"],
        "eligible_not_before": policy["eligible_not_before"],
        "candidate_registration_performed": False,
        "strategy_outcomes_computed": False,
    }


def register(prepared: Path, receipt: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    if pd.Timestamp(timestamp()) < pd.Timestamp(policy["eligible_not_before"]):
        raise QuantError(
            f"Six-family registration is not eligible before {policy['eligible_not_before']}."
        )
    prepared = prepared if prepared.is_absolute() else ROOT / prepared
    receipt = receipt if receipt.is_absolute() else ROOT / receipt
    if (
        prepared.is_symlink()
        or not prepared.resolve().is_relative_to(ROOT.resolve())
        or receipt.is_symlink()
        or not receipt.resolve().is_relative_to(ROOT.resolve())
    ):
        raise QuantError("Six-family registration files must remain inside the isolated workbench.")
    if receipt.exists():
        raise QuantError("Do not overwrite a completed six-family candidate registration.")
    readiness = read_json(safe_file(ROOT, (prepared / "readiness.json").relative_to(ROOT).as_posix()))
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        records = []
        for candidate in policy["candidates"]:
            spec = read_json(
                safe_file(
                    ROOT,
                    (prepared / f"{candidate['id']}-spec.json").relative_to(ROOT).as_posix(),
                )
            )
            previous = program.db.execute(
                "SELECT spec_sha FROM candidates WHERE id=?", (candidate["id"],)
            ).fetchone()
            if previous is None:
                registration = program.register_candidate(spec, readiness)
            else:
                program.verify()
                if previous["spec_sha"] != digest_json(spec):
                    raise QuantError("The interrupted candidate has a different frozen specification.")
                events = [
                    json.loads(row["body"])
                    for row in program.db.execute("SELECT body FROM events WHERE kind='candidate'")
                ]
                registration = next(row for row in events if row["candidate_id"] == candidate["id"])
            records.append({"spec": spec, "registration": registration})
        result = {
            "study": policy,
            "candidates": records,
            "readiness": readiness,
            "registered_before_strategy_outcomes": True,
            "order_authority": False,
        }
        write_json(receipt, result)
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or register two frozen six-family portfolios.")
    parser.add_argument("action", choices=("prepare", "register"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/six-factor-prepared-20261011")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare(args.prepared)
            print(
                json.dumps(
                    {
                        "prepared": [spec["id"] for spec in result["specs"]],
                        "eligible_not_before": result["eligible_not_before"],
                        "candidate_registration_performed": False,
                        "strategy_outcomes_computed": False,
                    }
                )
            )
        else:
            if args.receipt is None:
                raise QuantError("Candidate registration requires a new explicit receipt path.")
            result = register(args.prepared, args.receipt)
            print(json.dumps({"registered": [row["spec"]["id"] for row in result["candidates"]]}))
    except QuantError as exc:
        parser.exit(2, f"Six-family preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
