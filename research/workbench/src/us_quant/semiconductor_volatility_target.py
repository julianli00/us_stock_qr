from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.joint_risk_confirmation import build_targets as joint_targets
from us_quant.multifactor_stability import FACTORS, cap_volatility
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/semiconductor-volatility-target.json"
CANDIDATES = (
    {"id": "semiconductor_fixed_vol12", "source_baseline": "plain_sector"},
    {"id": "semiconductor_monthly_confirmation_fixed_vol12", "source_baseline": "monthly_joint_confirmation"},
)
REFERENCES = (
    "source_registration", "joint_source_policy", "original_volatility_policy",
    "original_volatility_implementation", "original_projection_helper",
)


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "semiconductor_fixed_volatility_transfer_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 128, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "volatility_sessions": 63, "target_volatility": 0.12,
        "target_change_band": 0.05, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False):
        raise QuantError("Keep the original volatility projection, hysteresis and two fixed monthly sources.")


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    plain = sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    confirmed = joint_targets(data, read_json(ROOT / policy["joint_source_policy"]["path"]))[
        "semiconductor_dollar_term_monthly_confirmation"
    ]
    if (not plain.index.equals(confirmed.index) or not plain.columns.equals(confirmed.columns)
        or not plain.isna().equals(confirmed.isna())):
        raise QuantError("Both volatility sources must preserve their original monthly target calendars.")
    return {CANDIDATES[0]["id"]: plain, CANDIDATES[1]["id"]: confirmed}


def build_from_baselines(
    data: MarketData, baselines: dict[str, pd.DataFrame], policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if ("BIL" not in data.close or not isinstance(baselines, dict) or set(baselines) != {row["id"] for row in CANDIDATES}
        or any(not isinstance(frame, pd.DataFrame) or not frame.index.equals(data.close.index)
               or not frame.columns.equals(data.close.columns)
               or (frame.isna().any(axis=1) & ~frame.isna().all(axis=1)).any()
               for frame in baselines.values())):
        raise QuantError("Volatility projection requires both complete original monthly source matrices.")
    daily = data.close.pct_change(fill_method=None)
    result, diagnostics = {}, []
    for candidate in CANDIDATES:
        source = baselines[candidate["id"]]
        active = source.dropna(how="all")
        if (active.empty or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Each monthly source must retain finite funded allocations.")
        output = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        monthly, previous = None, None
        for index, day in enumerate(data.close.index):
            if index < 63:
                continue
            update = not source.loc[day].isna().all()
            if update:
                monthly = source.loc[day].copy()
            if monthly is None:
                continue
            trailing = daily.iloc[index - 62:index + 1]
            if len(trailing) != 63 or not np.isfinite(trailing.to_numpy()).all():
                raise QuantError("Each daily projection needs all 63 actual already observed returns.")
            covariance = trailing.cov() * 252
            desired = cap_volatility(monthly, covariance, 0.12, 0.98)
            if ((desired < 0).any() or (desired.drop("BIL") > monthly.drop("BIL") + 1e-12).any()
                or desired["BIL"] + 1e-12 < monthly["BIL"]):
                raise QuantError("Risk projection cannot upscale or discard the source's existing bills.")
            deviation = None if previous is None else float(abs(desired - previous).max())
            emitted = previous is None or update or deviation + 1e-12 >= 0.05
            if emitted:
                output.loc[day] = desired
                previous = desired.copy()
            diagnostics.append({
                "candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                "source_monthly_update": update, "target_issued": bool(emitted),
                "desired_bil_weight": float(desired["BIL"]),
                "source_bil_weight": float(monthly["BIL"]),
                "desired_forecast_volatility": float(np.sqrt(desired.to_numpy() @ covariance.to_numpy() @ desired.to_numpy())),
                "target_change_from_previous_issued": deviation,
            })
        if output.dropna(how="all").empty:
            raise QuantError("No known complete source allocation can support the fixed volatility transfer.")
        result[candidate["id"]] = output
    return result, diagnostics


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_baselines(data, source_baselines(data, policy), policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    previous = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = [row for row in previous["candidates"]
               if row["spec"]["id"] == "semiconductor_dollar_term_monthly_confirmation"]
    if len(records) != 1:
        raise QuantError("Require the exact original monthly-confirmation registration and audited market.")
    original = records[0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    readiness_ref = records[0]["readiness"]
    readiness = read_json(safe_file(ROOT, readiness_ref["path"], readiness_ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-10, atol=1e-9):
            raise QuantError("Preserve every original actual-market and frozen risk-free observation.")
    baselines = source_baselines(data, policy)
    targets, diagnostics = build_from_baselines(data, baselines, policy)
    output = local_path(ROOT, output)
    if not output.is_relative_to(ROOT / "data") or output == ROOT / "data":
        raise QuantError("Prepare a new private volatility-transfer input directory.")
    new_output_directory(output)
    write_json(output / "projection-diagnostics.json", {"diagnostics": diagnostics})
    frozen = dict(original["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/semiconductor_volatility_target.py"] = file_digest(Path(__file__))
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    specs = []
    for candidate in CANDIDATES:
        spec = {
            "id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
            "market": original["market"], "asset_leverage": original["asset_leverage"],
            "frozen_files": frozen, "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False, "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        for suffix, frame in (("monthly-source", baselines[candidate["id"]]), ("targets", targets[candidate["id"]])):
            write_text_atomic(output / f"{candidate['id']}-{suffix}.csv", frame.to_csv(float_format="%.17g"))
        specs.append(spec)
    return {"specs": specs, "issued_target_rows": {name: len(frame.dropna(how="all")) for name, frame in targets.items()},
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the original volatility-control transfer, without orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/semiconductor-volatility-target-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({"prepared_candidates": len(result["specs"]), "issued_target_rows": result["issued_target_rows"],
                          "candidate_returns_or_metrics_computed": False, "investment_objective_verified": False}, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Volatility transfer preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
