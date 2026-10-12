from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.semiconductor_trend_transfer import build_targets as source_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/residual-treasury-rotation.json"
TREASURIES = ("IEF", "TLT")
CANDIDATES = (
    {"id": "semiconductor_relative_trend_treasury_momentum", "defense_method": "highest_positive_excess"},
    {"id": "semiconductor_relative_trend_treasury_diversified", "defense_method": "all_positive_inverse_volatility"},
)
REFERENCES = ("source_registration", "source_policy", "prior_defense_policy", "prior_defense_implementation")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "relative_trend_residual_treasury_rotation_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 132, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "treasury_assets": list(TREASURIES), "momentum_sessions": 126,
        "volatility_sessions": 63, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False):
        raise QuantError("Keep both fixed Treasury-only residual methods and unchanged original trend targets.")


def treasury_shares(history: pd.DataFrame, method: str) -> tuple[pd.Series, dict]:
    if (not isinstance(history, pd.DataFrame) or len(history) < 127
        or not {*TREASURIES, "BIL"}.issubset(history) or not history.index.is_unique
        or not history.index.is_monotonic_increasing or not history.columns.is_unique
        or not np.isfinite(history.to_numpy()).all() or (history <= 0).any().any()
        or method not in {row["defense_method"] for row in CANDIDATES}):
        raise QuantError("Residual defense needs complete prior observed Treasury/bill prices and a frozen method.")
    change = history.iloc[-1] / history.iloc[-127] - 1
    eligible = [name for name in TREASURIES if change[name] > change["BIL"]]
    result = pd.Series(0.0, index=(*TREASURIES, "BIL"))
    if not eligible:
        result["BIL"] = 1.0
    elif method == "highest_positive_excess":
        selected = sorted(eligible, key=lambda name: (-float(change[name]), name))[0]
        result[selected] = 1.0
    else:
        volatility = history.loc[:, eligible].pct_change(fill_method=None).iloc[1:].tail(63).std(ddof=1)
        if not np.isfinite(volatility).all() or (volatility <= 1e-10).any():
            raise QuantError("Eligible defensive volatility cannot be missing or replaced with a default share.")
        inverse = 1 / volatility
        result.loc[eligible] = inverse / inverse.sum()
    if (not np.isfinite(result).all() or (result < 0).any()
        or not np.isclose(result.sum(), 1.0, rtol=0, atol=1e-12)):
        raise QuantError("Residual Treasury shares must remain funded and finite.")
    return result, {"eligible_treasuries": eligible,
                    "treasury126sessionreturns": {name: float(change[name]) for name in TREASURIES},
                    "bil126sessionreturn": float(change["BIL"]),
                    "selected_defensive_shares": {name: float(value) for name, value in result.items()}}


def route_residual(source: pd.Series, shares: pd.Series) -> pd.Series:
    if (not isinstance(source, pd.Series) or not source.index.is_unique
        or not {*TREASURIES, "BIL"}.issubset(source.index)
        or not np.isfinite(source).all() or (source < 0).any()
        or not np.isclose(source.sum(), 0.98, rtol=0, atol=1e-12)
        or not isinstance(shares, pd.Series) or list(shares.index) != [*TREASURIES, "BIL"]
        or not np.isfinite(shares).all() or (shares < 0).any()
        or not np.isclose(shares.sum(), 1.0, rtol=0, atol=1e-12)):
        raise QuantError("Residual routing needs the complete original funded source and verified defensive shares.")
    result = source.copy()
    amount = float(source["BIL"])
    result["BIL"] = 0.0
    result.loc[shares.index] += amount * shares
    unchanged = source.index.difference(shares.index)
    if (not result.loc[unchanged].equals(source.loc[unchanged]) or (result < 0).any()
        or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Treasury rotation cannot alter original equity/gold targets or the funded budget.")
    return result


def build_from_source(
    data: MarketData, original: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns)
        or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
        raise QuantError("Keep the complete original monthly-relative source matrix and its decision dates.")
    active = original.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Original source targets must remain finite, positive and cash funded.")
    results, diagnostics = {}, []
    for candidate in CANDIDATES:
        targets = original.copy()
        for day in active.index:
            shares, audit = treasury_shares(data.close.loc[:day], candidate["defense_method"])
            targets.loc[day] = route_residual(original.loc[day], shares)
            diagnostics.append({"candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                                "original_residual_bil_weight": float(original.loc[day, "BIL"]), **audit})
        results[candidate["id"]] = targets
    return results, diagnostics


def source(data: MarketData, policy: dict) -> pd.DataFrame:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[
        "semiconductor_equity_gold_relative_trends"
    ]


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_source(data, source(data, policy), policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    registration = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = [row for row in registration["candidates"] if row["spec"]["id"] == "semiconductor_equity_gold_relative_trends"]
    if len(records) != 1:
        raise QuantError("Require the exact original monthly-relative-trend source and market registration.")
    original_spec = records[0]["spec"]
    data = review_market({"market": original_spec["market"]}, ROOT)
    ref = records[0]["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve all original actual-fund and frozen risk-free source observations.")
    original = source(data, policy)
    frames, diagnostics = build_from_source(data, original, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private residual-defensive input directory.")
    new_output_directory(output)
    write_json(output / "defensive-diagnostics.json", {"diagnostics": diagnostics})
    write_text_atomic(output / "original-source-targets.csv", original.to_csv(float_format="%.17g"))
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(original_spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/residual_treasury_rotation.py"] = file_digest(Path(__file__))
    specs = []
    for candidate in CANDIDATES:
        spec = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
                "market": original_spec["market"], "asset_leverage": original_spec["asset_leverage"],
                "frozen_files": frozen, "history_status": "exposed_history_not_independent_holdout",
                "leveraged_products_allowed": False, "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
        specs.append(spec)
    return {"specs": specs, "monthly_targets_per_candidate": len(original.dropna(how="all")),
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed residual Treasury rotation, without orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/residual-treasury-rotation-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({"prepared_candidates": len(result["specs"]),
                          "monthly_targets_per_candidate": result["monthly_targets_per_candidate"],
                          "candidate_returns_or_metrics_computed": False, "investment_objective_verified": False}, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Residual Treasury preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
