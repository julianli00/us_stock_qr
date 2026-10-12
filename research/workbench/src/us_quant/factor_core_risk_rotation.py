from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import sector_leadership_rotation as parent
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.downside_risk_balance import risk_estimate
from us_quant.multifactor_stability import FACTORS, bounded_factor_shares
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/factor-core-risk-rotation.json"
CANDIDATES = (
    {"id": "semiconductor_leadership_core_inverse_volatility", "risk_measure": "sample_volatility"},
    {"id": "semiconductor_leadership_core_inverse_semideviation", "risk_measure": "semideviation"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "prior_bounds_policy",
              "prior_bounds_implementation", "prior_downside_policy", "prior_downside_implementation")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "bounded_style_core_risk_transfer_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 144, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_leadership126_observed_band025",
        "risk_sessions": 63, "factor_share_min": 0.15, "factor_share_max": 0.35,
        "actual_weight_trade_band": 0.025, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep all four factors,original risk estimators,bounds and unchanged source execution.")


def factor_risk_shares(
    history: pd.DataFrame, rates: pd.Series, method: str,
) -> tuple[pd.Series, pd.Series]:
    if (not isinstance(history, pd.DataFrame) or list(history.columns) != list(FACTORS)
        or len(history) < 64 or not history.index.is_unique or not history.index.is_monotonic_increasing
        or not isinstance(rates, pd.Series) or not rates.index.equals(history.index)
        or not np.isfinite(history.to_numpy()).all() or (history <= 0).any().any()
        or not np.isfinite(rates).all() or method not in {row["risk_measure"] for row in CANDIDATES}):
        raise QuantError("Factor risk shares require complete actual fund prices and frozen risk-free observations.")
    recent = history.pct_change(fill_method=None).iloc[1:].tail(63)
    if method == "sample_volatility":
        risks = recent.std(ddof=1)
    else:
        excess = recent.sub(rates.loc[recent.index], axis=0)
        risks = pd.Series({name: risk_estimate(excess[name].to_numpy(), "semideviation") for name in FACTORS})
    shares = bounded_factor_shares(risks, 0.15, 0.35)
    return shares, risks


def build_from_source(
    data: MarketData, original: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns) or not set(FACTORS).issubset(original.columns)
        or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
        raise QuantError("Keep the original complete desired matrix and all undefined nondecision rows.")
    active = original.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not active.loc[:, list(FACTORS)].gt(0).all().all()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Use the original positive four-style core and complete funded monthly source.")
    frames, diagnostics = {}, []
    unchanged = original.columns.difference(FACTORS)
    for candidate in CANDIDATES:
        frame = original.copy()
        for day in active.index:
            history = data.close.loc[:day, list(FACTORS)]
            shares, risks = factor_risk_shares(history, data.risk_free.loc[history.index], candidate["risk_measure"])
            budget = float(original.loc[day, list(FACTORS)].sum())
            target = original.loc[day].copy()
            target.loc[list(FACTORS)] = budget * shares
            if (not target.loc[unchanged].equals(original.loc[day, unchanged]) or (target < 0).any()
                or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)
                or not np.isclose(target.loc[list(FACTORS)].sum(), budget, rtol=0, atol=1e-12)
                or (shares < 0.15-1e-12).any() or (shares > 0.35+1e-12).any()):
                raise QuantError("Only bounded within-core shares may change;keep all nongroup targets and budgets.")
            frame.loc[day] = target
            diagnostics.append({"candidate_id": candidate["id"], "session": str(day.date()), "core_budget": budget,
                                "style_risks": {name:float(risks[name]) for name in FACTORS},
                                "style_shares": {name:float(shares[name]) for name in FACTORS}})
        frames[candidate["id"]] = frame
    return frames, diagnostics


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    original = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    return build_from_source(data, original, policy)[0]


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use an exact fixed factor-core configuration.")
    for key in ("source_policy", "source_implementation"):
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return parent.run(data, desired, parent.CANDIDATES[0],
                      read_json(ROOT / policy["source_policy"]["path"]), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    registration = read_json(ROOT / policy["source_registration"]["path"])
    records = [row for row in registration["candidates"] if row["spec"]["id"] == policy["source_id"]]
    if len(records) != 1:
        raise QuantError("Use the exact original126leadership source registration.")
    entry = records[0]
    spec = entry["spec"]
    data = review_market({"market": spec["market"]}, ROOT)
    ref = entry["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve every original actual fund observation and frozen risk-free rate.")
    original = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    frames, diagnostics = build_from_source(data, original, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private factor-core directory without replacing old evidence.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_json(output / "factor-risk-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/factor_core_risk_rotation.py"] = file_digest(Path(__file__))
    for candidate in CANDIDATES:
        actual = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                  "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
                  "market": spec["market"], "asset_leverage": spec["asset_leverage"], "frozen_files": frozen,
                  "history_status": "exposed_history_not_independent_holdout",
                  "leveraged_products_allowed": False, "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"desired_monthly_targets_per_candidate": len(original.dropna(how="all")),
            "actual_path_dependent_trade_targets_computed": False, "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bounded four-factor core risks without accounts or orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/factor-core-risk-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Factor-core preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
