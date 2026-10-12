from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import observed_weight_execution as execution
from us_quant.backtest import BacktestResult
from us_quant.breadth_gold_relative import build_targets as source_targets
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-leadership-rotation.json"
CANDIDATES = (
    {"id": "semiconductor_leadership126_observed_band025", "method": "strict126_relative_winner"},
    {"id": "semiconductor_leadership_votes_observed_band025", "method": "equal_relative_horizon_votes"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "execution_policy",
              "execution_implementation", "prior_winner_policy", "prior_relative_policy")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "semiconductor_relative_leadership_equity_switch_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 142, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_breadth_half_gold_relative",
        "satellite_asset": "SOXX", "fallback_equity_asset": "SPY", "winner_sessions": 126,
        "relative_horizon_sessions": [63, 126, 252], "actual_weight_trade_band": 0.025,
        "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both fixed leadership methods,source budget and original2.5ppexecution law.")


def leadership_shares(close: pd.DataFrame, method: str) -> pd.Series:
    if (not isinstance(close, pd.DataFrame) or list(close.columns) != ["SOXX", "SPY"]
        or len(close) < 253 or not close.index.is_unique or not close.index.is_monotonic_increasing
        or not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any()
        or method not in {row["method"] for row in CANDIDATES}):
        raise QuantError("Relative equity leadership requires complete actual fund prices and a frozen method.")
    horizons = (126,) if method == "strict126_relative_winner" else (63, 126, 252)
    votes = pd.DataFrame({horizon: close["SOXX"] / close["SOXX"].shift(horizon)
                          > close["SPY"] / close["SPY"].shift(horizon) for horizon in horizons})
    shares = votes.sum(axis=1) / len(horizons)
    return shares.where(np.arange(len(close)) >= 252).rename("soxx_satellite_share")


def build_from_source(
    data: MarketData, original: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns) or not {"SOXX", "SPY"}.issubset(original.columns)
        or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
        raise QuantError("Keep the complete original source matrix,price calendar and target universe.")
    active = original.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not active["SPY"].eq(0).all()):
        raise QuantError("Use the unchanged funded monthly source with no original SPY satellite allocation.")
    frames, diagnostics = {}, []
    unchanged = original.columns.difference(["SOXX", "SPY"])
    for candidate in CANDIDATES:
        shares = leadership_shares(data.close.loc[:, ["SOXX", "SPY"]], candidate["method"])
        if shares.loc[active.index].isna().any():
            raise QuantError("Original source dates must retain the full relative-return warmup.")
        frame = original.copy()
        for day in active.index:
            target = original.loc[day].copy()
            budget, share = float(target["SOXX"]), float(shares.loc[day])
            target["SOXX"] = budget * share
            target["SPY"] = budget * (1 - share)
            if (not target.loc[unchanged].equals(original.loc[day, unchanged])
                or (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)
                or not np.isclose(target[["SOXX", "SPY"]].sum(), budget, rtol=0, atol=1e-12)):
                raise QuantError("Leadership may only redistribute the original equity satellite to SPY.")
            frame.loc[day] = target
            diagnostics.append({"candidate_id": candidate["id"], "session": str(day.date()),
                                "source_satellite_weight": budget, "soxx_share": share})
        frames[candidate["id"]] = frame
    return frames, diagnostics


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    original = source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    return build_from_source(data, original, policy)[0]


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use a frozen leadership configuration,not a fitted or unrelated candidate.")
    for key in ("execution_policy", "execution_implementation"):
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return execution.run(data, desired, execution.CANDIDATES[0],
                         read_json(ROOT / policy["execution_policy"]["path"]), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    registration = read_json(ROOT / policy["source_registration"]["path"])
    records = [row for row in registration["candidates"] if row["spec"]["id"] == policy["source_id"]]
    if len(records) != 1:
        raise QuantError("Require the exact original graded-gold half source registration.")
    parent = records[0]
    spec = parent["spec"]
    data = review_market({"market": spec["market"]}, ROOT)
    ref = parent["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve all original actual observations and frozen risk-free rates.")
    original = source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    frames, diagnostics = build_from_source(data, original, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private leadership input directory without overwriting evidence.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_json(output / "leadership-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    for name in ("src/us_quant/portfolio_refinement.py", "config/portfolio-refinement.json",
                 "src/us_quant/cash_funded_accounting_v2.py"):
        frozen[name] = file_digest(ROOT / name)
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/sector_leadership_rotation.py"] = file_digest(Path(__file__))
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
    parser = argparse.ArgumentParser(description="Prepare fixed equity-leadership rotation without accounts or orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/sector-leadership-rotation-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Sector-leadership preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
