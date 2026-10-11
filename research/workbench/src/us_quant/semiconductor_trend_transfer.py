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
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic
from us_quant.trend_factor_guard import gated_target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/semiconductor-trend-transfer.json"
CANDIDATES = (
    {"id": "semiconductor_equity_gold_trend200", "rule": "daily_component_sma"},
    {"id": "semiconductor_equity_gold_relative_trends", "rule": "monthly_relative_trends"},
)
REFERENCES = ("source_registration", "prior_sma_policy", "prior_sma_implementation", "prior_relative_policy")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "semiconductor_fixed_price_trend_transfer_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 130, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "sma_observations": 200, "relative_trend_sessions": [63, 126, 252],
        "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False):
        raise QuantError("Keep both prior trend laws, original funding and two fixed risk constructions.")


def equity_signal(close: pd.DataFrame) -> pd.Series:
    if (not close.index.is_unique or not close.index.is_monotonic_increasing
        or not close.columns.is_unique or not {*FACTORS, "SOXX"}.issubset(close)
        or len(close) < 2 or not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any()):
        raise QuantError("The composed equity signal needs complete actual positive source prices.")
    daily = close.pct_change(fill_method=None).iloc[1:]
    returns = daily["SOXX"] * 0.5 + daily.loc[:, list(FACTORS)].sum(axis=1) * 0.125
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise QuantError("Invalid observed returns cannot create a synthetic valid equity signal.")
    index = pd.Series(1.0, index=close.index, name="equity_signal")
    index.iloc[1:] = (1 + returns).cumprod()
    return index


def gate_components(source: pd.Series, state: tuple[bool, bool]) -> pd.Series:
    if not {*FACTORS, "SOXX", "GLD", "BIL"}.issubset(source.index):
        raise QuantError("Require the original four-style/SOXX/gold funded source.")
    result = gated_target(source, state)
    if not state[0]:
        result["BIL"] += result["SOXX"]
        result["SOXX"] = 0.0
    if not np.isfinite(result).all() or (result < 0).any() or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12):
        raise QuantError("Trend gating cannot increase surviving risk budgets or borrow capital.")
    return result


def build_from_baseline(
    data: MarketData, baseline: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(baseline, pd.DataFrame) or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()):
        raise QuantError("Keep the full original source target shape, dates and unfilled nondecision rows.")
    active = baseline.dropna(how="all")
    if (active.empty or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Original monthly allocations must be complete, positive and funded.")
    equity = equity_signal(data.close)
    levels = pd.DataFrame({"equity": equity, "gold": data.close["GLD"]})
    means = levels.rolling(200, min_periods=200).mean()
    result, diagnostics = {}, []
    for candidate in CANDIDATES:
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        source, previous_state = None, None
        for position, day in enumerate(data.close.index):
            updated = not baseline.loc[day].isna().all()
            if updated:
                source = baseline.loc[day].copy()
            if source is None:
                continue
            if candidate["rule"] == "daily_component_sma":
                if means.loc[day].isna().any():
                    continue
                state = tuple(bool(levels.loc[day, name] > means.loc[day, name]) for name in levels)
                issued = previous_state is None or updated or state != previous_state
                scales = tuple(float(value) for value in state)
                if issued:
                    target = gate_components(source, state)
            else:
                if position < 252 or not updated:
                    continue
                bill_returns = {lookback: data.close["BIL"].iloc[position] / data.close["BIL"].iloc[position - lookback] - 1
                                for lookback in (63, 126, 252)}
                scales = tuple(sum(
                    levels[name].iloc[position] / levels[name].iloc[position - lookback] - 1 > bill_returns[lookback]
                    for lookback in (63, 126, 252)
                ) / 3 for name in levels)
                issued = True
                target = source.copy()
                for scale, names in zip(scales, ([*FACTORS, "SOXX"], ["GLD"]), strict=True):
                    original_budget = float(target.loc[names].sum())
                    target.loc[names] *= scale
                    target["BIL"] += original_budget * (1 - scale)
                state = scales
            if issued:
                if (not np.isfinite(target).all() or (target < 0).any()
                    or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)):
                    raise QuantError("A fixed trend target violated original cash funding or risk removal.")
                frame.loc[day] = target
            diagnostics.append({"candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                                "source_monthly_update": updated, "target_issued": bool(issued),
                                "equity_scale": float(scales[0]), "gold_scale": float(scales[1])})
            previous_state = state
        if frame.dropna(how="all").empty:
            raise QuantError("A trend method lacks its complete original observation warmup.")
        result[candidate["id"]] = frame
    return result, diagnostics


def monthly_source(data: MarketData, policy: dict) -> pd.DataFrame:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_baseline(data, monthly_source(data, policy), policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    registration = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = [row for row in registration["candidates"] if row["spec"]["id"] == "semiconductor_dollar_half_guard"]
    if len(records) != 1:
        raise QuantError("Require the original audited twelve-asset sector market registration.")
    original = records[0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    verified = verified_etf_market(registration["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-10, atol=1e-9):
            raise QuantError("Keep every original actual market and frozen risk-free observation.")
    baseline = monthly_source(data, policy)
    frames, diagnostics = build_from_baseline(data, baseline, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private trend-transfer input directory.")
    new_output_directory(output)
    write_json(output / "trend-diagnostics.json", {"diagnostics": diagnostics})
    write_text_atomic(output / "equity-signal.csv", equity_signal(data.close).to_csv(float_format="%.17g"))
    write_text_atomic(output / "monthly-source.csv", baseline.to_csv(float_format="%.17g"))
    readiness = deepcopy(registration["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(original["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    for name in ("src/us_quant/factor_gold_risk.py", "config/factor-gold-risk.json"):
        frozen[name] = file_digest(ROOT / name)
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/semiconductor_trend_transfer.py"] = file_digest(Path(__file__))
    specs = []
    for candidate in CANDIDATES:
        spec = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
                "market": original["market"], "asset_leverage": original["asset_leverage"], "frozen_files": frozen,
                "history_status": "exposed_history_not_independent_holdout", "leveraged_products_allowed": False,
                "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
        specs.append(spec)
    return {"specs": specs, "issued_targets": {name: len(frame.dropna(how="all")) for name, frame in frames.items()},
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed equity/gold price-trend transfers, without orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/semiconductor-trend-transfer-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({"prepared_candidates": len(result["specs"]), "issued_targets": result["issued_targets"],
                          "candidate_returns_or_metrics_computed": False, "investment_objective_verified": False}, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Trend transfer preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
