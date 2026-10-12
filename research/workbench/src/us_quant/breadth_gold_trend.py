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
from us_quant.sector_breadth_risk import build_targets as source_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic
from us_quant.trend_factor_guard import gated_target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/breadth-gold-trend.json"
CANDIDATES = (
    {"id": "semiconductor_breadth_half_gold_trend200", "source_id": "semiconductor_sector_breadth_equity_half"},
    {"id": "semiconductor_breadth_off_gold_trend200", "source_id": "semiconductor_sector_breadth_equity_off"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "prior_gold_policy", "prior_gold_implementation", "published_risk_context")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "sector_breadth_independent_gold_trend_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 136, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "gold_sma_observations": 200, "gold_signal_asset": "GLD",
        "decision_clock": "original_source_month_ends_only", "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the original two breadth sources and strictly200-observation monthly gold law.")


def gold_states(close: pd.Series) -> pd.Series:
    if (not isinstance(close, pd.Series) or len(close) < 200
        or not close.index.is_unique or not close.index.is_monotonic_increasing
        or not np.isfinite(close).all() or (close <= 0).any()):
        raise QuantError("The gold gate needs complete positive actual adjusted fund prices.")
    mean = close.rolling(200, min_periods=200).mean()
    return close.gt(mean).astype(float).where(mean.notna()).rename("gold_active")


def build_from_sources(
    data: MarketData, sources: dict[str, pd.DataFrame], policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(sources, dict) or set(sources) != {row["source_id"] for row in CANDIDATES}
        or "GLD" not in data.close or "BIL" not in data.close):
        raise QuantError("Require both original breadth targets and actual gold/bill funds.")
    states = gold_states(data.close["GLD"])
    frames, diagnostics = {}, []
    for candidate in CANDIDATES:
        original = sources[candidate["source_id"]]
        if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
            or not original.columns.equals(data.close.columns)
            or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
            raise QuantError("Keep every original source row, target column and undefined nondecision.")
        active = original.dropna(how="all")
        if (active.empty or not all(is_month_end(day) for day in active.index)
            or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
            or states.loc[active.index].isna().any()):
            raise QuantError("Original monthly targets and gold observations must be complete and fully informed.")
        frame = original.copy()
        for day in active.index:
            gold_active = bool(states.loc[day])
            target = gated_target(original.loc[day], (True, gold_active))
            unchanged = original.columns.difference(["GLD", "BIL"])
            if not target.loc[unchanged].equals(original.loc[day, unchanged]):
                raise QuantError("Independent gold removal cannot change any original equity or other fund target.")
            frame.loc[day] = target
            diagnostics.append({"candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                                "gold_active": gold_active, "original_gold_weight": float(original.loc[day, "GLD"]),
                                "removed_gold_weight": 0.0 if gold_active else float(original.loc[day, "GLD"])})
        frames[candidate["id"]] = frame
    return frames, diagnostics


def original_sources(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_sources(data, original_sources(data, policy), policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    registration = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = {row["spec"]["id"]: row for row in registration["candidates"]}
    if set(records) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Use the exact original two source registrations, not substitute breadth variants.")
    parent = records[CANDIDATES[0]["source_id"]]
    spec = parent["spec"]
    if any(row["spec"]["market"] != spec["market"] or row["spec"]["frozen_files"] != spec["frozen_files"]
           for row in records.values()):
        raise QuantError("Both original sources must retain their complete identical frozen market and dependencies.")
    data = review_market({"market": spec["market"]}, ROOT)
    ref = parent["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Keep original actual portfolio observations and frozen risk-free rates.")
    sources = original_sources(data, policy)
    frames, diagnostics = build_from_sources(data, sources, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare one new private breadth/gold input directory without overwriting evidence.")
    new_output_directory(output)
    for identifier, original in sources.items():
        write_text_atomic(output / f"{identifier}-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output / "gold-states.csv", gold_states(data.close["GLD"]).to_csv(float_format="%.17g"))
    write_json(output / "gold-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    for name in ("src/us_quant/factor_gold_risk.py", "config/factor-gold-risk.json"):
        frozen[name] = file_digest(ROOT / name)
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/breadth_gold_trend.py"] = file_digest(Path(__file__))
    for candidate in CANDIDATES:
        actual = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                  "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"], "market": spec["market"],
                  "asset_leverage": spec["asset_leverage"], "frozen_files": frozen,
                  "history_status": "exposed_history_not_independent_holdout", "leveraged_products_allowed": False,
                  "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"issued_targets_per_candidate": len(next(iter(frames.values())).dropna(how="all")),
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independent monthly gold trends on original breadth sources.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/breadth-gold-trend-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Breadth/gold preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
