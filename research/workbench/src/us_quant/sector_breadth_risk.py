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
from us_quant.sector_basket_sources import SECTORS, verified_market as signal_market
from us_quant.sector_growth_balance import build_targets as source_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-breadth-risk.json"
CANDIDATES = (
    {"id": "semiconductor_sector_breadth_equity_half", "weak_equity_scale": 0.5},
    {"id": "semiconductor_sector_breadth_equity_off", "weak_equity_scale": 0.0},
)
REFERENCES = ("source_registration", "source_policy", "signal_policy", "signal_implementation",
              "signal_readiness", "signal_manifest", "prior_sector_trend_policy",
              "prior_half_scale_policy", "prior_zero_scale_policy",
              "source_identity_assumption_code", "source_identity_assumption_policy")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "original_sector_etf_breadth_equity_risk_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 134, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "signal_sector_symbols": list(SECTORS), "sma_observations": 200,
        "strong_breadth": "strict_majority_of_all_nine", "cash_reserve": 0.02, "candidates": list(CANDIDATES),
        "signal_xlk_source": "original_registered_sector_snapshot",
        "portfolio_xlk_target": "always_zero_in_original_semiconductor_source",
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the original nine-sector signal, strict majority,200observations and half/zero pair.")


def breadth_counts(close: pd.DataFrame) -> pd.Series:
    if (not isinstance(close, pd.DataFrame) or list(close.columns) != list(SECTORS)
        or not close.index.is_unique or not close.index.is_monotonic_increasing
        or len(close) < 200 or not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any()):
        raise QuantError("Sector breadth requires complete actual prices for all nine original funds.")
    means = close.rolling(200, min_periods=200).mean()
    return close.gt(means).sum(axis=1).astype(float).where(means.notna().all(axis=1)).rename("above200_count")


def scale_equity(source: pd.Series, scale: float) -> pd.Series:
    equity = [*FACTORS, "SOXX"]
    if (not isinstance(source, pd.Series) or not source.index.is_unique
        or not {*equity, "GLD", "BIL"}.issubset(source.index)
        or not np.isfinite(source).all() or (source < 0).any()
        or not np.isclose(source.sum(), 0.98, rtol=0, atol=1e-12) or scale not in (0.0, 0.5, 1.0)):
        raise QuantError("Breadth can only reduce the complete original funded equity component.")
    result = source.copy()
    budget = float(result.loc[equity].sum())
    result.loc[equity] *= scale
    result["BIL"] += budget * (1 - scale)
    unchanged = source.index.difference([*equity, "BIL"])
    if (not result.loc[unchanged].equals(source.loc[unchanged]) or (result < 0).any()
        or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Keep gold/other targets exact and route only removed equity budget to BIL.")
    return result


def build_from_source(
    data: MarketData, original: pd.DataFrame, sector_close: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(original, pd.DataFrame) or not isinstance(sector_close, pd.DataFrame)
        or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns) or not sector_close.index.equals(data.close.index)
        or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
        raise QuantError("Keep the complete original target and actual-price calendars without filling missing inputs.")
    active = original.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Original monthly risk allocations must remain complete, finite and funded.")
    counts = breadth_counts(sector_close)
    frames, diagnostics = {}, []
    for candidate in CANDIDATES:
        frame = pd.DataFrame(np.nan, index=original.index, columns=original.columns)
        for day in active.index:
            if pd.isna(counts.loc[day]):
                continue
            strong = counts.loc[day] * 2 > len(SECTORS)
            scale = 1.0 if strong else candidate["weak_equity_scale"]
            frame.loc[day] = scale_equity(original.loc[day], scale)
            diagnostics.append({"candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                                "above200_sector_count": int(counts.loc[day]),
                                "strong_breadth": bool(strong), "equity_scale": float(scale)})
        if frame.dropna(how="all").empty:
            raise QuantError("No original monthly decision has complete sector-trend warmup.")
        frames[candidate["id"]] = frame
    return frames, diagnostics


def inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    original = source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[
        "four_factor_semiconductor_sector50"
    ]
    signals = signal_market(read_json(ROOT / policy["signal_policy"]["path"]))
    if (not signals.close.index.equals(data.close.index)
        or not original["XLK"].dropna().eq(0).all()):
        raise QuantError("Require the original full calendar and non-traded portfolio XLK; do not mix quote versions.")
    shared = ["SPY", "IEF", "GLD", "BIL", *FACTORS]
    for name in ("open", "close", "raw_close", "volume"):
        if not np.allclose(getattr(data, name).loc[:, shared], getattr(signals, name).loc[:, shared],
                           rtol=1e-10, atol=1e-9):
            raise QuantError("The independent sector signal cannot substitute shared original core observations.")
    if not np.allclose(data.risk_free, signals.risk_free, rtol=0, atol=1e-12):
        raise QuantError("Sector information cannot replace original frozen risk-free observations.")
    return original, signals.close.loc[:, list(SECTORS)]


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    original, signals = inputs(data, policy)
    return build_from_source(data, original, signals, policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    registration = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = [row for row in registration["candidates"] if row["spec"]["id"] == "semiconductor_dollar_half_guard"]
    if len(records) != 1:
        raise QuantError("Require the exact original audited twelve-asset market registration.")
    spec = records[0]["spec"]
    data = review_market({"market": spec["market"]}, ROOT)
    verified = verified_etf_market(registration["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve every original portfolio price and frozen risk-free observation.")
    original, signals = inputs(data, policy)
    frames, diagnostics = build_from_source(data, original, signals, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare one new private sector-breadth input directory.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output / "sector-signal-close.csv", signals.to_csv(float_format="%.17g"))
    write_json(output / "breadth-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(registration["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    manifest = read_json(ROOT / policy["signal_manifest"]["path"])
    frozen.update(manifest["file_sha256"])
    for key in ("original_policy", "original_registration", "core_registration"):
        reference = read_json(ROOT / policy["signal_policy"]["path"])[key]
        frozen[reference["path"]] = reference["sha256"]
    for name in ("factor_research.py", "dual_horizon.py"):
        frozen[f"src/us_quant/{name}"] = file_digest(Path(__file__).with_name(name))
    frozen["config/dual-horizon.json"] = file_digest(ROOT / "config/dual-horizon.json")
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/sector_breadth_risk.py"] = file_digest(Path(__file__))
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
    parser = argparse.ArgumentParser(description="Prepare original-sector breadth risk rules, without orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/sector-breadth-risk-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Sector breadth preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
