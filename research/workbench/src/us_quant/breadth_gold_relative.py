from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import semiconductor_trend_transfer as prior
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_breadth_risk import build_targets as source_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/breadth-gold-relative.json"
CANDIDATES = (
    {"id": "semiconductor_breadth_half_gold_relative", "source_id": "semiconductor_sector_breadth_equity_half"},
    {"id": "semiconductor_breadth_off_gold_relative", "source_id": "semiconductor_sector_breadth_equity_off"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "prior_relative_registration", "prior_relative_policy", "prior_relative_implementation",
              "published_risk_context", "published_binary_gold_context")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "sector_breadth_prior_relative_gold_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 138, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "gold_relative_sessions": [63, 126, 252], "gold_relative_benchmark": "BIL",
        "decision_clock": "original_source_month_ends_only_after252_session_warmup",
        "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both original breadth sources and the unchanged three-horizon gold-relative law.")


def build_from_sources(
    data: MarketData, sources: dict[str, pd.DataFrame], gold_reference: pd.DataFrame, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(sources, dict) or set(sources) != {row["source_id"] for row in CANDIDATES}
        or not isinstance(gold_reference, pd.DataFrame)
        or not gold_reference.index.equals(data.close.index)
        or list(gold_reference.columns) != ["original_gold", "relative_gold"]
        or not np.isfinite(gold_reference.to_numpy()[~gold_reference.isna().to_numpy()]).all()):
        raise QuantError("Require complete original source matrices and the prior generator's full gold reference.")
    frames, diagnostics = {}, []
    for candidate in CANDIDATES:
        original = sources[candidate["source_id"]]
        if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
            or not original.columns.equals(data.close.columns)
            or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
            raise QuantError("Keep every original target column, date and undefined nondecision.")
        active = original.dropna(how="all")
        if (active.empty or not all(is_month_end(day) for day in active.index)
            or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
            or not active["GLD"].equals(gold_reference.loc[active.index, "original_gold"].rename("GLD"))):
            raise QuantError("The prior gold base budget must equal each complete funded original source exactly.")
        frame = pd.DataFrame(np.nan, index=original.index, columns=original.columns)
        unchanged = original.columns.difference(["GLD", "BIL"])
        for day in active.index:
            value = gold_reference.loc[day, "relative_gold"]
            if data.close.index.get_loc(day) < 252:
                if not pd.isna(value):
                    raise QuantError("Do not invent a fully informed long-horizon gold target during warmup.")
                continue
            if pd.isna(value) or not 0 <= value <= original.loc[day, "GLD"]:
                raise QuantError("The prior fully informed gold target cannot be missing, negative or increase risk.")
            target = original.loc[day].copy()
            target["GLD"] = value
            target["BIL"] += original.loc[day, "GLD"] - value
            if (not target.loc[unchanged].equals(original.loc[day, unchanged])
                or (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)):
                raise QuantError("Copy only the prior gold component; preserve equity and cash funding exactly.")
            frame.loc[day] = target
            diagnostics.append({"candidate_id": candidate["id"], "session": day.strftime("%Y-%m-%d"),
                                "original_gold_weight": float(original.loc[day, "GLD"]),
                                "relative_gold_weight": float(value)})
        if frame.dropna(how="all").empty:
            raise QuantError("No original monthly decision has the full prior relative-trend warmup.")
        frames[candidate["id"]] = frame
    return frames, diagnostics


def inputs(data: MarketData, policy: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    sources = source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))
    prior_policy = read_json(ROOT / policy["prior_relative_policy"]["path"])
    baseline = prior.monthly_source(data, prior_policy)
    relative = prior.build_from_baseline(data, baseline, prior_policy)[0]["semiconductor_equity_gold_relative_trends"]
    reference = pd.DataFrame({"original_gold": baseline["GLD"], "relative_gold": relative["GLD"]})
    return sources, reference


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    sources, reference = inputs(data, policy)
    return build_from_sources(data, sources, reference, policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    registration = read_json(ROOT / policy["source_registration"]["path"])
    records = {row["spec"]["id"]: row for row in registration["candidates"]}
    if set(records) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Use both exact original breadth registrations, not substitute source variants.")
    parent = records[CANDIDATES[0]["source_id"]]
    spec = parent["spec"]
    relative_registration = read_json(ROOT / policy["prior_relative_registration"]["path"])
    relative_records = [row for row in relative_registration["candidates"]
                        if row["spec"]["id"] == "semiconductor_equity_gold_relative_trends"]
    if (len(relative_records) != 1 or relative_records[0]["spec"]["market"] != spec["market"]
        or any(row["spec"]["market"] != spec["market"] or row["spec"]["frozen_files"] != spec["frozen_files"]
               for row in records.values())):
        raise QuantError("Original source and relative-law registrations must retain the same frozen market.")
    data = review_market({"market": spec["market"]}, ROOT)
    ref = parent["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve every original fund observation and frozen risk-free series.")
    sources, reference = inputs(data, policy)
    frames, diagnostics = build_from_sources(data, sources, reference, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare one new private graded-gold input directory without replacing old evidence.")
    new_output_directory(output)
    for identifier, original in sources.items():
        write_text_atomic(output / f"{identifier}-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output / "prior-relative-gold-targets.csv", reference.to_csv(float_format="%.17g"))
    write_json(output / "gold-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for name, digest in relative_records[0]["spec"]["frozen_files"].items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not replace conflicting original frozen helper or source versions.")
        frozen[name] = digest
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/breadth_gold_relative.py"] = file_digest(Path(__file__))
    for candidate in CANDIDATES:
        actual = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                  "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
                  "market": spec["market"], "asset_leverage": spec["asset_leverage"], "frozen_files": frozen,
                  "history_status": "exposed_history_not_independent_holdout",
                  "leveraged_products_allowed": False, "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"issued_targets_per_candidate": len(next(iter(frames.values())).dropna(how="all")),
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prior gold-relative components on original breadth sources.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/breadth-gold-relative-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Graded gold preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
