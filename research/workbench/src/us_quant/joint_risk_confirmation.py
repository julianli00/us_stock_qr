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
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.sector_term_guard import load_inputs
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/joint-risk-confirmation.json"
CANDIDATES = (
    {"id": "semiconductor_dollar_term_monthly_confirmation", "confirmation_schedule": "completed_month"},
    {"id": "semiconductor_dollar_term_daily_confirmation", "confirmation_schedule": "daily_option_state"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "joint_dollar_option_confirmation_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 126, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "option_ratio_risk_threshold": 1.0,
        "risk_off_investment_scale": 0.50, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Keep both fixed conjunction timings, original sources, thresholds and funding.")


def confirmation_targets(
    monthly: pd.DataFrame, terms: pd.DataFrame, dollar: pd.Series, schedule: str,
) -> pd.DataFrame:
    if (
        not isinstance(monthly, pd.DataFrame) or not monthly.index.is_unique
        or not monthly.index.is_monotonic_increasing or not monthly.columns.is_unique or "BIL" not in monthly
        or (monthly.isna().any(axis=1) & ~monthly.isna().all(axis=1)).any()
        or not isinstance(terms, pd.DataFrame) or not terms.index.equals(monthly.index)
        or list(terms.columns) != ["VIX", "VIX3M"]
        or not np.isfinite(terms.to_numpy()).all() or (terms <= 0).any().any()
        or not isinstance(dollar, pd.Series) or not dollar.index.equals(monthly.dropna(how="all").index)
        or not np.isfinite(dollar).all()
        or schedule not in {"completed_month", "daily_option_state"}
    ):
        raise QuantError("Require complete source targets, daily option levels and own-query monthly dollar inputs.")
    active = monthly.dropna(how="all")
    if (
        active.empty or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("Original joint-risk budgets must remain finite, positive and cash funded.")
    result = pd.DataFrame(np.nan, index=monthly.index, columns=monthly.columns)
    baseline, known_dollar, previous_state = None, None, None
    for day in monthly.index:
        update = not monthly.loc[day].isna().all()
        if update:
            baseline = monthly.loc[day].copy()
            known_dollar = bool(dollar.loc[day] > 0)
        if baseline is None or (schedule == "completed_month" and not update):
            continue
        state = bool(known_dollar and terms.loc[day, "VIX"] >= terms.loc[day, "VIX3M"])
        if update or previous_state is None or state != previous_state:
            target = baseline.copy()
            if state:
                risky = target.index != "BIL"
                freed = float(target.loc[risky].sum()) * 0.5
                target.loc[risky] *= 0.5
                target["BIL"] += freed
            if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
                raise QuantError("Joint confirmation cannot change original total investment or borrow cash.")
            result.loc[day] = target
        previous_state = state
    return result


def build_from_inputs(data: MarketData, terms: pd.DataFrame, dollar: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    baseline = sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    return {
        candidate["id"]: confirmation_targets(baseline, terms, dollar, candidate["confirmation_schedule"])
        for candidate in CANDIDATES
    }


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    terms, dollar, _ = load_inputs(data, read_json(ROOT / policy["source_policy"]["path"]))
    return build_from_inputs(data, terms, dollar, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    previous = read_json(safe_file(ROOT, policy["source_registration"]["path"], policy["source_registration"]["sha256"]))
    records = [entry for entry in previous["candidates"] if entry["spec"]["id"] == "semiconductor_term_or_dollar_half_guard"]
    if len(records) != 1:
        raise QuantError("Require the exact original option/dollar source registration.")
    original = records[0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    verified = verified_etf_market(previous["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-10, atol=1e-9):
            raise QuantError("Keep the original audited source panels and risk-free evidence.")
    terms, dollar, provenance = load_inputs(data, read_json(ROOT / policy["source_policy"]["path"]))
    frames = build_from_inputs(data, terms, dollar, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private joint-risk source directory.")
    new_output_directory(output)
    for name, frame in (("option-closes", terms), ("monthly-dollar", dollar), ("dollar-provenance", provenance)):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    frozen = dict(original["frozen_files"])
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/joint_risk_confirmation.py"] = file_digest(Path(__file__))
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    readiness = deepcopy(previous["readiness"])
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
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
        specs.append(spec)
    return {"specs": specs, "defined_target_rows": {name: len(frame.dropna(how="all")) for name, frame in frames.items()},
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed monthly/daily joint-risk confirmation; no orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/joint-risk-confirmation-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({"prepared_candidates": len(result["specs"]), "defined_target_rows": result["defined_target_rows"],
                          "candidate_returns_or_metrics_computed": False, "investment_objective_verified": False}, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Joint risk preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
