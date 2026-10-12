from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import profit_factor_guard as parent
from us_quant import sector_term_guard as term
from us_quant.backtest import BacktestResult
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/"config/profit-term-confirmation.json"
CANDIDATES = (
    {"id": "semiconductor_profit_level_or_term_confirmation", "profit_measure": "level"},
    {"id": "semiconductor_profit_share_or_term_confirmation", "profit_measure": "output_share"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "term_registration", "term_policy", "term_implementation")


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "fixed_profit_option_information_complementarity_20261012",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 168, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "option_ratio_risk_threshold": 1.0, "risk_off_scale": 0.50,
        "risk_change_sessions": 63, "actual_weight_trade_band": 0.025, "cash_reserve": 0.02,
        "confirmation_schedule": "completed_month",
        "information_rule": "positive_dollar_and_either_profit_decline_or_option_pressure",
        "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Preserve both fixed monthly DollarAND-profitORoption rules,threshold1/halfscale/old band and budgets.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    references(policy)
    original_policy = read_json(ROOT/policy["source_policy"]["path"])
    comparisons, common = parent.load_inputs(data, original_policy)
    terms = term.verified_terms(data.close.index, read_json(ROOT/policy["term_policy"]["path"]))
    monthly = terms.loc[comparisons.index]
    if (not monthly.index.equals(comparisons.index) or list(monthly.columns) != ["VIX", "VIX3M"]
        or not np.isfinite(monthly.to_numpy()).all() or (monthly <= 0).any().any()):
        raise QuantError("Every original month-end needs genuine finite native VIX/VIX3Mcloses.")
    return comparisons, common, monthly


def build_from_inputs(
    data: MarketData, original: pd.DataFrame, comparisons: pd.DataFrame,
    monthly_terms: pd.DataFrame, policy: dict,
) -> dict[str, pd.DataFrame]:
    validate(policy)
    original_policy = read_json(ROOT/policy["source_policy"]["path"])
    parent.build_from_inputs(data, original, comparisons, original_policy)
    if (not isinstance(monthly_terms, pd.DataFrame) or not monthly_terms.index.equals(comparisons.index)
        or list(monthly_terms.columns) != ["VIX", "VIX3M"]
        or not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   and not pd.api.types.is_complex_dtype(dtype) for dtype in monthly_terms.dtypes)
        or not np.isfinite(monthly_terms.to_numpy()).all() or (monthly_terms <= 0).any().any()):
        raise QuantError("Only complete actual monthly option-risk fields may complement the frozen earnings input.")
    outputs = {}
    assets = original.columns.drop("BIL")
    for candidate in CANDIDATES:
        column = "CPATAX" if candidate["profit_measure"] == "level" else "profit_share"
        targets = original.copy()
        for day in comparisons.index:
            decline = comparisons.loc[day, f"{column}_change"] < 0
            pressure = monthly_terms.loc[day, "VIX"] >= monthly_terms.loc[day, "VIX3M"]
            if comparisons.loc[day, "dollar_change"] > 0 and (decline or pressure):
                freed = float(targets.loc[day, assets].sum())*0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        active = targets.dropna(how="all")
        if (not targets.isna().equals(original.isna()) or len(active) != 131
            or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Complementary confirmation cannot doublehalve,drop a valid month or alter original cash funding.")
        outputs[candidate["id"]] = targets
    return outputs


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    original_policy = read_json(ROOT/policy["source_policy"]["path"])
    original = parent.parent.source_baselines(data, read_json(ROOT/original_policy["source_policy"]["path"]))[
        original_policy["source_id"]
    ]
    comparisons, _, monthly = load_inputs(data, policy)
    return build_from_inputs(data, original, comparisons, monthly, policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use one of the two exact fixed profit/option complementary configurations.")
    return parent.run(data, desired, parent.CANDIDATES[0], read_json(ROOT/policy["source_policy"]["path"]),
                      start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT/policy["source_registration"]["path"])
    entries = [entry for entry in packet["candidates"] if entry["spec"]["id"] == parent.CANDIDATES[0]["id"]]
    if len(entries) != 1:
        raise QuantError("Require the actual original common-quarter profit confirmation source graph.")
    entry, spec = entries[0], entries[0]["spec"]
    for name, digest in spec["frozen_files"].items():
        safe_file(ROOT, name, digest)
    data = review_market({"market": spec["market"]}, ROOT)
    reference = entry["readiness"]
    readiness = read_json(safe_file(ROOT, reference["path"], reference["sha256"]))
    audited = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Preserve all exact original fund observations and risk-free inputs.")
    original_policy = read_json(ROOT/policy["source_policy"]["path"])
    original = parent.parent.source_baselines(data, read_json(ROOT/original_policy["source_policy"]["path"]))[
        original_policy["source_id"]
    ]
    comparisons, common, monthly = load_inputs(data, policy)
    desired = build_from_inputs(data, original, comparisons, monthly, policy)
    term_packet = read_json(ROOT/policy["term_registration"]["path"])
    term_specs = [entry["spec"] for entry in term_packet["candidates"]]
    if (len(term_specs) != 2 or {entry["id"] for entry in term_specs} != {row["id"] for row in term.CANDIDATES}
        or term_specs[0]["frozen_files"] != term_specs[1]["frozen_files"]):
        raise QuantError("Freeze the complete actually registered identical option-term source graphs.")
    frozen = dict(spec["frozen_files"])
    additions = {
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        **term_specs[0]["frozen_files"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("A conflicting original quarterly/option/execution source hash cannot be overwritten.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    output = local_path(ROOT, output)
    if output == ROOT/"data" or not output.is_relative_to(ROOT/"data"):
        raise QuantError("Use a new isolated private complementary-information preparation directory.")
    new_output_directory(output)
    write_text_atomic(output/"original-source-desired.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output/"common-quarter-queries.csv", common.to_csv(float_format="%.17g"))
    write_text_atomic(output/"risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    write_text_atomic(output/"monthly-term-closes.csv", monthly.to_csv(float_format="%.17g"))
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output/"readiness.json", readiness)
    for candidate in CANDIDATES:
        actual = {**spec, "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output/f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output/f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {"actual_monthly_term_quote_cells": monthly.size, "actual_common_quarter_queries": len(common),
            "original_desired_monthly_calendar_per_candidate": 131,
            "actual_path_dependent_feedback_targets_or_accounts_computed": False,
            "candidate_returns_or_metrics_computed": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed complementary quarterly-profit/monthly-option inputs without accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/profit-term-confirmation-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Profit/option complementarity preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
