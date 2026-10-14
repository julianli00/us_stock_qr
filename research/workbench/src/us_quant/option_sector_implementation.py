from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import implied_risk_execution as parent
from us_quant.backtest import BacktestResult
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/option-sector-implementation.json"
ROUND_ID = "user_confirmed_etf_restructure_20261013"
CANDIDATES = (
    {"id": "bounded_option_dollar_technology_full", "replacement_symbol": "XLK", "optimization_round_id": ROUND_ID},
    {"id": "bounded_option_dollar_nasdaq_full", "replacement_symbol": "QQQ", "optimization_round_id": ROUND_ID},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation")


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "bounded_option_dollar_sector_implementation_20261014_batch2",
        "optimization_round_id": ROUND_ID, "approved_round_configuration_cap": 6,
        "approved_round_batch_cap": 3, "batch_number": 2, "data_scope": "factor_etf_portfolio",
        "data_start": "2015-08-10", "as_of": "2026-10-05", "prior_evaluated_configurations": 174,
        "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_option_proxy_dollar_observed_band025",
        "actual_weight_trade_band": 0.025, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False
    ):
        raise QuantError("Preserve the two fixed sector transfers,original costs/guards and six-slot limit.")


def references(policy: dict, root: Path = ROOT) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(root, policy[key]["path"], policy[key]["sha256"])


def source_entry(policy: dict, root: Path = ROOT) -> dict:
    references(policy, root)
    packet = read_json(root / policy["source_registration"]["path"])
    rows = [row for row in packet["candidates"] if row["spec"]["id"] == policy["source_id"]]
    if len(rows) != 1:
        raise QuantError("Require the exact original option/Dollar/observed-weight registration.")
    return rows[0]


def verified_market(policy: dict, root: Path = ROOT) -> MarketData:
    entry = source_entry(policy, root)
    original = review_market({"market": entry["spec"]["market"]}, root)
    ref = entry["readiness"]
    audited = verified_etf_market(read_json(safe_file(root, ref["path"], ref["sha256"])), root)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(original, name), getattr(audited, name)
        if (
            not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("Retain the complete original actual fund panel and risk-free inputs.")
    if len(original.close) != 2805 or set(original.close.columns) != {
        "SPY", "IEF", "TLT", "GLD", "BIL", *FACTORS, "QQQ", "XLK", "SOXX",
    }:
        raise QuantError("Both destinations require the original verified twelve-asset2805session context.")
    return original


def build_from_original(data: MarketData, original: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    if (
        not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns)
        or not {"SOXX", "XLK", "QQQ", "GLD", "BIL", *FACTORS}.issubset(original.columns)
    ):
        raise QuantError("Use the complete original desired matrix and both admitted destination funds.")
    active = original.dropna(how="all")
    if (
        len(active) != 131 or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not active[["XLK", "QQQ"]].eq(0).all().all()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not original.notna().any(axis=1).equals(original.notna().all(axis=1))
    ):
        raise QuantError("Retain all131complete monthly budgets without destination preallocation or defaults.")
    result = {}
    for candidate in CANDIDATES:
        destination = candidate["replacement_symbol"]
        target = original.copy()
        target[destination] = original["SOXX"]
        target["SOXX"] = original["SOXX"] * 0.0
        unchanged = original.columns.drop(["SOXX", destination])
        if (
            not target.isna().equals(original.isna())
            or not target[unchanged].equals(original[unchanged])
            or not np.allclose(target.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Only the industry sleeve implementation may change,not its budget or monthly clock.")
        result[candidate["id"]] = target
    return result


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    source = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    return build_from_original(data, source, policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use only one of the two fixed budgeted sector implementations.")
    return parent.run(data, desired, parent.CANDIDATES[1],
                      read_json(ROOT / policy["source_policy"]["path"]), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    entry = source_entry(policy)
    data = verified_market(policy)
    original = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    desired = build_from_original(data, original, policy)
    frozen = {
        **entry["spec"]["frozen_files"],
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
    }
    for name, expected in frozen.items():
        safe_file(ROOT, name, expected)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare in a new specific private directory without rewriting prior study inputs.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-desired.csv", original.to_csv(float_format="%.17g"))
    ref = entry["readiness"]
    readiness = deepcopy(read_json(safe_file(ROOT, ref["path"], ref["sha256"])))
    readiness.update(checked_at=utc_now(), revalidation_not_new_market_observation=True)
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**entry["spec"], "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {"monthly_requests_each": 131, "unchanged_source_assets": 12,
            "candidate_returns_or_metrics_computed": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed sector implementations without strategy accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/option-sector-implementation-20261014")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Sector implementation preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
