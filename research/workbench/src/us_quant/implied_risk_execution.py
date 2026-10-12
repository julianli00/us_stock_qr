from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import implied_risk_balance as source
from us_quant import observed_weight_execution as execution
from us_quant.backtest import BacktestResult
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/"config/implied-risk-execution.json"
CANDIDATES = (
    {"id": "semiconductor_option_proxy_observed_band025", "source_id": "semiconductor_option_proxy_risk_balance"},
    {"id": "semiconductor_option_proxy_dollar_observed_band025", "source_id": "semiconductor_option_proxy_risk_balance_dollar_half"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "execution_registration", "execution_policy", "execution_implementation")


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "fixed_option_proxy_observed_weight_execution_20261012",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 164, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "actual_weight_trade_band": 0.025, "capital_usd": 10000,
        "commission_per_order": 1.0, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both original option budgets,unchanged2.5pp observed band and original costs/capital.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def execution_policy(policy: dict) -> dict:
    references(policy)
    original = read_json(ROOT/policy["execution_policy"]["path"])
    execution.validate(original)
    band, _ = execution.pure_band(original)
    if band["weight_band"] != 0.025 or band["qld_cap"] is not None or band["volatility_target"] is not None:
        raise QuantError("Reuse only the exact original pure observed-weight law,not a risk override.")
    return original


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    execution_policy(policy)
    frames = source.build_targets(data, read_json(ROOT/policy["source_policy"]["path"]))
    if set(frames) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Both exact original option-proxy allocation sources must remain unchanged.")
    return {row["id"]: frames[row["source_id"]] for row in CANDIDATES}


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use one of the two exact frozen option-budget execution configurations.")
    return execution.run(data, desired, execution.CANDIDATES[0],
                         execution_policy(policy), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    source_packet = read_json(ROOT/policy["source_registration"]["path"])
    records = {entry["spec"]["id"]: entry for entry in source_packet["candidates"]}
    if set(records) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Require both exact original plain/dollar-half option-budget registrations.")
    original = records[CANDIDATES[0]["source_id"]]
    spec = original["spec"]
    if any(entry["spec"]["market"] != spec["market"] or entry["spec"]["frozen_files"] != spec["frozen_files"]
           for entry in records.values()):
        raise QuantError("Both original option sources must retain the same full frozen market and dependencies.")
    for name, digest in spec["frozen_files"].items():
        safe_file(ROOT, name, digest)
    data = review_market({"market": spec["market"]}, ROOT)
    reference = original["readiness"]
    readiness = read_json(safe_file(ROOT, reference["path"], reference["sha256"]))
    audited = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Preserve every original actual fund and frozen risk-free observation.")
    desired = source_baselines(data, policy)
    execution_packet = read_json(ROOT/policy["execution_registration"]["path"])
    execution_specs = [entry["spec"] for entry in execution_packet["candidates"]]
    if (len(execution_specs) != 2
        or {entry["id"] for entry in execution_specs} != {row["id"] for row in execution.CANDIDATES}
        or execution_specs[0]["frozen_files"] != execution_specs[1]["frozen_files"]):
        raise QuantError("Bind the complete actually registered original execution graph.")
    frozen = dict(spec["frozen_files"])
    additions = {
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        **{name: digest for entry in execution_specs for name, digest in entry["frozen_files"].items()},
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("A conflicting original allocation/execution source hash cannot be overwritten.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    output = local_path(ROOT, output)
    if output == ROOT/"data" or not output.is_relative_to(ROOT/"data"):
        raise QuantError("Use a new isolated private option-execution preparation directory without accounts.")
    new_output_directory(output)
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output/"readiness.json", readiness)
    for candidate in CANDIDATES:
        actual = {**spec, "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output/f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output/f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {"unchanged_monthly_desired_targets_per_candidate": 131,
            "actual_path_dependent_issued_targets_or_accounts_computed": False,
            "candidate_returns_or_metrics_computed": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed option-budget observed-holding execution without accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/implied-risk-execution-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Option-budget execution preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
