from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import implied_risk_execution as parent
from us_quant import momentum_implementation as momentum
from us_quant.backtest import BacktestResult
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/option-momentum-implementation.json"
ROUND_ID = "user_confirmed_etf_restructure_20261013"
CANDIDATES = (
    {"id": "bounded_option_dollar_pdp_momentum_full", "pdp_share_of_momentum": 1.0,
     "optimization_round_id": ROUND_ID},
    {"id": "bounded_option_dollar_pdp_momentum_equal_mix", "pdp_share_of_momentum": 0.5,
     "optimization_round_id": ROUND_ID},
)
REFERENCES = (
    "source_registration", "source_policy", "source_implementation", "momentum_registration",
    "momentum_policy", "momentum_implementation", "momentum_source_audit", "momentum_manifest",
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "bounded_option_dollar_momentum_implementation_20261013_batch1",
        "optimization_round_id": ROUND_ID, "approved_round_configuration_cap": 6,
        "approved_round_batch_cap": 3, "batch_number": 1,
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 172, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_option_proxy_dollar_observed_band025",
        "replacement_symbol": "PDP", "replacement_family": "momentum",
        "actual_weight_trade_band": 0.025, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(name) != value for name, value in expected.items())
        or any(not isinstance(policy.get(name), dict) or set(policy[name]) != {"path", "sha256"} for name in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False
    ):
        raise QuantError("Preserve the exact bounded two momentum implementations,original budget/execution and six-slot limit.")


def references(policy: dict, root: Path = ROOT) -> None:
    validate(policy)
    for name in REFERENCES:
        safe_file(root, policy[name]["path"], policy[name]["sha256"])


def source_entry(policy: dict, root: Path = ROOT) -> dict:
    references(policy, root)
    packet = read_json(root / policy["source_registration"]["path"])
    entries = [entry for entry in packet["candidates"] if entry["spec"]["id"] == policy["source_id"]]
    if len(entries) != 1:
        raise QuantError("Require the exact original guarded option-budget observed-band registration.")
    return entries[0]


def verified_market(policy: dict, root: Path = ROOT) -> MarketData:
    entry = source_entry(policy, root)
    original = review_market({"market": entry["spec"]["market"]}, root)
    reference = entry["readiness"]
    audited = verified_etf_market(read_json(safe_file(root, reference["path"], reference["sha256"])), root)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(original, name), getattr(audited, name)
        tolerance = 1e-12 if name == "risk_free" else 1e-9
        if (
            not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=tolerance)
        ):
            raise QuantError("The original twelve-asset quotes and risk-free source must remain exact.")
    if "PDP" in original.close.columns:
        raise QuantError("Do not overwrite an already present fund column.")
    extra = momentum.verified_market(read_json(root / policy["momentum_policy"]["path"]))
    if not extra.close.index.equals(original.close.index) or len(original.close) != 2805:
        raise QuantError("The new momentum implementation must retain every original2805context session.")
    panels = {}
    for name in ("open", "close", "raw_close", "volume"):
        frame = getattr(original, name).copy()
        frame["PDP"] = getattr(extra, name)["PDP"]
        panels[name] = frame
    data = MarketData(**panels, risk_free=original.risk_free.copy())
    data.validate()
    return data


def original_targets(data: MarketData, policy: dict) -> pd.DataFrame:
    references(policy)
    if "PDP" not in data.close.columns:
        raise QuantError("Require the appended genuine PDP data without replacing original asset prices.")
    original = MarketData(
        data.open.drop(columns="PDP"), data.close.drop(columns="PDP"),
        data.raw_close.drop(columns="PDP"), data.volume.drop(columns="PDP"), data.risk_free,
    )
    targets = parent.source_baselines(original, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    targets["PDP"] = targets["MTUM"] * 0.0
    return targets.reindex(columns=data.close.columns)


def build_from_original(
    data: MarketData, original: pd.DataFrame, policy: dict,
) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    if (
        not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns)
        or not {"MTUM", "PDP", "SOXX", "GLD", "BIL"}.issubset(original.columns)
    ):
        raise QuantError("Use the full unchanged original desired matrix with an initially zero PDP column.")
    active = original.dropna(how="all")
    if (
        len(active) != 131 or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any() or not active["PDP"].eq(0).all()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not original.notna().any(axis=1).equals(original.notna().all(axis=1))
    ):
        raise QuantError("All131funded original month requests must remain;no partial/default target rows.")
    frames = {}
    unchanged = original.columns.drop(["MTUM", "PDP"])
    for candidate in CANDIDATES:
        target = original.copy()
        share = candidate["pdp_share_of_momentum"]
        target["PDP"] = original["MTUM"] * share
        target["MTUM"] = original["MTUM"] * (1.0 - share)
        if (
            not target.isna().equals(original.isna())
            or not target[unchanged].equals(original[unchanged])
            or not np.allclose(target.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Only the original momentum sleeve implementation may change in this batch.")
        frames[candidate["id"]] = target
    return frames


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_original(data, original_targets(data, policy), policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use only one of the two user-budgeted fixed momentum constructions.")
    return parent.run(
        data, desired, parent.CANDIDATES[1],
        read_json(ROOT / policy["source_policy"]["path"]), start, end, cost, delay,
    )


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    entry = source_entry(policy)
    data = verified_market(policy)
    original = original_targets(data, policy)
    desired = build_from_original(data, original, policy)
    manifest_path = ROOT / policy["momentum_manifest"]["path"]
    manifest = read_json(manifest_path)
    packet = read_json(ROOT / policy["momentum_registration"]["path"])
    momentum_spec = packet["candidates"][0]["spec"]
    frozen = dict(entry["spec"]["frozen_files"])
    additions = {
        **momentum_spec["frozen_files"],
        **{policy[name]["path"]: policy[name]["sha256"] for name in REFERENCES},
        **{(manifest_path.parent / name).relative_to(ROOT).as_posix(): digest for name, digest in manifest["files"].items()},
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("A conflicting original source or implementation graph cannot be overwritten.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare in a new specific private data directory,never overwrite completed inputs.")
    new_output_directory(output)
    panels = {}
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        values = getattr(data, name)
        frame = values.to_frame() if name == "risk_free" else values
        path = output / f"market-{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        panels[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    write_text_atomic(output / "original-source-desired.csv", original.to_csv(float_format="%.17g"))
    reference = entry["readiness"]
    readiness = deepcopy(read_json(ROOT / reference["path"]))
    readiness.update(checked_at=utc_now(), revalidation_not_new_market_observation=True)
    readiness["verified_etf_source"] = {
        "adapter": "option_momentum_implementation_20261013",
        "policy": POLICY.relative_to(ROOT).as_posix(), "policy_sha256": file_digest(POLICY),
        "factor_manifest": policy["momentum_manifest"]["path"],
        "factor_manifest_sha256": policy["momentum_manifest"]["sha256"],
    }
    readiness["source_check"] = policy["momentum_source_audit"]
    readiness["new_economic_factor_definitions"] = 0
    readiness["full_historical_methodology_verified"] = False
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {
            **entry["spec"], "id": candidate["id"], "configuration": candidate,
            "market": panels, "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {
        "new_configurations_prepared": 2,
        "unchanged_original_month_requests": 131,
        "actual_fund_assets": len(data.close.columns),
        "new_source_requests": 0,
        "new_independent_factor_definitions": 0,
        "candidate_returns_or_metrics_computed": False,
        "actual_feedback_trade_targets_or_accounts_computed": False,
        "optimization_round_id": policy["optimization_round_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bounded current-architecture momentum implementations without accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/option-momentum-implementation-20261013")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Bounded momentum preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
