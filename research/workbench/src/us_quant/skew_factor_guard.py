from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import skew_inputs as inputs
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/skew-factor-guard.json"
CANDIDATES = (
    {"id":"semiconductor_skew130_equity_half", "risk_scope":"equity_only"},
    {"id":"semiconductor_skew130_portfolio_half", "risk_scope":"all_non_bil"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "skew_policy", "skew_implementation", "skew_readiness", "skew_manifest")
PAUSED = ("2024-11-29",)
EQUITY = (*FACTORS, "SOXX")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"published_monthly_tail_asymmetry_risk_scopes_20261012",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":160, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_candidate":"four_factor_semiconductor_sector50",
        "skew_reference_level":130.0, "risk_off_scale":0.50, "cash_reserve":0.02,
        "known_paused_months":list(PAUSED),
        "missing_data_rule":"only_exact_predeclared_monthly_source_absence_pauses_request_unknown_gap_fatal",
        "strict_source_readiness_relabelled":False, "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the fixed tail reference/asset scopes/halfscale/known pause and original funding.")


def references(policy: dict) -> None:
    validate(policy)
    for name in REFERENCES:
        safe_file(ROOT, policy[name]["path"], policy[name]["sha256"])


def load_inputs(data: MarketData, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["skew_policy"]["path"]))
    proof = read_json(ROOT / policy["skew_readiness"]["path"])
    manifest = read_json(ROOT / policy["skew_manifest"]["path"])
    if (proof["policy"] != policy["skew_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["skew_manifest"]
        or proof["full_source_readiness"] is not False or proof["full_monthly_query_readiness"] is not False
        or proof["known_missing_skew_daily_dates"] != list(inputs.MISSING)
        or proof["known_missing_skew_monthly_queries"] != list(PAUSED)
        or proof["actual_monthly_skew_queries_available"] != 130
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["skew_policy"]["sha256"]
        or manifest["source_validator_sha256"] != inputs.fingerprint()
        or manifest["full_source_readiness"] is not False or manifest["full_monthly_query_readiness"] is not False):
        raise QuantError("Use the exact blocked source proof;the new no-request law doesnot repair source availability.")
    for name, digest in manifest["source_files"].items():
        safe_file(ROOT, name, digest)
    histories = {}
    for entry in manifest["histories"]:
        name = entry["series"]
        if name not in inputs.SERIES or name in histories:
            raise QuantError("Only the exact two genuine native source series may enter this tail study.")
        at = pd.Timestamp(entry["acquired_at"])
        if at.tzinfo is None:
            raise QuantError("Preserve the old and new actual acquisition timestamps.")
        last = at.tz_convert("America/New_York").normalize().tz_localize(None)
        if str(last.date()) != entry["acquisition_date_limit"]:
            raise QuantError("Do not retime source histories or change their genuine date limits.")
        histories[name] = inputs.parse_history(safe_file(ROOT, entry["path"], entry["sha256"]), name, last)
    context = inputs.align_context(histories)
    monthly = inputs.monthly_queries(context)
    for name, frame in (("source_context", context), ("monthly_availability", monthly)):
        reference = manifest[name]
        stored = pd.read_csv(safe_file(ROOT, reference["path"], reference["sha256"]),
                             index_col=0, parse_dates=True, float_precision="round_trip")
        if (not stored.index.equals(frame.index) or not stored.columns.equals(frame.columns)
            or not stored.isna().equals(frame.isna())
            or any(not stored[column].dropna().equals(frame[column].dropna()) for column in frame)):
            raise QuantError("Actual native quotes,availability or missing-source reasons differ from the frozen source matrix.")
    if not data.close.index.equals(context.loc["2015-08-10":].index):
        raise QuantError("Retain every original market-context day ratherthan deleting source gaps.")
    return monthly


def build_from_inputs(data: MarketData, monthly: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    original = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_candidate"]]
    months = original.dropna(how="all").index
    columns = ["SKEW", "VIX", "source_available", "source_reason"]
    if (not isinstance(monthly, pd.DataFrame) or not monthly.index.equals(months) or list(monthly.columns) != columns
        or not all(pd.api.types.is_numeric_dtype(monthly[name])
                   and not pd.api.types.is_bool_dtype(monthly[name]) and not pd.api.types.is_complex_dtype(monthly[name])
                   for name in ("SKEW", "VIX"))
        or monthly["source_available"].dtype != bool or np.isinf(monthly[["SKEW", "VIX"]].to_numpy()).any()
        or not np.isfinite(monthly["VIX"]).all() or (monthly["VIX"] <= 0).any()
        or not np.array_equal(monthly["SKEW"].notna(), monthly["source_available"])
        or monthly.index[~monthly["source_available"]].strftime("%Y-%m-%d").tolist() != list(PAUSED)
        or (monthly.loc[monthly["source_available"], "SKEW"] <= 0).any()
        or monthly.loc[monthly["source_available"], "source_reason"].notna().any()
        or monthly.loc[list(PAUSED), "source_reason"].tolist() != ["actual_SKEW_quote_missing_no_safe_default"]):
        raise QuantError("Only the exact predeclared missing month may pause;unknown inputs cannot enter a safe default.")
    results = {}
    for candidate in CANDIDATES:
        target = original.copy()
        assets = list(EQUITY) if candidate["risk_scope"] == "equity_only" else list(target.columns.drop("BIL"))
        for day in months:
            if not monthly.loc[day, "source_available"]:
                target.loc[day] = np.nan
            elif monthly.loc[day, "SKEW"] >= 130:
                freed = float(target.loc[day, assets].sum())*0.5
                target.loc[day, assets] *= 0.5
                target.loc[day, "BIL"] += freed
        active = target.dropna(how="all")
        if (len(active) != len(months)-len(PAUSED) or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any() or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Tail transfer cannot alter original cash funding,short or drop a valid target.")
        results[candidate["id"]] = target
    return results


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_inputs(data, load_inputs(data, policy), policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT / policy["source_registration"]["path"])
    records = [entry for entry in packet["candidates"] if entry["spec"]["id"] == policy["source_candidate"]]
    if len(records) != 1:
        raise QuantError("Require the exact original four-style/semiconductor/gold source registration.")
    original = records[0]["spec"]
    for name, digest in original["frozen_files"].items():
        safe_file(ROOT, name, digest)
    data = review_market({"market":original["market"]}, ROOT)
    audited = verified_etf_market(packet["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Keep every original actual fund and frozen risk-free source observation.")
    monthly = load_inputs(data, policy)
    frames = build_from_inputs(data, monthly, policy)
    baseline = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_candidate"]]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private tail-transfer preparation directory.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", baseline.to_csv(float_format="%.17g"))
    write_text_atomic(output / "monthly-source-availability.csv", monthly.to_csv(float_format="%.17g"))
    manifest = read_json(ROOT / policy["skew_manifest"]["path"])
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[name]["path"]:policy[name]["sha256"] for name in REFERENCES}, **manifest["source_files"],
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py":file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("near_term_volatility_inputs", "dollar_risk_guard", "macro_factor_tilt", "calendar", "storage")},
    }
    source_policy = read_json(ROOT / policy["skew_policy"]["path"])
    additions.update({source_policy[name]["path"]:source_policy[name]["sha256"] for name in inputs.REFERENCES})
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not overwrite a conflicting original frozen source,policy or helper.")
        frozen[name] = digest
    readiness = deepcopy(packet["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_source_monthly_targets":len(baseline.dropna(how="all")), "issued_targets_per_candidate":130,
            "known_no_request_months":list(PAUSED), "source_full_daily_or_monthly_readiness_repaired":False,
            "candidate_returns_or_metrics_computed":False, "new_source_requests":0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed monthly SPXtail-asymmetry response scopes.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/skew-factor-guard-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Tail-scope preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
