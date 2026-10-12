from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import implied_risk_execution as parent
from us_quant import unemployment_inputs as inputs
from us_quant.activity_inputs import select as select_monthly
from us_quant.backtest import BacktestResult
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs
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
POLICY = ROOT / "config/unemployment-factor-guard.json"
CANDIDATES = (
    {"id": "semiconductor_option_unemployment_rise_half", "dollar_confirmation": False},
    {"id": "semiconductor_option_dollar_unemployment_rise_half", "dollar_confirmation": True},
)
REFERENCES = (
    "source_registration", "source_policy", "source_implementation", "unemployment_policy",
    "unemployment_implementation", "unemployment_readiness", "unemployment_manifest",
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "own_vintage_unemployment_change_confirmations_20261012",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 170, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_option_proxy_observed_band025",
        "required_unemployment_observations_per_query": "latest_actual_completed_month_only",
        "risk_change_sessions": 63, "risk_off_scale": 0.5, "actual_weight_trade_band": 0.025,
        "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False
    ):
        raise QuantError("Preserve both latest-only unemployment rules,own63query differences,halfscale and original band.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def select_latest(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (
        not isinstance(observed, pd.Series) or not pd.api.types.is_numeric_dtype(observed)
        or pd.api.types.is_bool_dtype(observed) or pd.api.types.is_complex_dtype(observed)
        or ((observed.dropna() < 0) | (observed.dropna() > 100)).any()
    ):
        raise QuantError("Use actual latest U3percent values,not Sahm states or invalid substituted levels.")
    value, audit = select_monthly(observed, query, cutoff)
    return value, {**audit, "consumed_history_scope": "latest_actual_completed_month_only",
                   "rolling_history_source_readiness_repaired": False}


def load_latest(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["unemployment_policy"]["path"]))
    proof = read_json(ROOT / policy["unemployment_readiness"]["path"])
    manifest_path = ROOT / policy["unemployment_manifest"]["path"]
    manifest = read_json(manifest_path)
    if (
        proof["policy"] != policy["unemployment_policy"]
        or proof["private_source_manifest"] != policy["unemployment_manifest"]
        or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["full_source_readiness"] is not False or proof["blocked_queries"] != 15
        or proof["fresh_complete_queries"] != 210 or proof["maximum_actual_completed_month_age_days"] != 61
        or proof["Sahm_rolling_gap_threshold_recession_label_or_portfolio_computed"] is not False
        or manifest["policy_sha256"] != policy["unemployment_policy"]["sha256"]
        or manifest["source_validator_sha256"] != inputs.fingerprint()
        or manifest["full_source_readiness"] is not False
        or manifest["all_requested_batches_complete"] is not True or len(manifest["batches"]) != 6
        or any(row["missing_history_months"] != ["2025-10-01"] for row in proof["blocked_query_metadata"])
    ):
        raise QuantError("Retain the sealed15month-history gap proof;the new latest-only rule does not repair it.")
    for name, digest in manifest["source_files"].items():
        safe_file(manifest_path.parent, name, digest)
    vintages = {}
    for row in manifest["batches"]:
        dates = row["vintage_dates"]
        request = read_json(safe_file(manifest_path.parent, row["request_path"], row["request_sha256"]))
        if (
            request["url"] != inputs.PRIMARY_DEFINITIONS["UNRATE_form"] or request["method"] != "POST"
            or request["advertised_form_fields"] != inputs.request_fields(dates)
            or request["http_success"] is not True or request["sha256"] != row["sha256"]
            or set(vintages).intersection(dates)
        ):
            raise QuantError("Keep exact original UNRATE requests and snapshots without fallback or duplicates.")
        frame = inputs.parse_archive(safe_file(manifest_path.parent, row["path"], row["sha256"]), dates)
        vintages.update({cutoff: frame[cutoff].dropna() for cutoff in dates})
    queries, dates = inputs.cohort(index)
    if set(vintages) != set(dates):
        raise QuantError("All225original latest-observation queries are required without omissions.")
    context = manifest["private_verified_queries"]
    stored = pd.read_csv(safe_file(ROOT, context["path"], context["sha256"]), float_precision="round_trip")
    records = []
    for query in queries:
        cutoff = str(previous_session(previous_session(query)).date())
        value, audit = select_latest(vintages[cutoff], query, cutoff)
        records.append({**audit, "unemployment_percent": value})
    latest = pd.DataFrame(records).set_index("query_session")
    stored = stored.set_index("query_session")
    fields = ["vintage_cutoff", "observation_month", "completed_period_end", "completed_period_age_calendar_days"]
    if (
        stored.index.tolist() != latest.index.tolist()
        or stored[fields].to_dict("records") != latest[fields].to_dict("records")
        or not np.array_equal(stored["unemployment_percent"], latest["unemployment_percent"])
    ):
        raise QuantError("Every latest unemployment value,month,age and earlier snapshot must match the sealed proof.")
    return latest


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = load_latest(data.close.index, policy)
    execution_policy = read_json(ROOT / policy["source_policy"]["path"])
    allocation_policy = read_json(ROOT / execution_policy["source_policy"]["path"])
    _, dollar, _ = parent.source.load_inputs(data, allocation_policy)
    pairs = decision_pairs(data.close.index)
    comparisons = pd.DataFrame(index=dollar.index)
    comparisons["unemployment_level"] = [float(latest.loc[str(day.date()), "unemployment_percent"]) for day in dollar.index]
    comparisons["unemployment_reference_level"] = [
        float(latest.loc[str(pairs[day].date()), "unemployment_percent"]) for day in dollar.index
    ]
    comparisons["unemployment_change"] = comparisons["unemployment_level"] - comparisons["unemployment_reference_level"]
    comparisons["dollar_change"] = dollar
    return comparisons, latest


def build_from_inputs(
    data: MarketData, original: pd.DataFrame, comparisons: pd.DataFrame, policy: dict,
) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    fields = ["unemployment_level", "unemployment_reference_level", "unemployment_change", "dollar_change"]
    if (
        not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns)
        or not isinstance(comparisons, pd.DataFrame)
        or not comparisons.index.equals(original.dropna(how="all").index)
        or list(comparisons.columns) != fields
        or not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   and not pd.api.types.is_complex_dtype(dtype) for dtype in comparisons.dtypes)
        or not np.isfinite(comparisons.to_numpy()).all()
        or (comparisons[fields[:2]] < 0).any().any() or (comparisons[fields[:2]] > 100).any().any()
        or not np.array_equal(comparisons["unemployment_change"],
                              comparisons["unemployment_level"] - comparisons["unemployment_reference_level"])
    ):
        raise QuantError("Require all original monthly latest-only comparisons with actual U3levels and own-query changes.")
    active = original.dropna(how="all")
    if (
        len(active) != 131 or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("All131original funded source updates must remain;no unemployment-history pause/default.")
    outputs = {}
    assets = original.columns.drop("BIL")
    for candidate in CANDIDATES:
        targets = original.copy()
        risk = comparisons["unemployment_change"].gt(0)
        if candidate["dollar_confirmation"]:
            risk &= comparisons["dollar_change"].gt(0)
        selected = comparisons.index[risk]
        freed = targets.loc[selected, assets].sum(axis=1) * 0.5
        targets.loc[selected, assets] *= 0.5
        targets.loc[selected, "BIL"] += freed
        if (
            not targets.isna().equals(original.isna())
            or not np.allclose(targets.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Unemployment confirmation cannot change source calendar or original funding.")
        outputs[candidate["id"]] = targets
    return outputs


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    original = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, original, comparisons, policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use only one of the two frozen unemployment-change configurations.")
    return parent.run(data, desired, parent.CANDIDATES[0], read_json(ROOT / policy["source_policy"]["path"]),
                      start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT / policy["source_registration"]["path"])
    entries = [entry for entry in packet["candidates"] if entry["spec"]["id"] == policy["source_id"]]
    if len(entries) != 1:
        raise QuantError("Require the exact plain original option-budget observed-band source registration.")
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
        if (
            not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)
        ):
            raise QuantError("Preserve all original actual fund quotes and frozen risk-free values.")
    original = parent.source_baselines(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_id"]]
    comparisons, latest = load_inputs(data, policy)
    desired = build_from_inputs(data, original, comparisons, policy)
    manifest_path = ROOT / policy["unemployment_manifest"]["path"]
    manifest = read_json(manifest_path)
    source_policy = read_json(ROOT / policy["unemployment_policy"]["path"])
    frozen = dict(spec["frozen_files"])
    additions = {
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        **{(manifest_path.parent / name).relative_to(ROOT).as_posix(): digest for name, digest in manifest["source_files"].items()},
        **{source_policy[key]["path"]: source_policy[key]["sha256"]
           for key in ("source_index", "research_snapshot", "previous_preservation")},
        manifest["private_verified_queries"]["path"]: manifest["private_verified_queries"]["sha256"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py": file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("activity_inputs", "yield_curve_inputs", "inflation_inputs", "financial_conditions_guard",
                        "dollar_risk_guard", "macro_factor_tilt", "calendar", "storage")},
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Never overwrite a conflicting frozen allocation/execution or source dependency.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private latest-unemployment preparation directory.")
    new_output_directory(output)
    for name, frame in (
        ("original-source-desired.csv", original), ("latest-unemployment-queries.csv", latest),
        ("risk-comparisons.csv", comparisons),
    ):
        write_text_atomic(output / name, frame.to_csv(float_format="%.17g"))
    readiness = deepcopy(readiness)
    readiness.update(checked_at=utc_now(), revalidation_not_new_market_observation=True)
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        actual = {**spec, "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {"latest_observation_queries": len(latest), "original_monthly_targets_per_candidate": 131,
            "candidate_returns_or_metrics_computed": False, "actual_feedback_accounts_computed": False,
            "rolling15month_source_repaired": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed latest-unemployment changes without portfolio accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/unemployment-factor-guard-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Unemployment preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
