from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import implied_risk_execution as parent
from us_quant import profit_inputs as inputs
from us_quant.backtest import BacktestResult
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/"config/profit-factor-guard.json"
CANDIDATES = (
    {"id": "semiconductor_option_dollar_profit_level_confirmation", "profit_measure": "level"},
    {"id": "semiconductor_option_dollar_profit_share_confirmation", "profit_measure": "output_share"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "profit_policy",
              "profit_implementation", "profit_readiness", "profit_manifest")


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "common_quarter_profit_dollar_confirmations_20261012",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 166, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_option_proxy_observed_band025",
        "source_allocation_id": "semiconductor_option_proxy_risk_balance",
        "profit_query_join": "latest_common_completed_quarter_in_both_exact_same_query_snapshots",
        "maximum_completed_quarter_age_days": 186, "risk_change_sessions": 63, "risk_off_scale": 0.50,
        "actual_weight_trade_band": 0.025, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Preserve both fixed common-quarter profit confirmations,186age/63horizon/halfscale/old band.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def select_common(
    profit: pd.Series, output: pd.Series, query: pd.Timestamp, cutoff: str,
) -> tuple[float, float, dict]:
    _, profit_audit = inputs.select(profit, query, cutoff)
    _, output_audit = inputs.select(output, query, cutoff)
    if (output.dropna() <= 0).any():
        raise QuantError("Common-quarter nominal output must remain finite and strictly positive.")
    common = profit.dropna().index.intersection(output.dropna().index)
    if common.empty:
        raise QuantError("No actual common reported quarter exists;do not mix latest periods or fill the join.")
    value, audit = inputs.select(profit.reindex(common), query, cutoff)
    quarter = pd.Timestamp(audit["observation_quarter"])
    gdp = float(output.loc[quarter])
    share = value/gdp
    if not np.isfinite(share):
        raise QuantError("The macro income/output proxy must remain finite,not an overflow default.")
    return value, gdp, {
        **audit, "independently_latest_profit_quarter": profit_audit["observation_quarter"],
        "independently_latest_output_quarter": output_audit["observation_quarter"],
        "same_quarter_selected_in_both_original_query_snapshots": True,
        "macro_proxy_claimed_company_margin": False,
    }


def load_common(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT/policy["profit_policy"]["path"]))
    proof = read_json(ROOT/policy["profit_readiness"]["path"])
    manifest_path = ROOT/policy["profit_manifest"]["path"]
    manifest = read_json(manifest_path)
    if (proof["policy"] != policy["profit_policy"] or proof["private_source_manifest"] != policy["profit_manifest"]
        or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["full_source_readiness"] is not True or proof["fresh_query_selections"] != 450
        or proof["blocked_query_selections"] != 0 or proof["queries_with_different_latest_profit_output_quarters"] != 87
        or proof["source_ratios_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["profit_policy"]["sha256"]
        or manifest["readiness_code_sha256"] != inputs.fingerprint()
        or manifest["all_requested_batches_complete"] is not True or manifest["full_source_readiness"] is not True
        or len(manifest["batches"]) != 12):
        raise QuantError("Use the exact complete quarterly input proof without relabelling its original87period differences.")
    for name, digest in manifest["source_files"].items():
        safe_file(manifest_path.parent, name, digest)
    snapshots = {name: {} for name in inputs.SERIES}
    for batch in manifest["batches"]:
        series, dates = batch["series"], batch["vintage_dates"]
        request = read_json(safe_file(manifest_path.parent, batch["request_path"], batch["request_sha256"]))
        if (series not in snapshots or request["url"] != f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            or request["method"] != "POST" or request["advertised_form_fields"] != inputs.request_fields(series, dates)
            or request["http_success"] is not True or request["sha256"] != batch["sha256"]
            or set(snapshots[series]).intersection(dates)):
            raise QuantError("A frozen quarterly source request changed fields,status or duplicated a snapshot.")
        parsed = inputs.parse_archive(safe_file(manifest_path.parent, batch["path"], batch["sha256"]), series, dates)
        snapshots[series].update({day: parsed[day].dropna() for day in dates})
    inputs.query_readiness(index, snapshots)
    context = proof["private_verified_query_matrix"]
    if manifest["source_context"] != context:
        raise QuantError("Preserve the original complete quarterly query matrix reference.")
    safe_file(ROOT, context["path"], context["sha256"])
    queries, _ = inputs.cohort(index)
    records = []
    for query in queries:
        cutoff = str(previous_session(previous_session(query)).date())
        profit, output, audit = select_common(snapshots["CPATAX"][cutoff], snapshots["GDP"][cutoff], query, cutoff)
        records.append({**audit, "CPATAX": profit, "GDP": output, "profit_share": profit/output})
    return pd.DataFrame(records).set_index("query_session")


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = load_common(data.close.index, policy)
    original_parent = read_json(ROOT/policy["source_policy"]["path"])
    allocation_policy = read_json(ROOT/original_parent["source_policy"]["path"])
    _, dollar, _ = parent.source.load_inputs(data, allocation_policy)
    pairs = decision_pairs(data.close.index)
    comparisons = pd.DataFrame({"dollar_change": dollar}, index=dollar.index)
    for column in ("CPATAX", "profit_share"):
        values = common[column]
        comparisons[f"{column}_level"] = [float(values.loc[str(day.date())]) for day in dollar.index]
        comparisons[f"{column}_reference_level"] = [float(values.loc[str(pairs[day].date())]) for day in dollar.index]
        comparisons[f"{column}_change"] = comparisons[f"{column}_level"]-comparisons[f"{column}_reference_level"]
    return comparisons, common


def build_from_inputs(
    data: MarketData, original: pd.DataFrame, comparisons: pd.DataFrame, policy: dict,
) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    columns = ["dollar_change", *(f"{name}_{suffix}" for name in ("CPATAX", "profit_share")
                                  for suffix in ("level", "reference_level", "change"))]
    if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
        or not original.columns.equals(data.close.columns)
        or not isinstance(comparisons, pd.DataFrame)
        or not comparisons.index.equals(original.dropna(how="all").index) or list(comparisons.columns) != columns
        or not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   and not pd.api.types.is_complex_dtype(dtype) for dtype in comparisons.dtypes)
        or not np.isfinite(comparisons.to_numpy()).all()):
        raise QuantError("Require every original monthly source target and complete own-query common-quarter comparisons.")
    for name in ("CPATAX", "profit_share"):
        if not np.array_equal(comparisons[f"{name}_change"],
                              comparisons[f"{name}_level"]-comparisons[f"{name}_reference_level"]):
            raise QuantError("A common-quarter comparison cannot use a change unrelated to its actual own-query levels.")
    active = original.dropna(how="all")
    if (len(active) != 131 or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Keep all131fully funded original plain option-budget updates.")
    outputs = {}
    assets = original.columns.drop("BIL")
    for candidate in CANDIDATES:
        column = "CPATAX" if candidate["profit_measure"] == "level" else "profit_share"
        targets = original.copy()
        for day in comparisons.index:
            if comparisons.loc[day, "dollar_change"] > 0 and comparisons.loc[day, f"{column}_change"] < 0:
                freed = float(targets.loc[day, assets].sum())*0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        if (not targets.isna().equals(original.isna())
            or not np.allclose(targets.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Earnings confirmation cannot change the calendar or original cash funding.")
        outputs[candidate["id"]] = targets
    return outputs


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    original = parent.source_baselines(data, read_json(ROOT/policy["source_policy"]["path"]))[policy["source_id"]]
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, original, comparisons, policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use one of the two exact frozen earnings-confirmation configurations.")
    return parent.run(data, desired, parent.CANDIDATES[0], read_json(ROOT/policy["source_policy"]["path"]),
                      start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT/policy["source_registration"]["path"])
    entries = [entry for entry in packet["candidates"] if entry["spec"]["id"] == policy["source_id"]]
    if len(entries) != 1:
        raise QuantError("Require the exact original plain option-budget observed-band source registration.")
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
            raise QuantError("Preserve every original actual fund and frozen risk-free quote.")
    original = parent.source_baselines(data, read_json(ROOT/policy["source_policy"]["path"]))[policy["source_id"]]
    comparisons, common = load_inputs(data, policy)
    desired = build_from_inputs(data, original, comparisons, policy)
    manifest_path = ROOT/policy["profit_manifest"]["path"]
    manifest = read_json(manifest_path)
    source_policy = read_json(ROOT/policy["profit_policy"]["path"])
    frozen = dict(spec["frozen_files"])
    additions = {
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        **{(manifest_path.parent/name).relative_to(ROOT).as_posix(): digest for name, digest in manifest["source_files"].items()},
        **{source_policy[key]["path"]: source_policy[key]["sha256"]
           for key in ("source_index", "research_snapshot", "previous_preservation")},
        manifest["source_context"]["path"]: manifest["source_context"]["sha256"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py": file_digest(ROOT/f"src/us_quant/{name}.py")
           for name in ("activity_inputs", "yield_curve_inputs", "inflation_inputs", "financial_conditions_guard",
                        "dollar_risk_guard", "macro_factor_tilt", "calendar", "storage")},
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("A conflicting frozen allocation/execution/quarterly source hash cannot be overwritten.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    output = local_path(ROOT, output)
    if output == ROOT/"data" or not output.is_relative_to(ROOT/"data"):
        raise QuantError("Use a new isolated private earnings-confirmation directory without accounts.")
    new_output_directory(output)
    write_text_atomic(output/"original-source-desired.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output/"common-quarter-queries.csv", common.to_csv(float_format="%.17g"))
    write_text_atomic(output/"risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output/"readiness.json", readiness)
    for candidate in CANDIDATES:
        actual = {**spec, "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output/f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output/f"{candidate['id']}-desired-targets.csv",
                          desired[candidate["id"]].to_csv(float_format="%.17g"))
    return {"actual_common_quarter_queries": len(common), "unchanged_desired_monthly_calendar_per_candidate": 131,
            "actual_path_dependent_feedback_targets_or_accounts_computed": False,
            "candidate_returns_or_metrics_computed": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed common-quarter earnings confirmations without accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/profit-factor-guard-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Earnings confirmation preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
