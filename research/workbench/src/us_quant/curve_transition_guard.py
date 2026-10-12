from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import yield_curve_inputs as inputs
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs, dollar_comparisons, load_vintages
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/curve-transition-guard.json"
CANDIDATES = (
    {"id":"semiconductor_dollar_t10y3m_resteepening_confirmation", "curve_series":"T10Y3M"},
    {"id":"semiconductor_dollar_t10y2y_resteepening_confirmation", "curve_series":"T10Y2Y"},
)
REFERENCES = ("source_registration", "original_control_policy", "original_control_implementation",
              "dollar_policy", "dollar_implementation", "curve_policy", "curve_implementation",
              "curve_readiness", "curve_manifest")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"published_treasury_inversion_resteepening_confirmation_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":154, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_candidate":"semiconductor_dollar_half_guard",
        "original_control_candidate":"four_factor_semiconductor_sector50", "risk_change_sessions":63,
        "risk_off_investment_scale":0.50, "cash_reserve":0.02, "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both fixed Treasury maturity views,original63session/halfscale and unchanged funding.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def load_curves(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["curve_policy"]["path"]))
    proof = read_json(ROOT / policy["curve_readiness"]["path"])
    manifest_path = ROOT / policy["curve_manifest"]["path"]
    manifest = read_json(manifest_path)
    if (proof["policy"] != policy["curve_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["curve_manifest"]
        or proof["full_source_readiness"] is not True or proof["fresh_query_selections"] != 450
        or proof["blocked_query_selections"] != 0 or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["curve_policy"]["sha256"]
        or manifest["readiness_code_sha256"] != inputs.fingerprint()
        or manifest["all_requested_batches_complete"] is not True or manifest["full_source_readiness"] is not True
        or len(manifest["batches"]) != 12):
        raise QuantError("Use the exact complete Treasury source proof,without repairing or rewriting earlier studies.")
    for name, digest in manifest["source_files"].items():
        safe_file(manifest_path.parent, name, digest)
    vintages = {name:{} for name in inputs.SERIES}
    for batch in manifest["batches"]:
        series, dates = batch["series"], batch["vintage_dates"]
        if series not in vintages:
            raise QuantError("Only the two declared Treasury maturity views may enter this rule.")
        request = read_json(safe_file(manifest_path.parent, batch["request_path"], batch["request_sha256"]))
        if (request["url"] != f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            or request["advertised_form_fields"] != inputs.request_fields(series, dates)
            or request["http_success"] is not True or request["sha256"] != batch["sha256"]
            or set(vintages[series]).intersection(dates)):
            raise QuantError("An original Treasury request changed or duplicated a historical snapshot.")
        frame = inputs.parse_archive(safe_file(manifest_path.parent, batch["path"], batch["sha256"]), series, dates)
        vintages[series].update({day:frame[day].dropna() for day in dates})
    return inputs.query_readiness(index, vintages)


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, dict]:
    selected = load_curves(data.close.index, policy)
    dollars, _ = load_vintages(data.close.index, read_json(ROOT / policy["dollar_policy"]["path"]))
    dollar, dollar_audit = dollar_comparisons(dollars, data.close.index)
    pairs = decision_pairs(data.close.index)
    comparisons = pd.DataFrame({"dollar_change":dollar}, index=dollar.index)
    for series in inputs.SERIES:
        values = selected.loc[selected["series"].eq(series)].set_index("query_session")["spread_percentage_points"]
        comparisons[f"{series}_level"] = [float(values.loc[str(day.date())]) for day in dollar.index]
        comparisons[f"{series}_change"] = [
            float(values.loc[str(day.date())] - values.loc[str(pairs[day].date())]) for day in dollar.index
        ]
    return comparisons, {"curve":selected, "dollar":dollar_audit}


def build_from_inputs(data: MarketData, comparisons: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    original = sector_targets(data, read_json(ROOT / policy["original_control_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    columns = ["dollar_change", *(f"{series}_{suffix}" for series in inputs.SERIES for suffix in ("level", "change"))]
    if (not isinstance(comparisons, pd.DataFrame) or not comparisons.index.equals(original.dropna(how="all").index)
        or list(comparisons.columns) != columns
        or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in comparisons.dtypes)
        or not np.isfinite(comparisons.to_numpy()).all()):
        raise QuantError("Require all original monthly dollar and own-vintage Treasury level/change inputs.")
    outputs = {}
    assets = original.columns.drop("BIL")
    for candidate in CANDIDATES:
        series = candidate["curve_series"]
        targets = original.copy()
        for day in comparisons.index:
            if (comparisons.loc[day, "dollar_change"] > 0 and comparisons.loc[day, f"{series}_level"] < 0
                and comparisons.loc[day, f"{series}_change"] > 0):
                freed = float(targets.loc[day, assets].sum()) * 0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        active = targets.dropna(how="all")
        if (len(active) != len(comparisons) or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any() or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Treasury confirmation cannot omit a month,short,borrow or change funded total.")
        outputs[candidate["id"]] = targets
    return outputs


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, comparisons, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT / policy["source_registration"]["path"])
    records = [row for row in packet["candidates"] if row["spec"]["id"] == policy["source_candidate"]]
    if len(records) != 1:
        raise QuantError("Require the exact original dollar-sector source registration.")
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
            raise QuantError("Preserve every original actual fund and risk-free observation.")
    comparisons, audits = load_inputs(data, policy)
    frames = build_from_inputs(data, comparisons, policy)
    baseline = sector_targets(data, read_json(ROOT / policy["original_control_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private Treasury-rule preparation directory.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", baseline.to_csv(float_format="%.17g"))
    write_text_atomic(output / "risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    for name, frame in audits.items():
        write_text_atomic(output / f"{name}-availability.csv", frame.to_csv(index=False) if name == "curve"
                          else frame.to_csv(float_format="%.17g"))
    manifest_path = ROOT / policy["curve_manifest"]["path"]
    manifest = read_json(manifest_path)
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES},
        **{(manifest_path.parent / name).relative_to(ROOT).as_posix():digest
           for name, digest in manifest["source_files"].items()},
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py":file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("inflation_inputs", "financial_conditions_guard", "calendar", "storage")},
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not rewrite any conflicting original frozen source or helper.")
        frozen[name] = digest
    readiness = deepcopy(packet["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_monthly_decisions":len(comparisons), "complete_curve_queries":len(audits["curve"]),
            "issued_monthly_targets_per_candidate":len(next(iter(frames.values())).dropna(how="all")),
            "new_source_requests":0, "candidate_returns_or_metrics_computed":False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed own-vintage Treasury transition confirmations.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/curve-transition-guard-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Treasury transition preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
