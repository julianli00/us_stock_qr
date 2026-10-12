from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import housing_inputs as inputs
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/housing-factor-guard.json"
CANDIDATES = (
    {"id": "semiconductor_housing_permit_decline_half", "housing_confirmation": "permit_decline"},
    {"id": "semiconductor_housing_permit_starts_decline_half", "housing_confirmation": "permit_and_starts_decline"},
)
KNOWN_STALE = (
    {"query_session": "2025-11-25", "vintage_cutoff": "2025-11-21", "observation_month": "2025-08-01",
     "completed_period_end": "2025-08-31", "completed_period_age_calendar_days": 86},
    {"query_session": "2025-11-28", "vintage_cutoff": "2025-11-25", "observation_month": "2025-08-01",
     "completed_period_end": "2025-08-31", "completed_period_age_calendar_days": 89},
    {"query_session": "2025-12-29", "vintage_cutoff": "2025-12-24", "observation_month": "2025-08-01",
     "completed_period_end": "2025-08-31", "completed_period_age_calendar_days": 120},
    {"query_session": "2025-12-31", "vintage_cutoff": "2025-12-29", "observation_month": "2025-08-01",
     "completed_period_end": "2025-08-31", "completed_period_age_calendar_days": 122},
    {"query_session": "2026-01-29", "vintage_cutoff": "2026-01-27", "observation_month": "2025-10-01",
     "completed_period_end": "2025-10-31", "completed_period_age_calendar_days": 90},
    {"query_session": "2026-01-30", "vintage_cutoff": "2026-01-28", "observation_month": "2025-10-01",
     "completed_period_end": "2025-10-31", "completed_period_age_calendar_days": 91},
    {"query_session": "2026-04-30", "vintage_cutoff": "2026-04-28", "observation_month": "2026-01-01",
     "completed_period_end": "2026-01-31", "completed_period_age_calendar_days": 89},
)
PAUSED = ("2025-11-28", "2025-12-31", "2026-01-30", "2026-02-27",
          "2026-03-31", "2026-04-30", "2026-07-31")
REFERENCES = ("source_registration", "source_policy", "source_implementation", "housing_policy",
              "housing_implementation", "housing_readiness", "housing_manifest")
PAUSE_REASON = "predeclared_housing_period_exceeds_unchanged62day_age"


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "published_housing_pipeline_slowdown_confirmations_20261012",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 162, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_candidate": "four_factor_semiconductor_sector50",
        "risk_change_sessions": 63, "risk_off_scale": 0.50, "cash_reserve": 0.02,
        "candidates": list(CANDIDATES), "known_stale_queries": list(KNOWN_STALE),
        "known_paused_months": list(PAUSED),
        "missing_data_rule": "only_exact_predeclared_stale_query_pairs_pause_original_monthly_request_unknown_gap_fatal",
        "strict_source_readiness_relabelled": False,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Preserve both fixed housing-stage rules,63session comparison,halfscale and exact source pauses.")


def references(policy: dict) -> None:
    validate(policy)
    for name in REFERENCES:
        safe_file(ROOT, policy[name]["path"], policy[name]["sha256"])


def select_queries(index: pd.DatetimeIndex, vintages: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    queries, dates = inputs.cohort(index)
    if set(vintages) != set(inputs.SERIES) or any(set(values) != set(dates) for values in vintages.values()):
        raise QuantError("Housing rules require every original450source snapshot without substitutions.")
    rows = []
    for series in inputs.SERIES:
        for query in queries:
            cutoff = str(previous_session(previous_session(query)).date())
            observed = vintages[series][cutoff]
            try:
                value, audit = inputs.select(observed, query, cutoff)
                row = {"series": series, **audit, "thousands_of_units_saar": value,
                       "source_available": True, "source_reason": None}
            except QuantError as exc:
                if (str(exc) != "Monthly completed-period freshness exceeds the frozen62day source contract."
                    or observed.dropna().empty):
                    raise
                month = observed.dropna().index[-1]
                completed = month+pd.offsets.MonthEnd(0)
                audit = {
                    "query_session": str(query.date()), "vintage_cutoff": cutoff,
                    "observation_month": str(month.date()), "completed_period_end": str(completed.date()),
                    "completed_period_age_calendar_days": int((query-completed).days),
                }
                if audit not in KNOWN_STALE:
                    raise QuantError("An unknown stale housing query cannot enter the predeclared no-request law.") from exc
                row = {"series": series, **audit, "thousands_of_units_saar": np.nan,
                       "source_available": False, "source_reason": PAUSE_REASON}
            rows.append(row)
    selected = pd.DataFrame(rows)
    keys = list(KNOWN_STALE[0])
    for series in inputs.SERIES:
        actual = selected.loc[selected["series"].eq(series) & ~selected["source_available"], keys].to_dict("records")
        if actual != list(KNOWN_STALE):
            raise QuantError("The exact known housing gaps changed;do not absorb or remove them silently.")
    return selected


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["housing_policy"]["path"]))
    proof = read_json(ROOT / policy["housing_readiness"]["path"])
    manifest_path = ROOT / policy["housing_manifest"]["path"]
    manifest = read_json(manifest_path)
    if (proof["policy"] != policy["housing_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["housing_manifest"]
        or proof["full_source_readiness"] is not False or proof["blocked_query_selections"] != 14
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["housing_policy"]["sha256"]
        or manifest["readiness_code_sha256"] != inputs.fingerprint()
        or manifest["all_requested_batches_complete"] is not True or manifest["full_source_readiness"] is not False
        or len(manifest["batches"]) != 12):
        raise QuantError("Use the exact strictly blocked housing proof;the separate pause rule does not repair it.")
    for name, digest in manifest["source_files"].items():
        safe_file(manifest_path.parent, name, digest)
    vintages = {name: {} for name in inputs.SERIES}
    for batch in manifest["batches"]:
        series, dates = batch["series"], batch["vintage_dates"]
        if series not in vintages:
            raise QuantError("Only the two frozen native housing stages may enter this rule.")
        request = read_json(safe_file(manifest_path.parent, batch["request_path"], batch["request_sha256"]))
        if (request["url"] != f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            or request["method"] != "POST" or request["advertised_form_fields"] != inputs.request_fields(series, dates)
            or request["http_success"] is not True or request["sha256"] != batch["sha256"]
            or set(vintages[series]).intersection(dates)):
            raise QuantError("An original housing request changed fields,status or duplicated historical snapshots.")
        frame = inputs.parse_archive(safe_file(manifest_path.parent, batch["path"], batch["sha256"]), series, dates)
        vintages[series].update({day: frame[day].dropna() for day in dates})
    selected = select_queries(data.close.index, vintages)
    context = proof["private_verified_query_matrix"]
    if manifest["source_context"] != context:
        raise QuantError("The original complete housing query matrix must remain bound to its source proof.")
    stored = pd.read_csv(safe_file(ROOT, context["path"], context["sha256"]), float_precision="round_trip")
    metadata = ["series", *KNOWN_STALE[0]]
    if (stored[metadata].to_dict("records") != selected[metadata].to_dict("records")
        or not np.array_equal(stored["status"].eq("verified_fresh_source"), selected["source_available"])
        or not np.array_equal(stored.loc[selected["source_available"], "thousands_of_units_saar"],
                              selected.loc[selected["source_available"], "thousands_of_units_saar"])):
        raise QuantError("Raw housing selections,period ages or actual source availability differ from the sealed450query matrix.")
    pairs = decision_pairs(data.close.index)
    comparisons = pd.DataFrame(index=pd.DatetimeIndex(pairs))
    for series in inputs.SERIES:
        values = selected.loc[selected["series"].eq(series)].set_index("query_session")["thousands_of_units_saar"]
        levels = pd.Series([float(values.loc[str(day.date())]) for day in comparisons.index], index=comparisons.index)
        prior = pd.Series([float(values.loc[str(pairs[day].date())]) for day in comparisons.index], index=comparisons.index)
        available = np.isfinite(levels) & np.isfinite(prior)
        comparisons[f"{series}_level"] = levels
        comparisons[f"{series}_reference_level"] = prior
        comparisons[f"{series}_change"] = levels-prior
        comparisons[f"{series}_available"] = available
        comparisons[f"{series}_source_reason"] = np.where(available, None, PAUSE_REASON)
    return comparisons, selected


def build_from_inputs(data: MarketData, comparisons: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    original = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_candidate"]]
    months = original.dropna(how="all").index
    columns = [f"{name}_{suffix}" for name in inputs.SERIES
               for suffix in ("level", "reference_level", "change", "available", "source_reason")]
    if not isinstance(comparisons, pd.DataFrame) or not comparisons.index.equals(months) or list(comparisons.columns) != columns:
        raise QuantError("Require every original monthly pair and both declared own-vintage housing stages.")
    for series in inputs.SERIES:
        level, prior, change = (comparisons[f"{series}_{suffix}"] for suffix in ("level", "reference_level", "change"))
        available, reason = comparisons[f"{series}_available"], comparisons[f"{series}_source_reason"]
        if (not all(pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
                    and not pd.api.types.is_complex_dtype(values) for values in (level, prior, change))
            or available.dtype != bool or np.isinf(comparisons[[f"{series}_{suffix}" for suffix in
                                                              ("level", "reference_level", "change")]].to_numpy()).any()
            or (level.dropna() < 0).any() or (prior.dropna() < 0).any()
            or not np.array_equal(np.isfinite(level) & np.isfinite(prior) & np.isfinite(change), available)
            or not np.array_equal(change.loc[available], (level-prior).loc[available])
            or months[~available].strftime("%Y-%m-%d").tolist() != list(PAUSED)
            or reason.loc[available].notna().any()
            or reason.loc[~available].tolist() != [PAUSE_REASON]*len(PAUSED)):
            raise QuantError("Only the exact seven known housing-pair pauses are allowed;unknown or changed data is fatal.")
    results = {}
    assets = original.columns.drop("BIL")
    for candidate in CANDIDATES:
        targets = original.copy()
        for day in months:
            if not comparisons.loc[day, "PERMIT_available"] or not comparisons.loc[day, "HOUST_available"]:
                targets.loc[day] = np.nan
                continue
            decline = comparisons.loc[day, "PERMIT_change"] < 0
            if candidate["housing_confirmation"] == "permit_and_starts_decline":
                decline = decline and comparisons.loc[day, "HOUST_change"] < 0
            if decline:
                freed = float(targets.loc[day, assets].sum())*0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        active = targets.dropna(how="all")
        if (len(active) != len(months)-len(PAUSED) or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any() or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Housing confirmation cannot alter funding,short,borrow or delete a valid target.")
        results[candidate["id"]] = targets
    return results


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, comparisons, policy)


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
    data = review_market({"market": original["market"]}, ROOT)
    audited = verified_etf_market(packet["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Preserve every original actual fund and frozen risk-free source observation.")
    comparisons, selected = load_inputs(data, policy)
    frames = build_from_inputs(data, comparisons, policy)
    baseline = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[policy["source_candidate"]]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private housing-rule preparation directory.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", baseline.to_csv(float_format="%.17g"))
    write_text_atomic(output / "risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    write_text_atomic(output / "housing-availability.csv", selected.to_csv(index=False, float_format="%.17g"))
    write_json(output / "source-pauses.json", {
        "candidate_pause_decisions": {candidate["id"]: list(PAUSED) for candidate in CANDIDATES},
        "known_stale_queries": list(KNOWN_STALE), "pause_means_no_request_not_cash_or_zero_return": True,
        "every_formal_return_session_retained": True, "strict_source_full_readiness_unchanged_false": True,
    })
    manifest_path = ROOT / policy["housing_manifest"]["path"]
    manifest = read_json(manifest_path)
    source_policy = read_json(ROOT / policy["housing_policy"]["path"])
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        **{(manifest_path.parent / name).relative_to(ROOT).as_posix(): digest
           for name, digest in manifest["source_files"].items()},
        **{source_policy[key]["path"]: source_policy[key]["sha256"]
           for key in ("source_index", "research_snapshot", "previous_preservation")},
        manifest["source_context"]["path"]: manifest["source_context"]["sha256"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py": file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("activity_inputs", "yield_curve_inputs", "inflation_inputs", "financial_conditions_guard",
                        "dollar_risk_guard", "macro_factor_tilt", "calendar", "storage")},
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not overwrite a conflicting original frozen source,policy or helper.")
        frozen[name] = digest
    readiness = deepcopy(packet["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_source_monthly_targets": len(baseline.dropna(how="all")),
            "issued_targets_per_candidate": 124, "known_no_request_months": list(PAUSED),
            "strict_housing_source_readiness_repaired": False, "candidate_returns_or_metrics_computed": False,
            "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed own-vintage housing pipeline confirmations.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/housing-factor-guard-20261012")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Housing confirmation preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
