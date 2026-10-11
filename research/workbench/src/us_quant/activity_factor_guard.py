from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import activity_inputs as inputs
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs, dollar_comparisons, load_vintages
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/activity-factor-guard.json"
CANDIDATES = (
    {"id":"semiconductor_dollar_activity_magnitude_confirmation", "activity_series":"CFNAIMA3", "low_activity_threshold":-0.70},
    {"id":"semiconductor_dollar_activity_diffusion_confirmation", "activity_series":"CFNAIDIFF", "low_activity_threshold":-0.35},
)
KNOWN_STALE = (
    {"query_session":"2023-01-27", "vintage_cutoff":"2023-01-25", "observation_month":"2022-10-01",
     "completed_period_end":"2022-10-31", "completed_period_age_calendar_days":88},
    {"query_session":"2025-11-25", "vintage_cutoff":"2025-11-21", "observation_month":"2025-08-01",
     "completed_period_end":"2025-08-31", "completed_period_age_calendar_days":86},
    {"query_session":"2025-11-28", "vintage_cutoff":"2025-11-25", "observation_month":"2025-08-01",
     "completed_period_end":"2025-08-31", "completed_period_age_calendar_days":89},
    {"query_session":"2025-12-29", "vintage_cutoff":"2025-12-24", "observation_month":"2025-09-01",
     "completed_period_end":"2025-09-30", "completed_period_age_calendar_days":90},
    {"query_session":"2025-12-31", "vintage_cutoff":"2025-12-29", "observation_month":"2025-09-01",
     "completed_period_end":"2025-09-30", "completed_period_age_calendar_days":92},
)
PAUSED = ("2023-04-28", "2025-11-28", "2025-12-31", "2026-02-27", "2026-03-31")
REFERENCES = ("source_registration", "original_control_policy", "original_control_implementation",
              "dollar_policy", "dollar_implementation", "activity_policy", "activity_implementation",
              "activity_readiness", "activity_manifest")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"published_real_activity_dollar_confirmation_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":150, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_candidate":"semiconductor_dollar_half_guard",
        "original_control_candidate":"four_factor_semiconductor_sector50", "risk_change_sessions":63,
        "risk_off_investment_scale":0.50, "cash_reserve":0.02, "candidates":list(CANDIDATES),
        "known_stale_queries":list(KNOWN_STALE), "known_paused_months":list(PAUSED),
        "missing_data_rule":"only_exact_predeclared_stale_query_pairs_pause_original_monthly_rebalance_unknown_gap_fatal",
        "original_source_full_readiness_relabelled":False,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both fixed activity confirmations,known-pair pauses,original source and unchanged funding.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def select_queries(index: pd.DatetimeIndex, vintages: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    queries, dates = inputs.cohort(index)
    if (set(vintages) != set(inputs.SERIES)
        or any(set(vintages[name]) != set(dates) for name in inputs.SERIES)):
        raise QuantError("Activity rules require every original450source snapshot without substitutions.")
    rows = []
    for series in inputs.SERIES:
        for query in queries:
            cutoff = str(previous_session(previous_session(query)).date())
            observed = vintages[series][cutoff]
            try:
                value, audit = inputs.select(observed, query, cutoff)
                row = {"series":series, **audit, "index_value":value, "source_available":True, "source_reason":None}
            except QuantError as exc:
                if (str(exc) != "Monthly completed-period freshness exceeds the frozen62day source contract."
                    or observed.dropna().empty):
                    raise
                month = observed.dropna().index[-1]
                completed = month + pd.offsets.MonthEnd(0)
                audit = {
                    "query_session":str(query.date()), "vintage_cutoff":cutoff,
                    "observation_month":str(month.date()), "completed_period_end":str(completed.date()),
                    "completed_period_age_calendar_days":int((query-completed).days),
                }
                if audit not in KNOWN_STALE:
                    raise QuantError("An unknown stale monthly source cannot enter the predeclared standby rule.") from exc
                row = {"series":series, **audit, "index_value":np.nan, "source_available":False,
                       "source_reason":"predeclared_activity_period_exceeds_unchanged62day_age"}
            rows.append(row)
    selected = pd.DataFrame(rows)
    keys = list(KNOWN_STALE[0])
    for series in inputs.SERIES:
        actual = selected.loc[selected["series"].eq(series) & ~selected["source_available"], keys].to_dict("records")
        if actual != list(KNOWN_STALE):
            raise QuantError("The exact published source gaps changed;do not absorb or remove them silently.")
    return selected


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, dict]:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["activity_policy"]["path"]))
    proof = read_json(ROOT / policy["activity_readiness"]["path"])
    manifest_path = ROOT / policy["activity_manifest"]["path"]
    manifest = read_json(manifest_path)
    if (proof["policy"] != policy["activity_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["activity_manifest"]
        or proof["full_source_readiness"] is not False or proof["blocked_query_selections"] != 10
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["activity_policy"]["sha256"]
        or manifest["readiness_code_sha256"] != inputs.fingerprint()
        or manifest["all_requested_batches_complete"] is not True or manifest["full_source_readiness"] is not False):
        raise QuantError("Use the exact strictly blocked source proof;the new pause rule does not repair source readiness.")
    for name, digest in manifest["source_files"].items():
        safe_file(manifest_path.parent, name, digest)
    vintages = {name:{} for name in inputs.SERIES}
    for batch in manifest["batches"]:
        series, dates = batch["series"], batch["vintage_dates"]
        request = read_json(safe_file(manifest_path.parent, batch["request_path"], batch["request_sha256"]))
        if (request["url"] != f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            or request["advertised_form_fields"] != inputs.request_fields(series, dates)
            or request["http_success"] is not True or request["sha256"] != batch["sha256"]
            or set(vintages[series]).intersection(dates)):
            raise QuantError("A frozen source request changed or duplicated historical activity snapshots.")
        frame = inputs.parse_archive(safe_file(manifest_path.parent, batch["path"], batch["sha256"]), series, dates)
        vintages[series].update({day:frame[day].dropna() for day in dates})
    selected = select_queries(data.close.index, vintages)
    dollars, _ = load_vintages(data.close.index, read_json(ROOT / policy["dollar_policy"]["path"]))
    dollar, dollar_audit = dollar_comparisons(dollars, data.close.index)
    pairs = decision_pairs(data.close.index)
    comparisons = pd.DataFrame({"dollar_change":dollar}, index=dollar.index)
    for series in inputs.SERIES:
        observations = selected.loc[selected["series"].eq(series)].set_index("query_session")["index_value"]
        levels = [float(observations.loc[str(day.date())]) for day in dollar.index]
        changes = [float(observations.loc[str(day.date())] - observations.loc[str(pairs[day].date())])
                   for day in dollar.index]
        comparisons[f"{series}_level"] = levels
        comparisons[f"{series}_change"] = changes
        comparisons[f"{series}_available"] = np.isfinite(levels) & np.isfinite(changes)
    return comparisons, {"activity":selected, "dollar":dollar_audit}


def build_from_inputs(data: MarketData, comparisons: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    original = sector_targets(data, read_json(ROOT / policy["original_control_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    columns = ["dollar_change", *(f"{series}_{suffix}" for series in inputs.SERIES
                                  for suffix in ("level", "change", "available"))]
    if (not isinstance(comparisons, pd.DataFrame) or not comparisons.index.equals(original.dropna(how="all").index)
        or list(comparisons.columns) != columns or not np.isfinite(comparisons["dollar_change"]).all()):
        raise QuantError("Require complete original monthly activity and published-dollar inputs.")
    results = {}
    for candidate in CANDIDATES:
        series = candidate["activity_series"]
        level, change, available = (comparisons[f"{series}_{suffix}"] for suffix in ("level", "change", "available"))
        if (available.dtype != bool or np.isinf(level).any() or np.isinf(change).any()
            or not np.array_equal(np.isfinite(level) & np.isfinite(change), available)
            or comparisons.index[~available].strftime("%Y-%m-%d").tolist() != list(PAUSED)):
            raise QuantError("Only the exact five known monthly pauses are allowed;unknown missing activity is fatal.")
        targets = original.copy()
        assets = targets.columns.drop("BIL")
        for day in comparisons.index:
            if not available.loc[day]:
                targets.loc[day] = np.nan
            elif (comparisons.loc[day, "dollar_change"] > 0 and level.loc[day] < candidate["low_activity_threshold"]
                  and change.loc[day] < 0):
                freed = float(targets.loc[day, assets].sum()) * 0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        active = targets.dropna(how="all")
        if (not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Real-activity confirmation cannot change the funded total,short or borrow cash.")
        results[candidate["id"]] = targets
    return results


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, comparisons, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT / policy["source_registration"]["path"])
    records = [row for row in packet["candidates"] if row["spec"]["id"] == policy["source_candidate"]]
    if len(records) != 1:
        raise QuantError("Require the exact original dollar-only sector-source registration.")
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
            raise QuantError("Activity rules must preserve every exact original fund and risk-free source observation.")
    comparisons, audits = load_inputs(data, policy)
    frames = build_from_inputs(data, comparisons, policy)
    baseline = sector_targets(data, read_json(ROOT / policy["original_control_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new isolated private activity-rule directory without accounts or overwrites.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", baseline.to_csv(float_format="%.17g"))
    write_text_atomic(output / "risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    for name, frame in audits.items():
        write_text_atomic(output / f"{name}-availability.csv", frame.to_csv(index=False) if name == "activity"
                          else frame.to_csv(float_format="%.17g"))
    write_json(output / "source-pauses.json", {
        "candidate_pause_decisions":{candidate["id"]:list(PAUSED) for candidate in CANDIDATES},
        "known_stale_queries":list(KNOWN_STALE),
        "pause_means_no_request_not_cash_or_zero_return":True, "every_formal_return_session_retained":True,
        "strict_source_full_readiness_unchanged_false":True,
    })
    source_manifest_path = ROOT / policy["activity_manifest"]["path"]
    manifest = read_json(source_manifest_path)
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES},
        **{(source_manifest_path.parent / name).relative_to(ROOT).as_posix():digest
           for name, digest in manifest["source_files"].items()},
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
        "src/us_quant/inflation_inputs.py":file_digest(ROOT / "src/us_quant/inflation_inputs.py"),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not rewrite a conflicting old frozen source/helper version.")
        frozen[name] = digest
    readiness = deepcopy(packet["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_monthly_source_decisions":len(baseline.dropna(how="all")),
            "issued_monthly_targets_per_candidate":len(next(iter(frames.values())).dropna(how="all")),
            "known_paused_months":list(PAUSED), "candidate_returns_or_metrics_computed":False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed real-activity confirmations and explicit source pauses.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/activity-factor-guard-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Activity confirmation preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
