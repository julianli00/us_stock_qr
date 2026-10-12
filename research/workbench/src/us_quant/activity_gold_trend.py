from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import activity_factor_guard as parent
from us_quant.breadth_gold_trend import gold_states
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic
from us_quant.trend_factor_guard import gated_target, validate_policy as validate_gold_policy

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/activity-gold-trend.json"
CANDIDATES = (
    {"id":"semiconductor_activity_magnitude_gold_trend200", "source_id":"semiconductor_dollar_activity_magnitude_confirmation"},
    {"id":"semiconductor_activity_diffusion_gold_trend200", "source_id":"semiconductor_dollar_activity_diffusion_confirmation"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "prior_gold_registration",
              "prior_gold_policy", "prior_gold_implementation", "prior_gold_states_implementation", "published_risk_context")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"real_activity_prior_monthly_gold_trend_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":152, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "gold_sma_observations":200, "gold_signal_asset":"GLD",
        "decision_clock":"original_informed_activity_source_month_ends_only",
        "gold_warmup_policy":"no_new_target_before_all200observations_keep_original_formal_return_windows",
        "cash_reserve":0.02, "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both original activity sources,strict200gold-only law,warmup,monthlyclock and funding.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    validate_gold_policy(read_json(ROOT / policy["prior_gold_policy"]["path"]))


def build_from_sources(
    data: MarketData, sources: dict[str, pd.DataFrame], policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (not isinstance(sources, dict) or set(sources) != {row["source_id"] for row in CANDIDATES}
        or not {"GLD", "BIL"}.issubset(data.close.columns)):
        raise QuantError("Both original activity targets and actual gold/bill sources must remain explicit.")
    states = gold_states(data.close["GLD"])
    frames, diagnostics = {}, []
    for candidate in CANDIDATES:
        original = sources[candidate["source_id"]]
        if (not isinstance(original, pd.DataFrame) or not original.index.equals(data.close.index)
            or not original.columns.equals(data.close.columns)
            or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()
            or not original.loc[list(parent.PAUSED)].isna().all().all()):
            raise QuantError("Preserve every original source row,column and known macro no-request month.")
        active = original.dropna(how="all")
        if (active.empty or not all(is_month_end(day) for day in active.index)
            or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Original activity targets must remain monthly,complete and cash funded.")
        frame = pd.DataFrame(np.nan, index=original.index, columns=original.columns)
        unchanged = original.columns.difference(["GLD", "BIL"])
        for day in active.index:
            if pd.isna(states.loc[day]):
                if data.close.index.get_loc(day) >= 199:
                    raise QuantError("A fully informed gold state is missing;do not silently enter source standby.")
                diagnostics.append({"candidate_id":candidate["id"], "session":str(day.date()),
                                    "status":"unissued_pre_window_gold200warmup", "gold_active":None})
                continue
            gold_active = bool(states.loc[day])
            target = gated_target(original.loc[day], (True, gold_active))
            if not target.loc[unchanged].equals(original.loc[day, unchanged]):
                raise QuantError("Gold removal cannot change original equity or other nongold targets.")
            frame.loc[day] = target
            diagnostics.append({"candidate_id":candidate["id"], "session":str(day.date()),
                                "status":"original_monthly_target_issued", "gold_active":gold_active,
                                "original_gold_weight":float(original.loc[day, "GLD"]),
                                "removed_gold_weight":0.0 if gold_active else float(original.loc[day, "GLD"])})
        if frame.dropna(how="all").empty:
            raise QuantError("Each source must have informed gold targets before both formal windows.")
        frames[candidate["id"]] = frame
    return frames, diagnostics


def original_sources(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    return parent.build_targets(data, read_json(ROOT / policy["source_policy"]["path"]))


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    return build_from_sources(data, original_sources(data, policy), policy)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    registration = read_json(ROOT / policy["source_registration"]["path"])
    entries = {row["spec"]["id"]:row for row in registration["candidates"]}
    if set(entries) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Use the exact two original reviewed activity confirmations,not replacement source rules.")
    entry = entries[CANDIDATES[0]["source_id"]]
    spec = entry["spec"]
    if any(row["spec"]["market"] != spec["market"] or row["spec"]["frozen_files"] != spec["frozen_files"]
           or row["readiness"] != entry["readiness"] for row in entries.values()):
        raise QuantError("Both activity sources must retain their complete identical market/dependency/readiness context.")
    prior_gold = read_json(ROOT / policy["prior_gold_registration"]["path"])
    prior = next(row["spec"] for row in prior_gold["candidates"]
                 if row["spec"]["id"] == "semiconductor_breadth_half_gold_trend200")
    if prior["market"] != spec["market"]:
        raise QuantError("Reuse the existing gold law only with the exact same original price versions.")
    data = review_market({"market":spec["market"]}, ROOT)
    ref = entry["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    audited = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Preserve every original actual-fund and frozen risk-free observation.")
    sources = original_sources(data, policy)
    frames, diagnostics = build_from_sources(data, sources, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new private activity/gold preparation without changing prior inputs or accounts.")
    new_output_directory(output)
    for identifier, original in sources.items():
        write_text_atomic(output / f"{identifier}-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output / "gold-states.csv", gold_states(data.close["GLD"]).to_csv(float_format="%.17g"))
    write_json(output / "gold-diagnostics.json", {"diagnostics":diagnostics, "known_macro_pause_months":list(parent.PAUSED)})
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    additions = {
        **prior["frozen_files"], **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES},
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not replace conflicting original registered code/source versions.")
        safe_file(ROOT, name, digest)
        frozen[name] = digest
    for candidate in CANDIDATES:
        actual = {**spec, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    counts = {
        candidate["id"]:{
            "original_source_targets":len(sources[candidate["source_id"]].dropna(how="all")),
            "issued_targets":len(frames[candidate["id"]].dropna(how="all")),
            "gold_warmup_unissued_months":[row["session"] for row in diagnostics
                if row["candidate_id"] == candidate["id"] and row["status"] == "unissued_pre_window_gold200warmup"],
        } for candidate in CANDIDATES
    }
    return {"candidate_counts":counts, "known_macro_pause_months":list(parent.PAUSED),
            "candidate_returns_or_metrics_computed":False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed monthly gold-only gates on original activity sources.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/activity-gold-trend-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Activity/gold preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
