from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import gold_volatility_inputs as inputs
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import dollar_comparisons, load_vintages
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/implied-risk-balance.json"
CANDIDATES = (
    {"id":"semiconductor_option_proxy_risk_balance", "dollar_half_guard":False},
    {"id":"semiconductor_option_proxy_risk_balance_dollar_half", "dollar_half_guard":True},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "dollar_policy",
              "dollar_implementation", "option_policy", "option_implementation", "option_readiness", "option_manifest")
EQUITY = (*FACTORS, "SOXX")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"published_gold_equity_option_proxy_budget_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":158, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_candidate":"semiconductor_dollar_half_guard",
        "original_control_candidate":"four_factor_semiconductor_sector50", "original_monthly_decisions":131,
        "budget_proxy_indices":list(inputs.SERIES), "equity_share_min":0.30, "equity_share_max":0.70,
        "growth_share_of_equity":0.50, "cash_reserve":0.02, "risk_off_investment_scale":0.50,
        "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the fixed monthly option-proxy budget,original bounds/composition and paired dollar guard.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def option_months(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["option_policy"]["path"]))
    proof = read_json(ROOT / policy["option_readiness"]["path"])
    manifest = read_json(ROOT / policy["option_manifest"]["path"])
    if (proof["policy"] != policy["option_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["option_manifest"]
        or proof["full_source_readiness"] is not False or proof["monthly_query_availability"] is not True
        or proof["known_missing_gvz_daily_dates"] != list(inputs.MISSING)
        or proof["independent_monthly_close_inputs_checked"] != 262
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["option_policy"]["sha256"]
        or manifest["source_validator_sha256"] != inputs.fingerprint()
        or manifest["full_source_readiness"] is not False or manifest["monthly_query_availability"] is not True):
        raise QuantError("Monthly availability must use actual quotes without repairing the blocked full daily source.")
    for name, digest in manifest["source_files"].items():
        safe_file(ROOT, name, digest)
    histories = {}
    for entry in manifest["histories"]:
        name = entry["series"]
        if name not in inputs.SERIES or name in histories:
            raise QuantError("Only the exact two genuine native option-risk views may enter this budget.")
        at = pd.Timestamp(entry["acquired_at"])
        if at.tzinfo is None:
            raise QuantError("Preserve the actual old and new acquisition timestamps.")
        last = at.tz_convert("America/New_York").normalize().tz_localize(None)
        if str(last.date()) != entry["acquisition_date_limit"]:
            raise QuantError("Do not change genuine source date limits or retime old quotes.")
        histories[name] = inputs.parse_history(safe_file(ROOT, entry["path"], entry["sha256"]), name, last)
    daily = inputs.align_closes(index, histories)
    monthly = inputs.monthly_closes(daily)
    for name, frame in (("daily_closes", daily), ("monthly_closes", monthly)):
        reference = manifest[name]
        saved = pd.read_csv(safe_file(ROOT, reference["path"], reference["sha256"]),
                            index_col=0, parse_dates=True, float_precision="round_trip")
        if not frame.equals(saved):
            raise QuantError("Reparsed native monthly inputs or retained daily source gaps differ from the source proof.")
    return monthly


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    options = option_months(data.close.index, policy)
    raw, _ = load_vintages(data.close.index, read_json(ROOT / policy["dollar_policy"]["path"]))
    dollar, provenance = dollar_comparisons(raw, data.close.index)
    return options, dollar, provenance


def build_from_inputs(data: MarketData, options: pd.DataFrame, dollar: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    original = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    months = original.dropna(how="all").index
    if (not isinstance(options, pd.DataFrame) or not options.index.equals(months)
        or list(options.columns) != list(inputs.SERIES)
        or not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   and not pd.api.types.is_complex_dtype(dtype) for dtype in options.dtypes)
        or not np.isfinite(options.to_numpy()).all() or (options <= 0).any().any()
        or not isinstance(dollar, pd.Series) or not dollar.index.equals(months)
        or not pd.api.types.is_numeric_dtype(dollar) or pd.api.types.is_bool_dtype(dollar)
        or pd.api.types.is_complex_dtype(dollar) or not np.isfinite(dollar.to_numpy()).all()):
        raise QuantError("Require every original monthly native option quote and published dollar comparison.")
    results = {}
    for candidate in CANDIDATES:
        targets = original.copy()
        for day in months:
            gold, equity = float(options.loc[day, "GVZ"]), float(options.loc[day, "VIX"])
            total = gold+equity
            if not np.isfinite(total):
                raise QuantError("Native option-risk proxies cannot overflow into a budget default.")
            share = float(np.clip(gold/total, 0.30, 0.70))
            source_equity = float(original.loc[day, list(EQUITY)].sum())
            if not np.isfinite(source_equity) or source_equity <= 0:
                raise QuantError("Original factor/SOXXrelative proportions must remain actually defined.")
            targets.loc[day, list(EQUITY)] = original.loc[day, list(EQUITY)]/source_equity*share*0.98
            targets.loc[day, "GLD"] = (1-share)*0.98
            if candidate["dollar_half_guard"] and dollar.loc[day] > 0:
                assets = targets.columns.drop("BIL")
                freed = float(targets.loc[day, assets].sum())*0.5
                targets.loc[day, assets] *= 0.5
                targets.loc[day, "BIL"] += freed
        active = targets.dropna(how="all")
        if (not targets.isna().equals(original.isna()) or len(active) != len(months)
            or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
            raise QuantError("Option-proxy allocation cannot change the original monthly clock,funding or return calendar.")
        results[candidate["id"]] = targets
    return results


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    options, dollar, _ = load_inputs(data, policy)
    return build_from_inputs(data, options, dollar, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    packet = read_json(ROOT / policy["source_registration"]["path"])
    records = [entry for entry in packet["candidates"] if entry["spec"]["id"] == policy["source_candidate"]]
    if len(records) != 1:
        raise QuantError("Require the exact original four-style/SOXX/gold and published dollar source registration.")
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
            raise QuantError("Preserve every original actual fund and frozen risk-free observation.")
    options, dollar, provenance = load_inputs(data, policy)
    frames = build_from_inputs(data, options, dollar, policy)
    baseline = sector_targets(data, read_json(ROOT / policy["source_policy"]["path"]))[
        policy["original_control_candidate"]
    ]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private option-budget preparation directory.")
    new_output_directory(output)
    for name, frame in (("option-monthly-closes", options), ("monthly-dollar", dollar),
                        ("dollar-provenance", provenance), ("original-source-targets", baseline)):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    manifest = read_json(ROOT / policy["option_manifest"]["path"])
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES}, **manifest["source_files"],
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py":file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("near_term_volatility_inputs", "calendar", "storage")},
    }
    source_policy = read_json(ROOT / policy["option_policy"]["path"])
    additions.update({source_policy[key]["path"]:source_policy[key]["sha256"] for key in inputs.REFERENCES})
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not rewrite a conflicting original frozen source,policy or helper.")
        frozen[name] = digest
    readiness = deepcopy(packet["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_monthly_targets":len(baseline.dropna(how="all")),
            "actual_monthly_option_inputs":options.size, "full_daily_GVZ_readiness_repaired":False,
            "candidate_returns_or_metrics_computed":False, "new_source_requests":0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed monthly gold/equity option-proxy budgets.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/implied-risk-balance-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Option-proxy risk budget preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
