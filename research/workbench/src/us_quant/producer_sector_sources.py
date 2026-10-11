from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import sector_basket_sources as sectors
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import digest_json, file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/producer-sector-sources.json"
ADDITIONAL = ("XLE", "XLB")
ORIGINAL = ("SPY", "IEF", "GLD", "BIL", *FACTORS, "TLT", "QQQ", "XLK", "SOXX")
REFERENCES = ("source_registration", "sector_policy", "sector_implementation", "sector_readiness", "sector_manifest")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"existing_producer_sector_price_preparation_20261011",
        "scope":"source_only_original_portfolio_plus_existing_energy_materials_funds",
        "data_start":"2015-08-10", "as_of":"2026-10-05",
        "original_source_id":"semiconductor_breadth_half_gold_relative",
        "additional_assets":list(ADDITIONAL),
        "asset_roles":{"XLE":"energy_producer_equity_not_direct_commodity_futures",
                       "XLB":"materials_sector_equity_not_direct_commodity_futures"},
        "new_registered_candidates":0, "new_strategy_evaluations":0, "new_economic_factor_definitions":0,
        "portfolio_targets_or_returns_computed":False, "existing_W42_queue_changed":False,
        "order_authority":False, "investment_objective_verified":False,
    }
    contract = policy.get("source_contract",{}) if isinstance(policy,dict) else {}
    limits = policy.get("limits",{}) if isinstance(policy,dict) else {}
    future = policy.get("conceptual_future_research",{}) if isinstance(policy,dict) else {}
    if (not isinstance(policy,dict) or any(policy.get(key) != value for key,value in constants.items())
        or any(not isinstance(policy.get(key),dict) or set(policy[key]) != {"path","sha256"} for key in REFERENCES)
        or not isinstance(contract,dict) or not isinstance(limits,dict) or not isinstance(future,dict)
        or any(contract.get(key) is not True for key in (
            "only_existing_registered_raw_csv_and_primary_documents",
            "complete_original12asset_panels_and_frozen_risk_free_preserved",
            "complete2805post_inception_observations_per_additional_fund_required",
            "original_raw_csv_corporate_action_and_issuer_pdf_guards_unchanged",
            "both_original_signal_and_unused_portfolio_xlk_versions_preserved","no_price_source_retrieval_time_rewrite"))
        or contract.get("new_quote_requests") != 0 or contract.get("access_or_rate_limit_bypass") is not False
        or any(limits.get(key) is not False for key in (
            "energy_materials_funds_are_guaranteed_inflation_hedges",
            "sector_funds_are_new_independent_economic_factor_families",
            "current_issuer_identity_is_full_historical_constituent_mandate_notional_proof",
            "source_readiness_is_strategy_or_investment_qualification"))
        or limits.get("research_program_admission_not_implemented_by_this_source_study") is not True
        or future.get("rules_not_prepared_registered_or_evaluated") is not True
        or future.get("candidate_count") != 2 or future.get("existing_standalone_sector_nav_cap") != 0.40
        or future.get("cash_origin") != "only_original_gold_trend_removed_budget_not_equity_standby_cash"
        or future.get("methods") != ["existing126positive_excess_vs_bil_winner",
                                     "existing126positive_excess_and63inverse_volatility"]):
        raise QuantError("Keep this source-only producer-stock scope,no newquote/trial/family/guarantee orhistory rewrite.")


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory/name) for name in (
        "producer_sector_sources.py","sector_basket_sources.py","sector_growth_balance.py",
        "multifactor_stability.py","factor_research.py","dual_horizon.py","data.py","calendar.py","storage.py")})


def extend_market(original: MarketData, sector: MarketData) -> MarketData:
    original.validate()
    sector.validate()
    if (tuple(original.close.columns) != ORIGINAL or not set(ADDITIONAL).issubset(sector.close)
        or not original.close.index.equals(sector.close.index)
        or not np.allclose(original.risk_free,sector.risk_free,rtol=0,atol=1e-12)):
        raise QuantError("Keep the complete original portfolio shape/calendar and frozen risk-free source.")
    shared = ["SPY","IEF","GLD","BIL",*FACTORS]
    if not set(shared).issubset(sector.close):
        raise QuantError("All original eight core observations must remain independently checked.")
    for name in ("open","close","raw_close","volume"):
        if not np.allclose(getattr(original,name).loc[:,shared],getattr(sector,name).loc[:,shared],
                           rtol=1e-10,atol=1e-9):
            raise QuantError("A shared original core source observation changed.")

    def panel(name: str) -> pd.DataFrame:
        result = getattr(original,name).copy()
        for symbol in ADDITIONAL:
            result[symbol] = getattr(sector,name)[symbol]
        if not result.loc[:,list(ORIGINAL)].equals(getattr(original,name)):
            raise QuantError("Do not rewrite original prices,including the separate unused portfolio XLK version.")
        return result

    result = MarketData(panel("open"),panel("close"),panel("raw_close"),panel("volume"),original.risk_free.copy())
    result.validate()
    return result


def verified_market(policy: dict) -> tuple[MarketData,dict]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT,policy[key]["path"],policy[key]["sha256"])
    registration = read_json(ROOT / policy["source_registration"]["path"])
    records = [row for row in registration["candidates"] if row["spec"]["id"] == policy["original_source_id"]]
    if len(records) != 1:
        raise QuantError("Require the exact original twelve-asset source registration.")
    entry = records[0]
    for name,digest in entry["spec"]["frozen_files"].items():
        safe_file(ROOT,name,digest)
    original = review_market({"market":entry["spec"]["market"]},ROOT)
    ref = entry["readiness"]
    verified = verified_etf_market(read_json(safe_file(ROOT,ref["path"],ref["sha256"])),ROOT)
    for name in ("open","close","raw_close","volume","risk_free"):
        left,right = getattr(original,name),getattr(verified,name)
        if (not left.index.equals(right.index)
            or isinstance(left,pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left,right,rtol=1e-10,atol=1e-9)):
            raise QuantError("Original portfolio observations differ from its exact audited adapter.")
    sector = sectors.verified_market(read_json(ROOT / policy["sector_policy"]["path"]))
    data = extend_market(original,sector)
    if len(data.close) != 2805 or len(data.close.columns) != 14:
        raise QuantError("Require complete original2805sessions for exactly fourteen actual funds.")
    manifest = read_json(ROOT / policy["sector_manifest"]["path"])
    identities = {symbol:manifest["funds"][symbol]["identity"] for symbol in ADDITIONAL}
    if (len({row["cusip"] for row in identities.values()}) != 2
        or any(row["ordinary_sector_index_fund_objective"] is not True
               or row["intended_equity_index_return_multiple"] != 1.0 for row in identities.values())):
        raise QuantError("Require two distinct primary-identified ordinary producer-sector funds.")
    return data,{
        "original_market":entry["spec"]["market"],"additional_fund_identities":identities,
        "original_sector_quote_acquired_at":manifest["original_cache_verification"]["original_supplement_acquired_at"],
        "original_supplement_fund_checks":{symbol:manifest["original_cache_verification"]["fund_checks"][symbol]
                                         for symbol in ADDITIONAL},
        "new_quote_requests":0,"portfolio_targets_or_returns_computed":False,
    }


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    data,context = verified_market(policy)
    output = local_path(ROOT,output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new private source-only producer-sector directory.")
    new_output_directory(output)
    market = {}
    for name in ("open","close","raw_close","volume","risk_free"):
        value = getattr(data,name)
        value = value.to_frame() if isinstance(value,pd.Series) else value
        path = output/f"market-{name}.csv"
        write_text_atomic(path,value.to_csv(float_format="%.17g"))
        market[name] = {"path":path.relative_to(ROOT).as_posix(),"sha256":file_digest(path)}
    record = {
        "schema_version":1,"recorded_at":utc_now(),"source_policy_sha256":file_digest(POLICY),
        "source_validator_sha256":fingerprint(),"market":market,"context":context,
        "original_assets":list(ORIGINAL),"additional_assets":list(ADDITIONAL),
        "prepared_assets":14,"complete_price_sessions_per_asset":2805,
        "ready_for_separate_rule_preparation":True,
        "research_program_14asset_admission_implemented_by_this_study":False,
        "new_registered_candidates":0,"new_strategy_evaluations":0,"new_economic_factor_definitions":0,
        "portfolio_targets_or_returns_computed":False,"W42_or_forward_rules_changed":False,
        "investment_objective_verified":False,"order_authority":False,
    }
    write_json(output / "manifest.json",record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify existing producer-sector sources without portfolio targets or returns.")
    parser.add_argument("action",choices=("prepare",))
    parser.add_argument("--prepared",type=Path,default=ROOT / "data/producer-sector-source-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({"prepared_assets":result["prepared_assets"],"sessions":2805,
                          "new_quote_requests":0,"portfolio_targets_or_returns_computed":False,
                          "research_program_admission_implemented_by_this_study":False},indent=2))
    except (QuantError,OSError,ValueError) as exc:
        parser.exit(2,f"Producer-sector source preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
