from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import dollar_comparisons, load_vintages
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.volatility_term_risk import SOURCES as OPTION_SOURCE
from us_quant.volatility_term_risk import load_terms

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-term-guard.json"
CONTROL = ROOT / "config/sector-growth-balance.json"
DOLLAR_POLICY = ROOT / "config/dollar-risk-guard.json"
SOURCE = ROOT / "data/sector-term-source-20261011"
PRIOR = ROOT / "evidence/dollar_risk_guard_20261011_registration.json"
OPTION_REGISTRATION = ROOT / "evidence/term_risk_20261010_registration.json"
METHODOLOGY_URL = (
    "https://cdn-api.cboe.com/api/global/us_indices/governance/"
    "Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf"
)
DOC_URLS = {
    "Cboe-history.html": "https://www.cboe.com/tradable-products/vix/vix-historical-data",
    "Cboe-VIX3M.html": "https://www.cboe.com/us/indices/dashboard/VIX3M/",
    "Cboe-term-methodology.pdf": METHODOLOGY_URL,
}
CANDIDATES = [
    {"id": "semiconductor_term_half_guard", "include_known_monthly_dollar_risk": False},
    {"id": "semiconductor_term_or_dollar_half_guard", "include_known_monthly_dollar_risk": True},
]


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "data_scope": "factor_etf_portfolio",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "new_configurations": 2,
        "new_economic_factor_definitions": 0,
        "factor_ids": [
            "price_momentum",
            "value_exposure",
            "quality_exposure",
            "low_volatility_exposure",
        ],
        "factor_symbols": list(FACTORS),
        "original_control_policy": CONTROL.relative_to(ROOT).as_posix(),
        "original_control_candidate": "four_factor_semiconductor_sector50",
        "dollar_source_policy": DOLLAR_POLICY.relative_to(ROOT).as_posix(),
        "option_source_policy": "config/volatility-term-risk.json",
        "option_ratio_risk_threshold": 1.0,
        "risk_off_investment_scale": 0.50,
        "cash_reserve": 0.02,
        "target_event_policy": "completed_month_or_binary_risk_state_change",
        "candidates": CANDIDATES,
    }
    if (
        any(policy.get(key) != value for key, value in constants.items())
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError(
            "The fixed sector term-risk source, threshold, event rule or budgets changed."
        )


def validate_primary_documents(directory: Path) -> dict:
    from pypdf import PdfReader

    records = read_json(safe_file(directory, "acquisition.json"))["sources"]
    if (
        len(records) != 3
        or {row["path"] for row in records} != set(DOC_URLS)
        or any(
            row["url"] != DOC_URLS[row["path"]] or row["http_success"] is not True
            for row in records
        )
    ):
        raise QuantError("Three actual official option-index source documents are required.")
    for row in records:
        safe_file(directory, row["path"], row["sha256"])
    history = (directory / "Cboe-history.html").read_text()
    dashboard = (directory / "Cboe-VIX3M.html").read_text()
    description = re.search(r"CTX\[cSymbol\]\s*=\s*'(.*?)';", dashboard, flags=re.S)
    if (
        "daily closing values" not in history
        or "var symbol = 'VIX3M';" not in dashboard
        or description is None
        or "three-month implied volatility of the S&P 500" not in description[1]
        or "September 18, 2017" not in description[1]
        or "VXV" not in description[1]
        or METHODOLOGY_URL not in description[1]
    ):
        raise QuantError(
            "The scoped current index definition and actual ticker rename are required."
        )
    path = directory / "Cboe-term-methodology.pdf"
    if not path.read_bytes().startswith(b"%PDF-"):
        raise QuantError("The primary methodology must be a real PDF, not a viewer.")
    text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    normalized = " ".join(text.split())
    if (
        "VIX3M 93 days" not in normalized
        or "VIX3M Every 15 seconds RTH Between 9:31 a.m. and 4:15 p.m. ET" not in normalized
        or "Cboe 3-Month Volatility Index VIX3M September, 2009 October, 2013" not in normalized
    ):
        raise QuantError(
            "The actual primary horizon, dissemination and launch entries are required."
        )
    return {
        "current_vix3m_horizon_days": 93,
        "current_regular_calculation_interval": "09:31-16:15America/New_York",
        "current_methodology_launch_entry": "October2013",
        "issuer_disclosed_ticker_rename": {"date": "2017-09-18", "from": "VXV", "to": "VIX3M"},
        "full_historical_methodology_continuity_verified": False,
        "historical_first_publication_instants_verified": False,
    }


def verified_terms(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    validate_policy(policy)
    manifest = read_json(safe_file(SOURCE, "verified-manifest.json"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("source_type") != "actual_official_option_index_closes"
        or manifest.get("source_returns_are_portfolio_returns") is not False
        or manifest.get("strategy_outcomes_computed") is not False
    ):
        raise QuantError("Option risk inputs require the exact sealed pre-outcome source manifest.")
    for name, digest in manifest["files"].items():
        safe_file(SOURCE, name, digest)
    validate_primary_documents(SOURCE)
    previous = read_json(
        safe_file(
            ROOT,
            OPTION_REGISTRATION.relative_to(ROOT).as_posix(),
            manifest["original_option_registration_sha256"],
        )
    )
    frozen = previous["candidates"][0]["spec"]["frozen_files"]
    for name in ("manifest.json", "VIX.csv", "VIX3M.csv"):
        relative = (OPTION_SOURCE / name).relative_to(ROOT).as_posix()
        safe_file(ROOT, relative, frozen[relative])
    return load_terms(OPTION_SOURCE, index)


def load_inputs(data: MarketData, policy: dict) -> tuple:
    terms = verified_terms(data.close.index, policy)
    vintages, _ = load_vintages(data.close.index, read_json(DOLLAR_POLICY))
    dollar, audit = dollar_comparisons(vintages, data.close.index)
    return terms, dollar, audit


def build_from_inputs(
    data: MarketData, terms: pd.DataFrame, dollar: pd.Series, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    monthly = sector_targets(data, read_json(CONTROL))[policy["original_control_candidate"]]
    if (
        not terms.index.equals(data.close.index)
        or list(terms.columns) != ["VIX", "VIX3M"]
        or not np.isfinite(terms.to_numpy()).all()
        or (terms <= 0).any().any()
        or not dollar.index.equals(monthly.dropna(how="all").index)
        or not np.isfinite(dollar).all()
    ):
        raise QuantError("Risk guards require complete official daily and fixed monthly inputs.")
    result = {}
    assets = data.close.columns.drop("BIL")
    for candidate in policy["candidates"]:
        target = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        baseline, previous_state, known_dollar = None, None, None
        for day in data.close.index:
            updated = not monthly.loc[day].isna().all()
            if updated:
                baseline = monthly.loc[day].copy()
                known_dollar = bool(dollar.loc[day] > 0)
            if baseline is None:
                continue
            state = bool(terms.loc[day, "VIX"] >= terms.loc[day, "VIX3M"])
            state = state or (candidate["include_known_monthly_dollar_risk"] and known_dollar)
            if updated or previous_state is None or state != previous_state:
                weights = baseline.copy()
                if state:
                    removed = float(weights.loc[assets].sum()) * 0.5
                    weights.loc[assets] *= 0.5
                    weights["BIL"] += removed
                if (weights < 0).any() or not np.isclose(weights.sum(), 0.98, rtol=0, atol=1e-12):
                    raise QuantError("A timely risk target cannot short, borrow or change funding.")
                target.loc[day] = weights
            previous_state = state
        result[candidate["id"]] = target
    return result


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    terms, dollar, _ = load_inputs(data, policy)
    return build_from_inputs(data, terms, dollar, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR)
    original = prior["candidates"][0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    actual = verified_etf_market(prior["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        if (
            not getattr(data, name).index.equals(getattr(actual, name).index)
            or not getattr(data, name).columns.equals(getattr(actual, name).columns)
            or not np.allclose(getattr(data, name), getattr(actual, name), rtol=1e-10, atol=1e-9)
        ):
            raise QuantError(
                "Term risk research cannot substitute the original actual sector market."
            )
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("Term risk research must retain the original frozen risk-free series.")
    terms, dollar, audit = load_inputs(data, policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Use a separate in-workbench term-risk preparation directory.")
    new_output_directory(output)
    for name, frame in (
        ("option-closes", terms),
        ("monthly-dollar", dollar),
        ("dollar-availability", audit),
    ):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    source = read_json(SOURCE / "verified-manifest.json")
    frozen = {
        **original["frozen_files"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        "src/us_quant/volatility_term_risk.py": file_digest(
            ROOT / "src/us_quant/volatility_term_risk.py"
        ),
        "config/volatility-term-risk.json": file_digest(ROOT / "config/volatility-term-risk.json"),
        OPTION_REGISTRATION.relative_to(ROOT).as_posix(): file_digest(OPTION_REGISTRATION),
        (SOURCE / "verified-manifest.json").relative_to(ROOT).as_posix(): file_digest(
            SOURCE / "verified-manifest.json"
        ),
    }
    frozen.update(
        {
            (SOURCE / name).relative_to(ROOT).as_posix(): digest
            for name, digest in source["files"].items()
        }
    )
    for name in ("manifest.json", "VIX.csv", "VIX3M.csv"):
        frozen[(OPTION_SOURCE / name).relative_to(ROOT).as_posix()] = file_digest(
            OPTION_SOURCE / name
        )
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            **original,
            "id": candidate["id"],
            "configuration": candidate,
            "frozen_files": frozen,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    summary = {
        "complete_daily_option_observations": len(terms),
        "completed_monthly_dollar_queries": len(dollar),
        "missing_required_option_observations": 0,
        "option_forward_fill_performed": False,
        "future_option_closes_used_for_an_earlier_open": False,
        "option_close_first_historical_publication_instants_authenticated": False,
        "all_monthly_dollar_releases_before_their_query": bool(
            (audit["current_release_date"] < audit.index).all()
        ),
        "options_sha256": file_digest(output / "option-closes.csv"),
        "monthly_dollar_sha256": file_digest(output / "monthly-dollar.csv"),
        "dollar_availability_sha256": file_digest(output / "dollar-availability.csv"),
        "pending_targets_are_absolute_and_never_overwritten": True,
        "new_economic_factor_definitions": 0,
        "existing_prospective_profiles_changed": False,
        "strategy_outcomes_computed": False,
    }
    write_json(output / "availability-summary.json", summary)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "availability_audit": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed timely sector-risk guards.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/sector-term-prepared-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {
                    "prepared": [spec["id"] for spec in result["specs"]],
                    "availability_audit": result["availability_audit"],
                    "strategy_outcomes_computed": False,
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Timely sector-risk research blocked: {exc}\n")


if __name__ == "__main__":
    main()
