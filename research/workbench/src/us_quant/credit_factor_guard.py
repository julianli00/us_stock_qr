from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.adaptive_factor_allocation import target as original_target
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.macro_factor_tilt import align_observations
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
POLICY = ROOT / "config/credit-factor-guard.json"
CONTROL = ROOT / "config/adaptive-factor-allocation.json"
SOURCE = ROOT / "data/credit-spread-source-20261011"
PRIOR = ROOT / "evidence/factor_gold_risk_20261010_registration.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("credit_series") != "BAA10Y"
        or policy.get("credit_level_sessions") != 252
        or policy.get("credit_change_sessions") != 21
        or policy.get("publication_delay_sessions") != 2
        or policy.get("maximum_observation_age_days") != 7
        or policy.get("original_control_policy") != CONTROL.relative_to(ROOT).as_posix()
        or policy.get("original_control_candidate") != "four_factor_static70_gold30"
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "credit_baa_level_guard", "allow_compression_recovery": False},
            {"id": "credit_baa_compression_recovery", "allow_compression_recovery": True},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed credit-risk source, guards or original portfolio changed.")


def load_credit(directory: Path, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    source = read_json(safe_file(directory, "acquisition.json"))
    if (
        source.get("schema_version") != 1
        or source.get("series") != "BAA10Y"
        or source.get("url")
        != "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y&cosd=2014-01-01&coed=2026-10-05"
        or source.get("raw_inputs_not_to_be_published") is not True
        or set(source.get("files", {}))
        != {"BAA10Y.csv", "BAA10Y-source.html", "ICE-source-restriction.html"}
    ):
        raise QuantError("Credit inputs require the exact official source and retained access limits.")
    for name, digest in source["files"].items():
        safe_file(directory, name, digest)
    frame = pd.read_csv(directory / "BAA10Y.csv", na_values=["."])
    if list(frame.columns) != ["observation_date", "BAA10Y"]:
        raise QuantError("The input is not the declared daily Baa/Treasury spread.")
    observed = pd.Series(
        pd.to_numeric(frame["BAA10Y"], errors="raise").to_numpy(),
        index=pd.to_datetime(frame["observation_date"]),
        name="BAA10Y",
    )
    return align_observations(observed, index)


def build_from_inputs(
    data: MarketData, credit: pd.Series, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if (
        set(data.close.columns) != {"SPY", "IEF", "GLD", "BIL", *FACTORS}
        or not credit.index.equals(data.close.index)
        or not np.isfinite(credit).all()
    ):
        raise QuantError("Credit targets need complete aligned observations and the original ETF panel.")
    median = credit.rolling(policy["credit_level_sessions"], min_periods=252).median()
    prior_credit = credit.shift(policy["credit_change_sessions"])
    control_policy = read_json(CONTROL)
    control_candidate = next(
        row
        for row in control_policy["candidates"]
        if row["id"] == policy["original_control_candidate"]
    )
    previous = pd.Series(0.0, index=data.close.columns)
    previous["BIL"] = 0.98
    outputs = {
        row["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for row in policy["candidates"]
    }
    for i, day in enumerate(data.close.index):
        if i < 252 or not is_month_end(day):
            continue
        if pd.isna(median.loc[day]) or pd.isna(prior_credit.loc[day]):
            raise QuantError("A complete pre-decision credit history is required.")
        original = original_target(
            data.close.iloc[: i + 1],
            data.risk_free.iloc[: i + 1],
            previous,
            control_candidate,
            control_policy,
        )
        level_safe = bool(credit.loc[day] < median.loc[day])
        compressing = bool(credit.loc[day] < prior_credit.loc[day])
        for candidate in policy["candidates"]:
            risk_on = level_safe or (candidate["allow_compression_recovery"] and compressing)
            target = original.copy()
            if not risk_on:
                removed = float(target.loc[list(FACTORS)].sum())
                target.loc[list(FACTORS)] = 0.0
                target["BIL"] += removed
            if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
                raise QuantError("Credit defense cannot short assets or borrow cash.")
            outputs[candidate["id"]].loc[day] = target
    return outputs


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    credit, _ = load_credit(SOURCE, data.close.index)
    return build_from_inputs(data, credit, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR)
    actual = verified_etf_market(prior["readiness"], ROOT)
    market = prior["candidates"][0]["spec"]["market"]
    data = review_market({"market": market}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        retained, verified = getattr(data, name), getattr(actual, name)
        if (
            not retained.index.equals(verified.index)
            or not retained.columns.equals(verified.columns)
            or not np.allclose(retained, verified, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("The retained market must match the audited original actual funds.")
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The credit study must retain the original frozen risk-free data.")
    _, provenance = load_credit(SOURCE, data.close.index)
    new_output_directory(output)
    availability = output / "availability.csv"
    write_text_atomic(availability, provenance.to_csv(index_label="decision_session"))
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            CONTROL,
            Path(__file__).with_name("adaptive_factor_allocation.py"),
            Path(__file__).with_name("macro_factor_tilt.py"),
            SOURCE / "acquisition.json",
            SOURCE / "BAA10Y.csv",
            SOURCE / "BAA10Y-source.html",
            SOURCE / "ICE-source-restriction.html",
        )
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    summary = {
        "decision_sessions": len(provenance),
        "maximum_observation_age_days": int(provenance["age_calendar_days"].max()),
        "carried_after_availability_sessions": int(provenance["carried_after_availability"].sum()),
        "all_available_no_later_than_decision": bool(
            (provenance["available_session"] <= provenance.index).all()
        ),
        "availability_sha256": file_digest(availability),
        "raw_or_aligned_credit_values_published": False,
        "current_historical_download_not_vintage_proof": True,
        "old_forward_archive_includes_this_series": False,
    }
    write_json(output / "availability-summary.json", summary)
    return {
        "specs": specs,
        "readiness": readiness,
        "availability_audit": summary,
        "returns_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed, causal credit-risk guards.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/credit-factor-guard-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {
                    "prepared": [row["id"] for row in result["specs"]],
                    "availability_audit": result["availability_audit"],
                    "returns_computed": False,
                }
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Credit study blocked: {exc}\n")


if __name__ == "__main__":
    main()
