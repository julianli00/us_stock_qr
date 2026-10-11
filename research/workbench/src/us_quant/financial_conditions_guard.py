from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs, dollar_comparisons, load_vintages
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

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/financial-conditions-guard.json"
CONTROL = ROOT / "config/sector-growth-balance.json"
DOLLAR_POLICY = ROOT / "config/dollar-risk-guard.json"
PRIOR = ROOT / "evidence/dollar_risk_guard_20261011_registration.json"
SOURCE = ROOT / "data/financial-conditions-vintages-20261011"
SOURCE_MANIFEST = "verified-manifest-v2.json"
CANDIDATES = [
    {"id": "semiconductor_dollar_nfci_tight_guard", "confirmation": "tighter_than_average"},
    {
        "id": "semiconductor_dollar_nfci_deterioration_guard",
        "confirmation": "increasing_as_published_conditions",
    },
]
FORM = {
    "url": "https://alfred.stlouisfed.org/series/downloaddata?seid=NFCI",
    "method": "POST",
    "units": "lin",
    "file_type": "2",
    "file_format": "csv",
    "maximum_entered_dates_characters": 500,
    "maximum_dates_per_request": 40,
}


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "data_scope": "factor_etf_portfolio",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "new_configurations": 2,
        "new_economic_factor_definitions": 0,
        "factor_symbols": list(FACTORS),
        "factor_ids": [
            "price_momentum",
            "value_exposure",
            "quality_exposure",
            "low_volatility_exposure",
        ],
        "original_control_policy": CONTROL.relative_to(ROOT).as_posix(),
        "original_control_candidate": "four_factor_semiconductor_sector50",
        "dollar_source_policy": DOLLAR_POLICY.relative_to(ROOT).as_posix(),
        "comparison_candidate": "semiconductor_dollar_half_guard",
        "financial_conditions_series": "NFCI",
        "vintage_cutoff": "previous_nyse_session",
        "observation_source_start": "2014-01-01",
        "observation_source_end": "2026-10-02",
        "maximum_observation_age_days": 14,
        "stale_input_behavior": "pause_rebalance_and_record_gap",
        "risk_change_sessions": 63,
        "tight_conditions_threshold": 0.0,
        "risk_off_investment_scale": 0.50,
        "cash_reserve": 0.02,
        "public_form": FORM,
        "candidates": CANDIDATES,
    }
    if (
        any(policy.get(key) != value for key, value in constants.items())
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError(
            "The fixed NFCI confirmations, historical vintages or risk budgets changed."
        )


def needed_vintage_dates(index: pd.DatetimeIndex) -> list[str]:
    pairs = decision_pairs(index)
    return sorted({str(previous_session(day).date()) for day in set(pairs) | set(pairs.values())})


def parse_vintage_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    if (
        not dates
        or dates != sorted(set(dates))
        or len(dates) > 40
        or len(" ".join(dates)) > 500
        or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in dates)
    ):
        raise QuantError(
            "The vintage batch must respect the advertised finite date-entry controls."
        )
    try:
        with ZipFile(path) as archive:
            expected = {"README.txt", f"vintages_starting_{dates[0]}.csv"}
            if set(archive.namelist()) != expected or any(
                entry.file_size > 5_000_000 for entry in archive.infolist()
            ):
                raise QuantError("The source is not the expected bounded ALFRED two-file archive.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            if (
                "Series ID: NFCI" not in readme
                or "Output Format: Observations by Vintage Date, All Observations" not in readme
                or "Federal Reserve Bank of Chicago" not in readme
                or "Weekly, Ending Friday" not in readme
            ):
                raise QuantError(
                    "The source archive is not the actual plain-index NFCI vintage export."
                )
            match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if match is None or match[1].split() != dates:
                raise QuantError(
                    "The archive README does not confirm every requested historical vintage."
                )
            content = archive.read(f"vintages_starting_{dates[0]}.csv")
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError(
            "An ALFRED vintage download must be an actual readable ZIP, not HTML."
        ) from exc
    frame = pd.read_csv(io.BytesIO(content), na_values=["."], dtype=str)
    names = ["observation_date", *(f"NFCI_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != names:
        raise QuantError("The actual NFCI columns do not encode exactly the requested as-of dates.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (
        index.empty
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or not (index.weekday == 4).all()
    ):
        raise QuantError("NFCI observations must retain unique chronological weekly Friday dates.")
    result = pd.DataFrame(index=index)
    for day, name in zip(dates, names[1:], strict=True):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy()
        known = np.isfinite(values)
        if not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any():
            raise QuantError(
                "A vintage has missing history, nonfinite values or future observations."
            )
        result[day] = values
    return result


def load_conditions(index: pd.DatetimeIndex, policy: dict) -> tuple[pd.Series, pd.DataFrame]:
    validate_policy(policy)
    manifest = read_json(safe_file(SOURCE, SOURCE_MANIFEST))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("series") != "NFCI"
        or manifest.get("current_histories_used_for_targets") is not False
        or manifest.get("raw_inputs_not_to_be_published") is not True
        or manifest.get("strategy_outcomes_computed") is not False
        or manifest.get("stale_input_behavior") != "pause_rebalance_and_record_gap"
    ):
        raise QuantError("NFCI inputs require the sealed pre-outcome actual vintage manifest.")
    for name, digest in manifest["files"].items():
        safe_file(SOURCE, name, digest)
    values, all_dates = {}, set()
    for batch in manifest["batches"]:
        dates = batch["vintage_dates"]
        if all_dates.intersection(dates):
            raise QuantError("A vintage is duplicated across source batches.")
        request = read_json(safe_file(SOURCE, batch["request_path"], batch["request_sha256"]))
        expected_fields = {
            "form[units]": "lin",
            "form[obs_start_date]": "2014-01-01",
            "form[obs_end_date]": "2026-10-02",
            "form[entered_vintage_dates]": " ".join(dates),
            "form[file_type]": "2",
            "form[file_format]": "csv",
            "form[download_data]": "",
        }
        if (
            request["url"] != FORM["url"]
            or request["method"] != "POST"
            or request["advertised_form_fields"] != expected_fields
            or request["http_success"] is not True
            or request["zip_sha256"] != batch["sha256"]
        ):
            raise QuantError(
                "The actual public-form request differs from the frozen source contract."
            )
        frame = parse_vintage_archive(safe_file(SOURCE, batch["path"], batch["sha256"]), dates)
        for day in dates:
            values[day] = frame[day].dropna()
        all_dates.update(dates)
    required = needed_vintage_dates(index)
    if set(required) != all_dates:
        raise QuantError(
            "The complete exact requested vintage cohort is required, without substitution."
        )
    pairs = decision_pairs(index)
    queries = sorted(set(pairs) | set(pairs.values()))
    aligned, provenance = {}, []
    for day in queries:
        cutoff = str(previous_session(day).date())
        observed = values[cutoff]
        latest = observed.index[-1]
        age = (day - latest).days
        if age < 0:
            raise QuantError("A historical NFCI query cannot contain a future observation.")
        fresh = age <= 14
        aligned[day] = float(observed.iloc[-1]) if fresh else np.nan
        provenance.append(
            {
                "query_session": day,
                "vintage_cutoff": cutoff,
                "observation_date": latest,
                "age_calendar_days": age,
                "fresh_required_reading": fresh,
                "standby_reason": None if fresh else "nfci_vintage_exceeds_frozen_14day_age",
                "available_on_a_strictly_earlier_session": pd.Timestamp(cutoff) < day,
            }
        )
    return pd.Series(aligned, name="NFCI"), pd.DataFrame(provenance).set_index("query_session")


def load_inputs(index: pd.DatetimeIndex, policy: dict) -> tuple[pd.DataFrame, dict]:
    conditions, provenance = load_conditions(index, policy)
    vintages, _ = load_vintages(index, read_json(DOLLAR_POLICY))
    dollar, dollar_audit = dollar_comparisons(vintages, index)
    pairs = decision_pairs(index)
    comparisons = pd.DataFrame(
        {
            "dollar_change": dollar,
            "conditions_level": conditions.loc[dollar.index],
            "conditions_change": [
                float(conditions.loc[day] - conditions.loc[pairs[day]]) for day in dollar.index
            ],
        },
        index=dollar.index,
    )
    comparisons["level_available"] = comparisons["conditions_level"].notna()
    comparisons["change_available"] = comparisons["conditions_change"].notna()
    if not np.isfinite(comparisons["dollar_change"]).all():
        raise QuantError("The original dollar risk input must be finite at every decision.")
    return comparisons, {"nfci": provenance, "dollar": dollar_audit}


def build_from_inputs(
    data: MarketData, comparisons: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    original = sector_targets(data, read_json(CONTROL))[policy["original_control_candidate"]]
    if (
        not comparisons.index.equals(original.dropna(how="all").index)
        or set(comparisons.columns)
        != {
            "dollar_change",
            "conditions_level",
            "conditions_change",
            "level_available",
            "change_available",
        }
        or not np.isfinite(comparisons["dollar_change"]).all()
    ):
        raise QuantError(
            "NFCI confirmations need complete risk inputs at every original month-end."
        )
    for field, available in (
        ("conditions_level", "level_available"),
        ("conditions_change", "change_available"),
    ):
        if (
            comparisons[available].dtype != bool
            or np.isinf(comparisons[field]).any()
            or not np.array_equal(np.isfinite(comparisons[field]), comparisons[available])
        ):
            raise QuantError("NFCI missingness must match its explicit source availability state.")
    if (comparisons["change_available"] & ~comparisons["level_available"]).any():
        raise QuantError("A change cannot be available without its required current reading.")
    assets = original.columns.drop("BIL")
    result = {}
    for candidate in policy["candidates"]:
        target = original.copy()
        for day in comparisons.index:
            available = (
                "level_available"
                if candidate["confirmation"] == "tighter_than_average"
                else "change_available"
            )
            if not comparisons.loc[day, available]:
                target.loc[day] = np.nan
                continue
            confirmed = (
                comparisons.loc[day, "conditions_level"] > 0
                if candidate["confirmation"] == "tighter_than_average"
                else comparisons.loc[day, "conditions_change"] > 0
            )
            if comparisons.loc[day, "dollar_change"] > 0 and confirmed:
                removed = float(target.loc[day, assets].sum()) * 0.5
                target.loc[day, assets] *= 0.5
                target.loc[day, "BIL"] += removed
        active = target.dropna(how="all")
        if (active < 0).any().any() or not np.allclose(
            active.sum(axis=1), 0.98, rtol=0, atol=1e-12
        ):
            raise QuantError(
                "NFCI defense cannot borrow, short or change the original target budget."
            )
        result[candidate["id"]] = target
    return result


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data.close.index, policy)
    return build_from_inputs(data, comparisons, policy)


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
            raise QuantError("The NFCI study cannot substitute the actual sector/factor market.")
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The NFCI study must preserve the original frozen risk-free series.")
    comparisons, audits = load_inputs(data.close.index, policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Use a separate in-workbench preparation directory.")
    new_output_directory(output)
    write_text_atomic(output / "risk-comparisons.csv", comparisons.to_csv(float_format="%.17g"))
    for name, audit in audits.items():
        write_text_atomic(output / f"{name}-availability.csv", audit.to_csv())
    pauses = {
        candidate["id"]: [
            str(day.date())
            for day in comparisons.index[
                ~comparisons[
                    "level_available"
                    if candidate["confirmation"] == "tighter_than_average"
                    else "change_available"
                ]
            ]
        ]
        for candidate in policy["candidates"]
    }
    write_json(
        output / "source-pauses.json",
        {
            "candidate_pause_decisions": pauses,
            "reason": "required_nfci_vintage_exceeds_frozen_14day_age",
            "pause_means_no_rebalance_not_zero_return_or_cash_fallback": True,
            "all_actual_mark_to_market_sessions_retained": True,
        },
    )
    manifest = read_json(SOURCE / SOURCE_MANIFEST)
    frozen = {
        **original["frozen_files"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        PRIOR.relative_to(ROOT).as_posix(): file_digest(PRIOR),
        (SOURCE / SOURCE_MANIFEST).relative_to(ROOT).as_posix(): file_digest(
            SOURCE / SOURCE_MANIFEST
        ),
    }
    frozen.update(
        {
            (SOURCE / name).relative_to(ROOT).as_posix(): digest
            for name, digest in manifest["files"].items()
        }
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
        "actual_monthly_decisions": len(comparisons),
        "actual_nfci_vintage_dates": len(needed_vintage_dates(data.close.index)),
        "actual_nfci_query_sessions": len(audits["nfci"]),
        "maximum_nfci_observation_age_days": int(audits["nfci"]["age_calendar_days"].max()),
        "maximum_consumed_nfci_observation_age_days": int(
            audits["nfci"].loc[audits["nfci"]["fresh_required_reading"], "age_calendar_days"].max()
        ),
        "stale_query_sessions": [
            str(day.date())
            for day in audits["nfci"].index[~audits["nfci"]["fresh_required_reading"]]
        ],
        "candidate_pause_decisions": pauses,
        "source_pause_log_sha256": file_digest(output / "source-pauses.json"),
        "all_vintage_cutoffs_strictly_before_queries": bool(
            audits["nfci"]["available_on_a_strictly_earlier_session"].all()
        ),
        "original_daily_equity_window_cutoff": policy["as_of"],
        "weekly_source_observation_end": policy["observation_source_end"],
        "current_histories_used_for_targets": False,
        "early_publication_rounding_preserved": True,
        "source_snapshot_is_not_authenticated_first_publication_time": True,
        "risk_comparisons_sha256": file_digest(output / "risk-comparisons.csv"),
        "nfci_availability_sha256": file_digest(output / "nfci-availability.csv"),
        "strategy_outcomes_computed": False,
        "new_factor_definition_count": 0,
        "existing_prospective_profiles_changed": False,
        "order_authority": False,
    }
    write_json(output / "availability-summary.json", summary)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "availability_audit": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare two fixed vintage-aware NFCI confirmations."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/financial-conditions-prepared-20261011"
    )
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
    except (QuantError, ValueError) as exc:
        parser.exit(2, f"NFCI research blocked: {exc}\n")


if __name__ == "__main__":
    main()
