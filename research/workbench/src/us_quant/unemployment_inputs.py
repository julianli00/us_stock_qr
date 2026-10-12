from __future__ import annotations

import io
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.activity_inputs import cohort
from us_quant.activity_inputs import select as select_monthly
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.inflation_inputs import request_fields as vintage_fields
from us_quant.storage import digest_json, file_digest
from us_quant.yield_curve_inputs import archive_metadata

SERIES = "UNRATE"
OBSERVATION_END = "2026-09-01"
HISTORY_MONTHS = 15
PRIMARY_DEFINITIONS = {
    "UNRATE": "https://fred.stlouisfed.org/series/UNRATE",
    "UNRATE_form": "https://alfred.stlouisfed.org/series/downloaddata?seid=UNRATE",
    "Sahm_realtime_definition": "https://fred.stlouisfed.org/series/SAHMREALTIME",
    "Sahm_current_definition": "https://fred.stlouisfed.org/series/SAHMCURRENT",
    "original_policy_proposal": "https://www.hamiltonproject.org/publication/policy-proposal/direct-stimulus-payments-to-individuals/",
}


def request_fields(dates: list[str]) -> dict:
    return {**vintage_fields(dates), "form[obs_end_date]": OBSERVATION_END}


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "unemployment_vintage_source_20261012",
        "scope": "historical_monthly_source_readiness_only", "series": SERIES,
        "data_start": "2015-08-10", "as_of": "2026-10-05",
        "source_observation_start": "2014-01-01", "source_observation_end": OBSERVATION_END,
        "monthly_reference_pairs": 131, "query_sessions": 225, "required_vintage_dates": 225,
        "vintage_cutoff": "second_previous_nyse_session", "maximum_completed_period_age_days": 62,
        "required_consecutive_months_per_query": HISTORY_MONTHS,
        "freshness_basis": "query_date_minus_latest_completed_observation_month_end",
        "missing_stale_or_unverified_input": "block_source_readiness_preserve_failure_no_fill_or_standby_strategy",
        "maximum_dates_per_request": 40, "maximum_entered_dates_characters": 500,
        "independent_probe_dates": ["2015-08-27", "2021-02-24", "2026-09-28"],
        "new_registered_factor_definitions": 0, "new_registered_candidates": 0,
        "new_strategy_evaluations": 0, "primary_definitions": PRIMARY_DEFINITIONS,
    }
    if (
        not isinstance(policy, dict)
        or any(policy.get(key) != value for key, value in expected.items())
        or any(policy.get(key) is not False for key in (
            "source_statistics_states_targets_or_returns_computed",
            "new_independent_factor_families_claimed", "existing_W42_or_forward_rules_changed",
            "investment_objective_verified", "order_authority",
        ))
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in ("source_index", "research_snapshot", "previous_preservation")
        )
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False
    ):
        raise QuantError("Preserve the frozen UNRATE vintage/history/freshness and source-only boundary.")


def parse_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    request_fields(dates)
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (
                len(archive.namelist()) != 2
                or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())
            ):
                raise QuantError("Use the exact bounded two-member unemployment vintage ZIP.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            metadata = archive_metadata(readme)
            titles = []
            if dates[0] <= "2019-09-05":
                titles.append("Civilian Unemployment Rate")
                if re.search(r"^Civilian Unemployment Rate\s+1960-03-15\s+2019-09-05\s*$", readme, re.M) is None:
                    raise QuantError("Require the exact original UNRATE title interval for its early vintages.")
            if dates[-1] >= "2019-09-06":
                titles.append("Unemployment Rate")
                if re.search(r"^Unemployment Rate\s+2019-09-06\s+Current\s*$", readme, re.M) is None:
                    raise QuantError("Require the exact current UNRATE title interval for its later vintages.")
            expected = {
                "Title": " ".join(titles),
                "Source": "U.S. Bureau of Labor Statistics",
                "Release": "Employment Situation", "Units": "Percent", "Frequency": "Monthly",
                "Seasonal Adjustment": "Seasonally Adjusted",
            }
            declared = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if (
                re.search(r"^Series ID:\s*UNRATE\s*$", readme, re.M) is None
                or "Output Format: Observations by Vintage Date, All Observations" not in readme
                or any(metadata[key] != value for key, value in expected.items())
                or "LNS14000000" not in metadata["Notes"]
                or declared is None or declared[1].split() != dates
            ):
                raise QuantError("Exact native UNRATE identity,units and requested vintages must match.")
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("Unemployment vintages require a readable ZIP,not a success-shaped HTML response.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    columns = ["observation_date", *(f"UNRATE_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != columns:
        raise QuantError("Do not substitute another labor series or current-history columns.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (
        index.empty or not index.is_unique or not index.is_monotonic_increasing
        or index.tz is not None or not (index.day == 1).all()
        or (index < pd.Timestamp("2014-01-01")).any()
        or (index > pd.Timestamp(OBSERVATION_END)).any()
    ):
        raise QuantError("Require distinct chronological month-start observation periods in the frozen range.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, columns[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if (
            not known.any() or np.isinf(values).any()
            or ((values[known] < 0) | (values[known] > 100)).any()
            or (index[known] + pd.offsets.MonthEnd(0) > pd.Timestamp(day)).any()
        ):
            raise QuantError("Unemployment percent values must be finite,in range and from completed months.")
        result[day] = values
    return result


def select(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (
        not isinstance(observed, pd.Series)
        or not pd.api.types.is_numeric_dtype(observed)
        or pd.api.types.is_bool_dtype(observed)
        or pd.api.types.is_complex_dtype(observed)
        or ((observed.dropna() < 0) | (observed.dropna() > 100)).any()
    ):
        raise QuantError("Use actual numeric unemployment percent observations,not a substituted risk state.")
    value, audit = select_monthly(observed, query, cutoff)
    last = pd.Timestamp(audit["observation_month"])
    required = pd.date_range(end=last, periods=HISTORY_MONTHS, freq="MS")
    history = observed.reindex(required)
    if not np.isfinite(history.to_numpy()).all():
        raise QuantError("All15consecutive monthly values must exist in the same original vintage;no fill or shortening.")
    return value, {
        **audit,
        "consecutive_history_months": HISTORY_MONTHS,
        "history_start_month": str(required[0].date()),
        "rolling_means_or_recession_state_computed": False,
    }


def query_readiness(index: pd.DatetimeIndex, vintages: dict[str, pd.Series]) -> pd.DataFrame:
    queries, dates = cohort(index)
    if set(vintages) != set(dates):
        raise QuantError("Keep all225actual unemployment snapshots without a smaller fallback cohort.")
    rows = []
    for query in queries:
        cutoff = str(previous_session(previous_session(query)).date())
        value, audit = select(vintages[cutoff], query, cutoff)
        rows.append({"series": SERIES, **audit, "unemployment_percent": value})
    return pd.DataFrame(rows)


def fingerprint() -> str:
    folder = Path(__file__).parent
    return digest_json({name: file_digest(folder / name) for name in (
        "unemployment_inputs.py", "activity_inputs.py", "yield_curve_inputs.py",
        "inflation_inputs.py", "dollar_risk_guard.py", "financial_conditions_guard.py",
        "macro_factor_tilt.py", "calendar.py", "storage.py", "config.py",
    )})
