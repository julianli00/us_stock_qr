from __future__ import annotations

import io
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.activity_inputs import cohort
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.inflation_inputs import request_fields as prior_date_fields
from us_quant.storage import digest_json, file_digest
from us_quant.yield_curve_inputs import archive_metadata

SERIES = {
    "CPATAX": "Corporate Profits After Tax with Inventory Valuation Adjustment (IVA) and Capital Consumption Adjustment (CCAdj)",
    "GDP": "Gross Domestic Product",
}


def request_fields(series: str, dates: list[str]) -> dict:
    if series not in SERIES:
        raise QuantError("Use only the exact adjusted after-tax profit and nominal output series.")
    return {**prior_date_fields(dates), "form[obs_end_date]": "2026-04-01"}


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "quarterly_profit_output_vintage_source_20261012",
        "scope": "historical_quarterly_source_readiness_only", "series": list(SERIES),
        "data_start": "2015-08-10", "as_of": "2026-10-05",
        "source_observation_start": "2014-01-01", "source_observation_end": "2026-04-01",
        "monthly_reference_pairs": 131, "query_sessions_per_series": 225,
        "required_vintage_dates_per_series": 225, "vintage_cutoff": "second_previous_nyse_session",
        "maximum_completed_period_age_days": 186,
        "freshness_basis": "query_date_minus_completed_observation_quarter_end_not_quarter_label_or_download_date",
        "missing_stale_or_unverified_input": "block_source_readiness_preserve_failure_no_fill_or_standby_strategy",
        "maximum_dates_per_request": 40, "maximum_entered_dates_characters": 500,
        "independent_probe_dates": ["2015-08-27", "2021-02-24", "2026-09-28"],
        "new_registered_factor_definitions": 0, "new_strategy_evaluations": 0, "new_registered_candidates": 0,
    }
    definitions = {
        **{name: f"https://fred.stlouisfed.org/series/{name}" for name in SERIES},
        **{f"{name}_form": f"https://alfred.stlouisfed.org/series/downloaddata?seid={name}" for name in SERIES},
    }
    if (not isinstance(policy, dict) or any(policy.get(name) != value for name, value in expected.items())
        or policy.get("source_definitions") != definitions
        or any(policy.get(name) is not False for name in (
            "source_ratios_states_targets_or_returns_computed", "new_independent_factor_families_claimed",
            "existing_W42_or_forward_rules_changed", "investment_objective_verified", "order_authority"))
        or any(not isinstance(policy.get(name), dict) or set(policy[name]) != {"path", "sha256"}
               for name in ("source_index", "research_snapshot", "previous_preservation"))
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Preserve the exact quarterly source identities,186day completed-quarter age and no-portfolio scope.")


def parse_archive(path: Path, series: str, dates: list[str]) -> pd.DataFrame:
    request_fields(series, dates)
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (len(archive.namelist()) != 2 or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())):
                raise QuantError("Use the exact bounded two-member quarterly vintage ZIP.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            metadata = archive_metadata(readme)
            expected = {
                "Title": SERIES[series], "Source": "U.S. Bureau of Economic Analysis",
                "Release": "Gross Domestic Product", "Units": "Billions of Dollars",
                "Frequency": "Quarterly", "Seasonal Adjustment": "Seasonally Adjusted Annual Rate",
            }
            if (re.search(rf"^Series ID:\s*{series}\s*$", readme, re.M) is None
                or "Output Format: Observations by Vintage Date, All Observations" not in readme
                or any(metadata[name] != value for name, value in expected.items())):
                raise QuantError("Scoped metadata must prove exact profit/output identities,native quarterly nominal units and annual rates.")
            declared = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if declared is None or declared[1].split() != dates:
                raise QuantError("Every exact requested quarterly vintage must appear in its primary README.")
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("Quarterly vintages require an actual readable ZIP,not a success-shaped HTML response.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    expected_columns = ["observation_date", *(f"{series}_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected_columns:
        raise QuantError("Quarterly columns must retain the exact native series and declared vintage cohort.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (index.empty or not index.is_unique or not index.is_monotonic_increasing or index.tz is not None
        or not (index.day == 1).all() or not index.month.isin((1, 4, 7, 10)).all()
        or (index < pd.Timestamp("2014-01-01")).any() or (index > pd.Timestamp("2026-04-01")).any()):
        raise QuantError("Profit/output series require distinct chronological quarter-start period labels in the frozen range.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected_columns[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if (not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any()
            or series == "GDP" and (values[known] <= 0).any()):
            raise QuantError("Quarterly snapshots cannot contain invalid/future levels or nonpositive nominal GDP.")
        result[day] = values
    return result


def select(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (not isinstance(observed, pd.Series) or not isinstance(observed.index, pd.DatetimeIndex)
        or not pd.api.types.is_numeric_dtype(observed) or pd.api.types.is_bool_dtype(observed)
        or pd.api.types.is_complex_dtype(observed) or not observed.index.is_unique
        or not observed.index.is_monotonic_increasing or observed.index.tz is not None
        or not (observed.index.day == 1).all() or not observed.index.month.isin((1, 4, 7, 10)).all()
        or np.isinf(observed.to_numpy()).any()
        or not isinstance(query, pd.Timestamp) or query.tz is not None or query != query.normalize()
        or query not in sessions(query.date().isoformat(), query.date().isoformat())
        or cutoff != str(previous_session(previous_session(query)).date())):
        raise QuantError("Select only exact second-prior-NYSE quarterly snapshots on actual query sessions.")
    known = observed.dropna()
    if known.empty or (known.index > pd.Timestamp(cutoff)).any():
        raise QuantError("Missing or future quarterly snapshots cannot use current-history fallback.")
    quarter = known.index[-1]
    completed = quarter+pd.offsets.QuarterEnd(startingMonth=3)
    if completed > pd.Timestamp(cutoff):
        raise QuantError("A first-of-quarter label is not a completed observation or authenticated release date.")
    age = (query-completed).days
    if not 0 <= age <= 186:
        raise QuantError("Completed-quarter freshness exceeds the frozen186day source contract.")
    return float(known.iloc[-1]), {
        "query_session": str(query.date()), "vintage_cutoff": cutoff,
        "observation_quarter": str(quarter.date()), "completed_period_end": str(completed.date()),
        "completed_period_age_calendar_days": int(age),
        "snapshot_available_two_nyse_sessions_before_query": True, "quarter_label_claimed_first_release": False,
    }


def query_readiness(index: pd.DatetimeIndex, vintages: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    queries, dates = cohort(index)
    if set(vintages) != set(SERIES) or any(set(values) != set(dates) for values in vintages.values()):
        raise QuantError("Every one of the450quarterly series/query snapshots is required without substitution.")
    records = []
    for name in SERIES:
        for query in queries:
            cutoff = str(previous_session(previous_session(query)).date())
            observed = vintages[name][cutoff]
            if name == "GDP" and (observed.dropna() <= 0).any():
                raise QuantError("The selected nominal GDPsource cannot contain nonpositive output levels.")
            value, audit = select(observed, query, cutoff)
            records.append({"series": name, **audit, "billions_of_dollars_saar": value})
    return pd.DataFrame(records)


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name: file_digest(directory/name) for name in (
        "profit_inputs.py", "activity_inputs.py", "yield_curve_inputs.py", "inflation_inputs.py",
        "dollar_risk_guard.py", "financial_conditions_guard.py", "macro_factor_tilt.py", "calendar.py", "storage.py")})
