from __future__ import annotations

import io
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.inflation_inputs import request_fields as bounded_vintage_fields
from us_quant.storage import digest_json, file_digest

FORM_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=ICSA"
INITIAL_MEMBER = "obs.,_initial_release_only.csv"


def request_fields(dates: list[str], *, initial_release: bool = False) -> dict:
    if not isinstance(dates, list) or any(not isinstance(day, str) for day in dates):
        raise QuantError("Claims vintages must be an explicit list of calendar-date strings.")
    return {
        **bounded_vintage_fields(dates),
        "form[obs_end_date]": "2026-10-03",
        "form[file_type]": "4" if initial_release else "2",
    }


def archive_data(path: Path, dates: list[str], *, initial_release: bool) -> pd.DataFrame:
    request_fields(dates, initial_release=initial_release)
    member = INITIAL_MEMBER if initial_release else f"vintages_starting_{dates[0]}.csv"
    output = (
        "Observations, Initial Release Only"
        if initial_release
        else "Observations by Vintage Date, All Observations"
    )
    try:
        with ZipFile(path) as archive:
            if (
                len(archive.namelist()) != 2
                or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())
            ):
                raise QuantError("Use only the exact bounded two-file claims source archive.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            if not all(
                term in readme
                for term in (
                    "Series ID: ICSA",
                    "Initial Claims",
                    output,
                    "U.S. Employment and Training Administration",
                    "Unemployment Insurance Weekly Claims Report",
                    "Number",
                    "Weekly, Ending Saturday",
                    "Seasonally Adjusted",
                )
            ):
                raise QuantError(
                    "Claims archive identity, units, frequency or output format changed."
                )
            match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if match is None or match[1].split() != dates:
                raise QuantError("Every actual requested claims date must appear in the README.")
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("Claims data must be an actual historical ZIP, not HTTP200 HTML.") from exc
    try:
        return pd.read_csv(
            io.BytesIO(content),
            dtype=str,
            keep_default_na=False,
            na_values=["", "."],
        )
    except (pd.errors.ParserError, UnicodeError) as exc:
        raise QuantError("The original claims CSV must be complete and parseable.") from exc


def weekly_index(values) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise QuantError("Claims observation dates must be actual calendar dates.") from exc
    if (
        index.empty
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or (index.weekday != 5).any()
        or (index < pd.Timestamp("2014-01-01")).any()
        or (index > pd.Timestamp("2026-10-03")).any()
    ):
        raise QuantError("Claims observations require unique chronological Saturday dates.")
    return index


def native_counts(values: pd.Series) -> pd.Series:
    try:
        counts = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise QuantError(
            "Native claims counts must be numeric, not a substituted return series."
        ) from exc
    known = counts.dropna()
    if np.isinf(counts).any() or (known <= 0).any() or (known != np.floor(known)).any():
        raise QuantError("Native claims values must be positive finite integer counts.")
    return counts


def parse_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    frame = archive_data(path, dates, initial_release=False)
    expected = ["observation_date", *(f"ICSA_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected:
        raise QuantError("The claims columns differ from the exact historical snapshot dates.")
    index = weekly_index(frame.pop("observation_date"))
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected[1:], strict=True):
        counts = native_counts(frame[column])
        known = counts.notna().to_numpy()
        if not known.any() or (index[known] > pd.Timestamp(day)).any():
            raise QuantError("An actual claims snapshot cannot contain a future observation.")
        result[day] = counts.to_numpy()
    return result


def parse_initial_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    frame = archive_data(path, dates, initial_release=True)
    if list(frame.columns) != ["period_start_date", "ICSA", "realtime_start_date"]:
        raise QuantError("First-release claims require their original release-date column.")
    index = weekly_index(frame["period_start_date"])
    counts = native_counts(frame["ICSA"])
    try:
        released = pd.DatetimeIndex(pd.to_datetime(frame["realtime_start_date"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise QuantError("Initial claims release dates must be actual calendar dates.") from exc
    if (
        counts.isna().any()
        or released.tz is not None
        or released.hasnans
        or (released <= index).any()
        or not set(released.strftime("%Y-%m-%d")).issubset(dates)
    ):
        raise QuantError(
            "First release must follow observation and belong to an actual requested date."
        )
    return pd.DataFrame(
        {
            "initial_claims": counts.to_numpy(),
            "first_release_date": released,
        },
        index=index,
    )


def required_queries(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if not index.equals(sessions("2015-08-10", "2026-10-05")):
        raise QuantError("Claims readiness retains the exact original 2,805-session market index.")
    pairs = decision_pairs(index)
    queries = sorted(set(pairs) | set(pairs.values()))
    if len(pairs) != 131 or len(queries) != 225:
        raise QuantError("Claims readiness cannot omit any original monthly or reference query.")
    return queries


def needed_vintage_dates(index: pd.DatetimeIndex) -> list[str]:
    return sorted(
        {str(previous_session(previous_session(day)).date()) for day in required_queries(index)}
    )


def inspect_queries(
    index: pd.DatetimeIndex,
    vintages: dict[str, pd.Series],
    initial_releases: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(vintages, dict) or set(vintages) != set(needed_vintage_dates(index)):
        raise QuantError("Every original claims snapshot is required; no current-history fallback.")
    if not isinstance(initial_releases, pd.DataFrame) or list(initial_releases.columns) != [
        "initial_claims",
        "first_release_date",
    ]:
        raise QuantError("Claims readiness requires independently sourced first-release dates.")
    weekly_index(initial_releases.index)
    initial_counts = native_counts(initial_releases["initial_claims"])
    if initial_counts.isna().any():
        raise QuantError("First-release counts cannot be filled or inferred.")
    records = []
    for query in required_queries(index):
        cutoff = previous_session(previous_session(query))
        observed = vintages[str(cutoff.date())]
        if not isinstance(observed, pd.Series):
            raise QuantError("Each claims snapshot must contain its actual dated observations.")
        weekly_index(observed.index)
        counts = native_counts(observed)
        known = counts.dropna()
        if known.empty or (known.index > cutoff).any():
            raise QuantError(
                "Use only claims observations present in their earlier actual snapshot."
            )
        day = known.index[-1]
        if day not in initial_releases.index:
            raise QuantError("The selected claims week lacks an actual first-release observation.")
        first = initial_releases.loc[day]
        try:
            released = pd.Timestamp(first["first_release_date"])
        except (TypeError, ValueError) as exc:
            raise QuantError("The selected first-release date is invalid.") from exc
        age = (query - day).days
        if pd.isna(released) or released.tzinfo is not None or not day < released <= cutoff:
            raise QuantError(
                "Selected claims information is future or not actually released by its cutoff."
            )
        records.append(
            {
                "query_session": str(query.date()),
                "vintage_cutoff": str(cutoff.date()),
                "observation_date": str(day.date()),
                "first_release_date": str(released.date()),
                "age_calendar_days": age,
                "information_delay_sessions": 2,
                "initial_claims": int(known.loc[day]),
                "first_released_initial_claims": int(initial_counts.loc[day]),
                "source_ready": age <= 14,
                "blocker": None if age <= 14 else "observation_age_exceeds_frozen_fourteen_days",
            }
        )
    return pd.DataFrame(records).set_index("query_session")


def query_readiness(
    index: pd.DatetimeIndex,
    vintages: dict[str, pd.Series],
    initial_releases: pd.DataFrame,
) -> pd.DataFrame:
    inspected = inspect_queries(index, vintages, initial_releases)
    blocked = inspected.index[~inspected["source_ready"]].tolist()
    if blocked:
        raise QuantError(
            "Claims readiness blocked by the frozen fourteen-day age at: " + ", ".join(blocked)
        )
    return inspected


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "jobless_claims_input_readiness_20261011",
        "scope": "historical_monthly_source_readiness_only",
        "series": "ICSA",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "source_observation_start": "2014-01-01",
        "source_observation_end": "2026-10-03",
        "native_units": "Number",
        "native_frequency": "Weekly, Ending Saturday",
        "seasonal_adjustment": "Seasonally Adjusted",
        "required_vintage_dates": 225,
        "monthly_comparison_grid": 131,
        "information_delay_sessions": 2,
        "maximum_observation_age_days": 14,
        "vintage_cutoff": "second_previous_nyse_session",
        "initial_release_dates": (
            "advertised_releases_after_selected_observation_through_snapshot_cutoff"
        ),
        "first_release_date_validation_required": True,
        "form_url": FORM_URL,
        "maximum_dates_per_request": 40,
        "maximum_entered_dates_characters": 500,
        "new_registered_factor_definitions": 0,
        "new_strategy_evaluations": 0,
        "new_factor_family_independence_claimed": False,
        "strategy_states_or_returns_computed": False,
        "forward_rules_changed": False,
        "existing_queued_proposals_replaced": False,
        "order_authority": False,
    }
    if any(
        policy.get(key) != value or (type(value) is bool and policy.get(key) is not value)
        for key, value in expected.items()
    ):
        raise QuantError(
            "The source-only claims scope, information lag or no-performance boundary changed."
        )


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json(
        {
            name: file_digest(folder / name)
            for name in (
                "jobless_claims_inputs.py",
                "inflation_inputs.py",
                "calendar.py",
                "dollar_risk_guard.py",
                "storage.py",
            )
        }
    )
